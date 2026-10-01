"""TWAP fair-value model bot for Polysimulator up/down 5m markets.

Resolution rule (market description): "Up" iff the Chainlink BTC/USD 60s-TWAP
over the 5-minute window >= the price at the start of the window. This is a
TWAP contract, not spot-vs-open. The existing bots price it as spot momentum;
this bot prices the actual instrument: the time-average of the price.

Model (Brownian motion, log scale):
  Given ref (price at t=0), the observed path P_0..P_t, current spot S, and
  time remaining r = T - t, with sigma the per-second log volatility:
    E[final_TWAP] = (1/T) * (int_0^t P_u du + r * S)
    Var[final_TWAP | past] = sigma^2 * (1/T^2) * int_0^t int_t^T s ds' du
                           = sigma^2 * r * (t^2 + r^2/3) / (3 * T^2)   [small-sigma]
  P(Up) = Phi( log(E_final/ref) / sqrt(Var_ln) )
  The dominant residual uncertainty is the future path's contribution; using
  the log-price BM: ln P_t = ln P_0 + sigma*W_t. Then
    E[ln final_TWAP] - ln ref  ~=  (E_final - ref)/ref   (small sigma)
    Var[ln final_TWAP] ~= sigma^2 * r * (t^2 + r^2/3) / (3 * T^2) / S^2... 

  We work in log space cleanly:
    ln S_t = ln S_0 + sigma W_t
    final_TWAP = (1/T) int_0^T S_u du
    E[final_TWAP | path to t] = A + (r/T) * S_t   where A = (1/T) int_0^t S_u du
    Var[int_t^T S_u du] = S_t^2 * sigma^2 * r^3/3   (driftless log-BM, small sigma)
    so Var[final_TWAP | past] = S_t^2 * sigma^2 * r^3 / (3 T^2)
  P(Up) = Phi( (E_final - ref) / sqrt(Var_final + ref^2 * sigma^2 * r/T^2 ... ))
  A simpler, well-behaved approximation that captures the TWAP-lag effect:
    z = (E_final - ref) / ( S_t * sigma * sqrt(r^3 / (3 T^2) + r * T/3) )
  The r*T/3 term adds the "the TWAP itself is an average" smoothing.

The exploitable edge: right after a sharp spot move, the market prices the
leader far too high because the TWAP lags the spot. The model's E_final stays
closer to ref, so the laggard is cheap.
"""
import argparse
import json
import logging
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("twap-bot")
API = "https://api.polysimulator.com/v1"
CB = "https://api.exchange.coinbase.com"
CB_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Accept": "application/json",
}
ASSETS = ("btc", "sol", "eth", "bnb")
CB_SYMBOL = {a: a.upper() + "-USD" for a in ASSETS}
WINDOW = 300
FEE_R = 0.05            # taker fee = 0.05 * min(p, 1-p) * shares
POLL = 0.5
EDGE_FRAC = 0.02        # min EV as fraction of stake to trade (~$0.20)
MIN_T = 90              # skip first 90s (early entries are spike noise: t<90 0W-4L last session)
SKIP_T_LO = 100         # skip the 100-110s rebalance dip (74% and 43% win in both sessions)
SKIP_T_HI = 110
MAX_T = 295             # endgame-only: the official Chainlink open (group_item_threshold) is
                        # only published at t~234s, so the model can only trade the last ~60s
SD_FLOOR_FRAC = 0.001   # floor sd at 0.1% of spot: Coinbase/Chainlink basis + model error
# Confident-tail strategy (calibrated on live TWAP outcomes):
#   Only buy the side the model is CONFIDENT in, and only when the market is
#   lagging (ask clearly below model prob). The model is accurate in the
#   confident tails (model>=0.80 -> favored side wins) but wrong in the
#   middle (model~0.50), so we never buy the cheap laggard at 0.25-0.50.
CONF_MIN_FAV = 0.76     # require model prob >= 0.85 (fresh $100 session: 0.80-0.84 band was 6W-3L -$182; >=0.85 is +$255)
ASK_LO = 0.65           # don't buy below this (0.50-0.60 cheap asks: 3 of 11 losses this session)
ASK_HI = 0.85           # don't pay near-certainty (no edge left after fee)
GAP_MIN = 0.08          # require model_prob - ask >= 8c (grid search: bigger lag = fatter wins; 0.08 was best across all 3 data chunks)
SESSION_LOSS_CAP = 1000.0 # stop trading for the session after losing this much (raised to 1000: user wants the full overnight run unobstructed)
WINDOW_EXPOSURE_CAP = 2 # max same-side fills per 5m window (3+ same-side clusters
                        # are correlated bets: worst cluster lost -$30 in one window)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def dec_list(v):
    return json.loads(v) if isinstance(v, str) else v


def sigma_1m(klines):
    closes = [float(k[4]) for k in klines]
    rets = [(b - a) / a for a, b in zip(closes, closes[1:])]
    return statistics.pstdev(rets) if len(rets) > 1 else 0.0


class WindowModel:
    """Tracks one 5-minute window for one asset.

    path entries are (offset_seconds_from_start, price).
    ref = group_item_threshold (Chainlink price at window start).
    """

    def __init__(self, asset, start, ref):
        self.asset = asset
        self.start = start
        self.end = start + WINDOW
        self.sigma_min = 0.0
        self.path = []
        self.ref = ref
        self.spot = None
        self.last_sample = 0.0
        self.last_fetch_ms = int(start * 1000)

    @property
    def sigma_s(self):
        return self.sigma_min / 60.0

    def sample(self, cb):
        now = time.time()
        if now - self.last_sample < 0.4:
            return
        self.last_sample = now
        try:
            self.spot = float(cb.get(CB + "/products/" + CB_SYMBOL[self.asset] + "/ticker",
                                     headers=CB_HEADERS, timeout=4).json()["price"])
        except (httpx.HTTPError, KeyError, ValueError):
            return
        # CB /trades returns the LAST N trades (no startTime cursor), so dedupe
        # by trade timestamp against the path already collected this window.
        try:
            r = cb.get(CB + "/products/" + CB_SYMBOL[self.asset] + "/trades",
                       headers=CB_HEADERS, params={"limit": 1000}, timeout=5)
            batch = r.json()
            last_ts = max((off for off, _ in self.path), default=None)
            added = 0
            for t in batch:
                try:
                    ts = datetime.fromisoformat(t["time"].replace("Z", "+00:00")).timestamp()
                except (KeyError, ValueError, TypeError):
                    continue
                off = ts - self.start
                if not (0 <= off <= now - self.start):
                    continue
                if last_ts is not None and ts <= last_ts:
                    continue  # already collected
                self.path.append((off, float(t["price"])))
                added += 1
            if added:
                self.path.sort(key=lambda x: x[0])
            self.path = self.path[-4000:]
        except (httpx.HTTPError, ValueError):
            pass

    def fair(self, now=None):
        """Return (p_up, info) or (None, None)."""
        if now is None:
            now = time.time()
        t = now - self.start
        r = self.end - now
        if self.ref is None or self.spot is None or self.sigma_min <= 0:
            return None, None
        if not (MIN_T <= t <= MAX_T):
            return None, None
        if SKIP_T_LO <= t < SKIP_T_HI:
            return None, None
        T = float(WINDOW)
        if self.path:
            integ = 0.0
            prev = 0.0
            for off, p in self.path:
                if off > t:
                    break
                integ += (off - prev) * p
                prev = off
            integ += (t - prev) * self.spot   # fill last gap to current
            a_int = integ / T
        else:
            a_int = self.spot * t / T
        e_final = a_int + r / T * self.spot
        sigma = self.sigma_s
        # Var[final_TWAP | past] in price^2 units
        var = self.spot ** 2 * sigma ** 2 * r ** 3 / (3 * T * T)
        sd = math.sqrt(max(var, 1e-18))
        # Floor: residual Coinbase-vs-Chainlink basis + discretisation + any
        # model error. Keeps z from saturating in the endgame.
        sd = max(sd, SD_FLOOR_FRAC * self.spot)
        z = (e_final - self.ref) / sd
        p_up = norm_cdf(z)
        return p_up, {"t": round(t, 1), "remaining": round(r, 1), "ref": self.ref,
                      "spot": self.spot, "e_final": round(e_final, 2),
                      "sd": round(sd, 2), "z": round(z, 3),
                      "sigma_min": round(self.sigma_min, 8)}


def discover(sim, asset):
    """Return (slug, condition_id, token_map, threshold) for the current window."""
    r = sim.get(API + "/markets/updown", params={"asset": asset.upper(), "interval": "5M"},
                timeout=8)
    r.raise_for_status()
    payload = r.json()
    rows = payload if isinstance(payload, list) else payload.get("markets", payload.get("items", []))
    start = int(time.time() // WINDOW * WINDOW)
    slug = f"{asset}-updown-5m-{start}"
    for m in rows:
        if m.get("slug") != slug or not m.get("active") or m.get("closed") or m.get("resolved"):
            continue
        cid = m["condition_id"]
        nested = next((x for x in m["markets"] if x.get("condition_id") == cid), None)
        if not nested:
            continue
        outcomes = dec_list(nested["outcomes"]); tokens = dec_list(nested["token_ids"])
        if set(outcomes) != {"Up", "Down"} or len(tokens) != 2:
            continue
        thr = m.get("group_item_threshold")
        return slug, cid, dict(zip(outcomes, tokens)), (float(thr) if thr else None)
    return None


def book_best(sim, token):
    r = sim.get(API + "/book", params={"token_id": token}, timeout=8)
    if r.status_code != 200:
        return None, None
    b = r.json()
    bids = [float(x["price"]) for x in b.get("bids", []) if float(x["size"]) > 0]
    asks = [float(x["price"]) for x in b.get("asks", []) if float(x["size"]) > 0]
    return (max(bids, default=None), min(asks, default=None))


def fee_for(price, shares):
    return FEE_R * min(price, 1 - price) * shares


class Logger:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("log", "paper"), default="log")
    parser.add_argument("--assets", nargs="+", choices=ASSETS, default=list(ASSETS))
    parser.add_argument("--poll", type=float, default=POLL)
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--edge", type=float, default=EDGE_FRAC,
                        help="min EV as fraction of stake to trade (paper mode)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set POLYSIM_API_KEY in .env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "twap_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sim = httpx.Client(timeout=10, headers={"X-API-Key": api_key})
    cb = httpx.Client(timeout=8)
    clob = httpx.Client(timeout=10)  # Polymarket public CLOB (settlement)
    log = Logger(ROOT / "price_logs" / "twap" / "windows.jsonl")
    journal_path = ROOT / "price_logs" / "twap" / "trades.jsonl"
    journal = open(journal_path, "a", encoding="utf-8")
    models = {}
    sigma_cache = {}
    thr_warned = set()
    traded = set()   # slugs already bought this session
    window_side_count = defaultdict(int)  # (window_start, side) -> fills this session
    session_pnl = 0.0  # realized PnL from closed fills
    pending = []      # open buys awaiting window resolution
    last_status = 0.0

    # Crash recovery: re-track any fill from the last 2h that has no
    # matching settle event in the journal (lost when the process died).
    # Settle events are counted per slug (a slug may have multiple fills).
    try:
        from collections import Counter
        settle_count = Counter()
        fills = []
        for line in Path(journal_path).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "settle":
                settle_count[d.get("slug")] += 1
            elif "slug" in d and "cost" in d:
                fills.append(d)
        cutoff = time.time() - 7200
        seen_recovered = Counter()
        for d in fills:
            try:
                start = int(d["slug"].rsplit("-", 1)[1])
            except (KeyError, ValueError):
                continue
            slug = d["slug"]
            if start < cutoff:
                continue
            if seen_recovered[slug] < settle_count.get(slug, 0):
                # this fill already has a matching settle event
                seen_recovered[slug] += 1
                continue
            pending.append({"slug": slug, "cid": d["cid"],
                            "shares": float(d.get("qty") or 0.0), "side": d["side"],
                            "cost": float(d["cost"]), "model_p": d.get("model_p")})
            traded.add(slug)
            window_side_count[(start, d["side"])] += 1
            LOG.info("RECOVER %s %s cost=%.2f", slug, d["side"], d["cost"])
    except OSError:
        pass

    def win_of(p):
        """Determine the resolved outcome for pending entry p (dict with
        'cid', 'slug'). CLOB-only sources (no Coinbase):
          'clob'    — official tokens[].winner flag (authoritative; posted
                      ~30 min after close).
          'polysim' — Polysimulator order book fully converged on one
                      outcome (the CLOB book pinning the winner; seconds
                      after close). Settled after a short persistence
                      guard in case of a transient quote.
        Returns (outcome, source) or (None, None) if not resolved yet."""
        cid = p.get("cid")
        if not cid:
            return None, None
        # 1) CLOB winner flag (authoritative)
        try:
            r = clob.get(f"https://clob.polymarket.com/markets/{cid}", timeout=10)
            m = r.json()
            for t in (m.get("tokens") or []):
                if t.get("winner"):
                    return t["outcome"], "clob"
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        # 2) Polysimulator rolling list: book converged on one outcome
        asset = (p.get("slug") or "").split("-")[0].upper()
        try:
            r = sim.get(API + "/markets/updown",
                        params={"asset": asset, "interval": "5M"}, timeout=8)
            rows = r.json()
            if isinstance(rows, dict):
                rows = rows.get("markets", [])
            for m in rows:
                if m.get("condition_id") != cid:
                    continue
                om = (m.get("markets") or [m])[0]
                op = json.loads(om["outcome_prices"]) if isinstance(om.get("outcome_prices"), str) else om.get("outcome_prices")
                oc = json.loads(om["outcomes"]) if isinstance(om.get("outcomes"), str) else om.get("outcomes")
                if not op or not oc:
                    continue
                op = [float(x) for x in op]
                if max(op) >= 0.995:
                    return oc[op.index(max(op))], "polysim"
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            pass
        return None, None


    def cb_win(slug, cb_client):
        """Recompute the 5m TWAP outcome from Coinbase Exchange 1-minute
        candles. Resolves 'Up' if TWAP(open->close) >= open price (>= rule,
        matches the market threshold definition)."""
        asset = slug.split("-updown")[0]
        prod = CB_SYMBOL.get(asset)
        if not prod:
            return None
        try:
            start = int(slug.rsplit("-", 1)[1])
        except ValueError:
            return None
        end = start + WINDOW
        try:
            r = cb_client.get(CB + "/products/" + prod + "/candles",
                              headers=CB_HEADERS,
                              params={"granularity": 60, "start": start, "end": end},
                              timeout=6)
            kl = r.json()
            if not kl:
                return None
            # CB candle: [ts, low, high, open, close, volume]
            open_p = float(kl[0][3])
            if open_p <= 0:
                return None
            integ = 0.0
            for i, k in enumerate(kl):
                ts = float(k[0])
                close = float(k[4])
                nxt = float(kl[i + 1][0]) if i + 1 < len(kl) else end
                lo = max(ts, start)
                hi = min(nxt, end)
                if hi <= lo:
                    continue
                if i == 0:
                    integ += (hi - lo) * open_p  # first candle: open is window start
                else:
                    integ += (hi - lo) * close
            twap = integ / float(WINDOW)
            return "Up" if twap >= open_p else "Down"
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            return None

    def settle_pending():
        """Mark resolved pending buys as won/lost; update session_pnl.
        Settles immediately on the authoritative CLOB flag; the faster
        Polysimulator-book result must be seen on 2 consecutive polls
        (persistence guard) in case of a transient quote."""
        nonlocal pending, session_pnl
        now = time.time()
        still = []
        for p in pending:
            slug = p["slug"]
            start = int(slug.rsplit("-", 1)[1])
            if now < start + WINDOW + 25:  # not yet past resolution
                still.append(p)
                continue
            winner, source = win_of(p)
            if winner is None:
                still.append(p)
                continue
            # persistence guard for the fast (book-convergence) source
            if source != "clob":
                if p.get("last_win") == winner and p.get("guard", 0) >= 1:
                    pass  # confirmed
                else:
                    p["last_win"] = winner
                    p["guard"] = p.get("guard", 0) + 1 if p.get("last_win") == winner else 1
                    still.append(p)
                    continue
            won = (winner == p["side"])
            shares = float(p.get("shares") or 0.0)
            cost = float(p.get("cost") or 0.0)
            pnl = (shares if won else 0.0) - cost
            session_pnl += pnl
            journal.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                      "event": "settle", "slug": slug, "side": p["side"],
                                      "winner": winner, "won": won, "source": source,
                                      "pnl": round(pnl, 4),
                                      "session_pnl": round(session_pnl, 4)}) + "\n")
            journal.flush()
            LOG.info("SETTLE %s %s winner=%s (%s) pnl=%.2f session_pnl=%.2f",
                     slug, p["side"], winner, source, pnl, session_pnl)
        pending = still

    def fit_sigma(asset):
        try:
            r = cb.get(CB + "/products/" + CB_SYMBOL[asset] + "/candles",
                       headers=CB_HEADERS,
                       params={"granularity": 60, "start": int(time.time()) - 1800,
                               "end": int(time.time())}, timeout=6)
            return sigma_1m(r.json())
        except httpx.HTTPError:
            return 0.0

    try:
        net_down_logged = False
        while True:
            net_fails = 0
            for asset in args.assets:
                slug = cid = None
                try:
                    mk = discover(sim, asset)
                    if not mk:
                        models = {k: v for k, v in models.items()
                                  if v.start > int(time.time())}
                        continue
                    slug, cid, tokens, thr = mk
                    start = int(slug.rsplit("-", 1)[1])
                    m = models.get(slug)
                    if m is None:
                        # Trust ONLY the official Chainlink window-open
                        # (group_item_threshold). If it isn't available yet,
                        # wait for it on the next poll — never substitute a
                        # Coinbase price (that reference is what settlement
                        # compares against, so it must be the real one).
                        ref = float(thr) if thr else None
                        if ref is None:
                            if slug not in thr_warned:
                                LOG.info("%s: waiting for official open price", slug)
                                thr_warned.add(slug)
                            continue
                        m = models[slug] = WindowModel(asset, start, ref)
                    if m.sigma_min <= 0 and \
                            (asset not in sigma_cache or
                             time.time() - sigma_cache[asset].get("ts", 0) > 120):
                        sigma_cache[asset] = {"val": fit_sigma(asset), "ts": time.time()}
                        m.sigma_min = sigma_cache[asset]["val"]
                    m.sample(cb)
                    p_up, info = m.fair()

                    rec = {"t": datetime.now(timezone.utc).isoformat(), "asset": asset,
                           "window": slug, "cid": cid, "p_up_model": p_up,
                           "spot": m.spot, "ref": m.ref, "info": info,
                           "mode": args.mode}
                    # Confident-tail strategy: only buy the side the model is
                    # confident in, and only when the market ask is clearly
                    # below that confidence (market lagging -> edge after fee).
                    if p_up is not None:
                        best = None
                        for outcome, token in tokens.items():
                            bb, ba = book_best(sim, token)
                            rec[f"bid_{outcome}"] = bb
                            rec[f"ask_{outcome}"] = ba
                            if not ba:
                                continue
                            model_p = p_up if outcome == "Up" else 1 - p_up
                            shares = args.stake / ba
                            fee = fee_for(ba, shares)
                            ev = (model_p * (shares - args.stake - fee)
                                  - (1 - model_p) * (args.stake + fee))
                            rec[f"ev_{outcome}"] = round(ev, 4)
                            ok = (model_p >= CONF_MIN_FAV
                                  and ASK_LO <= ba <= ASK_HI
                                  and (model_p - ba) >= GAP_MIN
                                  and ev > args.edge * args.stake)
                            if ok and (best is None or ev > best[0]):
                                best = (ev, model_p, outcome, ba, bb, shares, fee)
                        if best:
                            ev, model_p, outcome, ba, bb, shares, fee = best
                            rec["best"] = {"side": outcome, "model_p": round(model_p, 4),
                                           "ask": ba, "bid": bb, "ev": round(ev, 4),
                                           "shares": round(shares, 3), "fee": round(fee, 4)}
                            should_trade = (args.mode == "paper"
                                            and slug not in traded
                                            and session_pnl > -SESSION_LOSS_CAP
                                            and window_side_count[(start, outcome)] < WINDOW_EXPOSURE_CAP)
                            if should_trade:
                                # FOK orders fail when the book can't cover the
                                # full size. On FOK_ORDER_NOT_FILLED_ERROR, re-quote
                                # and retry (up to 3 tries) — the book often
                                # replenishes within a couple of seconds.
                                import uuid
                                filled = False
                                for attempt in (1, 2, 3):
                                    bb, ba = book_best(sim, tokens[outcome])
                                    rec[f"bid_{outcome}"] = bb
                                    rec[f"ask_{outcome}"] = ba
                                    if not ba or not (ASK_LO <= ba <= ASK_HI) or \
                                            (model_p - ba) < GAP_MIN:
                                        break  # edge gone — don't chase
                                    shares = args.stake / ba
                                    fee = fee_for(ba, shares)
                                    limit = f"{min(ba, ASK_HI):.2f}"
                                    body = {"market_id": cid, "side": "BUY", "outcome": outcome,
                                            "amount": str(args.stake), "price": limit,
                                            "order_type": "market", "time_in_force": "FOK"}
                                    try:
                                        resp = sim.post(API + "/orders", json=body, timeout=10,
                                                        headers={"Idempotency-Key": str(uuid.uuid4())})
                                        payload = resp.json()
                                    except httpx.HTTPError as exc:
                                        rec[f"buy_{outcome}"] = {"error": type(exc).__name__}
                                        break
                                    rec[f"buy_{outcome}"] = payload
                                    if resp.status_code in (200, 201) and \
                                            payload.get("status") == "FILLED":
                                        filled = True
                                        traded.add(slug)
                                        window_side_count[(start, outcome)] += 1
                                        # journal every trade for later reconciliation
                                        cost = args.stake + fee
                                        journal.write(json.dumps({
                                            "ts": datetime.now(timezone.utc).isoformat(),
                                            "slug": slug, "asset": asset, "side": outcome,
                                            "model_p": model_p, "ev": ev, "ask": ba,
                                            "fill": payload.get("fill_price"),
                                            "qty": payload.get("quantity"),
                                            "cost": cost, "fee": fee, "t": info.get("t"),
                                            "z": info.get("z"), "sd": info.get("sd"),
                                            "cid": cid,
                                            "order_id": payload.get("order_id"),
                                            "retries": attempt - 1,
                                        }) + "\n")
                                        journal.flush()
                                        pending.append({"slug": slug, "cid": cid,
                                                        "shares": shares, "side": outcome,
                                                        "cost": cost, "model_p": model_p})
                                        LOG.info("PAPER BUY %s %s model=%.3f ask=%.3f ev=%.3f -> FILLED (attempt %d)",
                                                 slug, outcome, model_p, ba, ev, attempt)
                                        break
                                    status = payload.get("status") or payload.get("error") or resp.status_code
                                    if status != "FOK_ORDER_NOT_FILLED_ERROR":
                                        LOG.info("PAPER BUY %s %s model=%.3f ask=%.3f ev=%.3f -> %s",
                                                 slug, outcome, model_p, ba, ev, status)
                                        break
                                    LOG.info("PAPER BUY %s %s model=%.3f ask=%.3f -> FOK not filled, retrying (%d/3)",
                                             slug, outcome, model_p, ba, attempt)
                                    time.sleep(1.0)
                    log.log(rec)
                except httpx.HTTPError as exc:
                    net_fails += 1
                    if net_fails >= len(args.assets):
                        # API unreachable: back off instead of hot-retrying
                        if not net_down_logged:
                            LOG.warning("network down (%s); backing off", type(exc).__name__)
                            net_down_logged = True
                        time.sleep(15)
                        continue
                    LOG.warning("asset %s cycle failed (%s)", asset, type(exc).__name__)
                except Exception as exc:
                    LOG.warning("asset %s unexpected (%s): %s", asset, type(exc).__name__, exc)
            if pending:
                try:
                    settle_pending()
                except Exception as exc:  # never let settlement kill the bot
                    LOG.warning("settle_pending failed: %s: %s", type(exc).__name__, exc)
            if args.once:
                break
            if time.time() - last_status > 60:
                LOG.info("mode=%s windows=%d session_pnl=%.2f pending=%d",
                         args.mode, len(models), session_pnl, len(pending))
                last_status = time.time()
            time.sleep(args.poll)
    finally:
        log.close()
        journal.close()
        sim.close()
        cb.close()


if __name__ == "__main__":
    main()
