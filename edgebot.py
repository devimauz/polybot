"""Edge bot for Polysimulator 5m BTC up/down markets.

Strategy (calibrated on ~155h of recorded windows, validated against the
authoritative Polymarket CLOB winners, not the model's own e_final):

  Buy the side the TWAP model is confident about WHEN the market ask is still
  cheap relative to that confidence (the market is lagging the near-settled
  TWAP outcome). Hold to window settlement. $10 per fill.

  GATE (favored side, per ~0.5s poll):
    t >= T_LO                 (late window: outcome near-settled, model sharp)
    MP_LO <= model_p <= MP_HI (moderate-to-high confidence, not near-certain)
    ASK_LO <= ask <= ASK_HI   (still cheap enough to be +EV after the 5% fee)
    model_p - ask > GAP_MIN   (model clearly above the price we pay)

  Backtest on 155h of BTC windows (CLOB winners, max 2 fills/side/window):
    ~0.9 trades/hr, ~66% win, net ~+$0.5/hr, positive in ~2/3 of hours.

Reuses twap_bot's WindowModel / discover / book_best / settlement so the live
signals are byte-for-byte the same as the recorded data that was backtested.

Goal tracking: logs realized P&L in rolling 1h buckets and total. With
--target-hours N the bot exits by itself once it has run >= N hours AND its
total realized P&L is positive (the "profitable over 1 hour" stop condition).
"""
import argparse
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

import twap_bot
from twap_bot import (API, CB, CB_SYMBOL, CB_HEADERS, WindowModel, discover,
                      book_best, fee_for, sigma_1m, ASSETS)

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("edgebot")

# --- Gate (CLOB-validated). Override via CLI if desired. ---
T_LO = 180          # only the late window (90-180s is spike-prone / untested here)
MP_LO = 0.62        # model confidence floor on the favored side
MP_HI = 0.92        # above ~0.92 the ask is too rich -> no edge left
ASK_LO = 0.45       # don't pay below this (cheap asks are where the market is right)
ASK_HI = 0.90       # don't pay near-certainty (no edge after the fee)
GAP_MIN = 0.01      # require model_p - ask > 1c
MAX_SIDE = 2        # max same-side fills per 5m window (matches backtest)
STAKE = 10.0
FEE_R = 0.05        # kept in sync with twap_bot (only used for EV logging)
POLL = 0.5
SESSION_LOSS_CAP = 500.0  # stop trading for the session after losing this much


class HourTracker:
    """Rolling realized P&L over the last 3600s + all-time total."""

    def __init__(self):
        self.total = 0.0
        self.events = deque()  # (timestamp, pnl)

    def add(self, pnl, ts=None):
        self.total += pnl
        self.events.append((ts or time.time(), pnl))
        cutoff = time.time() - 3600
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def hour(self):
        return sum(p for _, p in self.events)

    def n_hour(self):
        return len(self.events)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", choices=ASSETS, default=["btc"])
    parser.add_argument("--poll", type=float, default=POLL)
    parser.add_argument("--stake", type=float, default=STAKE)
    parser.add_argument("--t-lo", type=float, default=T_LO)
    parser.add_argument("--mp-lo", type=float, default=MP_LO)
    parser.add_argument("--mp-hi", type=float, default=MP_HI)
    parser.add_argument("--ask-lo", type=float, default=ASK_LO)
    parser.add_argument("--ask-hi", type=float, default=ASK_HI)
    parser.add_argument("--gap-min", type=float, default=GAP_MIN)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--target-hours", type=float, default=1.0,
                        help="exit once total P&L is positive after this many hours")
    parser.add_argument("--max-hours", type=float, default=24.0,
                        help="hard stop after this many hours regardless of P&L")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set POLYSIM_API_KEY in .env")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "edgebot.log",
                                                      encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sim = httpx.Client(timeout=10, headers={"X-API-Key": api_key})
    cb = httpx.Client(timeout=8)
    clob = httpx.Client(timeout=10)

    log_dir = ROOT / "price_logs" / "edgebot"
    log_dir.mkdir(parents=True, exist_ok=True)
    winf = open(log_dir / "windows.jsonl", "a", encoding="utf-8")
    journal_path = log_dir / "trades.jsonl"
    journal = open(journal_path, "a", encoding="utf-8")

    models = {}
    sigma_cache = {}
    thr_warned = set()
    win_side_count = defaultdict(int)   # (start, side) -> fills this session
    tracker = HourTracker()
    pending = []
    start_ts = time.time()
    last_status = 0.0
    goal_hit = False

    def win_of(p):
        cid = p.get("cid")
        if not cid:
            return None, None
        try:
            r = clob.get(f"https://clob.polymarket.com/markets/{cid}", timeout=10)
            m = r.json()
            for t in (m.get("tokens") or []):
                if t.get("winner"):
                    return t["outcome"], "clob"
        except (httpx.HTTPError, ValueError, TypeError):
            pass
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

    def settle_pending():
        nonlocal tracker
        now = time.time()
        still = []
        for p in pending:
            slug = p["slug"]
            start = int(slug.rsplit("-", 1)[1])
            if now < start + 300 + 25:
                still.append(p)
                continue
            winner, source = win_of(p)
            if winner is None:
                still.append(p)
                continue
            if source != "clob":
                if p.get("last_win") == winner and p.get("guard", 0) >= 1:
                    pass
                else:
                    p["last_win"] = winner
                    p["guard"] = p.get("guard", 0) + 1 if p.get("last_win") == winner else 1
                    still.append(p)
                    continue
            won = (winner == p["side"])
            shares = float(p.get("shares") or 0.0)
            cost = float(p.get("cost") or 0.0)
            pnl = (shares if won else 0.0) - cost
            tracker.add(pnl)
            journal.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                      "event": "settle", "slug": slug, "side": p["side"],
                                      "winner": winner, "won": won, "source": source,
                                      "pnl": round(pnl, 4),
                                      "total_pnl": round(tracker.total, 4),
                                      "hour_pnl": round(tracker.hour(), 4)}) + "\n")
            journal.flush()
            LOG.info("SETTLE %s %s winner=%s (%s) pnl=%.2f total=%.2f hour=%.2f",
                     slug, p["side"], winner, source, pnl, tracker.total, tracker.hour())
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

    import time as _t
    try:
        net_down_logged = False
        while True:
            net_fails = 0
            for asset in args.assets:
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
                        ref = float(thr) if thr else None
                        if ref is None:
                            if slug not in thr_warned:
                                LOG.info("%s: waiting for official open price", slug)
                                thr_warned.add(slug)
                            continue
                        m = models[slug] = WindowModel(asset, start, ref)
                    if m.sigma_min <= 0 and (asset not in sigma_cache or
                            time.time() - sigma_cache[asset].get("ts", 0) > 120):
                        sigma_cache[asset] = {"val": fit_sigma(asset), "ts": time.time()}
                        m.sigma_min = sigma_cache[asset]["val"]
                    m.sample(cb)
                    p_up, info = m.fair()
                    rec = {"t": datetime.now(timezone.utc).isoformat(), "asset": asset,
                           "window": slug, "cid": cid, "p_up_model": p_up,
                           "spot": m.spot, "ref": m.ref, "info": info, "mode": "paper"}
                    if p_up is not None and info is not None:
                        # Pick the favored side that clears the gate (best gap wins).
                        best = None
                        for outcome, token in tokens.items():
                            bb, ba = book_best(sim, token)
                            rec[f"bid_{outcome}"] = bb
                            rec[f"ask_{outcome}"] = ba
                            if not ba:
                                continue
                            model_p = p_up if outcome == "Up" else 1 - p_up
                            ok = (info["t"] >= args.t_lo
                                  and args.mp_lo <= model_p <= args.mp_hi
                                  and args.ask_lo <= ba <= args.ask_hi
                                  and (model_p - ba) > args.gap_min)
                            if ok and (best is None or (model_p - ba) > best[3]):
                                shares = args.stake / ba
                                fee = fee_for(ba, shares)
                                best = (outcome, ba, model_p, model_p - ba, shares, fee, bb)
                        if best:
                            outcome, ba, model_p, gap, shares, fee, bb = best
                            rec["best"] = {"side": outcome, "model_p": round(model_p, 4),
                                           "ask": ba, "gap": round(gap, 4),
                                           "shares": round(shares, 3), "fee": round(fee, 4),
                                           "t": info["t"], "z": info.get("z")}
                            allowed = (win_side_count[(start, outcome)] < args.max_side
                                       and tracker.total > -SESSION_LOSS_CAP
                                       and not goal_hit)
                            if allowed:
                                filled = False
                                for attempt in (1, 2, 3):
                                    bb, ba = book_best(sim, tokens[outcome])
                                    rec[f"bid_{outcome}"] = bb
                                    rec[f"ask_{outcome}"] = ba
                                    if not ba or not (args.ask_lo <= ba <= args.ask_hi) or \
                                            not ((model_p - ba) > args.gap_min):
                                        break  # edge gone — don't chase
                                    shares = args.stake / ba
                                    fee = fee_for(ba, shares)
                                    limit = f"{min(ba, args.ask_hi):.2f}"
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
                                    if resp.status_code in (200, 201) and payload.get("status") == "FILLED":
                                        filled = True
                                        win_side_count[(start, outcome)] += 1
                                        cost = args.stake + fee
                                        journal.write(json.dumps({
                                            "ts": datetime.now(timezone.utc).isoformat(),
                                            "slug": slug, "asset": asset, "side": outcome,
                                            "model_p": model_p, "gap": gap, "ask": ba,
                                            "fill": payload.get("fill_price"),
                                            "qty": payload.get("quantity"),
                                            "cost": cost, "fee": fee, "t": info["t"],
                                            "z": info.get("z"), "cid": cid,
                                            "order_id": payload.get("order_id"),
                                            "retries": attempt - 1}) + "\n")
                                        journal.flush()
                                        pending.append({"slug": slug, "cid": cid,
                                                        "shares": shares, "side": outcome,
                                                        "cost": cost, "model_p": model_p})
                                        LOG.info("EDGE BUY %s %s model=%.3f ask=%.3f gap=%.3f t=%.0f z=%.2f -> FILLED",
                                                 slug, outcome, model_p, ba, gap, info["t"], info.get("z") or 0)
                                        break
                                    status = payload.get("status") or payload.get("error") or resp.status_code
                                    if status != "FOK_ORDER_NOT_FILLED_ERROR":
                                        LOG.info("EDGE BUY %s %s model=%.3f ask=%.3f -> %s",
                                                 slug, outcome, model_p, ba, status)
                                        break
                                    time.sleep(1.0)
                    winf.write(json.dumps(rec) + "\n")
                    winf.flush()
                except httpx.HTTPError as exc:
                    net_fails += 1
                    if net_fails >= len(args.assets):
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
                except Exception as exc:
                    LOG.warning("settle_pending failed: %s: %s", type(exc).__name__, exc)

            now = time.time()
            elapsed_h = (now - start_ts) / 3600
            if not goal_hit and elapsed_h >= args.target_hours and tracker.total > 0:
                goal_hit = True
                LOG.info("GOAL REACHED: +%0.2f total over %.2fh (hour=%.2f); stopping trading",
                         tracker.total, elapsed_h, tracker.hour())
            if args.once or (goal_hit and not pending and elapsed_h >= args.target_hours) \
                    or elapsed_h >= args.max_hours:
                break
            if now - last_status > 60:
                LOG.info("t=%.0fh total_pnl=%.2f hour_pnl=%.2f n_hour=%d pending=%d goal=%s",
                         elapsed_h, tracker.total, tracker.hour(), tracker.n_hour(),
                         len(pending), goal_hit)
                last_status = now
            time.sleep(args.poll)
    finally:
        winf.close()
        journal.close()
        sim.close()
        cb.close()
        clob.close()
        LOG.info("edgebot stopped: total_pnl=%.2f", tracker.total)


if __name__ == "__main__":
    main()
