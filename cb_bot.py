#!/usr/bin/env python3
"""
cb_bot.py
=========
BTC 5-minute Up/Down bot driven by the CB Exchange (Coinbase Exchange) BTC-USD
spot price — the same "cboe-xchange" source from btc_multi_source.py.

Strategy (per user spec):
  * At each new 5-minute window, track the CB price every ~1s.
  * D10s = (current price - price 10s ago) / price 10s ago, as a percent.
  * When D10s first exceeds  +ENTRY_PCT  -> buy UP   (one order, then ignore
    that side for the rest of the window).
  * When D10s first falls to  -ENTRY_PCT -> buy DOWN (same, opposite side).
  * Mid-window exits: sell when the token bid has moved +TP_PCT (take profit)
    or -STOP_PCT (stop, same % on the opposite side) from our fill price.
  * Anything left at window end settles via the CLOB winner flag / the
    polysimulator book convergence (same logic as twap_bot.py).

Paper mode posts real FOK orders to polysimulator (needs POLYSIM_API_KEY in .env);
log mode only records what it would have done.

Usage:
    python cb_bot.py --mode paper
    python cb_bot.py --mode paper --stake 50 --entry-pct 0.005 --tp-pct 15
    python cb_bot.py --mode log --once
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("This script needs the 'httpx' package.  pip install httpx")
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("cb-bot")
API = "https://api.polysimulator.com/v1"

# --- CB Exchange price source (btc_multi_source.py "cboe-xchange") ---
CB_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
CB_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Accept": "application/json",
}


def parse_cb(payload) -> float | None:
    try:
        return float(payload["price"])
    except (TypeError, KeyError, ValueError):
        return None


# --- window / market ---
WINDOW = 300            # 5-minute updown windows
DELTA_S = 10.0          # D10s: delta measured over the last 10 seconds
ENTRY_PCT = 0.005       # |D10s| (percent) that triggers a side
TP_PCT = 15.0           # take profit when bid >= fill * (1 + TP_PCT/100)
STOP_PCT = 15.0         # stop when bid <= fill * (1 - STOP_PCT/100)
MAX_T = 290             # no new entries in the last 10s of the window
FEE_R = 0.05            # taker fee = 0.05 * min(p, 1-p) * shares (same as twap_bot)


def dec_list(v):
    return json.loads(v) if isinstance(v, str) else v


def fee_for(price: float, shares: float) -> float:
    return FEE_R * min(price, 1 - price) * shares


def discover(sim, asset):
    """Return (slug, condition_id, {outcome: token}, threshold) for the current window."""
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


class Jsonl:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    if load_dotenv:
        load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("log", "paper"), default="log")
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--entry-pct", type=float, default=ENTRY_PCT,
                        help="|D10s| in percent that triggers a side (default 0.005)")
    parser.add_argument("--tp-pct", type=float, default=TP_PCT,
                        help="take profit when bid is this %% above fill (default 15)")
    parser.add_argument("--stop-pct", type=float, default=STOP_PCT,
                        help="stop when bid is this %% below fill (default 15)")
    parser.add_argument("--poll", type=float, default=0.1,
                        help="CB price poll interval in seconds (default 0.1 = 100ms)")
    parser.add_argument("--delta-s", type=float, default=DELTA_S,
                        help="delta lookback in seconds (default 10)")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
    if args.mode == "paper" and not api_key:
        raise SystemExit("Set POLYSIM_API_KEY in .env (paper mode)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "cb_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sim = httpx.Client(timeout=10, headers={"X-API-Key": api_key or "log-mode"})
    cb = httpx.Client(timeout=args.timeout, headers=CB_HEADERS)
    clob = httpx.Client(timeout=10)

    journal_path = ROOT / "price_logs" / "cb" / "trades.jsonl"
    samples_path = ROOT / "price_logs" / "cb" / "windows.jsonl"
    journal = Jsonl(journal_path)
    samples = Jsonl(samples_path)

    # price history: (epoch, price)
    hist: deque = deque(maxlen=2048)
    pending = []          # open positions
    session_pnl = 0.0
    windows_done = 0
    current_slug = None
    entered = {}          # slug -> {"Up": bool, "Down": bool}
    last_status = 0.0

    # --- crash recovery: re-track fills from the last 2h without a settle/exit ---
    try:
        for line in Path(journal_path).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            ev = d.get("event")
            if ev in ("settle", "exit"):
                continue
            if "slug" in d and d.get("ts", "") > datetime.fromtimestamp(
                    time.time() - 7200, tz=timezone.utc).isoformat():
                pending.append({"slug": d["slug"], "cid": d.get("cid"),
                                "shares": float(d.get("qty") or 0.0),
                                "fill": float(d.get("fill") or d.get("ask") or 0.5),
                                "side": d["side"], "cost": float(d.get("cost") or 0.0),
                                "token": d.get("token")})
                LOG.info("RECOVER %s %s cost=%.2f", d["slug"], d["side"], d.get("cost", 0))
    except OSError:
        pass

    def delta_pct(now):
        """D{delta_s}s in percent: current price vs the price ~delta_s ago."""
        if not hist:
            return None
        cur = hist[-1][1]
        target = now - args.delta_s
        ref = None
        for ts, p in hist:
            if ts >= target:
                ref = p
                break
        if ref is None:
            ref = hist[0][1]
        if not ref:
            return None
        return (cur - ref) / ref * 100.0

    def win_of(p):
        """Same resolution logic as twap_bot: CLOB winner flag, then polysim book."""
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
        nonlocal pending, session_pnl
        now = time.time()
        still = []
        for p in pending:
            slug = p["slug"]
            start = int(slug.rsplit("-", 1)[1])
            # --- mid-window exit: TP/STOP relative to our fill price ---
            if now < start + WINDOW:
                tok = p.get("token")
                if tok:
                    try:
                        bid, _ask = book_best(sim, tok)
                    except (httpx.HTTPError, ValueError, TypeError):
                        bid = None
                    if bid is not None:
                        fill = float(p["fill"])
                        shares = float(p.get("shares") or 0.0)
                        tp_price = fill * (1 + args.tp_pct / 100.0)
                        stop_price = fill * (1 - args.stop_pct / 100.0)
                        if shares > 0 and (bid >= tp_price or bid <= stop_price):
                            reason = "tp" if bid >= tp_price else "stop"
                            proceeds = shares * bid
                            fee = fee_for(bid, shares)
                            cost = float(p.get("cost") or 0.0)
                            pnl = (proceeds - fee) - cost
                            session_pnl += pnl
                            if args.mode == "paper":
                                body = {"market_id": p["cid"], "side": "SELL",
                                        "outcome": p["side"], "quantity": str(shares),
                                        "price": f"{min(bid, 0.99):.2f}",
                                        "order_type": "market", "time_in_force": "FOK"}
                                try:
                                    resp = sim.post(API + "/orders", json=body, timeout=10,
                                                    headers={"Idempotency-Key": str(uuid.uuid4())})
                                    payload = resp.json()
                                except httpx.HTTPError as exc:
                                    payload = {"error": type(exc).__name__}
                            else:
                                payload = {"simulated": True}
                            journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                                           "event": "exit", "slug": slug, "side": p["side"],
                                           "reason": reason, "bid": bid, "fill": fill,
                                           "shares": round(shares, 3),
                                           "proceeds": round(proceeds, 4),
                                           "fee": round(fee, 4), "cost": cost,
                                           "pnl": round(pnl, 4),
                                           "session_pnl": round(session_pnl, 4),
                                           "order": payload})
                            LOG.info("EXIT(%s) %s %s bid=%.3f fill=%.3f shares=%.2f pnl=%.2f session_pnl=%.2f",
                                     reason, slug, p["side"], bid, fill, shares, pnl, session_pnl)
                            continue  # closed
            if now < start + WINDOW + 25:
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
            session_pnl += pnl
            journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                           "event": "settle", "slug": slug, "side": p["side"],
                           "winner": winner, "won": won, "source": source,
                           "pnl": round(pnl, 4), "session_pnl": round(session_pnl, 4)})
            LOG.info("SETTLE %s %s winner=%s (%s) pnl=%.2f session_pnl=%.2f",
                     slug, p["side"], winner, source, pnl, session_pnl)
        pending = still

    LOG.info("start mode=%s asset=%s stake=%.2f entry=±%.4f%% tp=+%.1f%% stop=-%.1f%%",
             args.mode, args.asset, args.stake, args.entry_pct, args.tp_pct, args.stop_pct)

    try:
        while True:
            now = time.time()
            # --- 1) CB price tick ---
            try:
                r = cb.get(CB_URL)
                if r.status_code == 200:
                    price = parse_cb(r.json())
                    if price:
                        hist.append((now, price))
            except httpx.HTTPError as exc:
                LOG.warning("CB fetch failed (%s)", type(exc).__name__)

            # --- 2) current window market ---
            try:
                mk = discover(sim, args.asset)
            except httpx.HTTPError:
                mk = None
            if not mk:
                if current_slug is not None:
                    windows_done += 1
                    LOG.info("window closed: %s (windows=%d)", current_slug, windows_done)
                current_slug = None
                entered = {}
            else:
                slug, cid, tokens, thr = mk
                start = int(slug.rsplit("-", 1)[1])
                t = now - start
                if slug != current_slug:
                    current_slug = slug
                    entered = {}
                    LOG.info("window open %s thr=%s", slug, thr)

                d = delta_pct(now)
                cur_price = hist[-1][1] if hist else None
                rec = {"t": datetime.now(timezone.utc).isoformat(), "window": slug,
                       "t_in": round(t, 1), "cb_price": cur_price, "d10s": (round(d, 5) if d is not None else None)}

                # --- 3) entry: D10s crossed +/- entry threshold ---
                if (d is not None and cur_price is not None and 0 <= t < MAX_T
                        and args.mode in ("paper", "log")):
                    side = None
                    if d >= args.entry_pct and not entered.get("Up"):
                        side = "Up"
                    elif d <= -args.entry_pct and not entered.get("Down"):
                        side = "Down"
                    if side:
                        entered[side] = True
                        ask = None
                        try:
                            _bid, ask = book_best(sim, tokens[side])
                        except httpx.HTTPError:
                            ask = None
                        rec["d10s"] = round(d, 5)
                        rec["side"] = side
                        rec["ask"] = ask
                        if ask:
                            stake = args.stake
                            shares = stake / ask
                            fee = fee_for(ask, shares)
                            cost = stake + fee
                            if args.mode == "paper":
                                body = {"market_id": cid, "side": "BUY", "outcome": side,
                                        "amount": str(stake), "price": f"{min(ask, 0.99):.2f}",
                                        "order_type": "market", "time_in_force": "FOK"}
                                resp = None
                                try:
                                    resp = sim.post(API + "/orders", json=body, timeout=10,
                                                    headers={"Idempotency-Key": str(uuid.uuid4())})
                                    payload = resp.json()
                                except httpx.HTTPError as exc:
                                    payload = {"error": type(exc).__name__}
                                if resp is None:
                                    LOG.warning("PAPER BUY %s %s failed: %s", slug, side, payload.get("error"))
                                    entered[side] = True  # keep one entry per side regardless
                                elif resp.status_code in (200, 201) and payload.get("status") == "FILLED":
                                    fill = float(payload.get("fill_price") or ask)
                                    qty = float(payload.get("quantity") or shares)
                                    journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                                                   "slug": slug, "asset": args.asset, "side": side,
                                                   "d10s": round(d, 5), "ask": ask, "fill": fill,
                                                   "qty": qty, "stake": stake, "cost": cost,
                                                   "fee": fee, "t": round(t, 1),
                                                   "cb_price": cur_price, "cid": cid,
                                                   "token": tokens[side],
                                                   "order_id": payload.get("order_id")})
                                    pending.append({"slug": slug, "cid": cid,
                                                    "shares": qty, "fill": fill,
                                                    "side": side, "cost": cost,
                                                    "token": tokens[side]})
                                    LOG.info("PAPER BUY %s %s d10s=%+.4f%% cb=%.2f ask=%.3f fill=%.3f stake=%.2f",
                                             slug, side, d, cur_price, ask, fill, stake)
                                else:
                                    status = payload.get("status") or payload.get("error") or resp.status_code
                                    LOG.info("PAPER BUY %s %s d10s=%+.4f%% ask=%.3f -> %s",
                                             slug, side, d, ask, status)
                            else:
                                journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                                               "slug": slug, "asset": args.asset, "side": side,
                                               "d10s": round(d, 5), "ask": ask, "fill": ask,
                                               "qty": shares, "stake": stake, "cost": cost,
                                               "fee": fee, "t": round(t, 1),
                                               "cb_price": cur_price, "cid": cid,
                                               "token": tokens[side], "simulated": True})
                                pending.append({"slug": slug, "cid": cid,
                                                "shares": shares, "fill": ask,
                                                "side": side, "cost": cost,
                                                "token": tokens[side]})
                                LOG.info("LOG BUY %s %s d10s=%+.4f%% cb=%.2f ask=%.3f stake=%.2f",
                                         slug, side, d, cur_price, ask, stake)
                        else:
                            LOG.info("ENTRY %s %s d10s=%+.4f%% but no ask in book", slug, side, d)

                if int(t) % 30 < int(args.poll) + 1:  # sample record ~every 30s
                    samples.write(rec)

                if pending:
                    try:
                        settle_pending()
                    except Exception as exc:
                        LOG.warning("settle_pending failed: %s: %s", type(exc).__name__, exc)
            if args.once:
                break
            if time.time() - last_status > 60:
                LOG.info("mode=%s asset=%s windows=%d session_pnl=%.2f pending=%d cb=%.2f d10s=%s",
                         args.mode, args.asset, windows_done, session_pnl, len(pending),
                         hist[-1][1] if hist else 0.0,
                         f"{delta_pct(time.time()):+.4f}%" if delta_pct(time.time()) is not None else "--")
                last_status = time.time()
            time.sleep(max(args.poll, 0.25))
    finally:
        samples.close()
        journal.close()
        sim.close()
        cb.close()
        clob.close()


if __name__ == "__main__":
    main()
