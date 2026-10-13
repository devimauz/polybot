"""Forward-test the BTC 5-minute Coinbase-momentum strategy (paper only).

The bot observes Coinbase from the beginning of each window, evaluates once at
150 seconds, and records a simulated Polymarket buy only when the backtested
rule and recorded order-book depth permit a full fill. It never places orders.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from btc5m_momentum_strategy import momentum_signal
from polymarket_btc5m_logger import CoinbaseFeed, decode_list

ROOT = Path(__file__).resolve().parent
GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
LOG = logging.getLogger("btc5m-momentum-paper")


class Journal:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def write(self, row):
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        self.stream.close()


def normalized_book(raw):
    levels = lambda side: [{"price": float(x["price"]), "size": float(x["size"])}
                           for x in raw.get(side, []) if float(x["size"]) > 0]
    asks, bids = levels("asks"), levels("bids")
    return {"asks": asks, "bids": bids,
            "ask": min((x["price"] for x in asks), default=None),
            "bid": max((x["price"] for x in bids), default=None)}


def discover(client, start):
    slug = f"btc-updown-5m-{start}"
    response = client.get(GAMMA, params={"slug": slug}); response.raise_for_status()
    for row in response.json():
        if row.get("slug") != slug or row.get("closed") or not row.get("acceptingOrders"):
            continue
        outcomes, tokens = decode_list(row.get("outcomes")), decode_list(row.get("clobTokenIds"))
        if len(outcomes) == 2 and len(tokens) == 2 and set(outcomes) == {"Up", "Down"}:
            return {"slug": slug, "start": start, "end": start + 300,
                    "condition_id": row.get("conditionId"),
                    "tokens": dict(zip(outcomes, map(str, tokens)))}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stake", type=float, default=100)
    parser.add_argument("--entry-second", type=float, default=150)
    parser.add_argument("--minimum-move-bps", type=float, default=2)
    parser.add_argument("--ask-min", type=float, default=.45)
    parser.add_argument("--ask-max", type=float, default=.90)
    parser.add_argument("--poll", type=float, default=.1)
    args = parser.parse_args()
    values = (args.stake, args.entry_second, args.minimum_move_bps, args.ask_min,
              args.ask_max, args.poll)
    if not all(math.isfinite(x) for x in values) or args.stake <= 0 or args.poll <= 0:
        raise SystemExit("invalid numeric setting")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    journal = Journal(ROOT / "price_logs" / "momentum" / "paper_trades.jsonl")
    feed = CoinbaseFeed(); feed.start()
    state = {"start": None, "market": None, "open": None, "evaluated": False}
    pending = []
    last_resolution_check = 0.0
    session_pnl = 0.0
    try:
        with httpx.Client(timeout=8, headers={"User-Agent": "btc5m-momentum-paper/1.0"}) as client:
            while True:
                now = time.time(); start = int(now) // 300 * 300; elapsed = now - start
                if state["start"] != start:
                    state = {"start": start, "market": None, "open": None, "evaluated": False}
                    LOG.info("new window btc-updown-5m-%d", start)
                cb = feed.snapshot(now)
                if state["open"] is None and elapsed <= 5 and cb.get("price") is not None:
                    state["open"] = cb["price"]
                    LOG.info("Coinbase open %.2f", state["open"])
                if state["market"] is None:
                    try: state["market"] = discover(client, start)
                    except httpx.HTTPError as exc: LOG.warning("discovery failed: %s", type(exc).__name__)
                if elapsed >= args.entry_second and not state["evaluated"]:
                    state["evaluated"] = True
                    market = state["market"]
                    reason = None; signal = None
                    try:
                        if state["open"] is None: reason = "missing_window_open"
                        elif cb.get("price") is None: reason = "missing_coinbase_price"
                        elif not market: reason = "missing_market"
                        else:
                            move = (cb["price"] / state["open"] - 1) * 10_000
                            side = "Up" if move > 0 else "Down"
                            response = client.get(f"{CLOB}/book", params={"token_id": market["tokens"][side]})
                            response.raise_for_status()
                            book = normalized_book(response.json())
                            signal = momentum_signal(state["open"], cb["price"], {side: book},
                                                     args.stake, args.minimum_move_bps,
                                                     args.ask_min, args.ask_max)
                            if signal is None: reason = "strategy_filter_or_depth"
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        reason = type(exc).__name__
                    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": "paper_trade" if signal else "skip",
                           "slug": f"btc-updown-5m-{start}", "elapsed": elapsed,
                           "coinbase_open": state["open"], "coinbase_now": cb.get("price"),
                           "reason": reason, "signal": signal}
                    journal.write(row)
                    if signal:
                        pending.append({"slug": market["slug"],
                                        "condition_id": market["condition_id"], **signal})
                        LOG.info("PAPER BUY %s $%.2f move=%+.2fbps avg=%.3f shares=%.2f fee=%.2f",
                                 signal["side"], args.stake, signal["move_bps"], signal["avg_price"],
                                 signal["shares"], signal["fee"])
                    else: LOG.info("SKIP %s", reason)
                if pending and now - last_resolution_check >= 30:
                    last_resolution_check = now
                    still = []
                    for trade in pending:
                        try:
                            response = client.get(f"{CLOB}/markets/{trade['condition_id']}")
                            response.raise_for_status(); data = response.json()
                            winners = [token.get("outcome") for token in data.get("tokens", [])
                                       if token.get("winner") is True]
                            if len(winners) != 1:
                                still.append(trade); continue
                            won = winners[0] == trade["side"]
                            pnl = (trade["shares"] if won else 0.0) - trade["cost"]
                            session_pnl += pnl
                            journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                                           "event": "paper_settle", "slug": trade["slug"],
                                           "side": trade["side"], "winner": winners[0],
                                           "won": won, "pnl": pnl, "session_pnl": session_pnl})
                            LOG.info("SETTLE %s winner=%s pnl=%+.2f session=%+.2f",
                                     trade["slug"], winners[0], pnl, session_pnl)
                        except (httpx.HTTPError, ValueError, KeyError):
                            still.append(trade)
                    pending = still
                time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("stopped")
    finally:
        feed.close(); journal.close()


if __name__ == "__main__":
    main()

