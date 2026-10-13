"""Polysimulator paper bot for the backtested BTC 5-minute momentum rule.

At 150 seconds into each window, buy the Coinbase-direction side when BTC has
moved at least 2 bps, the best ask is 0.45-0.90, and $100 of ask depth exists.
Orders are FOK paper orders sent to Polysimulator; no real funds are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from btc5m_momentum_strategy import fee_for, momentum_signal
from polymarket_btc5m_logger import CoinbaseFeed, decode_list

ROOT = Path(__file__).resolve().parent
API = "https://api.polysimulator.com/v1"
LOG = logging.getLogger("btc5m-momentum-polysim")


class Journal:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def write(self, row):
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        self.stream.close()


def discover(client, start):
    slug = f"btc-updown-5m-{start}"
    response = client.get(API + "/markets/updown",
                          params={"asset": "BTC", "interval": "5M"})
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("markets", payload.get("items", []))
    for row in rows:
        if row.get("slug") != slug or not row.get("active") or row.get("closed") or row.get("resolved"):
            continue
        condition_id = row.get("condition_id")
        nested = next((item for item in row.get("markets", [])
                       if item.get("condition_id") == condition_id), None)
        if not nested:
            continue
        outcomes = decode_list(nested.get("outcomes"))
        tokens = decode_list(nested.get("token_ids"))
        if len(outcomes) == 2 and len(tokens) == 2 and set(outcomes) == {"Up", "Down"}:
            return {"slug": slug, "start": start, "end": start + 300,
                    "condition_id": condition_id,
                    "tokens": dict(zip(outcomes, map(str, tokens)))}
    return None


def get_book(client, token):
    response = client.get(API + "/book", params={"token_id": token})
    response.raise_for_status()
    raw = response.json()
    asks = [{"price": float(x["price"]), "size": float(x["size"])}
            for x in raw.get("asks", []) if float(x["size"]) > 0]
    bids = [{"price": float(x["price"]), "size": float(x["size"])}
            for x in raw.get("bids", []) if float(x["size"]) > 0]
    return {"asks": asks, "bids": bids,
            "ask": min((x["price"] for x in asks), default=None),
            "bid": max((x["price"] for x in bids), default=None)}


def polysim_winner(client, condition_id):
    response = client.get(API + "/markets/updown",
                          params={"asset": "BTC", "interval": "5M"})
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("markets", [])
    for row in rows:
        if row.get("condition_id") != condition_id:
            continue
        nested = next((item for item in row.get("markets", [row])
                       if item.get("condition_id", condition_id) == condition_id), None)
        if not nested:
            continue
        outcomes = decode_list(nested.get("outcomes"))
        prices = decode_list(nested.get("outcome_prices"))
        if len(outcomes) == len(prices) == 2:
            prices = [float(x) for x in prices]
            if max(prices) >= .995:
                return outcomes[prices.index(max(prices))]
    return None


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stake", type=float, default=100)
    parser.add_argument("--entry-second", type=float, default=150)
    parser.add_argument("--minimum-move-bps", type=float, default=2)
    parser.add_argument("--ask-min", type=float, default=.45)
    parser.add_argument("--ask-max", type=float, default=.90)
    parser.add_argument("--poll", type=float, default=.1)
    parser.add_argument("--once", action="store_true",
                        help="check connectivity/current state without placing an order")
    args = parser.parse_args()
    values = (args.stake, args.entry_second, args.minimum_move_bps,
              args.ask_min, args.ask_max, args.poll)
    if not all(math.isfinite(x) for x in values) or args.stake <= 0 or args.poll <= 0:
        raise SystemExit("invalid numeric setting")
    key = os.environ.get("POLYSIM_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set POLYSIM_API_KEY in .env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "btc5m_momentum_polysim.log",
                                                      encoding="utf-8")])
    journal = Journal(ROOT / "price_logs" / "momentum-polysim" / "trades.jsonl")
    feed = CoinbaseFeed(); feed.start()
    state = {"start": None, "market": None, "open": None, "evaluated": False,
             "last_discovery": 0.0}
    pending = []
    session_pnl = 0.0
    last_settle = 0.0
    try:
        with httpx.Client(timeout=8, headers={"X-API-Key": key,
                                              "User-Agent": "btc5m-momentum-polysim/1.0"}) as client:
            while True:
                now = time.time(); start = int(now) // 300 * 300; elapsed = now - start
                if state["start"] != start:
                    state = {"start": start, "market": None, "open": None,
                             "evaluated": False, "last_discovery": 0.0}
                    LOG.info("new window btc-updown-5m-%d", start)
                cb = feed.snapshot(now)
                if state["open"] is None and elapsed <= 5 and cb.get("price") is not None:
                    state["open"] = cb["price"]
                    LOG.info("Coinbase window open %.2f", state["open"])
                if state["market"] is None and now - state["last_discovery"] >= 1:
                    state["last_discovery"] = now
                    try:
                        state["market"] = discover(client, start)
                    except httpx.HTTPError as exc:
                        LOG.warning("Polysimulator discovery failed: %s", type(exc).__name__)

                if args.once:
                    LOG.info("diagnostic market=%s cb=%s elapsed=%.1f open=%s",
                             bool(state["market"]), cb.get("price"), elapsed, state["open"])
                    break

                if elapsed >= args.entry_second and not state["evaluated"]:
                    state["evaluated"] = True
                    market, signal, reason, order = state["market"], None, None, None
                    try:
                        if state["open"] is None: reason = "started_after_window_open"
                        elif cb.get("price") is None: reason = "missing_coinbase_price"
                        elif not market: reason = "missing_polysim_market"
                        else:
                            move = (cb["price"] / state["open"] - 1) * 10_000
                            side = "Up" if move > 0 else "Down"
                            book = get_book(client, market["tokens"][side])
                            signal = momentum_signal(state["open"], cb["price"], {side: book},
                                                     args.stake, args.minimum_move_bps,
                                                     args.ask_min, args.ask_max)
                            if signal is None:
                                reason = "strategy_filter_or_insufficient_depth"
                            else:
                                body = {"market_id": market["condition_id"], "side": "BUY",
                                        "outcome": signal["side"], "amount": str(args.stake),
                                        "price": f"{signal['worst_price']:.2f}",
                                        "order_type": "market", "time_in_force": "FOK"}
                                response = client.post(API + "/orders", json=body,
                                                       headers={"Idempotency-Key": str(uuid.uuid4())})
                                try: order = response.json()
                                except ValueError: order = {"error": response.text[:500]}
                                if response.status_code not in (200, 201) or order.get("status") != "FILLED":
                                    reason = order.get("error") or order.get("status") or f"HTTP {response.status_code}"
                                    signal = None
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        reason = f"{type(exc).__name__}: {exc}"

                    record = {"ts": datetime.now(timezone.utc).isoformat(),
                              "event": "fill" if signal else "skip", "mode": "polysim",
                              "slug": f"btc-updown-5m-{start}", "elapsed": round(elapsed, 3),
                              "coinbase_open": state["open"], "coinbase_now": cb.get("price"),
                              "reason": reason, "signal": signal, "order": order}
                    if signal:
                        fill = float(order.get("fill_price") or signal["avg_price"])
                        shares = float(order.get("quantity") or signal["shares"])
                        fee = float(order.get("fee") or fee_for(fill, shares))
                        cost = args.stake + fee
                        record.update({"fill": fill, "shares": shares, "fee": fee, "cost": cost,
                                       "condition_id": market["condition_id"]})
                        pending.append({"slug": market["slug"], "condition_id": market["condition_id"],
                                        "side": signal["side"], "shares": shares, "cost": cost})
                        LOG.info("POLYSIM FILLED %s $%.2f move=%+.2fbps fill=%.3f shares=%.2f",
                                 signal["side"], args.stake, signal["move_bps"], fill, shares)
                    else:
                        LOG.info("SKIP %s", reason)
                    journal.write(record)

                if pending and now - last_settle >= 30:
                    last_settle = now; still = []
                    for trade in pending:
                        try: winner = polysim_winner(client, trade["condition_id"])
                        except (httpx.HTTPError, ValueError, KeyError): winner = None
                        if winner is None:
                            still.append(trade); continue
                        won = winner == trade["side"]
                        pnl = (trade["shares"] if won else 0.0) - trade["cost"]
                        session_pnl += pnl
                        journal.write({"ts": datetime.now(timezone.utc).isoformat(),
                                       "event": "settle", "mode": "polysim",
                                       "slug": trade["slug"], "side": trade["side"],
                                       "winner": winner, "won": won, "pnl": pnl,
                                       "session_pnl": session_pnl})
                        LOG.info("SETTLE %s winner=%s pnl=%+.2f session=%+.2f",
                                 trade["slug"], winner, pnl, session_pnl)
                    pending = still
                time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("stopped")
    finally:
        feed.close(); journal.close()


if __name__ == "__main__":
    main()

