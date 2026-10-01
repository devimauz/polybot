"""Polysimulator BTC 5-minute contrarian jump bot.

If either outcome's midpoint rises by at least 10 cents inside a rolling
300 ms window, buy the opposite outcome.  Hold only one position at a time
and sell it when its best bid is 10% above the actual entry fill.

Dry run (default):  python jump_bot.py
Paper orders:       python jump_bot.py --paper
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from decimal import ROUND_CEILING

import httpx
import websocket
from dotenv import load_dotenv

from bot import (Bot, D, Journal, LOG, Quote, ROOT, configure_simulator_pacing,
                 is_fok_rejection)

JUMP = D("0.10")
JUMP_WINDOW = 0.300
TAKE_PROFIT = D("0.10")
DEFAULT_STAKE = D("10")
ENTRY_CUTOFF_SECONDS = 12
ORDER_SLIPPAGE = D("0.04")
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class MarketFeed:
    """Public real-time Polymarket best bid/ask feed for two outcome tokens."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = {}
        self.books = {}
        self.generation = 0
        self.thread = None

    def subscribe(self, tokens):
        token_map = {str(token): outcome for outcome, token in tokens.items()}
        with self.lock:
            if token_map == self.tokens:
                return
            self.tokens = token_map
            self.books = {}
            self.generation += 1
            generation = self.generation
        self.thread = threading.Thread(target=self._run, args=(generation, token_map),
                                       name="polymarket-market-feed", daemon=True)
        self.thread.start()

    def _run(self, generation, tokens):
        while generation == self.generation:
            ws = None
            try:
                ws = websocket.create_connection(WS_URL, timeout=5)
                ws.send(json.dumps({"assets_ids": list(tokens), "type": "market",
                                    "custom_feature_enabled": True}))
                ws.settimeout(1)
                last_ping = time.monotonic()
                while generation == self.generation:
                    if time.monotonic() - last_ping >= 10:
                        ws.send("PING")
                        last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw or raw == "PONG":
                        continue
                    self._handle(json.loads(raw), tokens, generation)
            except Exception as exc:
                if generation == self.generation:
                    LOG.warning("Live market feed reconnecting (%s)", type(exc).__name__)
                    time.sleep(1)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _handle(self, message, tokens, generation):
        if isinstance(message, list):
            for item in message:
                self._handle(item, tokens, generation)
            return
        if not isinstance(message, dict):
            return
        event = message.get("event_type")
        # Initial subscription snapshots currently omit event_type.
        if event is None and "bids" in message and "asks" in message:
            event = "book"
        changes = message.get("price_changes", []) if event == "price_change" else [message]
        now = time.monotonic()
        with self.lock:
            if generation != self.generation:
                return
            for row in changes:
                token = str(row.get("asset_id", row.get("market", "")))
                if token not in tokens:
                    continue
                book = self.books.setdefault(token, {"bid": None, "ask": None,
                                                      "tick": D("0.01"), "stamp": 0.0})
                if event == "book":
                    bids = [D(x["price"]) for x in row.get("bids", []) if D(x["size"]) > 0]
                    asks = [D(x["price"]) for x in row.get("asks", []) if D(x["size"]) > 0]
                    book["bid"] = max(bids, default=None)
                    book["ask"] = min(asks, default=None)
                elif event in ("price_change", "best_bid_ask"):
                    if row.get("best_bid") not in (None, ""):
                        book["bid"] = D(str(row["best_bid"]))
                    if row.get("best_ask") not in (None, ""):
                        book["ask"] = D(str(row["best_ask"]))
                elif event == "tick_size_change" and row.get("new_tick_size"):
                    book["tick"] = D(str(row["new_tick_size"]))
                book["stamp"] = now

    def quotes(self):
        with self.lock:
            result = {}
            for token, outcome in self.tokens.items():
                book = self.books.get(token)
                if not book or book["bid"] is None or book["ask"] is None:
                    continue
                bid, ask = book["bid"], book["ask"]
                if not D(0) <= bid <= ask <= D(1) or ask - bid > D("0.10"):
                    continue
                result[outcome] = Quote((bid + ask) / 2, ask, ask, str(book["tick"]),
                                        False, bid)
            return result


@dataclass
class Position:
    market: str
    outcome: str
    entry_price: D
    shares: D
    opened_at: float
    state: str = "open"


class JumpJournal(Journal):
    """Durable single-position state, including crash/timeout locks."""

    BLOCKING_STATES = ("reserved", "open", "exit_reserved", "unknown", "holding")

    def __init__(self, path):
        super().__init__(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS jump_positions (
            market TEXT PRIMARY KEY, outcome TEXT NOT NULL,
            entry_price TEXT NOT NULL, shares TEXT NOT NULL,
            opened_at REAL NOT NULL, state TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')""")
        self.db.commit()

    def active(self):
        marks = ",".join("?" for _ in self.BLOCKING_STATES)
        row = self._retry(lambda: self.db.execute(
            f"SELECT market,outcome,entry_price,shares,opened_at,state "
            f"FROM jump_positions WHERE state IN ({marks}) "
            "ORDER BY opened_at DESC LIMIT 1", self.BLOCKING_STATES).fetchone())
        return Position(row[0], row[1], D(row[2]), D(row[3]), row[4], row[5]) if row else None

    def reserve_position(self, market, outcome):
        def do():
            self.db.execute("BEGIN IMMEDIATE")
            try:
                marks = ",".join("?" for _ in self.BLOCKING_STATES)
                if self.db.execute(
                        f"SELECT 1 FROM jump_positions WHERE state IN ({marks}) LIMIT 1",
                        self.BLOCKING_STATES).fetchone():
                    self.db.rollback()
                    return False
                self.db.execute(
                    "INSERT OR REPLACE INTO jump_positions "
                    "(market,outcome,entry_price,shares,opened_at,state,detail) "
                    "VALUES (?,?,?,?,?,'reserved','')",
                    (market, outcome, "0", "0", time.time()))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(do)

    def update_position(self, market, state, *, entry_price=None, shares=None, detail=""):
        fields, values = ["state=?", "detail=?"], [state, detail]
        if entry_price is not None:
            fields.append("entry_price=?")
            values.append(str(entry_price))
        if shares is not None:
            fields.append("shares=?")
            values.append(str(shares))
        values.append(market)
        self._retry(lambda: (self.db.execute(
            f"UPDATE jump_positions SET {','.join(fields)} WHERE market=?", values),
            self.db.commit()))


class JumpBot(Bot):
    journal_type = JumpJournal
    journal_prefix = "jump-"
    price_log_dir = ROOT / "price_logs" / "jump"

    def __init__(self, live, stake=DEFAULT_STAKE):
        super().__init__(live, "btc")
        self.stake = D(str(stake))
        self.history = {"Up": deque(), "Down": deque()}
        self.history_market = None
        self._status_width = 0
        self.feed = MarketFeed()

    def _live_status(self, slug, quotes, rises, position=None):
        """Refresh one compact terminal line without flooding the log file."""
        up_move = rises.get("Up") or D("0")
        down_move = rises.get("Down") or D("0")
        elapsed = max(0, min(300, int(time.time()) - int(slug.rsplit("-", 1)[1])))
        base = (f"\r[{elapsed:03d}/300s] "
                f"UP bid/ask/mid {quotes['Up'].bid:.2f}/{quotes['Up'].ask:.2f}/"
                f"{quotes['Up'].probability:.3f} Δ300ms {up_move:+.3f} | "
                f"DOWN {quotes['Down'].bid:.2f}/{quotes['Down'].ask:.2f}/"
                f"{quotes['Down'].probability:.3f} Δ300ms {down_move:+.3f}")
        if position and position.state == "open" and position.outcome in quotes:
            bid = quotes[position.outcome].bid
            target = position.entry_price * (D("1") + TAKE_PROFIT)
            base += (f" | OPEN {position.outcome} entry {position.entry_price:.3f} "
                     f"bid {bid:.3f} TP {target:.3f}")
        elif position:
            base += f" | LOCKED {position.outcome} ({position.state})"
        else:
            base += " | FLAT"
        padding = " " * max(0, self._status_width - len(base))
        sys.stdout.write(base + padding)
        sys.stdout.flush()
        self._status_width = len(base)

    def _status_newline(self):
        if self._status_width:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._status_width = 0

    def _record_and_detect(self, outcome, price, stamp):
        """Return rise from the lowest sample still inside the 300 ms window."""
        samples = self.history[outcome]
        cutoff = stamp - JUMP_WINDOW
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        prior = min((p for _, p in samples), default=None)
        samples.append((stamp, price))
        rise = price - prior if prior is not None else D("0")
        return rise if rise >= JUMP else None

    def _reset_history(self, market):
        if market != self.history_market:
            self.history = {"Up": deque(), "Down": deque()}
            self.history_market = market

    def _window_move(self, outcome):
        samples = self.history[outcome]
        if len(samples) < 2:
            return D("0")
        return samples[-1][1] - min(p for _, p in samples)

    def _buy(self, slug, end, outcome, quote):
        if not self.journal.reserve_position(slug, outcome):
            return False
        limit = min(D("0.99"), quote.ask + ORDER_SLIPPAGE)
        body = {"market_id": self.condition_id, "side": "BUY", "outcome": outcome,
                "amount": str(self.stake), "price": str(limit),
                "order_type": "market", "time_in_force": "FOK"}
        if not self.live:
            shares = self.stake / quote.ask
            self.journal.update_position(slug, "open", entry_price=quote.ask,
                                         shares=shares, detail="dry-run")
            LOG.info("DRY RUN BUY %s $%s @ %s (%s shares)", outcome, self.stake,
                     quote.ask, shares.quantize(D("0.0001")))
            return True
        try:
            response = self.post_with_retry(body, end)
        except Exception as exc:
            self.journal.update_position(slug, "unknown", detail=type(exc).__name__)
            raise
        if response is None:
            self.journal.update_position(slug, "unknown", detail="submission uncertain")
            LOG.error("BUY outcome uncertain; new entries locked pending reconciliation")
            return False
        if is_fok_rejection(response) or not response.get("success"):
            self.journal.update_position(slug, "rejected", detail=json.dumps(response))
            return False
        shares = D(response["takingAmount"])
        entry = D(response["makingAmount"]) / shares
        self.journal.update_position(slug, "open", entry_price=entry, shares=shares,
                                     detail=json.dumps(response))
        LOG.info("BUY FILLED %s @ %s (%s shares)", outcome, entry, shares)
        return True

    def _sell(self, position, quote, end):
        tick = D(quote.tick)
        limit = max(tick, quote.bid - ORDER_SLIPPAGE)
        limit = (limit / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        if not self.live:
            pnl = position.shares * (quote.bid - position.entry_price)
            self.journal.update_position(position.market, "closed_tp",
                                         detail=f"bid={quote.bid}; gross_pnl={pnl}")
            LOG.info("DRY RUN TAKE PROFIT %s @ %s; gross PnL %s",
                     position.outcome, quote.bid, pnl.quantize(D("0.01")))
            return True
        self.journal.update_position(position.market, "exit_reserved",
                                     detail=f"target hit at bid {quote.bid}")
        body = {"market_id": self.condition_id, "side": "SELL",
                "outcome": position.outcome, "quantity": str(position.shares),
                "price": str(limit), "order_type": "market", "time_in_force": "FOK"}
        try:
            response = self.post_with_retry(body, end)
        except Exception as exc:
            self.journal.update_position(position.market, "unknown", detail=type(exc).__name__)
            raise
        if response is None:
            self.journal.update_position(position.market, "unknown", detail="exit uncertain")
            return False
        if is_fok_rejection(response) or not response.get("success"):
            self.journal.update_position(position.market, "open", detail=json.dumps(response))
            return False
        self.journal.update_position(position.market, "closed_tp", detail=json.dumps(response))
        LOG.info("TAKE PROFIT FILLED %s bid=%s target=%s", position.outcome,
                 quote.bid, position.entry_price * (D("1") + TAKE_PROFIT))
        return True

    def step(self):
        market = self.discover(time.time())
        if not market:
            return
        slug, end, tokens = market
        self._reset_history(slug)
        self.feed.subscribe(tokens)
        now = time.time()
        quotes = self.feed.quotes()
        if len(quotes) != 2:
            return

        position = self.journal.active()
        if position:
            # Once its market has ended the simulator settles it; release the
            # strategy lock. Unknown submissions remain locked for safety.
            try:
                position_end = int(position.market.rsplit("-", 1)[1]) + 300
            except ValueError:
                position_end = end
            if position.state in ("open", "holding") and now >= position_end:
                self._status_newline()
                self.journal.update_position(position.market, "closed_settled",
                                             detail="market window ended")
                self.history = {"Up": deque(), "Down": deque()}
                return
            if position.state != "open" or position.market != slug:
                self._live_status(slug, quotes, {}, position)
                return
            held = quotes[position.outcome]
            target = position.entry_price * (D("1") + TAKE_PROFIT)
            self._live_status(slug, quotes, {}, position)
            if held.bid >= target and now < end - ENTRY_CUTOFF_SECONDS:
                self._status_newline()
                if self._sell(position, held, end):
                    self.history = {"Up": deque(), "Down": deque()}
            return

        stamp = time.monotonic()
        rises = {side: self._record_and_detect(side, q.probability, stamp)
                 for side, q in quotes.items()}
        moves = {side: self._window_move(side) for side in quotes}
        self._live_status(slug, quotes, moves)
        signals = [(rise, side) for side, rise in rises.items() if rise is not None]
        if signals and now < end - ENTRY_CUTOFF_SECONDS:
            rise, jumped = max(signals)
            buy = "Down" if jumped == "Up" else "Up"
            self._status_newline()
            LOG.info("SIGNAL %s rose %s in <=%.0fms; contrarian BUY %s",
                     jumped, rise, JUMP_WINDOW * 1000, buy)
            if self._buy(slug, end, buy, quotes[buy]):
                self.history = {"Up": deque(), "Down": deque()}
            return


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true",
                        help="submit paper orders to Polysimulator (default: dry run)")
    parser.add_argument("--stake", type=D, default=DEFAULT_STAKE,
                        help="dollars per entry (default: 10)")
    parser.add_argument("--poll", type=float,
                        default=float(os.environ.get("POLL_SECONDS", "0.05")),
                        help="delay after each cycle (default: 0.05s; API pacing still applies)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.stake <= 0 or not D("0.01") <= D(str(args.poll)) <= D("60"):
        parser.error("stake must be positive and poll must be between 0.01 and 60 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "jump_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = None
    try:
        bot = JumpBot(args.paper, args.stake)
        configure_simulator_pacing(bot)
        LOG.info("%s BTC jump bot: +$%s within %.0fms => buy opposite; TP +%.0f%%; "
                 "one position maximum; poll %.3fs",
                 "POLYSIMULATOR PAPER" if args.paper else "DRY RUN", JUMP,
                 JUMP_WINDOW * 1000, TAKE_PROFIT * 100, args.poll)
        while True:
            try:
                bot.step()
            except httpx.HTTPError as exc:
                LOG.warning("Market-data request failed (%s)", type(exc).__name__)
            if args.once:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        if bot is not None:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
