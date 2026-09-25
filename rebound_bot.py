"""Confirmed-rebound Polysimulator bot. Dry run unless --paper is supplied."""
import argparse
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import httpx
from dotenv import load_dotenv
from bot import Bot, Journal, D, ROOT, configure_simulator_pacing, quote_book
from edge_trader import meets_minimum_size

LOG = logging.getLogger("rebound-bot")


@dataclass
class ReboundSignal:
    history: deque = field(default_factory=deque)
    confirmation_start: float | None = None
    confirmations: int = 0

    def update(self, stamp, midpoint):
        if self.history and stamp <= self.history[-1][0]:
            return False  # Repeated cached snapshots do not confirm a move.
        if self.history and stamp - self.history[-1][0] > 5:
            self.history.clear()
            self.confirmation_start = None
            self.confirmations = 0
        self.history.append((stamp, midpoint))
        while self.history and stamp - self.history[0][0] > 60:
            self.history.popleft()
        if len(self.history) < 5:
            return False
        points = list(self.history)
        trough_index = min(range(len(points)), key=lambda i: (points[i][1], -i))
        low_time, low = points[trough_index]
        earlier = points[:trough_index]
        qualifies = (bool(earlier) and max(p for _, p in earlier) - low >= D("0.15")
                     and midpoint - low >= D("0.08"))
        if not qualifies:
            self.confirmation_start = None
            self.confirmations = 0
            return False
        if self.confirmation_start is None or low_time > self.confirmation_start:
            self.confirmation_start, self.confirmations = stamp, 1
            return False
        self.confirmations += 1
        return stamp - self.confirmation_start >= 3 and self.confirmations >= 3


def snapshot_time(book):
    raw = book["timestamp"]
    try:
        return float(raw) / 1000
    except (ValueError, TypeError):
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("book timestamp missing timezone")
        return parsed.timestamp()


class ReboundJournal(Journal):
    def __init__(self, path):
        super().__init__(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS rebound_entries (
            market TEXT, trade_market TEXT PRIMARY KEY, created REAL, snapshot REAL)""")
        self.db.commit()

    def session_entries(self, market):
        return self.db.execute("""SELECT o.market,o.status,r.created,r.snapshot FROM orders o
            LEFT JOIN rebound_entries r ON o.market=r.trade_market
            WHERE r.market=? OR o.market=?""", (market, market)).fetchall()

    def reserve_rebound(self, market, outcome, amount, maximum, snapshot):
        def reserve():
            self.db.execute("BEGIN IMMEDIATE")
            try:
                entries = self.session_entries(market)
                now = time.time()
                blocked = len(entries) >= maximum or any(row[1] in ("unknown", "reserved") for row in entries)
                if entries:
                    last_time = max((row[2] or 0) for row in entries)
                    last_snapshot = max((row[3] or 0) for row in entries)
                    blocked = blocked or now - last_time < 10 or snapshot <= last_snapshot
                if blocked:
                    self.db.rollback()
                    return None
                trade_market = f"{market}#rebound-{len(entries) + 1}"
                day = datetime.now(timezone.utc).date().isoformat()
                self.db.execute("INSERT INTO rebound_entries VALUES (?,?,?,?)", (market, trade_market, now, snapshot))
                self.db.execute("INSERT INTO orders(market,outcome,day,amount,status,detail) VALUES (?,?,?,?,'reserved','')",
                                (trade_market, outcome, day, float(amount)))
                self.db.commit()
                return trade_market
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(reserve)


class ReboundBot(Bot):
    journal_type = ReboundJournal
    journal_prefix = "rebound-"
    price_log_dir = ROOT / "price_logs" / "rebound"

    def __init__(self, live=False, asset="btc", amount=D("5"), tp_percent=D("20"), max_entries=5):
        super().__init__(live, asset)
        self.amount = amount
        self.tp_multiplier = 1 + tp_percent / 100
        self.max_entries = max_entries
        self.signal_market = None
        self.signals = {}
        self.next_status = 0

    def execute_entry(self, slug, end, outcome, quote, book):
        tick = D(quote.tick)
        limit = (min(quote.ask + D("0.02"), D("0.65")) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        if limit < quote.ask:
            return False
        target_ceiling = ((limit * self.tp_multiplier) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        if target_ceiling >= 1:
            return False
        enough, minimum, shares = meets_minimum_size(book, self.amount, limit)
        if not enough:
            LOG.info("SKIP %s %s: $%s covers %s shares; minimum=%s", slug, outcome, self.amount, shares, minimum)
            return False
        # Require at least a minute remaining and a recent decision snapshot.
        if not 30 <= time.time() - (end - 300) < 240 or time.time() - snapshot_time(book) > 5:
            return False
        trade_key = self.journal.reserve_rebound(slug, outcome, self.amount, self.max_entries, snapshot_time(book))
        if trade_key is None:
            return False
        # Consume both histories: another entry must form from a new drop/rebound.
        self.signals = {side: ReboundSignal() for side in ("Up", "Down")}
        if not self.live:
            self.journal.finish(trade_key, outcome, "simulated", json.dumps({"limit": str(limit), "tp_multiplier": str(self.tp_multiplier)}))
            LOG.info("DRY RUN REBOUND %s %s $%s limit=%s; TP %.0f%% above actual fill",
                     slug, outcome, self.amount, limit, (self.tp_multiplier - 1) * 100)
            return True
        body = {"market_id": self.condition_id, "side": "BUY", "outcome": outcome,
                "amount": str(self.amount), "price": str(limit), "order_type": "market", "time_in_force": "FOK"}
        try:
            # The inherited deadline is end-10; move it to the strategy's end-60.
            response = self.post_with_retry(body, end - 50)
            if not response or not response.get("success") or response.get("status") != "matched":
                self.journal.finish(trade_key, outcome, "unknown", json.dumps(response))
                LOG.warning("Unconfirmed entry %s %s; retained in journal", slug, outcome)
                return True
            price = D(response["makingAmount"]) / D(response["takingAmount"])
            target = ((price * self.tp_multiplier) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            self.journal.record_profit(trade_key, outcome, self.condition_id, end, quote.tick, response, target=target)
            self.log_prices(slug, outcome, quote, "buy_matched", response)
            LOG.info("FILLED %s %s price=%s TP=%s", slug, outcome, price, target)
            self.place_profits()
        except Exception as exc:
            rejected = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404, 422, 429)
            detail = exc.response.text.replace(self.api_key, "[REDACTED]")[:2000] if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            # Never replay an ambiguous buy or allow the opposite side after an attempt.
            self.journal.finish(trade_key, outcome, "rejected" if rejected else "unknown", detail)
            LOG.warning("Entry %s %s: %s; no resubmission", slug, outcome, detail)
        return True

    def step(self):
        self.place_profits()
        market = self.discover(time.time())
        if not market:
            return
        slug, end, tokens = market
        if slug != self.signal_market:
            self.signal_market = slug
            self.signals = {side: ReboundSignal() for side in tokens}
        age = time.time() - (end - 300)
        entries = self.journal.session_entries(slug)
        seen = len(entries) >= self.max_entries or any(row[1] in ("unknown", "reserved") for row in entries)
        if time.monotonic() >= self.next_status:
            LOG.info("%s %s entries=%s/%s remaining=%.0fs mode=%s", self.asset.upper(), slug, len(entries), self.max_entries, end - time.time(),
                     "limit reached or uncertain entry" if seen else "watching rebounds" if 30 <= age < 240 else "outside entry window")
            self.next_status = time.monotonic() + 15
        if seen or not 30 <= age < 240:
            return
        for outcome, token in tokens.items():
            try:
                book = self.simulator_get("book", token_id=token)
                now = time.time()
                stamp = snapshot_time(book)
                if not -2 <= now - stamp <= 5:
                    raise ValueError(f"snapshot age {now - stamp:.2f}s outside allowed -2 to 5s")
                quote = quote_book(book, now)
                if quote.ask - quote.bid > D("0.06"):
                    raise ValueError(f"spread {quote.ask - quote.bid} exceeds 0.03")
                ready = self.signals[outcome].update(stamp, quote.probability)
                self.log_prices(slug, outcome, quote)
                if ready and D("0.15") <= quote.ask <= D("0.65"):
                    LOG.info("REBOUND %s %s midpoint=%s ask=%s", slug, outcome, quote.probability, quote.ask)
                    if self.execute_entry(slug, end, outcome, quote, book):
                        break
            except (httpx.HTTPError, ValueError, KeyError, ArithmeticError) as exc:
                self.signals[outcome] = ReboundSignal()
                reason = str(exc).replace(self.api_key, "[REDACTED]")
                LOG.info("Skip %s %s (%s): %s", slug, outcome, type(exc).__name__, reason)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="Submit Polysimulator paper orders")
    parser.add_argument("--asset", choices=Bot.assets, help="Trade only this asset (default: all of " + "/".join(Bot.assets) + ")")
    parser.add_argument("--amount", type=Decimal, default=D("10"))
    parser.add_argument("--tp-percent", type=Decimal, default=D("20"))
    parser.add_argument("--poll", type=float, default=0.1)
    parser.add_argument("--max-entries", type=int, default=5, help="Entry attempts per asset/session (default 5)")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.amount.is_finite() or args.amount <= 0 or not args.tp_percent.is_finite() or not 0 < args.tp_percent <= 50 or not 0.1 <= args.poll <= 60 or not 1 <= args.max_entries <= 20:
        parser.error("amount must be positive, tp-percent 0-50, poll 0.1-60")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "rebound_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    traders = []
    try:
        assets = (args.asset,) if args.asset else Bot.assets
        for asset in assets:
            traders.append(ReboundBot(args.paper, asset, args.amount, args.tp_percent, args.max_entries))
        configure_simulator_pacing(traders[0])
        LOG.info("%s %s $%s; TP %s%% before fees; max %s entries/session, 10s cooldown",
                 "PAPER" if args.paper else "DRY RUN", "/".join(a.upper() for a in assets),
                 args.amount, args.tp_percent, args.max_entries)
        # Each asset gets its own bot/journal; never overlap cycles within a pass.
        while True:
            for trader in traders:
                try:
                    trader.step()
                except httpx.HTTPError as exc:
                    LOG.warning("%s market request failed (%s)", trader.asset.upper(), type(exc).__name__)
            if args.once:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        LOG.info("Stopped; submitted sell orders remain active")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        for trader in traders:
            trader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
