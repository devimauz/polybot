"""Polysimulator five-minute UP/DOWN threshold bot. Dry run unless --paper is supplied.

Use --asset sol for one asset, or omit --asset to check all configured assets.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
D = Decimal
LOG = logging.getLogger("btc-bot")
AMOUNT = D("10")
SESSION_ENTRY_DELAY = 30
_SIMULATOR_LOCK = threading.Lock()
_SIMULATOR_LAST_REQUEST = 0.0
_SIMULATOR_INTERVAL = 0.55


def simulator_interval(profile, tiers):
    tier = next(row for row in tiers if row["tier"] == profile["tier"])
    rps, rpm = D(str(tier["requests_per_second"])), D(str(tier["requests_per_minute"]))
    if not rps.is_finite() or not rpm.is_finite() or rps <= 0 or rpm <= 0:
        raise ValueError("Invalid API rate limits")
    return float(max(1 / rps, 60 / rpm) * D("1.10"))


def configure_simulator_pacing(bot):
    global _SIMULATOR_INTERVAL
    try:
        profile = bot.simulator_get("me")
        tiers = bot.simulator_get("keys/tiers")
        interval = simulator_interval(profile, tiers)
        with _SIMULATOR_LOCK:
            _SIMULATOR_INTERVAL = interval
        LOG.info("Polysimulator tier=%s; shared request spacing %.3fs", profile["tier"], interval)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration, ArithmeticError):
        LOG.warning("Could not detect API tier; keeping conservative request pacing")


def pace_simulator():
    """Share the detected API budget across all asset workers in this process."""
    global _SIMULATOR_LAST_REQUEST
    with _SIMULATOR_LOCK:
        time.sleep(max(0, _SIMULATOR_INTERVAL - (time.monotonic() - _SIMULATOR_LAST_REQUEST)))
        _SIMULATOR_LAST_REQUEST = time.monotonic()


@dataclass(frozen=True)
class Quote:
    probability: Decimal
    ask: Decimal
    limit: Decimal
    tick: str
    neg_risk: bool
    bid: Decimal


def triggered(outcome, midpoint):
    return outcome in ("Up", "Down") and D("0.82") <= midpoint < D("0.97")


def transient_error(exc):
    if isinstance(exc, (httpx.RequestError, TimeoutError, ConnectionError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code in (408, 429) or exc.response.status_code >= 500)

def is_fok_rejection(payload):
    if not isinstance(payload, dict) or payload.get("success") is True or payload.get("status") == "matched":
        return False
    message = payload.get("error") or payload.get("errorMsg") or ""
    return isinstance(message, str) and "order couldn't be fully filled" in message.lower() and "fok" in message.lower()


def quote_book(book, now):
    if book.get("stale") is True:
        raise ValueError("stale order book")
    raw_stamp = book["timestamp"]
    try:
        stamp = float(raw_stamp) / 1000
    except (ValueError, TypeError):
        try:
            parsed = datetime.fromisoformat(str(raw_stamp).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timestamp must include a timezone")
            stamp = parsed.timestamp()
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid order book timestamp") from exc
    if not -5 <= now - stamp <= 30:
        raise ValueError("stale order book")
    bids = [D(x["price"]) for x in book["bids"] if D(x["size"]) > 0]
    asks = [(D(x["price"]), D(x["size"])) for x in book["asks"] if D(x["size"]) > 0]
    if not bids or not asks:
        raise ValueError("empty order book")
    bid, ask = max(bids), min(p for p, _ in asks)
    if not D(0) < bid <= ask < D(1) or ask - bid > D("0.05"):
        raise ValueError("invalid or wide spread")
    tick = D(book["tick_size"])
    if tick <= 0:
        raise ValueError("invalid tick size")
    limit = (min(ask + D("0.01"), D("0.97") - tick, 1 - tick) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    return Quote((bid + ask) / 2, ask, limit, str(tick),
                 bool(book.get("neg_risk", False)), bid)


class Journal:
    def __init__(self, path):
        # check_same_thread=False: in multi_bot each Journal is built on the main
        # thread but used only by its own worker thread (one connection per bot).
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("""CREATE TABLE IF NOT EXISTS orders (
            market TEXT, outcome TEXT, day TEXT, amount REAL,
            status TEXT, detail TEXT,
            filled INTEGER NOT NULL DEFAULT 0, fill_price REAL NOT NULL DEFAULT 0,
            shares REAL NOT NULL DEFAULT 0, PRIMARY KEY(market, outcome))""")
        self._migrate("orders", {"filled": "INTEGER NOT NULL DEFAULT 0",
                                 "fill_price": "REAL NOT NULL DEFAULT 0",
                                 "shares": "REAL NOT NULL DEFAULT 0"})
        self.db.execute("""CREATE TABLE IF NOT EXISTS exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold TEXT NOT NULL DEFAULT '0', retry_after REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(market, outcome))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS reversal_watch (
            market TEXT, outcome TEXT, PRIMARY KEY(market, outcome))""")
        self.db.commit()

    def _migrate(self, table, columns):
        # Add columns that an older schema is missing (CREATE IF NOT EXISTS cannot).
        existing = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in existing:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _locked(self, exc):
        return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()

    def _retry(self, fn):
        # Several bots share one WAL file; retry briefly if another connection
        # holds the write lock (checkpoint, another bot's BEGIN IMMEDIATE).
        for attempt in range(5):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if not self._locked(exc) or attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError("unreachable")

    def seen(self, market, outcome):
        return self._retry(lambda: self.db.execute("SELECT 1 FROM orders WHERE market=? AND outcome=?", (market, outcome)).fetchone() is not None)

    def seen_market(self, market):
        return self._retry(lambda: self.db.execute("SELECT 1 FROM orders WHERE market=? LIMIT 1", (market,)).fetchone() is not None)

    def observe_reversal(self, market, outcome, probability):
        """Remember 80c per market/direction across restarts; signal at <=60c."""
        def observe():
            if probability >= D("0.80"):
                self.db.execute("INSERT OR IGNORE INTO reversal_watch VALUES (?,?)", (market, outcome))
                self.db.commit()
            armed = self.db.execute("SELECT 1 FROM reversal_watch WHERE market=? AND outcome=?",
                                    (market, outcome)).fetchone() is not None
            return armed and probability <= D("0.60")
        return self._retry(observe)

    def reserve(self, market, outcome, amount=AMOUNT, *, once_per_market=False):
        # Commit before sending: a timeout or crash must never cause an automatic duplicate.
        def do():
            self.db.execute("BEGIN IMMEDIATE")
            try:
                day = datetime.now(timezone.utc).date().isoformat()
                if (self.seen_market(market) if once_per_market else self.seen(market, outcome)):
                    self.db.rollback()
                    return False
                self.db.execute("INSERT INTO orders(market, outcome, day, amount, status, detail) VALUES (?,?,?,?,?,?)", (market, outcome, day, float(amount), "reserved", ""))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(do)

    def finish(self, market, outcome, status, detail):
        self._retry(lambda: (
            self.db.execute("UPDATE orders SET status=?, detail=? WHERE market=? AND outcome=?", (status, detail, market, outcome)),
            self.db.commit()))

    def reserve_exit(self, market, outcome):
        def do():
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT status,retry_after FROM exits WHERE market=? AND outcome=?", (market, outcome)).fetchone()
                if row and (row[0] not in ("rejected", "partial") or row[1] > time.time()):
                    self.db.rollback()
                    return False
                self.db.execute("""INSERT INTO exits(market,outcome,status,detail) VALUES (?,?,'reserved','')
                    ON CONFLICT(market,outcome) DO UPDATE SET status='reserved',detail=''""", (market, outcome))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(do)

    def finish_exit(self, market, outcome, status, detail, sold):
        self._retry(lambda: (
            self.db.execute("UPDATE exits SET status=?,detail=?,sold=?,retry_after=? WHERE market=? AND outcome=?",
                            (status, json.dumps(detail), str(sold), time.time() + 5, market, outcome)),
            self.db.commit()))


class Bot:
    journal_type = Journal
    amount = AMOUNT
    assets = ("btc", "sol", "doge", "eth", "bnb")
    asset = "sol"
    journal_prefix = ""
    price_log_dir = ROOT / "price_logs" / "polysimulator"

    def log_prices(self, slug, outcome, quote, event="quote", response=None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "polysimulator" if self.live else "dry-run", "event": event,
            "buy_price": str(quote.ask), "bid_price": str(quote.bid),
            "sell_price": str(quote.bid), "midpoint": str(quote.probability),
            "buy_limit": str(quote.limit), "fill_price": None,
        }
        if response and event == "buy_matched":
            try:
                paid = D(str(response["makingAmount"]))
                shares = D(str(response["takingAmount"]))
                if paid.is_finite() and shares.is_finite() and paid > 0 and shares > 0:
                    record["fill_price"] = str(paid / shares)
            except (KeyError, ValueError, ArithmeticError):
                pass
            record["order_id"] = response.get("orderID")
        try:
            self.price_log_dir.mkdir(parents=True, exist_ok=True)
            with (self.price_log_dir / f"{self.asset.upper()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            LOG.warning("Could not write %s price log", self.asset.upper())

    def __init__(self, live, asset="sol"):
        if asset not in Bot.assets:
            raise ValueError(f"asset must be one of {Bot.assets}")
        self.asset = asset
        self.live = live
        self.http = httpx.Client(timeout=10)
        self.journal = self.journal_type(ROOT / (self.journal_prefix + ("polysimulator.sqlite3" if live else "polysimulator-dry-run.sqlite3")))
        self.market = None
        self.client = None
        self.api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
        if not self.api_key:
            self.close()
            raise ValueError("Set POLYSIM_API_KEY in .env for Polysimulator")
        self.condition_id = None

    def simulator_get(self, path, **params):
        pace_simulator()
        response = self.http.get(
            "https://api.polysimulator.com/v1/" + path,
            params=params, headers={"X-API-Key": self.api_key})
        response.raise_for_status()
        return response.json()
    def get(self, url, **params):
        response = self.http.get(url, params=params)
        response.raise_for_status()
        return response.json()


    def discover(self, now):
        start = int(now) // 300 * 300
        slug = f"{self.asset}-updown-5m-{start}"
        # Outcome IDs do not change within a session. Avoid spending an API
        # request rediscovering them on every price check.
        cached = getattr(self, "market", None)
        if cached and cached[0] == slug and now < cached[1]:
            return cached
        payload = self.simulator_get("markets/updown", asset=self.asset.upper(), interval="5M")
        rows = payload if isinstance(payload, list) else payload.get("markets", payload.get("items", []))
        for m in rows:
            if m.get("slug") != slug or m.get("closed") or not m.get("active") or m.get("resolved"):
                continue
            end = datetime.fromisoformat(m["end_date"].replace("Z", "+00:00")).timestamp()
            if abs(end - (start + 300)) > 1:
                raise ValueError("market does not match the expected five-minute window")
            nested = next((row for row in m.get("markets", [])
                           if row.get("condition_id") == m.get("condition_id")), None)
            if nested is None:
                raise ValueError("Polysimulator outcome metadata is missing")
            outcomes, tokens = nested["outcomes"], nested["token_ids"]
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
            if len(tokens) != 2 or len(outcomes) != 2 or set(outcomes) != {"Up", "Down"} or len(set(tokens)) != 2:
                raise ValueError("unexpected market outcomes")
            self.condition_id = m.get("condition_id")
            if not self.condition_id:
                raise ValueError("Polysimulator market is missing its condition ID")
            self.market = (slug, end, dict(zip(outcomes, tokens)))
            LOG.info("Market: %s", slug)
            return self.market
        return None

    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        if all(self.journal.seen(slug, outcome) for outcome in tokens):
            now = time.monotonic()
            if now >= getattr(self, "_next_wait_log", 0):
                LOG.info("%s all directions already attempted this session; still running, next session in %.0fs",
                         self.asset.upper(), max(0, end - time.time()))
                self._next_wait_log = now + 15
            return
        # Sessions are five-minute windows; use their start, not process uptime.
        remaining = end - 300 + SESSION_ENTRY_DELAY - time.time()
        if remaining > 0:
            LOG.info("Waiting %.1fs before new buys in %s", remaining, slug)
            return
        quotes = {}
        for outcome, token in tokens.items():
            try:
                q = quote_book(self.simulator_get("book", token_id=token), time.time())
            except ValueError as exc:
                LOG.info("%s skipped: %s", outcome, exc)
                continue
            LOG.info("%s %s midpoint=%.1f%% ask=%.3f", self.asset.upper(), outcome, q.probability * 100, q.ask)
            self.log_prices(slug, outcome, q)
            quotes[outcome] = q
        for outcome, q in quotes.items():
            if self.journal.seen(slug, outcome):
                continue
            if not triggered(outcome, q.probability):
                continue
            tick = D(q.tick)
            if q.ask > q.limit:
                LOG.info("Skip %s: ask exceeds the maximum entry price", outcome)
                continue
            if time.time() >= end - 10:
                LOG.info("Skip %s: market closing", outcome)
                continue
            LOG.info("SIGNAL %s: %s midpoint=%.1f%%; buy ask=%s limit=%s",
                     slug, outcome, q.probability * 100, q.ask, q.limit)
            amount = self.amount
            if not self.journal.reserve(slug, outcome, amount):
                LOG.info("Already attempted this direction for this asset/session")
                continue
            signed = {
                "market_id": self.condition_id, "side": "BUY",
                "outcome": outcome, "amount": str(amount),
                "price": str(q.limit), "order_type": "market",
                "time_in_force": "FOK",
            }
            if not self.live:
                self.journal.finish(slug, outcome, "simulated", "Signal only; no order submitted")
                LOG.info("DRY RUN: buy $%s %s %s, maximum price %s", amount, self.asset.upper(), outcome, q.limit)
                self.log_prices(slug, outcome, q, "buy_simulated")
                continue
            try:
                response = self.post_with_retry(signed, end)
            except Exception as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 404, 409, 422, 429):
                    self.journal.finish(slug, outcome, "rejected", exc.response.text)
                    LOG.warning("%s %s paper order rejected (HTTP %s); see journal",
                                self.asset.upper(), outcome, exc.response.status_code)
                    continue
                self.journal.finish(slug, outcome, "unknown", "Submission failed or timed out; reconcile in Polysimulator")
                raise RuntimeError("Order outcome unknown. Check Polysimulator before restarting; attempt remains reserved") from None
            if response is None:
                self.journal.finish(slug, outcome, "unknown", "Submission disconnected or market closing; reconcile in Polysimulator")
                LOG.warning("%s %s submission uncertain; skipping this direction and continuing", slug, outcome)
                continue
            if is_fok_rejection(response) or response.get("status") in ("rejected", "cancelled", "canceled", "expired"):
                self.journal.finish(slug, outcome, "rejected", json.dumps(response))
                LOG.warning("%s %s FOK rejected: no fill; continuing", self.asset.upper(), outcome)
                continue
            LOG.info("Order response: %s", response)
            if not response.get("success") or response.get("status") != "matched":
                self.journal.finish(slug, outcome, "unknown", json.dumps(response))
                raise RuntimeError("Order not confirmed matched; review journal and Polysimulator account")
            self.log_prices(slug, outcome, q, "buy_matched", response)

    def post_with_retry(self, signed, end):
        # Submit once. A lost response leaves the journal reservation intact.
        pace_simulator()
        if time.time() >= end - 10:
            return None
        try:
            result = self.http.post(
                "https://api.polysimulator.com/v1/orders",
                headers={"X-API-Key": self.api_key, "Idempotency-Key": str(uuid.uuid4())},
                json=signed)
            result.raise_for_status()
        except Exception as exc:
            if transient_error(exc) and not isinstance(exc, httpx.HTTPStatusError):
                return None
            raise
        payload = result.json()
        status = str(payload.get("status", "")).lower()
        if status not in ("filled", "matched"):
            return {"success": False, "status": status, "raw": payload}
        price = D(str(payload["price"]))
        shares = D(str(payload["quantity"]))
        limit = D(signed["price"])
        price_ok = (price <= limit if signed.get("side") == "BUY" else price >= limit)
        if (not price.is_finite() or not shares.is_finite() or not limit.is_finite()
                or not 0 < price < D(1) or not price_ok or shares <= 0):
            raise ValueError("Invalid Polysimulator fill; reconcile the journal")
        return {
            "success": True, "status": "matched",
            "orderID": payload.get("order_id", payload.get("id")),
            "makingAmount": str(price * shares), "takingAmount": str(shares),
            "raw": payload,
        }
    def close(self):
        self.http.close()
        self.journal.db.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", "--live", dest="live", action="store_true", help="Submit Polysimulator paper orders only (--live is a compatibility alias)")
    parser.add_argument("--asset", choices=Bot.assets, help="Trade only this asset (default: all of " + "/".join(Bot.assets) + ")")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots, failed = [], False
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.05 <= poll <= 60:
            raise ValueError("Polling must be 0.05-60 seconds")
        assets = (args.asset,) if args.asset else Bot.assets
        for asset in assets:
            bots.append(Bot(args.live, asset))
        configure_simulator_pacing(bots[0])
        LOG.info("%s %s; $%s per direction, max two entries per asset/session; sell at 0.97; polling delay %ss",
                 "POLYSIMULATOR PAPER" if args.live else "DRY RUN", "/".join(asset.upper() for asset in assets), AMOUNT, poll)
        # Each worker owns one bot/SQLite connection; never overlap its cycles.
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            while True:
                failed = False
                futures = [(bot, workers.submit(bot.step)) for bot in bots]
                for bot, future in futures:
                    try:
                        future.result()
                    except httpx.HTTPError as exc:
                        LOG.warning("%s market data request failed (%s)", bot.asset.upper(), type(exc).__name__)
                        failed = True
                if args.once:
                    break
                time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        # SDK exceptions can include request data; do not log private signing inputs.
        LOG.error("Stopped (%s): %s. Check configuration, journal and account before retrying.", type(exc).__name__, exc)
        return 1
    finally:
        for bot in bots:
            bot.close()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
