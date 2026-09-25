"""Experimental BTC bot with execution checks. Dry run unless --live is supplied."""

import argparse
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
D = Decimal
LOG = logging.getLogger("btc-bot")
AMOUNT = D("1.5")
STRATEGY_VERSION = "bottest-v3"
LOG_LOCK = threading.Lock()
MAX_FUTURE_CLOCK_SKEW = 5.0


@dataclass(frozen=True)
class Settings:
    min_midpoint: Decimal = D("0.88")
    max_midpoint: Decimal = D("0.89")
    max_price: Decimal = D("0.93")
    max_spread: Decimal = D("0.05")
    slippage: Decimal = D("0.01")
    max_book_age: float = 5.0
    max_request_seconds: float = 2.0
    entry_delay: float = 30.0
    close_buffer: float = 10.0
    confirmation_seconds: float = 0.0
    one_direction: bool = True
    max_attempts: int = 2
    rejection_cooldown: float = 5.0

    def __post_init__(self):
        prices = (self.min_midpoint, self.max_midpoint, self.max_price, self.max_spread, self.slippage)
        times = (self.max_book_age, self.max_request_seconds, self.entry_delay,
                 self.close_buffer, self.confirmation_seconds, self.rejection_cooldown)
        if not all(x.is_finite() for x in prices) or not all(math.isfinite(x) for x in times):
            raise ValueError("Settings must be finite")
        if not D(0) < self.min_midpoint < self.max_midpoint <= D(1):
            raise ValueError("Require 0 < min midpoint < max midpoint <= 1; upper bound is exclusive")
        if not D(0) < self.min_midpoint <= self.max_price < D(1):
            raise ValueError("Require 0 < min midpoint <= max price < 1")
        if not D(0) < self.max_spread < D(1) or not D(0) <= self.slippage < D(1):
            raise ValueError("Invalid spread or slippage")
        if not 0 < self.max_book_age <= 30 or not 0 < self.max_request_seconds <= 10:
            raise ValueError("Invalid quote freshness limits")
        if not 0 <= self.entry_delay < 300 or not 1 <= self.close_buffer < 300 - self.entry_delay:
            raise ValueError("Invalid entry window")
        if not 0 <= self.confirmation_seconds < 300 - self.entry_delay - self.close_buffer:
            raise ValueError("Invalid confirmation period")
        if not 1 <= self.max_attempts <= 3 or self.rejection_cooldown < 1:
            raise ValueError("Use 1-3 attempts and a cooldown of at least one second")


def decimal(value):
    try:
        result = D(str(value))
        if not result.is_finite():
            raise ValueError("Non-finite number")
        return result
    except ArithmeticError as exc:
        raise ValueError("Invalid number") from exc


@dataclass(frozen=True)
class Quote:
    probability: Decimal
    ask: Decimal
    limit: Decimal
    tick: str
    neg_risk: bool
    bid: Decimal
    asks: tuple
    book_timestamp: float

    def depth(self, limit=None):
        limit = self.limit if limit is None else limit
        return sum((price * size for price, size in self.asks if price <= limit), D(0))


def triggered(outcome, probability, settings=None):
    settings = settings or Settings()
    return settings.min_midpoint <= probability < settings.max_midpoint


def transient_error(exc):
    from py_clob_client_v2.exceptions import PolyApiException
    if isinstance(exc, (httpx.RequestError, TimeoutError, ConnectionError)):
        return True
    return isinstance(exc, PolyApiException) and (
        exc.status_code is None or exc.status_code in (408, 429)
        or 500 <= exc.status_code < 600)


def is_fok_rejection(payload):
    if not isinstance(payload, dict) or payload.get("success") is True or payload.get("status") == "matched":
        return False
    message = payload.get("error") or payload.get("errorMsg") or ""
    return isinstance(message, str) and "order couldn't be fully filled" in message.lower() and "fok" in message.lower()


def quote_book(book, now, settings=None):
    settings = settings or Settings()
    stamp = float(book["timestamp"]) / 1000
    if not math.isfinite(stamp):
        raise ValueError("non-finite order book timestamp")
    age = now - stamp
    if age < -MAX_FUTURE_CLOCK_SKEW:
        raise ValueError(f"book timestamp {-age:.2f}s ahead of local clock (allowed {MAX_FUTURE_CLOCK_SKEW:g}s)")
    if age > settings.max_book_age:
        raise ValueError(f"order book {age:.2f}s old (maximum {settings.max_book_age:g}s)")
    def levels(items):
        result = []
        for item in items:
            price, size = decimal(item["price"]), decimal(item["size"])
            if not 0 < price < 1 or size < 0:
                raise ValueError("Invalid book level")
            if size:
                result.append((price, size))
        return result
    bids = [price for price, _ in levels(book["bids"])]
    asks = levels(book["asks"])
    if not bids or not asks:
        raise ValueError("empty order book")
    bid, ask = max(bids), min(p for p, _ in asks)
    if bid > ask:
        raise ValueError("crossed order book")
    tick = decimal(book["tick_size"]).normalize()
    if str(tick) not in ("0.1", "0.01", "0.005", "0.0025", "0.001", "0.0001"):
        raise ValueError("invalid tick size")
    limit = (min(ask + settings.slippage, settings.max_price, 1 - tick) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    return Quote((bid + ask) / 2, ask, limit, str(tick),
                 bool(book.get("neg_risk", False)), bid, tuple(sorted(asks)), stamp)


def signed_order_id(client, signed, neg_risk):
    """Use the pinned SDK's EIP-712 hash; persist no private key or signature."""
    from py_clob_client_v2.config import get_contract_config
    from py_clob_client_v2.order_utils import ExchangeOrderBuilderV1, ExchangeOrderBuilderV2
    from py_clob_client_v2.order_utils.model.order_data_v1 import SignedOrderV1
    from py_clob_client_v2.order_utils.model.order_data_v2 import SignedOrderV2
    signer = client.builder.signer
    config = get_contract_config(signer.get_chain_id())
    if isinstance(signed, SignedOrderV2):
        builder_type = ExchangeOrderBuilderV2
        address = config.neg_risk_exchange_v2 if neg_risk else config.exchange_v2
    elif isinstance(signed, SignedOrderV1):
        builder_type = ExchangeOrderBuilderV1
        address = config.neg_risk_exchange if neg_risk else config.exchange
    else:
        raise ValueError("Unsupported signed order type")
    builder = builder_type(address, signer.get_chain_id(), signer)
    return builder.build_order_hash(builder.build_order_typed_data(signed))


def resolved_winner(market, tokens):
    """A closed book alone or a near-$1 quote does not establish resolution."""
    if market.get("closed") is not True:
        return None
    winners = [t for t in market.get("tokens", []) if t.get("winner") is True]
    if len(winners) != 1:
        return None
    winner = winners[0]
    outcome = winner.get("outcome")
    return outcome if outcome in tokens and str(winner.get("token_id")) == str(tokens[outcome]) else None


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS orders (
            market TEXT, outcome TEXT, day TEXT, amount REAL,
            status TEXT, detail TEXT, PRIMARY KEY(market, outcome))""")
        self.db.commit()

        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS bottest_attempts (
                id INTEGER PRIMARY KEY, market TEXT, outcome TEXT, created REAL,
                status TEXT, updated REAL, metadata TEXT, response TEXT);
            CREATE INDEX IF NOT EXISTS bottest_attempt_market
                ON bottest_attempts(market, outcome);
            CREATE TABLE IF NOT EXISTS bottest_markets (
                market TEXT PRIMARY KEY, end REAL, condition_id TEXT,
                tokens TEXT, winner TEXT, checked REAL DEFAULT 0);
        """)

        self.db.execute("""CREATE TABLE IF NOT EXISTS exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold TEXT NOT NULL DEFAULT '0', retry_after REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(market, outcome))""")
        self.db.commit()

    def seen(self, market, outcome):
        return self.db.execute("SELECT 1 FROM orders WHERE market=? AND outcome=?", (market, outcome)).fetchone() is not None

    def eligible(self, market, outcome, settings, now=None):
        now = time.time() if now is None else now
        rows = self.db.execute("SELECT outcome,status FROM orders WHERE market=?", (market,)).fetchall()
        for direction, status in rows:
            if status != "rejected" and (direction == outcome or settings.one_direction):
                return False
        prior = self.db.execute("SELECT status,updated FROM bottest_attempts WHERE market=? AND outcome=? ORDER BY id",
                                (market, outcome)).fetchall()
        if any(direction == outcome for direction, _ in rows) and not prior:
            return False  # Do not automatically retry legacy attempts with no audit trail.
        return (len(prior) < settings.max_attempts and
                (not prior or (prior[-1][0] == "rejected" and
                               now >= prior[-1][1] + settings.rejection_cooldown)))

    def reserve(self, market, outcome, amount=AMOUNT, settings=None, metadata=None):
        # Commit before sending: a timeout or crash must never cause an automatic duplicate.
        settings = settings or Settings()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            day = datetime.now(timezone.utc).date().isoformat()
            if not self.eligible(market, outcome, settings):
                self.db.rollback()
                return False
            self.db.execute("""INSERT INTO orders VALUES (?,?,?,?,?,?)
                ON CONFLICT(market,outcome) DO UPDATE SET day=excluded.day,
                amount=excluded.amount,status='reserved',detail=excluded.detail""",
                (market, outcome, day, float(amount), "reserved", json.dumps(metadata or {})))
            self.db.execute("INSERT INTO bottest_attempts(market,outcome,created,status,updated,metadata,response) VALUES (?,?,?,?,?,?,?)",
                            (market, outcome, time.time(), "reserved", time.time(), json.dumps(metadata or {}), "{}"))
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, market, outcome, status, detail):
        self.db.execute("UPDATE orders SET status=?, detail=? WHERE market=? AND outcome=?", (status, detail, market, outcome))
        self.db.execute("""UPDATE bottest_attempts SET status=?,response=?,updated=? WHERE id=(
            SELECT MAX(id) FROM bottest_attempts WHERE market=? AND outcome=?)""",
            (status, detail, time.time(), market, outcome))
        self.db.commit()

    def reserve_exit(self, market, outcome):
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

    def finish_exit(self, market, outcome, status, detail, sold):
        self.db.execute("UPDATE exits SET status=?,detail=?,sold=?,retry_after=? WHERE market=? AND outcome=?",
                        (status, json.dumps(detail), str(sold), time.time() + 5, market, outcome))
        self.db.commit()


class Bot:
    amount = AMOUNT
    asset = "btc"
    journal_prefix = ""
    price_log_dir = ROOT / "price_logs"
    settings = Settings()

    def log_prices(self, slug, outcome, quote, event="quote", response=None, **extra):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "live" if self.live else "dry-run", "event": event,
            "strategy_version": STRATEGY_VERSION, "configured_amount": str(self.amount),
            "settings": {k: str(v) if isinstance(v, D) else v for k, v in asdict(self.settings).items()},
            "buy_price": str(quote.ask) if quote else None,
            "bid_price": str(quote.bid) if quote else None,
            "sell_price": str(quote.bid) if quote else None,
            "midpoint": str(quote.probability) if quote else None,
            "buy_limit": str(quote.limit) if quote else None, "fill_price": None,
            "book_timestamp": quote.book_timestamp if quote else None,
            "depth_usd_at_limit": str(quote.depth()) if quote else None,
            "fees_paid": None, "net_pnl": None,
        }
        if response and event == "buy_matched":
            try:
                paid = D(str(response["makingAmount"]))
                shares = D(str(response["takingAmount"]))
                if paid.is_finite() and shares.is_finite() and paid > 0 and shares > 0:
                    record["fill_price"] = str(paid / shares)
                    record["paid"] = str(paid)
                    record["shares"] = str(shares)
            except (KeyError, ValueError, ArithmeticError):
                pass
            record["order_id"] = response.get("orderID")
        record.update(extra)
        try:
            self.price_log_dir.mkdir(parents=True, exist_ok=True)
            with LOG_LOCK:
                with (self.price_log_dir / f"{self.asset.upper()}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            return True
        except OSError:
            LOG.warning("Could not write %s price log", self.asset.upper())
            return False

    def __init__(self, live, settings=None, amount=AMOUNT):
        self.live = live
        self.settings = settings or Settings()
        self.amount = decimal(amount)
        if self.amount <= 0 or not math.isfinite(float(self.amount)):
            raise ValueError("Order amount must be positive")
        self.http = httpx.Client(timeout=3)
        # Share the live journal with bot.py to respect previous live attempts.
        self.journal_path = ROOT / (self.journal_prefix + ("live.sqlite3" if live else "bottest-dry-run.sqlite3"))
        self.journal = Journal(self.journal_path)
        self.market = None
        self.client = None
        self.signals = {}
        if live:
            if self.get("https://polymarket.com/api/geoblock").get("blocked") is not False:
                raise RuntimeError("Trading is unavailable from this location")
            from py_clob_client_v2 import ClobClient
            key = os.environ.get("POLY_PRIVATE_KEY")
            if not key:
                raise ValueError("Set POLY_PRIVATE_KEY in .env before using --live")
            signature = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
            funder = os.environ.get("POLY_FUNDER") or None
            if signature not in (0, 1, 2, 3) or (signature != 0 and not funder):
                raise ValueError("Use signature type 0/1/2/3 and supply POLY_FUNDER for smart wallets")
            self.client = ClobClient("https://clob.polymarket.com", chain_id=137,
                                     key=key, signature_type=signature, funder=funder)
            self.client.set_api_creds(self.client.create_or_derive_api_key())

    def get(self, url, **params):
        response = self.http.get(url, params=params)
        response.raise_for_status()
        return response.json()


    def discover(self, now):
        start = int(now) // 300 * 300
        slug = f"{self.asset}-updown-5m-{start}"
        if self.market and self.market[0] == slug and now < self.market[1]:
            return self.market
        rows = self.get("https://gamma-api.polymarket.com/markets", slug=slug)
        for m in rows:
            if m.get("slug") != slug or m.get("closed") or not m.get("active") or not m.get("acceptingOrders"):
                continue
            end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            if abs(end - (start + 300)) > 1:
                raise ValueError("market does not match the expected five-minute window")
            outcomes, tokens = m["outcomes"], m["clobTokenIds"]
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
            if len(tokens) != 2 or len(outcomes) != 2 or set(outcomes) != {"Up", "Down"} or len(set(tokens)) != 2:
                raise ValueError("unexpected market outcomes")
            self.market = (slug, end, dict(zip(outcomes, tokens)))
            self.signals.clear()
            self.journal.db.execute("""INSERT INTO bottest_markets(market,end,condition_id,tokens)
                VALUES (?,?,?,?) ON CONFLICT(market) DO UPDATE SET
                condition_id=excluded.condition_id,tokens=excluded.tokens""",
                (slug, end, m.get("conditionId"), json.dumps(self.market[2])))
            self.journal.db.commit()
            LOG.info("Market: %s", slug)
            return self.market
        return None

    def read_quote(self, token):
        started = time.monotonic()
        book = self.get("https://clob.polymarket.com/book", token_id=token)
        if time.monotonic() - started > self.settings.max_request_seconds:
            raise ValueError("slow order book request")
        return quote_book(book, time.time(), self.settings)

    def entry_reason(self, quote, end, limit=None):
        now = time.time()
        limit = quote.limit if limit is None else limit
        if now < end - 300 + self.settings.entry_delay:
            return "entry_delay"
        if now >= end - self.settings.close_buffer:
            return "market_closing"
        if not -MAX_FUTURE_CLOCK_SKEW <= now - quote.book_timestamp <= self.settings.max_book_age:
            return "stale_book"
        if not triggered(None, quote.probability, self.settings):
            return "outside_signal_band"
        if quote.ask - quote.bid > self.settings.max_spread:
            return "wide_spread"
        if quote.ask > limit or limit > self.settings.max_price:
            return "ask_above_limit"
        if quote.depth(limit) < self.amount:
            return "insufficient_depth"
        return None

    def signal_confirmed(self, slug, outcome):
        now = time.monotonic()
        key = (slug, outcome)
        first, previous = self.signals.get(key, (now, now))
        if now - previous > self.settings.max_book_age:
            first = now
        self.signals[key] = (first, now)
        return now - first >= self.settings.confirmation_seconds

    def still_executable(self, token, original, end):
        quote = self.read_quote(token)
        return (quote.tick == original.tick and quote.neg_risk == original.neg_risk
                and self.entry_reason(quote, end, original.limit) is None)

    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        for outcome, token in tokens.items():
            try:
                quote = self.read_quote(token)
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                self.signals.pop((slug, outcome), None)
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                self.log_prices(slug, outcome, None, "quote_unavailable", reason=reason)
                LOG.info("%s %s quote unavailable: %s", self.asset.upper(), outcome, reason)
                continue
            reason = self.entry_reason(quote, end)
            eligible = self.journal.eligible(slug, outcome, self.settings)
            # Observe both directions even after a fill, rejection, or ambiguous attempt.
            self.log_prices(slug, outcome, quote, reason=reason or (None if eligible else "journal_blocked"),
                            seconds_remaining=round(end - time.time(), 3))
            if reason:
                self.signals.pop((slug, outcome), None)
                continue
            if not eligible:
                continue
            if not self.signal_confirmed(slug, outcome):
                continue
            signed, order_id = None, None
            if self.live:
                from py_clob_client_v2 import MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
                try:
                    signed = self.client.create_market_order(
                        MarketOrderArgs(token_id=token, amount=float(self.amount), side=Side.BUY,
                                        price=float(quote.limit), order_type=OrderType.FOK),
                        PartialCreateOrderOptions(tick_size=quote.tick, neg_risk=quote.neg_risk))
                    order_id = signed_order_id(self.client, signed, quote.neg_risk)
                except Exception as exc:
                    if not transient_error(exc):
                        raise
                    self.log_prices(slug, outcome, quote, "prepare_retry", reason=type(exc).__name__)
                    continue
                # Signing may fetch SDK metadata: recheck the market AFTER preparing the order.
                try:
                    valid = self.still_executable(token, quote, end)
                except (ValueError, KeyError, TypeError, httpx.HTTPError):
                    valid = False
                if not valid:
                    self.signals.pop((slug, outcome), None)
                    self.log_prices(slug, outcome, quote, "buy_skipped", reason="book_changed_before_submit")
                    continue
            metadata = {"strategy_version": STRATEGY_VERSION, "order_id": order_id,
                        "token_id": token, "limit": str(quote.limit), "end": end,
                        "configured_amount": str(self.amount), "signal_price": str(quote.probability)}
            if not self.journal.reserve(slug, outcome, self.amount, self.settings, metadata):
                continue
            if not self.live:
                self.journal.finish(slug, outcome, "simulated", json.dumps(metadata))
                self.log_prices(slug, outcome, quote, "buy_simulated")
                LOG.info("DRY RUN: signal for $%s %s %s at limit %s", self.amount, self.asset.upper(), outcome, quote.limit)
                continue
            try:
                response = self.post_with_retry(signed, end,
                    validate=lambda: self.still_executable(token, quote, end))
            except Exception as exc:
                from py_clob_client_v2.exceptions import PolyApiException
                if isinstance(exc, PolyApiException) and exc.status_code == 400 and is_fok_rejection(exc.error_msg):
                    self.journal.finish(slug, outcome, "rejected", json.dumps(exc.error_msg))
                    self.log_prices(slug, outcome, quote, "buy_rejected", order_id=order_id, reason="confirmed_fok_no_fill")
                    continue
                # No fresh signed order after an ambiguous failure. Reconcile by its saved hash.
                self.journal.finish(slug, outcome, "unknown", json.dumps({"orderID": order_id, "error_type": type(exc).__name__}))
                self.log_prices(slug, outcome, quote, "buy_unknown", order_id=order_id)
                LOG.warning("Order uncertain (%s); preserving reservation and observing prices", type(exc).__name__)
                continue
            if is_fok_rejection(response):
                self.journal.finish(slug, outcome, "rejected", json.dumps(response))
                self.log_prices(slug, outcome, quote, "buy_rejected", order_id=order_id, reason="confirmed_fok_no_fill")
                continue
            if not isinstance(response, dict) or response.get("success") is not True or response.get("status") != "matched":
                self.journal.finish(slug, outcome, "unknown", json.dumps({"orderID": order_id, "response": response}))
                self.log_prices(slug, outcome, quote, "buy_unknown", order_id=order_id)
                continue
            if response.get("orderID") != order_id:
                self.journal.finish(slug, outcome, "unknown", json.dumps({"orderID": order_id, "response": response}))
                self.log_prices(slug, outcome, quote, "buy_unknown", order_id=order_id, reason="response_order_id_mismatch")
                continue
            self.journal.finish(slug, outcome, "submitted", json.dumps(response))
            self.log_prices(slug, outcome, quote, "buy_matched", response)

    def post_with_retry(self, signed, end, validate=None):
        from py_clob_client_v2 import OrderType
        for attempt in range(3):
            if time.time() >= end - self.settings.close_buffer:
                return None
            if attempt and validate:
                try:
                    if not validate():
                        return None
                except (ValueError, KeyError, TypeError, httpx.HTTPError):
                    return None
                if time.time() >= end - self.settings.close_buffer:
                    return None

            try:
                response = self.client.post_order(signed, OrderType.FOK)
                # A rejection after a lost response cannot disprove an earlier fill.
                if attempt and (not isinstance(response, dict) or not response.get("success")
                                or response.get("status") != "matched"):
                    return None
                return response
            except Exception as exc:
                if not transient_error(exc):
                    if attempt:
                        return None
                    raise
                delay = 2 ** attempt
                if attempt == 2 or time.time() + delay >= end - self.settings.close_buffer:
                    return None
                LOG.warning("Submission disconnected; retrying same signed order in %ss (%s/3)", delay, attempt + 2)
                time.sleep(delay)
                if time.time() >= end - self.settings.close_buffer:
                    return None

    def reconcile_one(self):
        if not self.live:
            return
        row = self.journal.db.execute("""SELECT id,market,outcome,metadata FROM bottest_attempts
            WHERE status IN ('reserved','unknown') AND updated<? ORDER BY updated LIMIT 1""",
            (time.time() - 15,)).fetchone()
        if not row:
            return
        attempt_id, slug, outcome, encoded = row
        metadata = json.loads(encoded)
        # Back off missing/unavailable orders; absence is never proof of no fill.
        self.journal.db.execute("UPDATE bottest_attempts SET updated=? WHERE id=?", (time.time(), attempt_id))
        self.journal.db.commit()
        order_id = metadata.get("order_id")
        if not order_id:
            return
        order = self.client.get_order(order_id)
        if (not isinstance(order, dict) or order.get("id") != order_id or
                str(order.get("asset_id")) != str(metadata.get("token_id")) or order.get("side") != "BUY"):
            return
        size = decimal(order.get("size_matched"))
        if size < 0:
            return
        state = str(order.get("status", "")).upper()
        if size > 0:
            # An order's limit is NOT its average fill price. Do not invent paid amounts.
            detail = {"orderID": order_id, "matched_shares": str(size), "exchange_status": state}
            status = "reconciled"
        elif state == "CANCELED":
            detail = {"orderID": order_id, "exchange_status": state, "matched_shares": "0"}
            status = "rejected"
        else:
            return
        # Avoid overwriting the submission thread's newer result.
        self.journal.db.execute("BEGIN IMMEDIATE")
        current = self.journal.db.execute("SELECT status FROM bottest_attempts WHERE id=?", (attempt_id,)).fetchone()
        if not current or current[0] not in ("reserved", "unknown"):
            self.journal.db.rollback()
            return
        self.journal.finish(slug, outcome, status, json.dumps(detail))
        self.log_prices(slug, outcome, None, "order_reconciled", order_id=order_id,
                        matched_shares=str(size), reconciled_status=status)

    def settle_one(self):
        row = self.journal.db.execute("""SELECT market,condition_id,tokens FROM bottest_markets
            WHERE end<? AND winner IS NULL AND condition_id IS NOT NULL AND checked<?
            ORDER BY checked, end LIMIT 1""", (time.time(), time.time() - 60)).fetchone()
        if not row:
            return
        slug, condition_id, tokens_json = row
        self.journal.db.execute("UPDATE bottest_markets SET checked=? WHERE market=?", (time.time(), slug))
        self.journal.db.commit()
        market = self.get(f"https://clob.polymarket.com/markets/{condition_id}")
        if market.get("condition_id") != condition_id:
            return
        tokens = json.loads(tokens_json)
        winner = resolved_winner(market, tokens)
        if winner is None:
            return
        self.journal.db.execute("UPDATE bottest_markets SET winner=? WHERE market=?", (winner, slug))
        self.journal.db.commit()
        for outcome in tokens:
            detail = self.journal.db.execute("""SELECT response FROM bottest_attempts
                WHERE market=? AND outcome=? AND status='submitted' ORDER BY id DESC LIMIT 1""",
                (slug, outcome)).fetchone()
            gross = None
            if detail:
                try:
                    fill = json.loads(detail[0])
                    paid, shares = decimal(fill["makingAmount"]), decimal(fill["takingAmount"])
                    if paid > 0 and shares > 0:
                        gross = str((shares if outcome == winner else D(0)) - paid)
                except (KeyError, TypeError, ValueError):
                    pass
            self.log_prices(slug, outcome, None, "market_resolved", winner=winner,
                            payout_per_share="1" if outcome == winner else "0",
                            gross_pnl_before_fees=gross,
                            pnl_basis="hold_to_resolution; excludes fees and any external sells")

    def start_maintenance(self):
        self.stop_event = threading.Event()

        def run():
            # SQLite connections belong to the thread that creates them.
            worker = type(self).__new__(type(self))
            worker.live, worker.client = self.live, self.client
            worker.settings, worker.amount = self.settings, self.amount
            worker.asset, worker.price_log_dir = self.asset, self.price_log_dir
            try:
                with httpx.Client(timeout=3) as http:
                    worker.http = http
                    worker.journal = Journal(self.journal_path)
                    try:
                        while not self.stop_event.is_set():
                            for action in (worker.reconcile_one, worker.settle_one):
                                try:
                                    action()
                                except Exception as exc:
                                    LOG.warning("Background %s deferred (%s)", action.__name__, type(exc).__name__)
                            if self.stop_event.wait(15):
                                break
                    finally:
                        worker.journal.db.close()
            except Exception as exc:
                LOG.warning("Background monitoring stopped (%s)", type(exc).__name__)

        self.worker = threading.Thread(target=run, name="bottest-monitor", daemon=True)
        self.worker.start()

    def close(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.worker.join(timeout=7)
        self.http.close()
        self.journal.db.close()


def argument_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--live", action="store_true", help="Submit real orders using .env wallet credentials")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    parser.add_argument("--amount", type=decimal, default=AMOUNT, help="Dollars per buy, before fees")
    parser.add_argument("--min-midpoint", type=decimal, default=Settings.min_midpoint, help="Inclusive buy-signal lower bound (default: 0.88)")
    parser.add_argument("--max-midpoint", type=decimal, default=Settings.max_midpoint, help="Exclusive buy-signal upper bound (default: 0.89)")
    parser.add_argument("--max-price", type=decimal, default=Settings.max_price, help="Maximum executable purchase price, separate from midpoint (default: 0.93)")
    parser.add_argument("--max-spread", type=decimal, default=D("0.05"))
    parser.add_argument("--slippage", type=decimal, default=D("0.01"))
    parser.add_argument("--entry-delay", type=float, default=30, help="Seconds after the five-minute window starts")
    parser.add_argument("--close-buffer", type=float, default=10, help="Stop new submissions this many seconds before close")
    parser.add_argument("--confirmation-seconds", type=float, default=0, help="Optional continuous signal period; unvalidated strategy filter")
    parser.add_argument("--max-book-age", type=float, default=5)
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum fresh orders after confirmed no-fill rejections")
    parser.add_argument("--allow-both-directions", action="store_true", help="Disable the default per-market direction lock")
    return parser


def settings_from_args(args):
    return Settings(min_midpoint=args.min_midpoint, max_midpoint=args.max_midpoint, max_price=args.max_price,
                    max_spread=args.max_spread, slippage=args.slippage,
                    entry_delay=args.entry_delay, close_buffer=args.close_buffer,
                    confirmation_seconds=args.confirmation_seconds, max_book_age=args.max_book_age,
                    max_attempts=args.max_attempts, one_direction=not args.allow_both_directions)


def main():
    load_dotenv(ROOT / ".env")
    args = argument_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = None
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.5 <= poll <= 60:
            raise ValueError("Polling must be 0.5-60 seconds")
        settings = settings_from_args(args)
        bot = Bot(args.live, settings, args.amount)
        if not args.once:
            bot.start_maintenance()
        LOG.info("%s %s; $%s per order plus fees; one direction=%s; polling=%ss",
                 STRATEGY_VERSION, "LIVE" if args.live else "DRY RUN", bot.amount, settings.one_direction, poll)
        LOG.info("Buy signal: %s <= midpoint < %s; maximum purchase price=%s",
                 settings.min_midpoint, settings.max_midpoint, settings.max_price)
        while True:
            try:
                bot.step()
            except httpx.HTTPError as exc:
                LOG.warning("Market data request failed (%s)", type(exc).__name__)
                if args.once:
                    return 1
            if args.once:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        # SDK exceptions can include request data; do not log private signing inputs.
        LOG.error("Stopped (%s). Check configuration, journal and account before retrying.", type(exc).__name__)
        return 1
    finally:
        if bot:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
