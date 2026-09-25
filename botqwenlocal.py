"""BTC five-minute UP/DOWN threshold bot. Dry run unless --live is supplied."""

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import httpx
from dotenv import load_dotenv
import threading

ROOT = Path(__file__).resolve().parent
D = Decimal
LOG = logging.getLogger("btc-bot")
AMOUNT = D("2")
SESSION_ENTRY_DELAY = 60
MAX_ASK = D("0.90")          # never pay more than this to enter
SECOND_SIDE_MAX_ASK = D("0.88")  # stricter bar when the other outcome is already claimed
STOP_LOSS_DROP = None        # disabled: 5c stops fire inside normal noise and
                             # cost more than the flips they guard against (backtest 2026-09-17)
TAKE_PROFIT_RISE = D("0.075")  # sell when bid rises this far above the fill price
EXIT_CLOSE_MARGIN = 30        # stop exits at most this far before settlement
PRICE_LOG_MAX_BYTES = 5_000_000  # rotate per-asset price logs above this size


@dataclass(frozen=True)
class Quote:
    probability: Decimal
    ask: Decimal
    limit: Decimal
    tick: str
    neg_risk: bool
    bid: Decimal


def triggered(outcome, probability):
    return D("0.86") <= probability <= D("0.93")


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


def fill_detail(cost, fill_price, shares):
    return json.dumps({"cost": str(cost), "fill_price": str(fill_price), "shares": str(shares)})


def quote_book(book, now):
    stamp = float(book["timestamp"]) / 1000
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
    limit = (min(ask + D("0.01"), D("0.93"), 1 - tick) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    return Quote((bid + ask) / 2, ask, limit, str(tick),
                 bool(book.get("neg_risk", False)), bid)


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS orders (
            market TEXT, outcome TEXT, day TEXT, amount REAL,
            status TEXT, detail TEXT,
            filled INTEGER NOT NULL DEFAULT 0,
            fill_price REAL NOT NULL DEFAULT 0,
            shares REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(market, outcome))""")
        self.db.commit()
        for column, decl in (("filled", "INTEGER NOT NULL DEFAULT 0"),
                             ("fill_price", "REAL NOT NULL DEFAULT 0"),
                             ("shares", "REAL NOT NULL DEFAULT 0")):
            try:
                self.db.execute(f"ALTER TABLE orders ADD COLUMN {column} {decl}")
                self.db.commit()
            except sqlite3.OperationalError:
                pass

        self.db.execute("""CREATE TABLE IF NOT EXISTS exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold TEXT NOT NULL DEFAULT '0', retry_after REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(market, outcome))""")
        self.db.commit()

    def seen(self, market, outcome):
        with self._lock:
            return self.db.execute("SELECT 1 FROM orders WHERE market=? AND outcome=?", (market, outcome)).fetchone() is not None

    def first_outcome(self, market):
        with self._lock:
            row = self.db.execute("SELECT outcome FROM orders WHERE market=? ORDER BY rowid LIMIT 1", (market,)).fetchone()
            return row[0] if row else None

    def filled_cost_today(self, day):
        with self._lock:
            rows = self.db.execute("SELECT amount FROM orders WHERE day=? AND filled=1", (day,)).fetchall()
            return sum(r[0] for r in rows)

    def reserve(self, market, outcome, amount=AMOUNT):
        # Commit before sending: a timeout or crash must never cause an automatic duplicate.
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                day = datetime.now(timezone.utc).date().isoformat()
                if self.seen(market, outcome):
                    self.db.rollback()
                    return False
                self.db.execute("INSERT INTO orders(market, outcome, day, amount, status, detail) VALUES (?,?,?,?,?,?)",
                                (market, outcome, day, float(amount), "reserved", ""))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise

    def finish(self, market, outcome, status, detail):
        with self._lock:
            self.db.execute("UPDATE orders SET status=?, detail=? WHERE market=? AND outcome=?", (status, detail, market, outcome))
            self.db.commit()

    def record_fill(self, market, outcome, cost, fill_price, shares):
        with self._lock:
            self.db.execute("""UPDATE orders SET filled=1, detail=?, fill_price=?, shares=?
                WHERE market=? AND outcome=?""", (fill_detail(cost, fill_price, shares), float(fill_price), float(shares), market, outcome))
            self.db.commit()

    def open_positions(self, slug):
        # Positions filled in this window that are still open (not yet sold).
        with self._lock:
            return self.db.execute(
                "SELECT market, outcome, fill_price FROM orders WHERE market=? AND filled=1 AND fill_price>0",
                (slug,)).fetchall()

    def sold_shares(self, market, outcome):
        with self._lock:
            row = self.db.execute("SELECT shares FROM orders WHERE market=? AND outcome=?", (market, outcome)).fetchone()
            return D(str(row[0])) if row else None

    def sold_exit(self, market, outcome):
        with self._lock:
            return self.db.execute("SELECT 1 FROM exits WHERE market=? AND outcome=? AND status='sold'", (market, outcome)).fetchone() is not None

    def reserve_exit(self, market, outcome):
        with self._lock:
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
        with self._lock:
            self.db.execute("UPDATE exits SET status=?,detail=?,sold=?,retry_after=? WHERE market=? AND outcome=?",
                            (status, json.dumps(detail), str(sold), time.time() + 5, market, outcome))
            self.db.commit()


class Bot:
    amount = AMOUNT
    asset = "btc"
    journal_prefix = ""
    price_log_dir = ROOT / "price_logs"
    market = None
    usdc_balance = None
    daily_budget = D(200)
    _log_lock = threading.Lock()

    def log_prices(self, slug, outcome, quote, event="quote", response=None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "live" if self.live else "dry-run", "event": event,
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
            with self._log_lock:
                self.price_log_dir.mkdir(parents=True, exist_ok=True)
                path = self.price_log_dir / f"{self.asset.upper()}.jsonl"
                if path.exists() and path.stat().st_size > PRICE_LOG_MAX_BYTES:
                    day = datetime.now(timezone.utc).strftime("%Y%m%d")
                    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
                    rotated = path.with_name(f"{self.asset.upper()}-{day}-{stamp}.jsonl")
                    path.rename(rotated)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        except OSError:
            LOG.warning("Could not write %s price log", self.asset.upper())

    def __init__(self, live):
        self.live = live
        self.http = httpx.Client(timeout=10)
        self.journal = Journal(ROOT / (self.journal_prefix + ("live.sqlite3" if live else "dry-run.sqlite3")))
        self.market = None
        self.client = None
        self._log_lock = threading.Lock()
        self.daily_budget = self._daily_budget()
        self.funder = os.environ.get("POLY_FUNDER") or None
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
            try:
                from py_clob_client_v2 import BalanceAllowanceParams
                balance = self.client.get_balance_allowance(BalanceAllowanceParams())
                self.usdc_balance = D(str(balance.get("balance", "0"))) / D(10**int(balance.get("balance_decimals", 6)))
            except Exception as exc:
                LOG.warning("Could not read USDC balance (%s); continuing without balance guard", type(exc).__name__)
                self.usdc_balance = None
            self.reconcile_unknowns()

    @staticmethod
    def _daily_budget():
        raw = os.environ.get("DAILY_BUDGET_USD")
        if not raw:
            return D(200)  # $2 per side x 6 assets x 2 sides, plus headroom
        try:
            value = D(raw)
        except ArithmeticError:
            raise ValueError(f"DAILY_BUDGET_USD must be a number, got {raw!r}")
        if value <= 0:
            raise ValueError("DAILY_BUDGET_USD must be positive")
        return value

    def reconcile_unknowns(self):
        # Close out orders whose submission outcome was uncertain by checking real fills.
        for market, outcome in self.journal.db.execute(
                "SELECT market, outcome FROM orders WHERE status='unknown' AND day=? ORDER BY rowid",
                (datetime.now(timezone.utc).date().isoformat(),)).fetchall():
            try:
                rows = self.get("https://gamma-api.polymarket.com/markets", slug=market)
                row = next((m for m in rows if m.get("slug") == market), None)
                if not row:
                    continue
                tokens = row.get("clobTokenIds")
                tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
                outcomes = row.get("outcomes")
                outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                token = dict(zip(outcomes, tokens)).get(outcome)
                trades = self.get("https://data-api.polymarket.com/trades",
                                  user=self.funder, market=row.get("conditionId"), limit=100)
                filled = any(t.get("asset") == token and t.get("side") == "BUY" for t in trades) if isinstance(trades, list) else False
            except Exception as exc:
                LOG.warning("Reconciliation check failed for %s (%s)", market, type(exc).__name__)
                continue
            if filled:
                self.journal.finish(market, outcome, "submitted", json.dumps({"reconciled": True}))
                LOG.info("Reconciled %s %s: filled", market, outcome)
            else:
                self.journal.finish(market, outcome, "not_filled", "Reconciled: no fill found")
                LOG.info("Reconciled %s %s: no fill", market, outcome)

    def get(self, url, **params):
        response = self.http.get(url, params=params)
        response.raise_for_status()
        return response.json()


    def discover(self, now):
        start = int(now) // 300 * 300
        slug = f"{self.asset}-updown-5m-{start}"
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
            LOG.info("Market: %s", slug)
            return self.market
        return None

    def open_positions(self, slug):
        return self.journal.open_positions(slug)

    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        # Sessions are five-minute windows; use their start, not process uptime.
        remaining = end - 300 + SESSION_ENTRY_DELAY - time.time()
        if remaining > 0:
            LOG.info("Waiting %.1fs before new buys in %s", remaining, slug)
            return
        self.check_stop_losses(slug, end)
        day = datetime.now(timezone.utc).date().isoformat()
        max_ask = SECOND_SIDE_MAX_ASK if self.journal.first_outcome(slug) else MAX_ASK
        for outcome, token in tokens.items():
            if self.journal.seen(slug, outcome):
                continue
            try:
                q = quote_book(self.get("https://clob.polymarket.com/book", token_id=token), time.time())
            except ValueError as exc:
                LOG.info("%s skipped: %s", outcome, exc)
                continue
            LOG.info("%s %s midpoint=%.1f%% ask=%.3f", self.asset.upper(), outcome, q.probability * 100, q.ask)
            self.log_prices(slug, outcome, q)
            if not triggered(outcome, q.probability):
                continue
            if q.ask > max_ask:
                LOG.info("Skip %s: ask %.3f above limit %.3f", outcome, q.ask, max_ask)
                continue
            if time.time() >= end - 10:
                LOG.info("Skip %s: market closing", outcome)
                continue
            if self.live:
                filled_cost = self.journal.filled_cost_today(day)
                if filled_cost + float(self.amount) > self.daily_budget:
                    LOG.warning("Daily budget $%s reached ($%s filled); no more buys today", self.daily_budget, filled_cost)
                    return
                if self.usdc_balance is not None and self.usdc_balance < D("1") + self.amount:
                    LOG.warning("USDC balance $%s too low for further buys", self.usdc_balance)
                    return
                signed = self.prepare_order(token, q)
                if signed is None:
                    continue
                if time.time() >= end - 10:
                    continue
            if not self.journal.reserve(slug, outcome, self.amount):
                LOG.info("Already attempted this market/outcome")
                continue
            if not self.live:
                self.journal.finish(slug, outcome, "simulated", "Signal only; no order submitted")
                LOG.info("DRY RUN: buy $%s %s %s, maximum price %s", self.amount, self.asset.upper(), outcome, q.limit)
                self.log_prices(slug, outcome, q, "buy_simulated")
                continue
            try:
                response = self.post_with_retry(signed, end)
            except Exception as exc:
                from py_clob_client_v2.exceptions import PolyApiException
                if isinstance(exc, PolyApiException) and exc.status_code == 400 and is_fok_rejection(exc.error_msg):
                    self.journal.finish(slug, outcome, "rejected", json.dumps(exc.error_msg))
                    LOG.warning("%s %s FOK rejected: no fill; continuing", self.asset.upper(), outcome)
                    continue
                self.journal.finish(slug, outcome, "unknown", "Submission failed or timed out; reconcile in Polymarket")
                raise RuntimeError("Order outcome unknown. Check Polymarket before restarting; attempt remains reserved") from None
            if response is None:
                self.journal.finish(slug, outcome, "unknown", "Submission disconnected; retries exhausted or market closing; reconcile in Polymarket")
                LOG.warning("%s %s submission uncertain; skipping this direction and continuing", slug, outcome)
                continue
            if is_fok_rejection(response):
                self.journal.finish(slug, outcome, "rejected", json.dumps(response))
                LOG.warning("%s %s FOK rejected: no fill; continuing", self.asset.upper(), outcome)
                continue
            self.journal.finish(slug, outcome, "submitted", json.dumps(response))
            LOG.info("Order response: %s", response)
            if not response.get("success") or response.get("status") != "matched":
                raise RuntimeError("Order not confirmed matched; review journal and Polymarket account")
            self.log_prices(slug, outcome, q, "buy_matched", response)
            fill = self.fill_response(response, self.amount, q.ask)
            if fill is None:
                continue
            cost, price, shares = fill
            self.journal.record_fill(slug, outcome, cost, price, shares)
            self.journal.finish(slug, outcome, "submitted", fill_detail(cost, price, shares))
            LOG.info("%s %s ACTIVE: $%s at %.3f (%s shares)", self.asset.upper(), outcome, cost, price, shares)

    def prepare_order(self, token, quote):
        from py_clob_client_v2 import MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
        try:
            return self.client.create_market_order(
                MarketOrderArgs(token_id=token, amount=float(self.amount), side=Side.BUY,
                                price=float(quote.limit), order_type=OrderType.FOK),
                PartialCreateOrderOptions(tick_size=quote.tick, neg_risk=quote.neg_risk))
        except Exception as exc:
            if not transient_error(exc):
                raise
            LOG.warning("Order preparation disconnected; retrying on next cycle")
            return None

    @staticmethod
    def fill_response(response, amount, fallback):
        try:
            paid = D(str(response["makingAmount"]))
            shares = D(str(response["takingAmount"]))
            if not (paid.is_finite() and shares.is_finite() and paid > 0 and shares > 0):
                return None
            price = paid / shares
        except (KeyError, ValueError, ArithmeticError):
            return None
        if price <= D(0) or price >= D(1):
            return None
        return paid, min(price, fallback), shares

    def check_stop_losses(self, slug, end):
        if not self.live:
            return
        now = time.time()
        if now >= end - EXIT_CLOSE_MARGIN:
            return
        for market, outcome, fill_price in self.open_positions(slug):
            if self.journal.sold_exit(market, outcome):
                continue
            token = self.market[2].get(outcome) if self.market else None
            if not token:
                continue
            try:
                quote = quote_book(self.get("https://clob.polymarket.com/book", token_id=token), now)
            except Exception as exc:
                LOG.debug("%s %s stop check skipped: %s", market, outcome, exc)
                continue
            fill_price = D(str(fill_price))
            if STOP_LOSS_DROP is not None and quote.bid <= fill_price - STOP_LOSS_DROP:
                self.close_position(market, outcome, token, quote, fill_price, end, "stop-loss")
            elif quote.bid >= fill_price + TAKE_PROFIT_RISE:
                self.close_position(market, outcome, token, quote, fill_price, end, "take-profit")

    def close_position(self, market, outcome, token, quote, fill_price, end, reason):
        shares = self.journal.sold_shares(market, outcome)
        if shares is None or shares <= 0:
            LOG.warning("%s %s: no share count on record; skipping %s", market, outcome, reason)
            return
        from py_clob_client_v2 import MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
        if not self.journal.reserve_exit(market, outcome):
            return
        LOG.info("%s %s: %s triggered (bid %.3f vs fill %.3f); selling %s shares",
                 market, outcome, reason, quote.bid, fill_price, shares)
        self.log_prices(market, outcome, quote, "sell_signal")
        try:
            signed = self.client.create_market_order(
                MarketOrderArgs(token_id=token, amount=float(shares), side=Side.SELL,
                                price=float(max(quote.bid, D("0.01"))), order_type=OrderType.FOK),
                PartialCreateOrderOptions(tick_size=quote.tick, neg_risk=quote.neg_risk))
        except Exception as exc:
            if not transient_error(exc):
                raise
            LOG.warning("Sell preparation failed; retrying next cycle")
            self.journal.finish_exit(market, outcome, "rejected", json.dumps(str(exc)), 0)
            return
        try:
            response = self.post_with_retry(signed, end)
        except Exception as exc:
            self.journal.finish_exit(market, outcome, "rejected", json.dumps(str(exc)), 0)
            LOG.warning("%s %s sell submission failed (%s); retrying next cycle", market, outcome, type(exc).__name__)
            return
        if response is None:
            self.journal.finish_exit(market, outcome, "rejected", json.dumps("sell submission disconnected"), 0)
            return
        if is_fok_rejection(response) or not response.get("success") or response.get("status") != "matched":
            self.journal.finish_exit(market, outcome, "rejected", json.dumps(response), 0)
            LOG.warning("%s %s sell not matched; retrying next cycle", market, outcome)
            return
        sold = D(str(response.get("takingAmount", shares)))
        self.journal.finish_exit(market, outcome, "sold", json.dumps(response), sold)
        self.journal.finish(market, outcome, f"submitted-sold-{reason}",
                            json.dumps({"reason": reason, "sell_bid": str(quote.bid)}))
        self.log_prices(market, outcome, quote, "sell_matched", response)
        LOG.info("%s %s sold %s shares for %s (%s)", market, outcome, sold,
                 response.get("makingAmount"), reason)

    def post_with_retry(self, signed, end):
        from py_clob_client_v2 import OrderType
        for attempt in range(3):
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
                if attempt == 2 or time.time() + delay >= end - 10:
                    return None
                LOG.warning("Submission disconnected; retrying same signed order in %ss (%s/3)", delay, attempt + 2)
                time.sleep(delay)
                if time.time() >= end - 10:
                    return None

    def close(self):
        self.http.close()
        self.journal.db.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Submit real orders using .env wallet credentials")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = None
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.5 <= poll <= 60:
            raise ValueError("Polling must be 0.5-60 seconds")
        bot = Bot(args.live)
        LOG.info("%s; $%s per direction per market plus fees; daily budget $%s; polling delay %ss",
                 "LIVE" if args.live else "DRY RUN", AMOUNT, bot.daily_budget, poll)
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
