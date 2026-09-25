"""BTC five-minute UP/DOWN bot v2 - strategy-aware entries and exits.

R1 hard time exit: flatten by T-EXIT_TAU unless the distance test passes.
R2 distance test: hold to settlement only if |spot-strike| >= HOLD_K * SIGMA*sqrt(tau).
R3 no opinion entries at the open: entry only on model-vs-book mispricing.
R4 cross entry: spot crosses the strike while the book is still stale -> buy the cross side.
R5 base-rate journal: leader snapshot at T-180s and settlement winner are stored.
R6 catalyst clock: SKIP_UTC_RANGES="HH:MM-HH:MM,..." blocks entries in those windows.

Dry run unless --live is supplied. Spot feed (Binance/Coinbase) is a PROXY for the
Chainlink settlement oracle; basis risk remains. Tune SIGMA_PER_SQRT_SEC to realized vol.
"""
import argparse
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
D = Decimal
LOG = logging.getLogger("btc-bot")

AMOUNT = D(os.environ.get("ORDER_USDC", "2"))
SESSION_ENTRY_DELAY = 30
SIGMA = float(os.environ.get("SIGMA_PER_SQRT_SEC", "0.00012"))  # ~3.5% daily vol
MIN_EDGE = D(os.environ.get("MIN_EDGE", "0.06"))
EXIT_TAU = float(os.environ.get("EXIT_TAU", "120"))
HOLD_K = float(os.environ.get("HOLD_K", "3"))
STOP_MID = D(os.environ.get("STOP_MID", "0.25"))
CROSS_MAX_ASK = D(os.environ.get("CROSS_MAX_ASK", "0.60"))
CROSS_TAU = (60.0, 240.0)
MISP_TAU = (150.0, 270.0)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class Quote:
    probability: Decimal
    ask: Decimal
    bid: Decimal
    limit: Decimal
    sell_limit: Decimal
    tick: str
    neg_risk: bool


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
    sell_limit = (max(bid - D("0.01"), tick) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    return Quote((bid + ask) / 2, ask, bid, limit, sell_limit, str(tick), bool(book.get("neg_risk", False)))


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS orders (
            market TEXT, outcome TEXT, day TEXT, amount REAL,
            status TEXT, detail TEXT, PRIMARY KEY(market, outcome))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold TEXT NOT NULL DEFAULT '0', retry_after REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(market, outcome))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS positions (
            market TEXT, outcome TEXT, shares REAL, cost REAL, entry_price REAL,
            status TEXT, PRIMARY KEY(market, outcome))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS settlements (
            market TEXT PRIMARY KEY, leader_outcome TEXT, leader_mid REAL,
            winner TEXT, done INTEGER NOT NULL DEFAULT 0)""")
        self.db.commit()

    def seen(self, market, outcome):
        return self.db.execute("SELECT 1 FROM orders WHERE market=? AND outcome=?", (market, outcome)).fetchone() is not None

    def reserve(self, market, outcome, amount=AMOUNT):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            day = datetime.now(timezone.utc).date().isoformat()
            if self.seen(market, outcome):
                self.db.rollback()
                return False
            self.db.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                            (market, outcome, day, float(amount), "reserved", ""))
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, market, outcome, status, detail):
        self.db.execute("UPDATE orders SET status=?, detail=? WHERE market=? AND outcome=?",
                        (status, detail, market, outcome))
        self.db.commit()

    def reserve_exit(self, market, outcome):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = time.time()
            row = self.db.execute("SELECT status,retry_after FROM exits WHERE market=? AND outcome=?",
                                  (market, outcome)).fetchone()
            if row and not (row[0] in ("rejected", "partial") or row[1] <= now):
                self.db.rollback()
                return False
            self.db.execute("""INSERT INTO exits(market,outcome,status,detail,retry_after) VALUES (?,?,'reserved','',?)
                ON CONFLICT(market,outcome) DO UPDATE SET status='reserved', detail='', retry_after=?""",
                            (market, outcome, now + 10, now + 10))
            self.db.commit()
            return True
        except BaseException:
            self.db.rollback()
            raise

    def finish_exit(self, market, outcome, status, detail, sold):
        self.db.execute("UPDATE exits SET status=?,detail=?,sold=?,retry_after=? WHERE market=? AND outcome=?",
                        (status, json.dumps(detail), str(sold), time.time() + 5, market, outcome))
        self.db.commit()

    def open_position(self, market, outcome, shares, cost, price):
        self.db.execute("INSERT INTO positions VALUES (?,?,?,?,?,'open')
                         ON CONFLICT(market,outcome) DO UPDATE SET shares=?, cost=?, entry_price=?, status='open'",
                        (market, outcome, shares, cost, price, shares, cost, price))
        self.db.commit()

    def open_positions(self, market):
        return self.db.execute("SELECT outcome, shares FROM positions WHERE market=? AND status='open'",
                               (market,)).fetchall()

    def close_position(self, market, outcome):
        self.db.execute("UPDATE positions SET status='closed' WHERE market=? AND outcome=?", (market, outcome))
        self.db.commit()

    def note_leader(self, market, outcome, mid):
        self.db.execute("INSERT OR IGNORE INTO settlements(market, leader_outcome, leader_mid) VALUES (?,?,?)",
                        (market, outcome, float(mid)))
        self.db.commit()

    def leader(self, market):
        return self.db.execute("SELECT leader_outcome FROM settlements WHERE market=?", (market,)).fetchone()

    def mark_settled(self, market, winner):
        self.db.execute("UPDATE settlements SET winner=?, done=1 WHERE market=?", (winner, market))
        self.db.commit()

    def settled(self, market):
        row = self.db.execute("SELECT done FROM settlements WHERE market=?", (market,)).fetchone()
        return bool(row and row[0])


class Bot:
    amount = AMOUNT
    asset = "btc"
    journal_prefix = ""
    price_log_dir = ROOT / "price_logs"

    def __init__(self, live):
        self.live = live
        self.http = httpx.Client(timeout=10)
        self.journal = Journal(ROOT / (self.journal_prefix + ("live.sqlite3" if live else "dry-run.sqlite3")))
        self.market = None
        self.client = None
        self._spot = (0.0, None)
        self._strikes = {}
        self._sign = {}
        self.skip = []
        for chunk in os.environ.get("SKIP_UTC_RANGES", "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            a, b = chunk.split("-")
            h, m = map(int, a.split(":"))
            h2, m2 = map(int, b.split(":"))
            self.skip.append((h * 60 + m, h2 * 60 + m2))
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

    # ---- market data -------------------------------------------------
    def spot(self):
        now = time.time()
        if self._spot[1] is not None and now - self._spot[0] < 1.0:
            return self._spot[1]
        sources = (
            ("https://api.binance.com/api/v3/ticker/price", {"symbol": "BTCUSDT"}, lambda j: j["price"]),
            ("https://api.coinbase.com/v2/prices/BTC-USD/spot", {}, lambda j: j["data"]["amount"]),
        )
        for url, params, parse in sources:
            try:
                value = D(str(parse(self.get(url, **params))))
                self._spot = (now, value)
                return value
            except (httpx.HTTPError, KeyError, ValueError, ArithmeticError):
                continue
        return None

    def strike(self, slug, meta, tau):
        for container in (meta, *(meta.get("events") or [{}])):
            for key in ("strikePrice", "priceToBeat", "strike"):
                value = container.get(key)
                if value not in (None, ""):
                    return D(str(value))
        known = self._strikes.get(slug)
        if known:
            return known
        if tau >= 295:  # observed from the open: capture the proxy strike ourselves
            value = self.spot()
            if value:
                self._strikes[slug] = value
                return value
        return None

    def model_down(self, spot, strike, tau):
        z = (strike - spot) / (spot * SIGMA * math.sqrt(max(tau, 1.0)))
        return D(format(phi(max(-8.0, min(8.0, z))), ".6f"))

    def in_skip(self, now):
        if not self.skip:
            return False
        dt = datetime.fromtimestamp(now, timezone.utc)
        minutes = dt.hour * 60 + dt.minute
        return any(a <= minutes <= b for a, b in self.skip)

    def cross_side(self, slug, spot, strike, now):
        sign = 1 if spot > strike else -1
        prev = self._sign.get(slug)
        self._sign[slug] = (now, sign)
        if prev and now - prev[0] <= 20 and prev[1] != sign:
            return "Down" if sign < 0 else "Up"
        return None

    # ---- logging -----------------------------------------------------
    def log_record(self, record):
        try:
            self.price_log_dir.mkdir(parents=True, exist_ok=True)
            with (self.price_log_dir / f"{self.asset.upper()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            LOG.warning("Could not write %s price log", self.asset.upper())

    def log_prices(self, slug, outcome, quote, event="quote", response=None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "live" if self.live else "dry-run", "event": event,
            "buy_price": str(quote.ask), "bid_price": str(quote.bid),
            "sell_price": str(quote.bid), "midpoint": str(quote.probability),
            "buy_limit": str(quote.limit), "sell_limit": str(quote.sell_limit),
            "fill_price": None,
        }
        if response and event in ("buy_matched", "sell_matched"):
            try:
                paid = D(str(response["makingAmount"]))
                taken = D(str(response["takingAmount"]))
                if paid.is_finite() and taken.is_finite() and paid > 0 and taken > 0:
                    record["fill_price"] = str(paid / taken)
            except (KeyError, ValueError, ArithmeticError):
                pass
            record["order_id"] = response.get("orderID")
        self.log_record(record)

    def log_event(self, slug, event, **fields):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "asset": self.asset.upper(), "market": slug,
                  "mode": "live" if self.live else "dry-run", "event": event}
        record.update(fields)
        self.log_record(record)

    # ---- market discovery --------------------------------------------
    def discover(self, now):
        start = int(now) // 300 * 300
        slug = f"{self.asset}-updown-5m-{start}"
        for m in self.get("https://gamma-api.polymarket.com/markets", slug=slug):
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
            self.market = (slug, end, dict(zip(outcomes, tokens)), m)
            LOG.info("Market: %s", slug)
            return self.market
        return None

    # ---- order submission --------------------------------------------
    def sign_fok(self, side, token, amount, limit, quote):
        from py_clob_client_v2 import MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side
        return self.client.create_market_order(
            MarketOrderArgs(token_id=token, amount=amount,
                            side=Side.BUY if side == "buy" else Side.SELL,
                            price=float(limit), order_type=OrderType.FOK),
            PartialCreateOrderOptions(tick_size=quote.tick, neg_risk=quote.neg_risk))

    def post_with_retry(self, signed, end):
        from py_clob_client_v2 import OrderType
        for attempt in range(3):
            try:
                response = self.client.post_order(signed, OrderType.FOK)
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

    # ---- main loop body ----------------------------------------------
    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens, meta = market
        now = time.time()
        tau = end - now
        if end - 300 + SESSION_ENTRY_DELAY - now > 0:
            LOG.info("Waiting %.1fs before new buys in %s", end - 300 + SESSION_ENTRY_DELAY - now, slug)
            return

        spot = self.spot()
        strike = self.strike(slug, meta, tau)
        model_down = self.model_down(spot, strike, tau) if spot and strike else None

        quotes = {}
        for outcome, token in tokens.items():
            try:
                quotes[outcome] = quote_book(self.get("https://clob.polymarket.com/book", token_id=token), time.time())
            except ValueError as exc:
                LOG.info("%s skipped: %s", outcome, exc)
                continue
            self.log_prices(slug, outcome, quotes[outcome])

        # R5: leader snapshot at T-180s
        if 0 < tau <= 180 and self.journal.leader(slug) is None and len(quotes) == 2:
            leader = max(quotes, key=lambda o: quotes[o].probability)
            self.journal.note_leader(slug, leader, quotes[leader].probability)
            self.log_event(slug, "leader_snapshot", outcome=leader, mid=str(quotes[leader].probability))

        # R3/R4: entries
        if model_down and not self.in_skip(now):
            cross = self.cross_side(slug, spot, strike, now)
            for outcome, token in tokens.items():
                quote = quotes.get(outcome)
                if quote is None or self.journal.seen(slug, outcome):
                    continue
                p_model = model_down if outcome == "Down" else 1 - model_down
                reason = None
                if cross == outcome and CROSS_TAU[0] <= tau <= CROSS_TAU[1] and quote.ask <= CROSS_MAX_ASK:
                    reason = "cross"
                elif MISP_TAU[0] <= tau <= MISP_TAU[1] and p_model - quote.ask >= MIN_EDGE:
                    reason = "mispricing"
                if not reason or tau < 60:
                    continue
                LOG.info("Entry signal %s %s: %s (model=%s ask=%s)", slug, outcome, reason, p_model, quote.ask)
                self.buy(slug, outcome, token, quote, end, reason)

        # R1/R2: exit management for open positions
        for outcome, shares in self.journal.open_positions(slug):
            self.manage_position(slug, outcome, tokens[outcome], shares, quotes.get(outcome), tau, spot, strike)

        # R5: settlement result
        if tau <= 0 and not self.journal.settled(slug):
            self.record_settlement(slug, meta, outcomes_order=list(tokens))

    def buy(self, slug, outcome, token, quote, end, reason):
        signed = None
        if self.live:
            try:
                signed = self.sign_fok("buy", token, float(self.amount), quote.limit, quote)
            except Exception as exc:
                if not transient_error(exc):
                    raise
                LOG.warning("Order preparation disconnected; retrying on next cycle")
                return
            if time.time() >= end - 10:
                return
        if not self.journal.reserve(slug, outcome, self.amount):
            LOG.info("Already attempted this market/outcome")
            return
        if not self.live:
            self.journal.finish(slug, outcome, "simulated", f"signal={reason}; no order submitted")
            shares = float(self.amount / quote.ask)
            self.journal.open_position(slug, outcome, shares, float(self.amount), float(quote.ask))
            LOG.info("DRY RUN: buy $%s %s %s, maximum price %s", self.amount, self.asset.upper(), outcome, quote.limit)
            self.log_prices(slug, outcome, quote, "buy_simulated")
            return
        try:
            response = self.post_with_retry(signed, end)
        except Exception as exc:
            from py_clob_client_v2.exceptions import PolyApiException
            if isinstance(exc, PolyApiException) and exc.status_code == 400 and is_fok_rejection(exc.error_msg):
                self.journal.finish(slug, outcome, "rejected", json.dumps(exc.error_msg))
                LOG.warning("%s %s FOK rejected: no fill; continuing", self.asset.upper(), outcome)
                return
            self.journal.finish(slug, outcome, "unknown", "Submission failed or timed out; reconcile in Polymarket")
            raise RuntimeError("Order outcome unknown. Check Polymarket before restarting; attempt remains reserved") from None
        if response is None:
            self.journal.finish(slug, outcome, "unknown", "Submission disconnected; reconcile in Polymarket")
            LOG.warning("%s %s submission uncertain; skipping this direction", slug, outcome)
            return
        if is_fok_rejection(response):
            self.journal.finish(slug, outcome, "rejected", json.dumps(response))
            LOG.warning("%s %s FOK rejected: no fill; continuing", self.asset.upper(), outcome)
            return
        self.journal.finish(slug, outcome, "submitted", json.dumps(response))
        if not response.get("success") or response.get("status") != "matched":
            raise RuntimeError("Order not confirmed matched; review journal and Polymarket account")
        paid = D(str(response["makingAmount"]))
        shares = D(str(response["takingAmount"]))
        self.journal.open_position(slug, outcome, float(shares), float(paid), float(paid / shares))
        self.log_prices(slug, outcome, quote, "buy_matched", response)

    def manage_position(self, slug, outcome, token, shares, quote, tau, spot, strike):
        if tau <= 10 or quote is None:
            return
        if spot and strike:
            distance = abs(spot - strike) / spot
            sigma_left = SIGMA * math.sqrt(max(tau, 1.0))
            favorable = (outcome == "Down") == (spot < strike)
            hold = favorable and distance >= HOLD_K * sigma_left   # R2
        else:
            hold = quote.probability >= D("0.985")                 # market-implied fallback
        sell = (tau <= EXIT_TAU and not hold) or quote.probability <= STOP_MID
        if not sell:
            return
        LOG.info("Exit signal %s %s: tau=%.0fs mid=%s hold=%s", slug, outcome, tau, quote.probability, hold)
        if not self.journal.reserve_exit(slug, outcome):
            return
        if not self.live:
            proceeds = shares * float(quote.bid)
            self.journal.close_position(slug, outcome)
            self.journal.finish_exit(slug, outcome, "simulated", "dry-run sell", shares)
            self.log_prices(slug, outcome, quote, "sell_simulated")
            LOG.info("DRY RUN: sell %s shares of %s %s at ~%s", shares, self.asset.upper(), outcome, quote.bid)
            return
        try:
            signed = self.sign_fok("sell", token, shares, quote.sell_limit, quote)
        except Exception as exc:
            if not transient_error(exc):
                raise
            LOG.warning("Sell preparation disconnected; retrying next cycle")
            return
        end = tau + time.time()
        try:
            response = self.post_with_retry(signed, end)
        except Exception as exc:
            from py_clob_client_v2.exceptions import PolyApiException
            if isinstance(exc, PolyApiException) and exc.status_code == 400 and is_fok_rejection(exc.error_msg):
                self.journal.finish_exit(slug, outcome, "rejected", exc.error_msg, 0)
                return
            self.journal.finish_exit(slug, outcome, "unknown", "sell failed; reconcile", 0)
            raise RuntimeError("Sell outcome unknown; reconcile before restarting") from None
        if response is None:
            self.journal.finish_exit(slug, outcome, "unknown", "sell disconnected; reconcile", 0)
            LOG.warning("%s %s sell uncertain; position stays open", slug, outcome)
            return
        if is_fok_rejection(response):
            self.journal.finish_exit(slug, outcome, "rejected", response, 0)
            LOG.warning("%s %s sell FOK rejected; will retry", self.asset.upper(), outcome)
            return
        if not response.get("success") or response.get("status") != "matched":
            self.journal.finish_exit(slug, outcome, "unknown", response, 0)
            raise RuntimeError("Sell not confirmed matched; review journal and account")
        self.journal.close_position(slug, outcome)
        self.journal.finish_exit(slug, outcome, "submitted", response, shares)
        self.log_prices(slug, outcome, quote, "sell_matched", response)

    def record_settlement(self, slug, meta, outcomes_order):
        try:
            rows = self.get("https://gamma-api.polymarket.com/markets", slug=slug)
            prices = None
            for m in rows:
                if m.get("slug") == slug:
                    prices = m.get("outcomePrices")
                    break
            if not prices:
                return
            prices = json.loads(prices) if isinstance(prices, str) else prices
            winner = outcomes_order[int(max(range(2), key=lambda i: float(prices[i])))]
            self.journal.mark_settled(slug, winner)
            leader = self.journal.leader(slug)
            self.log_event(slug, "settlement", winner=winner,
                           leader_outcome=leader[0] if leader else None,
                           leader_held=bool(leader and leader[0] == winner))
            LOG.info("Settlement %s: winner=%s leader_held=%s", slug, winner, bool(leader and leader[0] == winner))
        except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
            LOG.warning("Settlement logging failed for %s (%s)", slug, type(exc).__name__)

    def close(self):
        self.http.close()
        self.journal.db.close()


def main():
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
        LOG.info("%s; $%s per direction per market plus fees; polling delay %ss; exit at T-%.0fs unless |d|>=%.0f*sigma",
                 "LIVE" if args.live else "DRY RUN", AMOUNT, poll, EXIT_TAU, HOLD_K)
        while True:
            try:
                bot.step()
            except httpx.HTTPError as exc:
                LOG.warning("Market data request failed (%s)", type(exc).__name__)
            if args.once:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s). Check configuration, journal and account before retrying.", type(exc).__name__)
        return 1
    finally:
        if bot:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())