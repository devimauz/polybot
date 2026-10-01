"""Live-capable Polymarket version of the final-minute NineFive strategy.

The bot watches BTC/ETH/SOL/BNB/DOGE five-minute UP/DOWN markets. During
the final minute, it buys one side using a fixed $10 stake after that side's
ask remains strictly above $0.90 (and no higher than $0.98) for five continuous
seconds. After entry, it sells the purchased side if its best bid falls to
$0.80 or lower.

Dry run (default):
  python ninefive_polymarket_bot.py

Real-money trading:
  python ninefive_polymarket_bot.py --live
"""

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR

import httpx
from dotenv import load_dotenv

from bot import Bot, D, LOG, ROOT, is_fok_rejection
from ninefive_bot import (ABOVE_HOLD_SECONDS, FAV_ASK_MAX, FAV_ASK_MIN,
                          FAV_SLIPPAGE, MAX_LEG_ATTEMPTS, NineFiveBot,
                          NineFiveJournal, TRADE_WINDOW_SECONDS)
from polymarket_feed import MarketFeed


STAKE = D("10")
MAX_OPEN_TRADES = 12
MIN_LIVE_STAKE = D("1.00")
STOP_LOSS_PRICE = D("0.50")
STOP_SLIPPAGE = D("0.05")
DEFAULT_POLL_SECONDS = 0.05
GUI_REFRESH_SECONDS = 0.25
_BALANCE_ORDER_LOCK = threading.Lock()


class TelegramNotifier:
    """Best-effort Telegram alerts; notification failures never stop trading."""

    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, message):
        if not self.enabled:
            return False
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message}, timeout=8)
            response.raise_for_status()
            if response.json().get("ok") is not True:
                raise ValueError("Telegram returned ok=false")
            return True
        except Exception as exc:
            # Do not log the exception text: an HTTP error may contain the URL/token.
            LOG.warning("Telegram notification failed (%s)", type(exc).__name__)
            return False


class NineFivePolymarketBot(NineFiveBot):
    """NineFive strategy with direct Polymarket discovery and execution."""

    amount = STAKE
    journal_type = NineFiveJournal
    journal_prefix = "ninefive-polymarket-"
    price_log_dir = ROOT / "price_logs" / "ninefive-polymarket"

    def __init__(self, live, asset="btc"):
        if asset not in Bot.assets:
            raise ValueError(f"asset must be one of {Bot.assets}")
        self.asset = asset
        self.live = live
        self.http = httpx.Client(timeout=10)
        suffix = "live.sqlite3" if live else "dry-run.sqlite3"
        self.journal = self.journal_type(ROOT / (self.journal_prefix + suffix))
        order_columns = {
            column[1] for column in self.journal.db.execute("PRAGMA table_info(orders)")
        }
        if "condition_id" not in order_columns:
            self.journal.db.execute("ALTER TABLE orders ADD COLUMN condition_id TEXT")
            self.journal.db.commit()
        self.market = None
        self.condition_id = None
        self.neg_risk = False
        self.client = None
        self.feed = MarketFeed()
        self.above_since = {"Up": None, "Down": None}
        self.timer_market = None
        self.last_quotes = ("", {})
        self.last_balance = None
        self.last_stake = STAKE if not live else None
        self.last_open_trade_count = 0
        self._logged_bbo = {}
        self.notifier = TelegramNotifier()
        self._next_settlement_check = 0.0
        self.journal.db.execute("""CREATE TABLE IF NOT EXISTS telegram_notifications (
            market TEXT, outcome TEXT, event TEXT, sent REAL,
            PRIMARY KEY(market, outcome, event))""")
        self.journal.db.execute("""CREATE TABLE IF NOT EXISTS polymarket_settlements (
            market TEXT, outcome TEXT, winner TEXT, pnl REAL, resolved REAL,
            PRIMARY KEY(market, outcome))""")
        self.journal.db.execute("""CREATE TABLE IF NOT EXISTS polymarket_exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold REAL DEFAULT 0, proceeds REAL DEFAULT 0, pnl REAL,
            updated REAL, PRIMARY KEY(market, outcome))""")
        self.journal.db.commit()

        if not live:
            return
        try:
            if self.get("https://polymarket.com/api/geoblock").get("blocked") is not False:
                raise RuntimeError("Polymarket trading is unavailable from this location")
            from py_clob_client_v2 import ClobClient
            key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
            if not key:
                raise ValueError("Set POLY_PRIVATE_KEY in .env before using --live")
            signature = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
            funder = os.environ.get("POLY_FUNDER", "").strip() or None
            if signature not in (0, 1, 2, 3) or (signature != 0 and not funder):
                raise ValueError("Use signature type 0/1/2/3 and set POLY_FUNDER for smart wallets")
            self.client = ClobClient("https://clob.polymarket.com", chain_id=137,
                                     key=key, signature_type=signature, funder=funder)
            self.client.set_api_creds(self.client.create_or_derive_api_key())
        except BaseException:
            self.close()
            raise

    def get(self, url, **params):
        response = self.http.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def discover(self, now):
        """Discover the current market directly through Polymarket Gamma."""
        start = int(now) // 300 * 300
        slug = f"{self.asset}-updown-5m-{start}"
        if self.market and self.market[0] == slug and now < self.market[1]:
            return self.market
        rows = self.get("https://gamma-api.polymarket.com/markets", slug=slug)
        for market in rows:
            if (market.get("slug") != slug or market.get("closed")
                    or not market.get("active") or not market.get("acceptingOrders")):
                continue
            end = datetime.fromisoformat(
                market["endDate"].replace("Z", "+00:00")).timestamp()
            if abs(end - (start + 300)) > 1:
                raise ValueError("market does not match the expected five-minute window")
            outcomes = market["outcomes"]
            tokens = market["clobTokenIds"]
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
            if (len(tokens) != 2 or len(outcomes) != 2
                    or set(outcomes) != {"Up", "Down"} or len(set(tokens)) != 2):
                raise ValueError("unexpected Polymarket outcomes")
            self.condition_id = market.get("conditionId")
            if not self.condition_id:
                raise ValueError("Polymarket market is missing its condition ID")
            self.neg_risk = bool(market.get("negRisk", False))
            self.market = (slug, end, dict(zip(outcomes, tokens)))
            LOG.info("Market: %s", slug)
            return self.market
        return None

    def log_prices(self, slug, outcome, quote, event="quote", response=None):
        token = None
        if self.market and self.market[0] == slug:
            token = self.market[2].get(outcome)
        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "polymarket-live" if self.live else "dry-run", "event": event,
            "ask": str(quote.ask), "bid": str(quote.bid),
            "midpoint": str(quote.probability), "tick": str(quote.tick),
            "condition_id": self.condition_id, "token_id": token,
            "seconds_remaining": (None if not self.market else
                                  round(self.market[1] - time.time(), 3)),
            "configured_stake": str(STAKE),
            "stop_loss_bid": str(STOP_LOSS_PRICE),
        }
        if response:
            record["response"] = response
        try:
            self.price_log_dir.mkdir(parents=True, exist_ok=True)
            with (self.price_log_dir / f"{self.asset.upper()}.jsonl").open(
                    "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            LOG.warning("Could not write %s price log", self.asset.upper())

    def audit_quotes(self):
        """Persist every changed BBO for both sides throughout the full market."""
        market = self.discover(time.time())
        if not market:
            return
        slug, _end, tokens = market
        self.feed.subscribe(tokens)
        quotes = self.feed.quotes()
        stamp = time.time()
        if quotes:
            self.last_quotes = (slug, {name: (quote.ask, stamp)
                                       for name, quote in quotes.items()})
        for outcome, quote in quotes.items():
            key = (slug, outcome)
            bbo = (quote.bid, quote.ask, str(quote.tick))
            if self._logged_bbo.get(key) == bbo:
                continue
            self._logged_bbo[key] = bbo
            self.log_prices(slug, outcome, quote, "bbo_change")

    def log_audit_event(self, slug, outcome, event, **fields):
        """Write a structured non-price event without secrets or signatures."""
        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "asset": self.asset.upper(), "market": slug, "outcome": outcome,
            "mode": "polymarket-live" if self.live else "dry-run",
            "event": event, "condition_id": getattr(self, "condition_id", None),
        }
        record.update(fields)
        try:
            self.price_log_dir.mkdir(parents=True, exist_ok=True)
            with (self.price_log_dir / f"{self.asset.upper()}.jsonl").open(
                    "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
        except OSError:
            LOG.warning("Could not write %s audit event", self.asset.upper())

    def _notify_once(self, slug, outcome, event, message):
        sent = self.journal.db.execute(
            "SELECT 1 FROM telegram_notifications WHERE market=? AND outcome=? AND event=?",
            (slug, outcome, event)).fetchone()
        if sent or not self.notifier.send(message):
            return False
        self.journal.db.execute(
            "INSERT OR IGNORE INTO telegram_notifications(market,outcome,event,sent) VALUES (?,?,?,?)",
            (slug, outcome, event, time.time()))
        self.journal.db.commit()
        return True

    def check_settlements(self):
        """Notify one newly resolved live trade, rate-limited per asset bot."""
        now = time.time()
        if not self.live or now < self._next_settlement_check:
            return
        self._next_settlement_check = now + 15
        prefix = self.asset + "-updown-5m-%"
        rows = self.journal.db.execute("""
            SELECT o.market,o.outcome,o.detail,o.condition_id,e.status,e.pnl
            FROM orders o LEFT JOIN telegram_notifications n
              ON n.market=o.market AND n.outcome=o.outcome AND n.event='settlement'
            LEFT JOIN polymarket_exits e
              ON e.market=o.market AND e.outcome=o.outcome
            WHERE o.status='matched' AND o.market LIKE ? AND n.market IS NULL
            ORDER BY o.market LIMIT 10""", (prefix,)).fetchall()
        for slug, outcome, detail, condition_id, exit_status, exit_pnl in rows:
            try:
                end = int(slug.rsplit("-", 1)[1]) + 300
            except (ValueError, IndexError):
                continue
            if now < end or not condition_id:
                continue
            try:
                market = self.get(f"https://clob.polymarket.com/markets/{condition_id}")
            except (httpx.HTTPError, ValueError, TypeError):
                continue
            if market.get("closed") is not True:
                continue
            winners = [token for token in market.get("tokens", [])
                       if token.get("winner") is True]
            if len(winners) != 1 or winners[0].get("outcome") not in ("Up", "Down"):
                continue
            winner = winners[0]["outcome"]
            try:
                response = json.loads(detail)
                paid = D(str(response["makingAmount"]))
                shares = D(str(response["takingAmount"]))
                pnl = (D(str(exit_pnl)) if exit_status == "sold"
                       else (shares if outcome == winner else D(0)) - paid)
            except (KeyError, TypeError, ValueError, ArithmeticError):
                continue
            result = ("STOPPED" if exit_status == "sold"
                      else ("WIN" if outcome == winner else "LOSS"))
            pnl_text = f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}"
            self.journal.db.execute("""
                INSERT INTO polymarket_settlements(market,outcome,winner,pnl,resolved)
                VALUES (?,?,?,?,?) ON CONFLICT(market,outcome) DO UPDATE SET
                winner=excluded.winner,pnl=excluded.pnl,resolved=excluded.resolved""",
                (slug, outcome, winner, float(pnl), now))
            self.journal.db.commit()
            self.log_audit_event(
                slug, outcome, "settlement", winner=winner, result=result,
                pnl=str(pnl), stopped=(exit_status == "sold"))
            self._notify_once(
                slug, outcome, "settlement",
                f"{result} | {self.asset.upper()} {outcome}\n"
                f"Winner: {winner}\nP&L before gas: {pnl_text}\nMarket: {slug}")

    def step(self):
        try:
            self.audit_quotes()
            self.monitor_stop_loss()
            return super().step()
        finally:
            self.check_settlements()

    def _reserve_exit(self, slug, outcome):
        def reserve():
            self.journal.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.journal.db.execute(
                    "SELECT status FROM polymarket_exits WHERE market=? AND outcome=?",
                    (slug, outcome)).fetchone()
                if row and row[0] in ("reserved", "unknown", "sold"):
                    self.journal.db.rollback()
                    return False
                self.journal.db.execute("""
                    INSERT INTO polymarket_exits(market,outcome,status,detail,updated)
                    VALUES (?,?,?,?,?) ON CONFLICT(market,outcome) DO UPDATE SET
                    status=excluded.status,detail=excluded.detail,updated=excluded.updated""",
                    (slug, outcome, "reserved", "", time.time()))
                self.journal.db.commit()
                return True
            except BaseException:
                self.journal.db.rollback()
                raise
        return self.journal._retry(reserve)

    def _finish_exit(self, slug, outcome, status, detail,
                     sold=0, proceeds=0, pnl=None):
        def finish():
            self.journal.db.execute("""
                UPDATE polymarket_exits
                SET status=?,detail=?,sold=?,proceeds=?,pnl=?,updated=?
                WHERE market=? AND outcome=?""",
                (status, detail, float(sold), float(proceeds),
                 None if pnl is None else float(pnl), time.time(), slug, outcome))
            self.journal.db.commit()
        return self.journal._retry(finish)

    def monitor_stop_loss(self):
        """Keep watching the filled side and sell when its bid reaches $0.80."""
        market = self.market
        if not market:
            return
        slug, end, tokens = market
        if time.time() >= end:
            return
        row = self.journal.db.execute("""
            SELECT o.outcome,o.shares,o.fill_price
            FROM orders o LEFT JOIN polymarket_exits e
              ON e.market=o.market AND e.outcome=o.outcome
            WHERE o.market=? AND o.status='matched' AND o.shares>0
              AND COALESCE(e.status,'') NOT IN ('sold','unknown','reserved')
            LIMIT 1""", (slug,)).fetchone()
        if not row:
            return
        outcome, shares, fill_price = row
        quotes = self.feed.quotes()
        stamp = time.time()
        self.last_quotes = (slug, {name: (item.ask, stamp)
                                   for name, item in quotes.items()})
        quote = quotes.get(outcome)
        if quote is None:
            return
        self.log_prices(slug, outcome, quote, "position_watch")
        if quote.bid > STOP_LOSS_PRICE:
            return
        self._execute_stop_loss(slug, end, tokens[outcome], outcome,
                                D(str(shares)), D(str(fill_price)), quote)

    def _execute_stop_loss(self, slug, end, token, outcome,
                           shares, fill_price, quote):
        if not self._reserve_exit(slug, outcome):
            return False
        LOG.warning("STOP LOSS %s %s: bid %.3f <= %.2f; selling %s shares",
                    self.asset.upper(), outcome, quote.bid,
                    STOP_LOSS_PRICE, shares)
        self.log_audit_event(
            slug, outcome, "stop_triggered", bid=str(quote.bid),
            shares=str(shares), fill_price=str(fill_price))
        self._notify_once(
            slug, outcome, "stop_triggered",
            f"STOP LOSS TRIGGERED | {self.asset.upper()} {outcome}\n"
            f"Bid: ${quote.bid}\nShares: {shares:.4f}\nMarket: {slug}")

        if not self.live:
            proceeds = shares * quote.bid
            pnl = proceeds - shares * fill_price
            self._finish_exit(slug, outcome, "sold", "dry-run stop loss",
                              shares, proceeds, pnl)
            self.log_audit_event(slug, outcome, "stop_sold_dry_run",
                                 proceeds=str(proceeds), pnl=str(pnl))
            return True

        last_detail = "FOK sell not filled"
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= end:
                self._finish_exit(slug, outcome, "rejected", "market closed")
                return False
            current = self.feed.quotes().get(outcome)
            if current is None:
                time.sleep(0.05)
                continue
            tick = D(current.tick)
            limit = max(D("0.01"), current.bid - STOP_SLIPPAGE)
            limit = (limit / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            try:
                from py_clob_client_v2 import (MarketOrderArgs, OrderType,
                                               PartialCreateOrderOptions, Side)
                signed = self.client.create_market_order(
                    MarketOrderArgs(token_id=token, amount=float(shares),
                                    side=Side.SELL, price=float(limit),
                                    order_type=OrderType.FOK),
                    PartialCreateOrderOptions(tick_size=current.tick,
                                              neg_risk=self.neg_risk))
                response = self.client.post_order(signed, OrderType.FOK)
            except Exception as exc:
                message = str(exc).lower()
                last_detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                if "fully filled or killed" in message:
                    if attempt < MAX_LEG_ATTEMPTS:
                        time.sleep(0.05)
                        continue
                    self._finish_exit(slug, outcome, "rejected", last_detail)
                    return False
                self._finish_exit(slug, outcome, "unknown", last_detail)
                self._notify_once(
                    slug, outcome, "stop_unknown",
                    f"STOP LOSS UNCERTAIN | {self.asset.upper()} {outcome}\n"
                    f"Check Polymarket manually.\nMarket: {slug}")
                return False
            if is_fok_rejection(response) or response.get("status") in (
                    "rejected", "cancelled", "canceled", "expired"):
                last_detail = json.dumps(response)
                if attempt < MAX_LEG_ATTEMPTS:
                    time.sleep(0.05)
                    continue
                self._finish_exit(slug, outcome, "rejected", last_detail)
                return False
            if not response.get("success") or response.get("status") != "matched":
                self._finish_exit(slug, outcome, "unknown", json.dumps(response))
                return False

            sold = D(str(response.get("makingAmount", shares)))
            proceeds = D(str(response.get("takingAmount", D(0))))
            pnl = proceeds - shares * fill_price
            self._finish_exit(slug, outcome, "sold", json.dumps(response),
                              sold, proceeds, pnl)
            self.log_prices(slug, outcome, current, "stop_loss_sold", response)
            pnl_text = f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}"
            self._notify_once(
                slug, outcome, "stop_sold",
                f"STOP LOSS SOLD | {self.asset.upper()} {outcome}\n"
                f"Proceeds: ${proceeds:.2f}\nP&L: {pnl_text}\nMarket: {slug}")
            self.log_audit_event(
                slug, outcome, "stop_sold", sold=str(sold),
                proceeds=str(proceeds), pnl=str(pnl), response=response)
            LOG.warning("STOP LOSS SOLD %s %s: proceeds $%.2f, P&L %+.2f",
                        self.asset.upper(), outcome, proceeds, pnl)
            return True

        self._finish_exit(slug, outcome, "rejected", last_detail)
        return False

    def open_trade_count(self):
        """Count this bot's matched trades whose five-minute market is active."""
        now = time.time()
        count = 0
        for (market,) in self.journal.db.execute("""
                SELECT o.market FROM orders o
                LEFT JOIN polymarket_exits e
                  ON e.market=o.market AND e.outcome=o.outcome
                WHERE o.status='matched' AND COALESCE(e.status,'')!='sold'"""):
            try:
                if int(market.rsplit("-", 1)[1]) + 300 > now:
                    count += 1
            except (ValueError, IndexError):
                continue
        return count

    def available_stake(self):
        """Return the fixed stake when balance and trade-slot capacity allow it."""
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        result = self.client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        balance = D(str(result["balance"])) / D("1000000")
        if not balance.is_finite() or balance < 0:
            raise ValueError("invalid Polymarket collateral balance")
        open_trades = self.open_trade_count()
        if open_trades >= MAX_OPEN_TRADES or balance < STAKE:
            self.last_balance, self.last_stake = balance, D(0)
            self.last_open_trade_count = open_trades
            return D(0)
        stake = STAKE
        self.last_balance, self.last_stake = balance, stake
        self.last_open_trade_count = open_trades
        return stake

    def _buy_leg(self, slug, end, tokens, outcome, max_price, slippage=FAV_SLIPPAGE):
        """Serialize balance sizing and live submission across all asset bots."""
        quote = self.feed.quotes().get(outcome)
        self.log_audit_event(
            slug, outcome, "entry_signal",
            ask=None if quote is None else str(quote.ask),
            bid=None if quote is None else str(quote.bid),
            seconds_remaining=round(end - time.time(), 3))
        if self.live:
            with _BALANCE_ORDER_LOCK:
                return self._buy_leg_impl(slug, end, tokens, outcome,
                                          max_price, slippage)
        return self._buy_leg_impl(slug, end, tokens, outcome, max_price, slippage)

    def _buy_leg_impl(self, slug, end, tokens, outcome, max_price,
                      slippage=FAV_SLIPPAGE):
        """Place a direct Polymarket FOK market buy with a protected limit."""
        if self.journal.seen(slug, outcome):
            return True

        token = tokens[outcome]
        stake = self.amount
        if self.live:
            try:
                stake = self.available_stake()
            except Exception as exc:
                LOG.warning("%s balance lookup failed (%s); order skipped",
                            self.asset.upper(), type(exc).__name__)
                return False
            if stake < MIN_LIVE_STAKE:
                LOG.warning("%s available balance $%.2f gives stake $%.2f below minimum; skipped",
                            self.asset.upper(), self.last_balance, stake)
                return False
            LOG.info("%s live balance $%.2f; %d/%d slots open; next stake $%.2f",
                     self.asset.upper(), self.last_balance,
                     self.last_open_trade_count, MAX_OPEN_TRADES, stake)
        last_detail = "no fill in attempts"
        reserved = False
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= end:
                if reserved:
                    self.journal.finish(slug, outcome, "unknown", "market closed before fill")
                return False
            quote = self.feed.quotes().get(outcome)
            if quote is None or not (FAV_ASK_MIN < quote.ask <= max_price):
                LOG.info("%s %s attempt %d: signal no longer executable",
                         self.asset.upper(), outcome, attempt)
                return False

            tick = D(quote.tick)
            limit = min(D("0.99"), quote.ask + slippage)
            limit = (limit / tick).to_integral_value(rounding=ROUND_FLOOR) * tick

            if not self.live:
                if not self.journal.reserve(slug, outcome, stake,
                                            condition_id=self.condition_id or ""):
                    return False
                self.journal.finish(slug, outcome, "simulated",
                                    f"dry run, ask {quote.ask}, limit {limit}")
                LOG.info("DRY RUN: buy $%s %s %s on Polymarket at limit %s",
                         stake, self.asset.upper(), outcome, limit)
                self.log_prices(slug, outcome, quote, "buy_simulated")
                self._notify_once(
                    slug, outcome, "dry_run",
                    f"DRY RUN | {self.asset.upper()} {outcome}\n"
                    f"Stake: ${stake}\nAsk: {quote.ask}\nLimit: {limit}\nMarket: {slug}")
                return True

            try:
                from py_clob_client_v2 import (MarketOrderArgs, OrderType,
                                               PartialCreateOrderOptions, Side)
                signed = self.client.create_market_order(
                    MarketOrderArgs(token_id=token, amount=float(stake),
                                    side=Side.BUY, price=float(limit),
                                    order_type=OrderType.FOK),
                    PartialCreateOrderOptions(tick_size=quote.tick,
                                              neg_risk=self.neg_risk))
                if not reserved:
                    if not self.journal.reserve(slug, outcome, stake,
                                                condition_id=self.condition_id or ""):
                        return False
                    reserved = True
                response = self.client.post_order(signed, OrderType.FOK)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:300]}"
                error_text = str(exc).lower()
                if "fully filled or killed" in error_text:
                    last_detail = detail
                    if attempt < MAX_LEG_ATTEMPTS:
                        time.sleep(0.1)
                        continue
                    if reserved:
                        self.journal.finish(slug, outcome, "rejected", detail)
                        self._notify_once(
                            slug, outcome, "order_rejected",
                            f"ORDER REJECTED | {self.asset.upper()} {outcome}\n"
                            f"Stake: ${stake}\nReason: FOK liquidity unavailable\n"
                            f"Market: {slug}")
                    LOG.warning("%s %s FOK rejected after %d attempts",
                                self.asset.upper(), outcome, MAX_LEG_ATTEMPTS)
                    return False
                if "not enough balance / allowance" in error_text:
                    if reserved:
                        self.journal.finish(slug, outcome, "rejected", detail)
                        self._notify_once(
                            slug, outcome, "order_rejected",
                            f"ORDER REJECTED | {self.asset.upper()} {outcome}\n"
                            f"Stake: ${stake}\nReason: insufficient available balance/allowance\n"
                            f"Market: {slug}")
                    LOG.warning("%s %s rejected: insufficient available balance/allowance",
                                self.asset.upper(), outcome)
                    return False
                if reserved:
                    self.journal.finish(slug, outcome, "unknown", detail)
                    self._notify_once(
                        slug, outcome, "order_unknown",
                        f"ORDER UNCERTAIN | {self.asset.upper()} {outcome}\n"
                        f"Stake: ${stake}\nMarket: {slug}\nCheck Polymarket manually.")
                LOG.warning("%s %s live order failed: %s",
                            self.asset.upper(), outcome, detail)
                return False

            if is_fok_rejection(response) or response.get("status") in (
                    "rejected", "cancelled", "canceled", "expired"):
                last_detail = json.dumps(response)
                if attempt < MAX_LEG_ATTEMPTS:
                    time.sleep(0.1)
                    continue
                self.journal.finish(slug, outcome, "rejected", last_detail)
                self._notify_once(
                    slug, outcome, "order_rejected",
                    f"ORDER REJECTED | {self.asset.upper()} {outcome}\n"
                    f"Stake: ${stake}\nMarket: {slug}")
                return False
            if not response.get("success") or response.get("status") != "matched":
                self.journal.finish(slug, outcome, "unknown", json.dumps(response))
                self._notify_once(
                    slug, outcome, "order_unknown",
                    f"ORDER UNCERTAIN | {self.asset.upper()} {outcome}\n"
                    f"Stake: ${stake}\nMarket: {slug}\nCheck Polymarket manually.")
                LOG.warning("%s %s order outcome uncertain: %s",
                            self.asset.upper(), outcome, response)
                return False

            self.journal.finish(slug, outcome, "matched", json.dumps(response))
            self.log_prices(slug, outcome, quote, "buy_matched", response)
            try:
                paid = D(str(response["makingAmount"]))
                shares = D(str(response["takingAmount"]))
                fill = paid / shares
                fill_text = f"${fill:.4f} | {shares:.4f} shares | paid ${paid:.2f}"
            except (KeyError, TypeError, ValueError, ArithmeticError):
                fill_text = f"stake ${stake} | signal ask {quote.ask}"
            self._notify_once(
                slug, outcome, "order_filled",
                f"ORDER FILLED | {self.asset.upper()} {outcome}\n"
                f"{fill_text}\nMarket: {slug}")
            LOG.info("%s %s live Polymarket order filled: %s",
                     self.asset.upper(), outcome, response)
            # Give the CLOB balance endpoint time to reflect this fill before
            # another asset sizes from the shared wallet.
            time.sleep(0.5)
            return True

        if reserved:
            self.journal.finish(slug, outcome, "rejected", last_detail)
        return False


def _journal_rows(bot, limit=10):
    """Newest direct-Polymarket attempts with any known settlement."""
    sql = """SELECT o.market,o.outcome,o.status,o.detail,o.fill_price,o.shares,
                    s.winner,COALESCE(e.pnl,s.pnl),e.status
             FROM orders o LEFT JOIN polymarket_settlements s
               ON s.market=o.market AND s.outcome=o.outcome
             LEFT JOIN polymarket_exits e
               ON e.market=o.market AND e.outcome=o.outcome
             ORDER BY o.market DESC LIMIT ?"""
    return bot.journal.db.execute(sql, (limit,)).fetchall()


def run_dashboard(args, assets):
    """Rich terminal dashboard backed only by Polymarket and the local journal."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    bots = [NineFivePolymarketBot(args.live, asset) for asset in assets]
    log_file = ROOT / "ninefive_polymarket_bot.log"
    cursor = log_file.stat().st_size if log_file.exists() else 0
    events = []
    try:
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            console.show_cursor(False)
            next_render = 0.0
            try:
                while True:
                    futures = [(bot, workers.submit(bot.step)) for bot in bots]
                    for bot, future in futures:
                        try:
                            future.result()
                        except httpx.HTTPError as exc:
                            LOG.warning("%s request failed (%s)",
                                        bot.asset.upper(), type(exc).__name__)
                        except Exception as exc:
                            LOG.exception("%s worker failed (%s)",
                                          bot.asset.upper(), type(exc).__name__)

                    frame_time = time.monotonic()
                    if not args.once and frame_time < next_render:
                        time.sleep(args.poll_seconds)
                        continue
                    next_render = frame_time + GUI_REFRESH_SECONDS

                    try:
                        if log_file.stat().st_size < cursor:
                            cursor = 0
                        with log_file.open("r", encoding="utf-8", errors="replace") as stream:
                            stream.seek(cursor)
                            for line in stream:
                                cursor = stream.tell()
                                if any(key in line for key in (
                                        "SIGNAL", "filled", "rejected", "uncertain",
                                        "Telegram", "ERROR", "CRITICAL")):
                                    events.append(line.strip())
                        events[:] = events[-12:]
                    except OSError:
                        pass

                    now = time.time()
                    market_table = Table(expand=True, pad_edge=False,
                                         header_style="bold cyan")
                    for name, justify in (("asset", "left"), ("close", "right"),
                                          ("up ask", "right"), ("down ask", "right"),
                                          ("favorite", "right"), ("state", "left")):
                        market_table.add_column(name, justify=justify)
                    for bot in bots:
                        market = bot.market
                        if not market:
                            market_table.add_row(bot.asset.upper(), "-", "-", "-", "-",
                                                 Text("waiting", style="dim"))
                            continue
                        slug, end, _tokens = market
                        remaining = max(0, end - now)
                        asks = {}
                        quote_slug, last_quotes = bot.last_quotes
                        if quote_slug == slug:
                            for outcome, (ask, stamp) in last_quotes.items():
                                if now - stamp < 5:
                                    asks[outcome] = ask
                        up = f"{asks['Up']:.2f}" if "Up" in asks else "-"
                        down = f"{asks['Down']:.2f}" if "Down" in asks else "-"
                        favorite = "-"
                        state = Text("waiting final minute", style="yellow")
                        if asks:
                            fav_out, fav_ask = max(asks.items(), key=lambda item: item[1])
                            favorite = f"{fav_out} {fav_ask:.2f}"
                            since = bot.above_since.get(fav_out)
                            held = max(0, now - since) if since is not None else 0
                            if remaining > TRADE_WINDOW_SECONDS:
                                state = Text("waiting final minute", style="yellow")
                            elif held >= ABOVE_HOLD_SECONDS:
                                state = Text("TRIGGER", style="bold green")
                            elif since is not None:
                                state = Text(
                                    f"arming {held:.1f}/{ABOVE_HOLD_SECONDS:.0f}s",
                                    style="yellow")
                            else:
                                state = Text("price outside range", style="white")
                        if bot.journal.seen_market(slug):
                            state = Text("attempt recorded", style="bold magenta")
                        market_table.add_row(bot.asset.upper(), f"{remaining:4.0f}s",
                                             up, down, favorite, state)

                    order_table = Table(expand=True, pad_edge=False,
                                        header_style="bold green")
                    for name, justify in (("market", "left"), ("side", "left"),
                                          ("status", "left"), ("fill", "right"),
                                          ("shares", "right"), ("result", "right")):
                        order_table.add_column(name, justify=justify)
                    try:
                        for (market, outcome, status, detail, fill, shares,
                             winner, pnl, exit_status) in _journal_rows(bots[0]):
                            fill_text = f"{fill:.4f}" if fill else "-"
                            shares_text = f"{shares:.4f}" if shares else "-"
                            result = "open"
                            style = "yellow"
                            if exit_status == "sold":
                                result = f"STOP {pnl:+.2f}"
                                style = "yellow" if pnl >= 0 else "red"
                            elif winner:
                                result = f"{'WIN' if pnl >= 0 else 'LOSS'} {pnl:+.2f}"
                                style = "green" if pnl >= 0 else "red"
                            order_table.add_row(market, outcome, status, fill_text,
                                                shares_text, Text(result, style=style))
                    except Exception as exc:
                        order_table.add_row("journal unavailable", "-", type(exc).__name__,
                                            "-", "-", "-")

                    event_panel = Panel(
                        Group(*[Text(line, style="cyan") for line in events[-7:]])
                        if events else Text("no order events yet", style="dim"),
                        title="Events", title_align="left")
                    mode = "REAL MONEY" if args.live else "dry run"
                    telegram = "Telegram on" if bots[0].notifier.enabled else "Telegram off"
                    stake_label = f"${STAKE} fixed stake"
                    if args.live and bots[0].last_stake is not None:
                        stake_label += f" (last ${bots[0].last_stake:.2f})"
                    header = Text(
                        f"NineFive Polymarket - {mode} - {stake_label} - {telegram} - "
                        f"{time.strftime('%H:%M:%S')}", style="bold")
                    with console.capture() as capture:
                        console.print(Group(header, market_table,
                                            Panel(order_table, title="Direct Polymarket Orders",
                                                  title_align="left"), event_panel))
                    import sys
                    sys.stdout.write("\x1b[H" + capture.get() + "\x1b[J")
                    sys.stdout.flush()
                    if args.once:
                        return
                    time.sleep(args.poll_seconds)
            finally:
                console.show_cursor(True)
    finally:
        for bot in bots:
            bot.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Place REAL orders using a fixed $10 stake (default: dry run)")
    parser.add_argument("--asset", choices=Bot.assets,
                        help="Trade one asset only (default: all assets)")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    parser.add_argument("--gui", action="store_true",
                        help="Show the direct Polymarket terminal dashboard")
    args = parser.parse_args()
    assets = (args.asset,) if args.asset else Bot.assets
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "ninefive_polymarket_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots = []
    try:
        poll = float(os.environ.get("POLYMARKET_POLL_SECONDS",
                                    str(DEFAULT_POLL_SECONDS)))
        if not 0.01 <= poll <= 60:
            raise ValueError("POLYMARKET_POLL_SECONDS must be between 0.01 and 60")
        args.poll_seconds = poll
        if args.gui:
            run_dashboard(args, assets)
            return 0
        bots = [NineFivePolymarketBot(args.live, asset) for asset in assets]
        sizing = f"fixed ${STAKE} stake"
        LOG.warning("%s: %s direct Polymarket orders; final %ss; ask >%s to %s for %ss; stop bid <=%s; polling %.3fs",
                    "REAL MONEY" if args.live else "DRY RUN", sizing,
                    TRADE_WINDOW_SECONDS, FAV_ASK_MIN, FAV_ASK_MAX,
                    ABOVE_HOLD_SECONDS, STOP_LOSS_PRICE, poll)
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            while True:
                futures = [(bot, workers.submit(bot.step)) for bot in bots]
                for bot, future in futures:
                    try:
                        future.result()
                    except httpx.HTTPError as exc:
                        LOG.warning("%s request failed (%s)", bot.asset.upper(), type(exc).__name__)
                if args.once:
                    break
                time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        for bot in bots:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
