"""Headless UP-bid-trend trader for Polymarket BTC five-minute markets.

Dry run:
    python btc_bid_trend_polymarket_headless.py

Real-money trading:
    python btc_bid_trend_polymarket_headless.py --live
"""

import argparse
import json
import os
import sqlite3
import threading
import time
from collections import deque
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import httpx
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from bot import ROOT
from btc_price_chart_bid_trend_polysim import (
    BID_TREND_ENTRY, BID_TREND_EXIT, BID_TREND_SECONDS,
    MIN_ENTRY_PRICE, MAX_ENTRY_PRICE, Discovery, MarketFeed,
    bid_trend_action, entry_price_allowed, up_bid_trend,
)


STAKE = Decimal("1.10")
MAX_SLIPPAGE = Decimal("0.03")
REFRESH_SECONDS = 0.10
SELL_DUST = Decimal("0.01")
SELL_RETRY_SECONDS = 0.020


def buy_limit(ask, tick, slippage=MAX_SLIPPAGE):
    ask, tick = Decimal(str(ask)), Decimal(str(tick))
    ceiling = min(Decimal("0.99"), ask + Decimal(str(slippage)))
    return (ceiling / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def sell_limit(bid, tick, slippage=MAX_SLIPPAGE):
    bid, tick = Decimal(str(bid)), Decimal(str(tick))
    floor = max(tick, bid - Decimal(str(slippage)))
    return (floor / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def remaining_after_sell(requested, sold, dust=SELL_DUST):
    """Return zero when only an unsellable rounding remainder is left."""
    remaining = max(Decimal("0"), Decimal(str(requested)) - Decimal(str(sold)))
    return Decimal("0") if remaining <= Decimal(str(dust)) else remaining


class LiveBidTrendTrader:
    def __init__(self, live=False):
        self.live = live
        suffix = "live" if live else "dry"
        self.db_path = ROOT / f"btc-bid-trend-polymarket-{suffix}.sqlite3"
        self.lock = threading.RLock()
        self.events = deque(maxlen=20)
        self.inflight = False
        self.retry_after = 0.0
        self.status = "waiting for UP bid trend outside +/-0.200"
        self.position = None
        self.last_trade = None
        self.client = None
        self.api = None
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            db.execute("""CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, market TEXT NOT NULL,
                outcome TEXT NOT NULL, token TEXT NOT NULL, tick TEXT NOT NULL,
                state TEXT NOT NULL, entry TEXT, shares TEXT, exit TEXT,
                pnl TEXT, created REAL NOT NULL, detail TEXT)""")
            row = db.execute("""SELECT * FROM positions
                WHERE state IN ('buying','open','selling','unknown')
                ORDER BY id DESC LIMIT 1""").fetchone()
            if row:
                self.position = dict(row)
                self.status = "recovered active position; new entries locked"
        if live:
            self._connect()
        self._log("LIVE trader ready" if live else "DRY-RUN trader ready")

    @staticmethod
    def _short(value, limit=240):
        return " ".join(str(value).split())[:limit]

    @classmethod
    def _error_detail(cls, exc):
        status = getattr(exc, "status_code", None)
        message = getattr(exc, "error_msg", None)
        if message not in (None, ""):
            try:
                message = json.dumps(message, separators=(",", ":"))
            except (TypeError, ValueError):
                pass
            prefix = f"HTTP {status}: " if status is not None else ""
            return cls._short(f"{type(exc).__name__}: {prefix}{message}")
        detail = repr(exc)
        return cls._short(f"{type(exc).__name__}: {detail}")

    def _log(self, message):
        with self.lock:
            self.events.append(
                f"{time.strftime('%H:%M:%S')}  {self._short(message)}")

    def log_snapshot(self):
        with self.lock:
            return list(self.events)

    def _connect(self):
        response = httpx.get("https://polymarket.com/api/geoblock", timeout=8)
        response.raise_for_status()
        if response.json().get("blocked") is not False:
            raise RuntimeError("Polymarket trading is unavailable from this location")
        key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
        if not key:
            raise ValueError("Set POLY_PRIVATE_KEY in .env before using --live")
        signature = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
        funder = os.environ.get("POLY_FUNDER", "").strip() or None
        if signature not in (0, 1, 2, 3) or (signature != 0 and not funder):
            raise ValueError("Set a valid POLY_SIGNATURE_TYPE and POLY_FUNDER")
        from py_clob_client_v2 import ClobClient
        self.client = ClobClient(
            "https://clob.polymarket.com", chain_id=137, key=key,
            signature_type=signature, funder=funder)
        self.client.set_api_creds(self.client.create_or_derive_api_key())
        from py_clob_client_v2 import (MarketOrderArgs, OrderType,
                                       PartialCreateOrderOptions, Side)
        self.api = (MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side)
        self._log("Authenticated with Polymarket CLOB; geoblock check passed")

    def snapshot(self):
        with self.lock:
            return self.status, None if self.position is None else dict(self.position)

    def _update(self, position_id, **fields):
        with sqlite3.connect(self.db_path) as db:
            assignments = ",".join(f"{name}=?" for name in fields)
            db.execute(f"UPDATE positions SET {assignments} WHERE id=?",
                       (*fields.values(), position_id))

    def evaluate(self, market, tokens, quotes, trend):
        trend = Decimal(str(trend))
        with self.lock:
            position = None if self.position is None else dict(self.position)
            if self.inflight or time.monotonic() < self.retry_after:
                return
            if position:
                action = bid_trend_action(trend, position["outcome"])
                if action != "SELL":
                    return
                quote = quotes.get(position["outcome"])
                if quote is None:
                    self.status = "exit waiting for outcome quote"
                    return
                self.inflight = True
                self.status = (f"trend {trend:+.3f}; FAK selling "
                               f"{position['outcome'].upper()}")
                self._log(self.status)
                threading.Thread(target=self._sell, args=(position, quote),
                                 daemon=True, name="live-bid-trend-sell").start()
                return
            action = bid_trend_action(trend)
            if action not in ("BUY_UP", "BUY_DOWN"):
                self.status = f"waiting; UP bid trend {trend:+.3f}"
                return
            outcome = "Up" if action == "BUY_UP" else "Down"
            quote = quotes.get(outcome)
            if quote is None:
                self.status = f"{outcome} entry waiting for quote"
                return
            if not entry_price_allowed(quote.ask):
                self.status = (f"{outcome} ask {quote.ask:.3f} outside "
                               f"{MIN_ENTRY_PRICE:.2f}-{MAX_ENTRY_PRICE:.2f}")
                return
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("""INSERT INTO positions
                    (market,outcome,token,tick,state,created,detail)
                    VALUES (?,?,?,?,'buying',?,'')""",
                                    (market, outcome, str(tokens[outcome]),
                                     str(quote.tick), time.time()))
                position_id = cursor.lastrowid
            position = {
                "id": position_id, "market": market, "outcome": outcome,
                "token": str(tokens[outcome]), "tick": str(quote.tick),
                "state": "buying", "entry": None, "shares": None,
                "exit": None, "pnl": None, "created": time.time(), "detail": "",
            }
            self.position = position
            self.inflight = True
            self.status = (f"trend {trend:+.3f}; FOK buying "
                           f"${STAKE} {outcome.upper()}")
            self._log(self.status)
        threading.Thread(target=self._buy, args=(position, quote), daemon=True,
                         name="live-bid-trend-buy").start()

    def _buy(self, position, quote):
        try:
            tick = Decimal(str(position["tick"]))
            limit = buy_limit(quote.ask, tick)
            if not self.live:
                paid, shares = STAKE, STAKE / Decimal(str(quote.ask))
                response = {"mode": "dry-run", "limit": str(limit)}
            else:
                MarketOrderArgs, OrderType, Options, Side = self.api
                signed = self.client.create_market_order(
                    MarketOrderArgs(token_id=position["token"], amount=float(STAKE),
                                    side=Side.BUY, price=float(limit),
                                    order_type=OrderType.FOK),
                    Options(tick_size=str(tick), neg_risk=False))
                response = self.client.post_order(signed, OrderType.FOK,
                                                  defer_exec=True)
                if (not response.get("success") or
                        str(response.get("status", "")).lower() != "matched"):
                    self._log(f"BUY FOK rejected: {response}")
                    raise ValueError(f"buy not matched: {str(response)[:200]}")
                paid = Decimal(str(response["makingAmount"]))
                shares = Decimal(str(response["takingAmount"]))
            entry = paid / shares
            self._update(position["id"], state="open", entry=str(entry),
                         shares=str(shares), detail=json.dumps(response)[:4000])
            with self.lock:
                self.position.update(state="open", entry=str(entry),
                                     shares=str(shares), detail=str(response)[:4000])
                self.last_trade = dict(self.position)
                self.status = (f"{position['outcome'].upper()} filled @ {entry:.4f}; "
                               "software FOK exit armed")
            self._log(f"BUY FILLED: {position['outcome'].upper()} "
                      f"entry={entry:.4f} shares={shares} paid=${paid}")
        except Exception as exc:
            self._update(position["id"], state="rejected",
                         detail=f"{type(exc).__name__}: {str(exc)[:500]}")
            with self.lock:
                self.position = None
                self.status = f"buy failed: {type(exc).__name__}: {str(exc)[:100]}"
            self._log(f"BUY ERROR: {self._error_detail(exc)}")
        finally:
            with self.lock:
                self.inflight = False

    def _sell(self, position, quote):
        try:
            tick = Decimal(str(position["tick"]))
            limit = sell_limit(quote.bid, tick)
            shares = Decimal(str(position["shares"]))
            if not self.live:
                sold, proceeds = shares, shares * Decimal(str(quote.bid))
                response = {"mode": "dry-run", "limit": str(limit)}
            else:
                from py_clob_client_v2 import AssetType, BalanceAllowanceParams
                self.client.update_balance_allowance(BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=position["token"]))
                MarketOrderArgs, OrderType, Options, Side = self.api
                signed = self.client.create_market_order(
                    MarketOrderArgs(token_id=position["token"], amount=float(shares),
                                    side=Side.SELL, price=float(limit),
                                    order_type=OrderType.FAK),
                    Options(tick_size=str(tick), neg_risk=False))
                response = self.client.post_order(signed, OrderType.FAK,
                                                  defer_exec=True)
                if (not response.get("success") or
                        str(response.get("status", "")).lower() != "matched"):
                    self._log(f"SELL FAK not filled: {response}")
                    raise ValueError(f"sell not matched: {str(response)[:200]}")
                sold = Decimal(str(response.get("makingAmount", shares)))
                proceeds = Decimal(str(response.get("takingAmount", "0")))
            if sold <= 0 or proceeds < 0:
                raise ValueError("FAK sell reported no filled shares")
            exit_price = proceeds / sold
            fill_pnl = proceeds - sold * Decimal(str(position["entry"]))
            total_pnl = Decimal(str(position.get("pnl") or "0")) + fill_pnl
            remaining = remaining_after_sell(shares, sold)
            if remaining:
                self._update(position["id"], state="open", shares=str(remaining),
                             exit=str(exit_price), pnl=str(total_pnl),
                             detail=json.dumps(response)[:4000])
                with self.lock:
                    self.position.update(state="open", shares=str(remaining),
                                         exit=str(exit_price), pnl=str(total_pnl))
                    self.retry_after = time.monotonic() + SELL_RETRY_SECONDS
                    self.status = (f"{position['outcome'].upper()} FAK partial: "
                                   f"sold {sold}, remaining {remaining}")
                self._log(f"SELL PARTIAL: {position['outcome'].upper()} "
                          f"exit={exit_price:.4f} sold={sold} remaining={remaining} "
                          f"P&L=${total_pnl:+.2f}")
            else:
                self._update(position["id"], state="closed", shares="0",
                             exit=str(exit_price), pnl=str(total_pnl),
                             detail=json.dumps(response)[:4000])
                closed = dict(position, state="closed", shares="0",
                              exit=str(exit_price), pnl=str(total_pnl))
                with self.lock:
                    self.position = None
                    self.last_trade = closed
                    self.status = (f"{position['outcome'].upper()} FAK sold @ "
                                   f"{exit_price:.4f}; P&L ${total_pnl:+.2f}")
                self._log(f"SELL FILLED: {position['outcome'].upper()} "
                          f"exit={exit_price:.4f} shares={sold} "
                          f"P&L=${total_pnl:+.2f}")
        except Exception as exc:
            self._update(position["id"], state="open",
                         detail=f"sell retry: {type(exc).__name__}: {str(exc)[:500]}")
            with self.lock:
                if self.position:
                    self.position["state"] = "open"
                self.retry_after = time.monotonic() + SELL_RETRY_SECONDS
                self.status = f"FAK sell failed; retrying: {type(exc).__name__}"
            self._log(f"SELL ERROR (position remains open): "
                      f"{self._error_detail(exc)}")
        finally:
            with self.lock:
                self.inflight = False

    def close(self):
        return None


def dashboard(real_money, market, quotes, trend_direction, trend_move, trader):
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan")
    table.add_column()
    slug = market[0] if market else "discovering"
    left = max(0, int(market[1] - time.time())) if market else 0
    table.add_row("Market", slug)
    table.add_row("Time left", f"{left}s")
    for outcome in ("Up", "Down"):
        quote = quotes.get(outcome)
        table.add_row(outcome, "--" if quote is None else
                      f"bid {quote.bid:.3f} / ask {quote.ask:.3f}")
    table.add_row("UP bid trend", f"{trend_direction} {trend_move:+.3f}")
    table.add_row("Entry", f"> +{BID_TREND_ENTRY:.3f} UP; < -{BID_TREND_ENTRY:.3f} DOWN")
    table.add_row("Exit", f"UP <= +{BID_TREND_EXIT:.3f}; DOWN >= -{BID_TREND_EXIT:.3f}")
    table.add_row("Orders", f"${STAKE} FOK buy / FAK sell / max 3c slippage")
    status, position = trader.snapshot()
    table.add_row("Status", status)
    if position:
        table.add_row("Position", f"{position['outcome']} {position['state']} "
                      f"entry={position.get('entry') or '--'} "
                      f"shares={position.get('shares') or '--'}")
    logs = Table.grid(expand=True)
    logs.add_column(style="dim")
    events = trader.log_snapshot()
    for event in events[-10:]:
        logs.add_row(event)
    if not events:
        logs.add_row("No events yet")
    mode = "[bold red]LIVE REAL-MONEY[/]" if real_money else "[bold yellow]DRY RUN[/]"
    return Panel(Group(table, Panel(logs, title="Event log")),
                 title=f"BTC BID-TREND POLYMARKET {mode}")


def run(real_money=False):
    feed = MarketFeed()
    discovery = Discovery(feed)
    trader = LiveBidTrendTrader(live=real_money)
    console = Console()
    samples = deque()
    current_slug = None
    discovery.start()
    try:
        with Live(console=console, refresh_per_second=10) as display:
            while True:
                market, error = discovery.snapshot()
                quotes = feed.quotes()
                now = time.monotonic()
                if market and market[0] != current_slug:
                    current_slug = market[0]
                    samples.clear()
                up = quotes.get("Up")
                if up is not None:
                    samples.append((now, up.bid))
                while samples and samples[0][0] < now - BID_TREND_SECONDS:
                    samples.popleft()
                direction, move = up_bid_trend(samples, now)
                if market and direction != "WAIT":
                    trader.evaluate(market[0], market[2], quotes, move)
                view = dashboard(real_money, market, quotes, direction, move, trader)
                if error:
                    view.subtitle = error
                display.update(view)
                time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping trader...[/yellow]")
    finally:
        discovery.stop.set()
        feed.generation += 1
        trader.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="submit real-money Polymarket orders")
    run(parser.parse_args().live)


if __name__ == "__main__":
    main()
