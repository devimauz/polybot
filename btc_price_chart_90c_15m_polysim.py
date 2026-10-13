"""Headless 15-minute BTC PolySimulator trader using a symmetric 90c trigger.

This keeps the chart/feed implementation from btc_price_chart_bid_trend_polysim
but uses the 15-minute Polymarket series and a symmetric 90c entry rule.
Entries are $100 FOK buys with no take-profit order.  The position is left for
market settlement.  At most one entry is made per 15-minute market.
"""

import json
import argparse
import sqlite3
import threading
import time
from decimal import Decimal

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import btc_price_chart_bid_trend_polysim as chart
from bot import ROOT
from opening_bot import OpeningBot


MARKET_SECONDS = 15 * 60
ENTRY_PRICE = Decimal("0.90")
REVERSAL_PRICE = Decimal("0.60")
MARKET_META = {}


def market_from_payload(payload, now):
    """Return the active 15-minute BTC market in the chart's tuple format."""
    rows = payload if isinstance(payload, list) else payload.get(
        "markets", payload.get("items", []))
    start = int(now) // MARKET_SECONDS * MARKET_SECONDS
    slug = f"btc-updown-15m-{start}"
    for market in rows:
        if (market.get("slug") != slug or market.get("closed") or
                not market.get("active")):
            continue
        condition = market.get("conditionId") or market.get("condition_id")
        nested = next((item for item in market.get("markets", [])
                       if item.get("condition_id") == condition), None)
        source = nested or market
        outcomes = source.get("outcomes")
        tokens = source.get("clobTokenIds", source.get("token_ids"))
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
        if (not condition or set(outcomes or []) != {"Up", "Down"} or
                len(tokens or []) != 2 or len(set(tokens)) != 2):
            continue
        MARKET_META[slug] = condition
        return slug, start + MARKET_SECONDS, dict(zip(outcomes, tokens))
    return None


class FifteenMinuteOpeningBot(OpeningBot):
    """Adapt OpeningBot's order journal to a 900-second market."""

    def discover(self, now):
        start = int(now) // MARKET_SECONDS * MARKET_SECONDS
        slug = f"btc-updown-15m-{start}"
        response = self.http.get(chart.GAMMA_API, params={"slug": slug})
        response.raise_for_status()
        market = market_from_payload(response.json(), now)
        self.condition_id = MARKET_META.get(slug) if market else None
        self.market = market
        return market

    def enter(self, slug, end, outcome, token, direct_tick=None,
              direct_limit=Decimal("0.99")):
        # OpeningBot calculates start as end-300. Moving its temporary end back
        # 600 seconds makes that start equal the real 15-minute market start.
        # Restore the true expiry in the journal before TP management runs.
        original_window = self.entry_window
        try:
            self.entry_window = MARKET_SECONDS
            super().enter(slug, end - 600, outcome, token,
                          direct_tick=direct_tick, direct_limit=direct_limit)
            row = self.db.execute(
                "SELECT 1 FROM trades WHERE market=? AND outcome=?",
                (slug, outcome)).fetchone()
            if row:
                self.update(slug, outcome, end=end)
        finally:
            self.entry_window = original_window

    def place_profits(self):
        """Disable TP placement and mark the journal clearly for the UI."""
        self.db.execute("""UPDATE trades SET tp_state='disabled'
            WHERE entry_state='filled' AND tp_state='needed'""")
        self.db.commit()


class FifteenMinuteDiscovery(chart.Discovery):
    def run(self):
        with httpx.Client(timeout=8) as client:
            while not self.stop.is_set():
                try:
                    now = time.time()
                    start = int(now) // MARKET_SECONDS * MARKET_SECONDS
                    slug = f"btc-updown-15m-{start}"
                    response = client.get(chart.GAMMA_API, params={"slug": slug})
                    response.raise_for_status()
                    found = market_from_payload(response.json(), now)
                    with self.lock:
                        changed = found and (not self.market or
                                             found[0] != self.market[0])
                        self.market = found
                        self.error = "" if found else "Waiting for active 15m market..."
                    if changed:
                        self.feed.subscribe(found[2])
                except Exception as exc:
                    with self.lock:
                        self.error = f"Polymarket {type(exc).__name__}"
                self.stop.wait(1)


class NinetyCentPaperTrader(chart.JumpPaperTrader):
    """Enter at 90c, then reverse when the held side falls to 60c."""

    venue_name = "POLYSIM 15M 90C"

    def __init__(self):
        self.session_started = time.time()
        self.db_path = ROOT / "chart-90c-15m-polysim.sqlite3"
        self.http = httpx.Client(timeout=10)
        setup = FifteenMinuteOpeningBot(
            asset="btc", amount=chart.JUMP_STAKE, paper=True,
            entry_window=MARKET_SECONDS, db_path=self.db_path,
            http_client=self.http, take_profit=chart.TAKE_PROFIT_RATE)
        setup.close()
        with sqlite3.connect(self.db_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jump_entries (
                market TEXT, slot INTEGER, trade_key TEXT UNIQUE, outcome TEXT,
                move REAL, created REAL, PRIMARY KEY(market,slot))""")
            db.execute("""CREATE TABLE IF NOT EXISTS local_pnl_closures (
                trade_key TEXT PRIMARY KEY, exit_price TEXT NOT NULL,
                realized_pnl TEXT NOT NULL, closed REAL NOT NULL)""")
            db.execute("DELETE FROM local_pnl_closures")
        self.lock = threading.Lock()
        self.inflight = set()
        self.preparing = set()
        self.prepared = {}
        self.position_cache = (None, 0.0, None)
        self.latest_cache = (0.0, None)
        self.pnl_cache = (0.0, None, None)
        self.stop = threading.Event()
        self.reversing = False
        self.status = "waiting for UP or DOWN ask to reach 0.90"

    def active_trade(self, slug):
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT * FROM trades
                WHERE market LIKE ? AND entry_state='filled'
                  AND tp_state='disabled'
                ORDER BY rowid DESC LIMIT 1""", (slug + "#jump-%",)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def should_reverse(outcome, bid):
        return outcome in ("Up", "Down") and Decimal(str(bid)) <= REVERSAL_PRICE

    def evaluate(self, slug, end, tokens, quotes, unused_trend):
        del end, unused_trend
        active = self.active_trade(slug)
        if active:
            outcome = active["outcome"]
            quote = quotes.get(outcome)
            opposite = "Down" if outcome == "Up" else "Up"
            opposite_quote = quotes.get(opposite)
            if quote is None or opposite_quote is None:
                self.status = f"holding {outcome.upper()}; waiting for both quotes"
                return False
            if not self.should_reverse(outcome, quote.bid):
                self.status = (f"holding {outcome.upper()}; bid {quote.bid:.3f}, "
                               f"reverse at <= {REVERSAL_PRICE:.2f}")
                return False
            with self.lock:
                if self.reversing or self.inflight:
                    return False
                self.reversing = True
                self.status = (f"{outcome.upper()} bid {quote.bid:.3f}; selling "
                               f"and reversing to {opposite.upper()}")
            threading.Thread(
                target=self._reverse,
                args=(slug, active, quote, opposite, opposite_quote,
                      tokens.get(opposite)),
                name="chart-15m-60c-reversal", daemon=True).start()
            return True

        last_slot, _ = self.trade_window_state(slug)
        if last_slot:
            self.status = "initial 90c entry already used; no open position"
            return False
        candidates = []
        for outcome in ("Up", "Down"):
            quote = quotes.get(outcome)
            if quote is not None and quote.ask >= ENTRY_PRICE:
                candidates.append((quote.ask, outcome, quote))
        if not candidates:
            self.status = "waiting for UP or DOWN ask to reach 0.90"
            return False
        ask, outcome, quote = max(candidates, key=lambda item: item[0])
        self.status = f"{outcome.upper()} ask {ask:.3f}; submitting 90c entry"
        return self.submit(slug, outcome, float(ask), tick=quote.tick,
                           token=tokens.get(outcome), entry_limit=ask)

    def _reverse(self, slug, active, quote, opposite, opposite_quote,
                 opposite_token):
        bot = None
        sold = False
        try:
            bot = FifteenMinuteOpeningBot(
                asset="btc", amount=chart.JUMP_STAKE, paper=True,
                entry_window=MARKET_SECONDS, db_path=self.db_path,
                http_client=self.http, take_profit=chart.TAKE_PROFIT_RATE)
            payload = bot.post({
                "market_id": active["cid"], "side": "SELL",
                "outcome": active["outcome"],
                "quantity": active["shares"], "price": str(quote.bid),
                "order_type": "market", "time_in_force": "FOK",
            })
            if str(payload.get("status", "")).lower() not in ("filled", "matched"):
                raise ValueError("reversal sell did not fill")
            exit_price = Decimal(str(payload.get("price", quote.bid)))
            shares = Decimal(str(active["shares"]))
            entry = Decimal(str(active["price"]))
            pnl = chart.marked_position_pnl(entry, shares, exit_price)
            with sqlite3.connect(self.db_path) as db:
                db.execute("""UPDATE trades SET tp_state='sold',detail=?
                    WHERE market=? AND outcome=?""",
                           (json.dumps(payload), active["market"], active["outcome"]))
                db.execute("""INSERT OR REPLACE INTO local_pnl_closures
                    VALUES (?,?,?,?)""",
                           (active["market"], str(exit_price), str(pnl), time.time()))
            sold = True
            with self.lock:
                self.position_cache = (None, 0.0, None)
                self.latest_cache = (0.0, None)
                self.pnl_cache = (0.0, None, None)
                self.reversing = False
                self.status = (f"{active['outcome'].upper()} sold @ {exit_price:.3f}; "
                               f"buying {opposite.upper()} @ {opposite_quote.ask:.3f}")
            if not self.submit(
                    slug, opposite, float(opposite_quote.ask),
                    tick=opposite_quote.tick, token=opposite_token,
                    entry_limit=opposite_quote.ask):
                with self.lock:
                    self.status = (f"{active['outcome'].upper()} sold; "
                                   f"{opposite.upper()} reversal was not submitted")
        except Exception as exc:
            with self.lock:
                self.status = (f"reversal {'buy' if sold else 'sell'} failed: "
                               f"{type(exc).__name__}: {str(exc)[:160]}")
        finally:
            if bot is not None:
                bot.close()
            with self.lock:
                self.reversing = False


def dashboard(market, quotes, trader, error=""):
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan")
    table.add_column()
    slug = market[0] if market else "discovering"
    remaining = max(0, int(market[1] - time.time())) if market else 0
    table.add_row("Market", slug)
    table.add_row("Time left", f"{remaining}s")
    for outcome in ("Up", "Down"):
        quote = quotes.get(outcome)
        value = "--" if quote is None else (
            f"bid {quote.bid:.3f} / ask {quote.ask:.3f}")
        table.add_row(outcome, value)
    table.add_row("Trigger", "buy UP/DOWN when that ask >= 0.900")
    table.add_row("Reversal", "held side bid <= 0.600: sell, buy opposite")
    table.add_row("Order", f"${chart.JUMP_STAKE} FOK; no take-profit")
    table.add_row("Status", trader.snapshot())
    latest = trader.latest_trade()
    if latest:
        table.add_row(
            "Latest trade",
            f"{latest['outcome']} {latest.get('entry_state') or 'signal'} "
            f"price={latest.get('price') or '--'} "
            f"shares={latest.get('shares') or '--'}")
    if error:
        table.add_row("Feed", error)
    return Panel(table, title="BTC 15M 90C POLYSIM (HEADLESS)")


def run():
    # Inherited order helpers resolve OpeningBot through the source module.
    chart.Discovery = FifteenMinuteDiscovery
    chart.OpeningBot = FifteenMinuteOpeningBot
    feed = chart.MarketFeed()
    discovery = FifteenMinuteDiscovery(feed)
    trader = NinetyCentPaperTrader()
    console = Console()
    current_slug = None
    discovery.start()
    try:
        with Live(console=console, refresh_per_second=10) as display:
            while True:
                market, error = discovery.snapshot()
                quotes = feed.quotes()
                if market:
                    if market[0] != current_slug:
                        current_slug = market[0]
                        trader.prepare(current_slug)
                    trader.evaluate(
                        market[0], market[1], market[2], quotes, Decimal("0"))
                display.update(dashboard(market, quotes, trader, error))
                time.sleep(0.10)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping trader...[/yellow]")
    finally:
        discovery.stop.set()
        feed.generation += 1
        trader.close()


def main():
    load_dotenv(ROOT / ".env")
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
