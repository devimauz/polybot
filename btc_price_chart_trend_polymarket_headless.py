"""Headless Kraken-move trader for live Polymarket BTC 5-minute markets.

Dry run (default):
    python btc_price_chart_trend_polymarket_headless.py

Live real-money trading:
    python btc_price_chart_trend_polymarket_headless.py --live

Strategy: a single Kraken BTC/USD WebSocket move of +$5 buys UP; -$5 buys
DOWN. Entries target at least 5.1 shares and arm an 8% GTC take-profit on
Polymarket. The existing spread and $0.20-$0.80 entry-price guards apply.
"""

import argparse
import time
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from bot import ROOT
import btc_price_chart_trend_polymarket as live_strategy
from btc_price_chart_trend_polysim import (
    KRAKEN_MOVE_THRESHOLD,
    MAX_ENTRY_SPREAD,
    CoinbasePriceFeed,
    Discovery,
    KrakenPriceFeed,
    MarketFeed,
    PolymarketBTCFeed,
    kraken_move_signal,
)

ENTRY_SHARES = Decimal("5.10")
TAKE_PROFIT = Decimal("0.08")
MIN_ENTRY_PRICE = Decimal("0.20")
MAX_ENTRY_PRICE = Decimal("0.80")
REFRESH_SECONDS = 0.10


def configure_live_strategy():
    """Apply this terminal strategy's sizing and execution configuration."""
    live_strategy.ENTRY_SHARES = ENTRY_SHARES
    live_strategy.TAKE_PROFIT = TAKE_PROFIT
    live_strategy.MIN_ENTRY_PRICE = MIN_ENTRY_PRICE
    live_strategy.MAX_ENTRY_PRICE = MAX_ENTRY_PRICE
    live_strategy.MAX_TRADES_PER_WINDOW = 300
    live_strategy.TRADE_COOLDOWN_SECONDS = 5.0


def money(value):
    return "--" if value is None else f"${float(value):,.2f}"


def quote_text(quote):
    return "--" if quote is None else f"{quote.bid:.3f} / {quote.ask:.3f}"


def dashboard(real_money, market, btc, beat, coinbase, kraken, kraken_move,
              quotes, trader, errors):
    slug = market[0] if market else "waiting for market"
    remaining = max(0, int(market[1] - time.time())) if market else 0

    summary = Table.grid(expand=True)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Market", slug, "Remaining", f"{remaining}s")
    summary.add_row("Polymarket BTC", money(btc), "Price to beat", money(beat))
    summary.add_row("Coinbase", money(coinbase), "Kraken", money(kraken))
    summary.add_row(
        "Last Kraken move",
        "--" if kraken_move is None else f"{kraken_move:+.2f} USD",
        "Trigger", f"+/-${KRAKEN_MOVE_THRESHOLD:.2f}")

    book = Table(expand=True, header_style="bold")
    for heading in ("Outcome", "Bid", "Ask", "Mid"):
        book.add_column(heading, justify="right" if heading != "Outcome" else "left")
    for outcome, style in (("Up", "green"), ("Down", "red")):
        quote = quotes.get(outcome)
        values = ("--", "--", "--") if quote is None else (
            f"{quote.bid:.3f}", f"{quote.ask:.3f}",
            f"{quote.probability:.3f}")
        book.add_row(outcome.upper(), *values, style=style)

    config = Table.grid(expand=True)
    config.add_column(style="bold magenta")
    config.add_column()
    config.add_row("Entry size", f">= {ENTRY_SHARES} shares")
    config.add_row("Take profit", f"{TAKE_PROFIT * 100:.0f}% GTC")
    config.add_row("Entry ask", f"{MIN_ENTRY_PRICE:.2f} - {MAX_ENTRY_PRICE:.2f}")
    config.add_row("Max spread", f"{MAX_ENTRY_SPREAD:.2f}")
    config.add_row("Trader", trader.snapshot())

    latest = trader.latest_trade()
    if latest:
        trade = (f"{latest.get('outcome')} / {latest.get('entry_state')}\n"
                 f"entry={latest.get('price') or '--'}  "
                 f"shares={latest.get('shares') or '--'}\n"
                 f"target={latest.get('target') or '--'}  "
                 f"GTC={latest.get('tp_state') or '--'}")
    else:
        trade = "No trade submitted in this process."

    mode = "[bold red]LIVE REAL-MONEY[/]" if real_money else "[bold yellow]DRY RUN[/]"
    return Panel(Group(
        Panel(summary, title="Market and feeds", border_style="blue"),
        Panel(book, title="Polymarket order book", border_style="magenta"),
        Panel(config, title="Kraken +/-$5 strategy", border_style="yellow"),
        Panel(trade, title="Latest trade / 8% GTC", border_style="red"),
        Panel(errors or "All feeds connected", title="Feed status",
              border_style="red" if errors else "green"),
    ), title=(f"BTC POLYMARKET TERMINAL  {mode}  "
              f"{datetime.now().astimezone():%H:%M:%S %Z}"),
       border_style="bright_blue")


def run(real_money=False):
    configure_live_strategy()
    feed = MarketFeed()
    btc_feed = PolymarketBTCFeed()
    coinbase_feed = CoinbasePriceFeed()
    kraken_feed = KrakenPriceFeed()
    discovery = Discovery(feed)
    trader = live_strategy.PolymarketJumpTrader(live=real_money)
    console = Console()
    current_slug = None
    last_kraken = None
    last_move = None
    last_status = None
    last_trade_state = None

    discovery.start()
    btc_feed.start()
    coinbase_feed.start()
    kraken_feed.start()
    console.log("[bold red]LIVE REAL-MONEY[/]" if real_money
                else "[bold yellow]DRY RUN[/]")
    try:
        with Live(console=console, refresh_per_second=10, screen=False) as display:
            while True:
                detected_at = time.monotonic()
                market, market_error = discovery.snapshot()
                btc, beat, btc_error = btc_feed.snapshot()
                coinbase, coinbase_error = coinbase_feed.snapshot()
                kraken, kraken_error, _ = kraken_feed.snapshot()
                quotes = feed.quotes()

                if market and market[0] != current_slug:
                    current_slug = market[0]
                    last_kraken = None
                    last_move = None
                    trader.prepare(current_slug, market[2])
                    console.log(f"[bold blue]MARKET[/] {current_slug}")

                signal = None
                if kraken is not None and kraken != last_kraken:
                    signal = kraken_move_signal(last_kraken, kraken)
                    last_move = None if last_kraken is None else kraken - last_kraken
                    last_kraken = kraken

                if signal and market:
                    outcome, move = signal
                    quote = quotes.get(outcome)
                    reason = None
                    if quote is None:
                        reason = f"{outcome} quote unavailable"
                    elif quote.ask - quote.bid > MAX_ENTRY_SPREAD:
                        reason = f"spread {quote.ask - quote.bid:.3f} too wide"
                    elif not MIN_ENTRY_PRICE <= quote.ask <= MAX_ENTRY_PRICE:
                        reason = f"ask {quote.ask:.3f} outside entry range"
                    if reason:
                        console.log(f"[yellow]SKIP[/] Kraken {move:+.2f}: {reason}")
                    else:
                        request_id = trader.submit(
                            market[0], outcome, move, quote.tick,
                            token=market[2][outcome], ask=quote.ask,
                            detected_at=detected_at)
                        if request_id:
                            console.log(
                                f"[bold green]ORDER[/] Kraken {move:+.2f} -> "
                                f"{outcome}; ask {quote.ask:.3f}")

                if market:
                    trader.observe(market[0], quotes)

                status = trader.snapshot()
                if status != last_status:
                    console.log(f"[bold magenta]TRADER[/] {status}")
                    last_status = status
                latest = trader.latest_trade()
                trade_state = None if not latest else (
                    latest.get("request_id"), latest.get("entry_state"),
                    latest.get("tp_state"), latest.get("shares"))
                if trade_state != last_trade_state and latest:
                    console.log(
                        f"[bold red]TRADE[/] {latest.get('outcome')} "
                        f"entry={latest.get('price') or '--'} "
                        f"shares={latest.get('shares') or '--'} "
                        f"target={latest.get('target') or '--'} "
                        f"GTC={latest.get('tp_state') or '--'}")
                last_trade_state = trade_state

                errors = " | ".join(filter(None, (
                    market_error, btc_error, coinbase_error, kraken_error)))
                display.update(dashboard(
                    real_money, market, btc, beat, coinbase, kraken,
                    last_move, quotes, trader, errors))
                time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping terminal trader...[/yellow]")
    finally:
        discovery.stop.set()
        btc_feed.stop.set()
        coinbase_feed.stop.set()
        kraken_feed.stop.set()
        trader.close()
        feed.generation += 1


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="Submit real-money Polymarket orders (default: dry run)")
    run(real_money=parser.parse_args().live)


if __name__ == "__main__":
    main()
