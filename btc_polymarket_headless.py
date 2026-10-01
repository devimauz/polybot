"""Rich terminal monitor and Polymarket jump trader with persistent logs.

Dry run: python btc_polymarket_headless.py
Live:    python btc_polymarket_headless.py --live
"""
import argparse
import os
import time
from collections import deque
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bot import ROOT
from btc_price_chart_polymarket import (
    FRAME_MS, MAX_ENTRY_PRICE, MAX_TRADES_PER_WINDOW, MIN_ENTRY_PRICE,
    TRADE_START_DELAY_SECONDS,
    CoinbasePriceFeed, Discovery, EventDrivenJumpDetector, MarketFeed,
    PolymarketBTCFeed,
    PolymarketJumpTrader, directional_jump, price_trend,
)

REFRESH_SECONDS = 0.10
TREND_SECONDS = 10.0
LOG_SNAPSHOT_SECONDS = 1.0


def money(value):
    return "--" if value is None else f"${value:,.2f}"


def probability(value):
    return "--" if value is None else f"{float(value):.3f}"


def dashboard(real_money, market, btc, beat, coinbase, offset, last_move,
              quotes, samples, moves, threshold, trader, errors):
    slug = market[0] if market else "waiting for market"
    remaining = max(0, int(market[1] - time.time())) if market else 0
    synced = None if coinbase is None or offset is None else coinbase + offset
    delta = None if btc is None or beat is None else btc - beat
    trend = price_trend(list(samples))
    trend_text = "--" if not trend else f"{trend[0]} {trend[1]:+.2f} USD / 10s"
    sequence = ", ".join(f"{move:+.2f}" for move in moves) or "--"

    market_table = Table.grid(expand=True)
    market_table.add_column(style="bold cyan"); market_table.add_column()
    market_table.add_column(style="bold cyan"); market_table.add_column()
    market_table.add_row("Market", slug, "Remaining", f"{remaining}s")
    market_table.add_row("Polymarket BTC", money(btc), "Price to beat", money(beat))
    market_table.add_row("Coinbase", money(coinbase), "Synced Coinbase", money(synced))
    market_table.add_row("BTC delta", "--" if delta is None else f"{delta:+,.2f} USD",
                         "Price offset", "--" if offset is None else f"{offset:+,.2f} USD")

    book = Table(expand=True, header_style="bold")
    for heading in ("Outcome", "Bid", "Ask", "Mid"):
        book.add_column(heading, justify="right" if heading != "Outcome" else "left")
    for outcome, style in (("Up", "green"), ("Down", "red")):
        quote = quotes.get(outcome)
        values = ("--", "--", "--") if quote is None else (
            f"{quote.bid:.3f}", f"{quote.ask:.3f}", f"{quote.probability:.3f}")
        book.add_row(outcome.upper(), *values, style=style)

    strategy = Table.grid(expand=True)
    strategy.add_column(style="bold magenta"); strategy.add_column()
    strategy.add_row("Latest Coinbase move", "--" if last_move is None else f"{last_move:+.2f} USD")
    strategy.add_row("10-second trend", trend_text)
    strategy.add_row("Jump threshold", f"{threshold:.2f} USD")
    strategy.add_row("Jump sequence", f"[{sequence}] total {sum(moves):+.2f}")
    strategy.add_row("Entry filter", f"{MIN_ENTRY_PRICE:.2f} - {MAX_ENTRY_PRICE:.2f}")
    strategy.add_row("Trades per window", str(MAX_TRADES_PER_WINDOW))
    strategy.add_row("Trader status", trader.snapshot())

    latest = trader.latest_trade()
    if latest:
        shares = "--" if latest.get("shares") is None else f"{float(latest['shares']):.6f}"
        position = (f"{latest['outcome']} / {latest['entry_state']} | trigger {float(latest['move']):+.2f} USD\n"
                    f"entry {probability(latest.get('price'))} | shares {shares}\n"
                    f"sell target {probability(latest.get('target'))} | "
                    f"state {latest.get('tp_state') or '--'}")
    else:
        position = "No trade submitted in this process yet."

    mode = "[bold red]LIVE REAL-MONEY[/]" if real_money else "[bold yellow]DRY RUN[/]"
    title = f"BTC POLYMARKET TERMINAL  {mode}  {datetime.now().astimezone():%H:%M:%S %Z}"
    return Panel(Group(
        Panel(market_table, title="Market", border_style="blue"),
        Panel(book, title="Polymarket order book", border_style="magenta"),
        Panel(strategy, title="Signal and strategy", border_style="yellow"),
        Panel(position, title="Position / software sell", border_style="red"),
        Panel(errors or "All feeds connected", title="Feed status",
              border_style="red" if errors else "green"),
        Text("Persistent event logs appear above. Press Ctrl+C to stop.", style="dim"),
    ), title=title, border_style="bright_blue")


def run(real_money=False):
    feed, btc_feed, coinbase_feed = MarketFeed(), PolymarketBTCFeed(), CoinbasePriceFeed()
    trader = PolymarketJumpTrader(live=real_money)
    discovery = Discovery(feed)
    threshold = float(os.environ.get("JUMP_THRESHOLD_USD", "10"))
    if threshold <= 0:
        raise ValueError("JUMP_THRESHOLD_USD must be positive")
    current_slug = offset = None
    samples = deque()
    last_render = last_log = 0.0
    last_status = last_errors = last_trade_state = None
    console = Console()
    def signal_event(kind, message):
        color = {"SIGNAL": "yellow", "SKIP": "yellow", "ORDER": "green"}.get(kind, "white")
        console.log(f"[bold {color}]{kind}[/] {message}")
    jump_detector = EventDrivenJumpDetector(
        discovery, feed, trader, threshold, on_event=signal_event)
    coinbase_feed.set_callback(jump_detector.on_price)
    discovery.start(); btc_feed.start(); coinbase_feed.start()
    try:
        mode = "LIVE REAL-MONEY" if real_money else "DRY RUN"
        console.log(f"[bold]{mode}[/bold] terminal trader started")
        with Live(console=console, refresh_per_second=10, screen=False) as display:
            while True:
                monotonic_now = time.monotonic()
                market, market_error = discovery.snapshot()
                btc, beat, btc_error = btc_feed.snapshot()
                coinbase, coinbase_error = coinbase_feed.snapshot()
                quotes = feed.quotes()
                if market and market[0] != current_slug:
                    current_slug, offset = market[0], None
                    samples.clear()
                    trader.prepare(current_slug, market[2])
                    jump_detector.reset(current_slug, market[2])
                    console.log(f"[bold blue]MARKET[/] {current_slug} started")
                if btc is not None and coinbase is not None and offset is None:
                    offset = btc - coinbase
                _, last_move, moves = jump_detector.snapshot()
                if coinbase is not None:
                    samples.append((monotonic_now, coinbase))
                    while samples and samples[0][0] < monotonic_now - TREND_SECONDS:
                        samples.popleft()
                if market:
                    trader.observe(market[0], quotes)

                status = trader.snapshot()
                if status != last_status:
                    console.log(f"[bold magenta]TRADER[/] {status}")
                    last_status = status
                latest = trader.latest_trade()
                trade_state = None if not latest else (
                    latest.get("market"), latest.get("entry_state"), latest.get("tp_state"),
                    latest.get("price"), latest.get("shares"), latest.get("target"))
                if trade_state != last_trade_state:
                    if latest:
                        console.log("[bold red]POSITION[/] "
                                    f"{latest['outcome']} entry={latest.get('price') or '--'} "
                                    f"shares={latest.get('shares') or '--'} "
                                    f"target={latest.get('target') or '--'} "
                                    f"entry_state={latest.get('entry_state')} "
                                    f"sell_state={latest.get('tp_state') or '--'}")
                    last_trade_state = trade_state
                errors = " | ".join(filter(None, (market_error, btc_error, coinbase_error)))
                if errors != last_errors:
                    style = "red" if errors else "green"
                    console.log(f"[bold {style}]FEEDS[/] {errors or 'all feeds connected'}")
                    last_errors = errors
                if monotonic_now - last_log >= LOG_SNAPSHOT_SECONDS:
                    trend = price_trend(list(samples))
                    trend_text = "--" if not trend else f"{trend[0]} {trend[1]:+.2f}"
                    up_quote, down_quote = quotes.get("Up"), quotes.get("Down")
                    up_text = "--" if up_quote is None else f"{up_quote.bid:.3f}/{up_quote.ask:.3f}"
                    down_text = "--" if down_quote is None else f"{down_quote.bid:.3f}/{down_quote.ask:.3f}"
                    move_text = "--" if last_move is None else f"{last_move:+.2f}"
                    console.log("[cyan]TICK[/] "
                                f"CB={money(coinbase)} POLY={money(btc)} BEAT={money(beat)} "
                                f"move={move_text} trend={trend_text} "
                                f"UP(b/a)={up_text} DOWN(b/a)={down_text}")
                    last_log = monotonic_now
                if monotonic_now - last_render >= REFRESH_SECONDS:
                    display.update(dashboard(real_money, market, btc, beat, coinbase, offset,
                                             last_move, quotes, samples, moves,
                                             threshold, trader, errors))
                    last_render = monotonic_now
                time.sleep(FRAME_MS / 1000)
    except KeyboardInterrupt:
        console.print("[yellow]Stopping terminal trader...[/yellow]")
    finally:
        discovery.stop.set(); btc_feed.stop.set(); coinbase_feed.stop.set()
        trader.close(); feed.generation += 1


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Submit real-money orders")
    run(real_money=parser.parse_args().live)


if __name__ == "__main__":
    main()
