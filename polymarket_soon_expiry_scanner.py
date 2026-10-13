"""Terminal scanner for soon-expiring Polymarket outcomes above a price.

Examples:
    python polymarket_soon_expiry_scanner.py
    python polymarket_soon_expiry_scanner.py --hours 3 --threshold 0.90
    python polymarket_soon_expiry_scanner.py --once
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import httpx
from rich.console import Console
from rich.table import Table


GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
# Gamma currently caps market pages at 100 even when a larger limit is sent.
# Matching that cap is important: otherwise a short first page looks final and
# later crypto UP/DOWN markets are never scanned.
PAGE_SIZE = 100
CLOB_BOOKS = "https://clob.polymarket.com/books"
CLOB_BATCH_SIZE = 100


def utc_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def json_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def qualifying_rows(markets, now, deadline, threshold, up_down_only=True):
    """Normalize qualifying outcomes from Gamma market records."""
    rows = []
    threshold = Decimal(str(threshold))
    for market in markets:
        expiry = utc_datetime(market.get("endDate") or market.get("endDateIso"))
        if expiry is None or not now < expiry <= deadline:
            continue
        outcomes = json_list(market.get("outcomes"))
        # _liveAskPrices is populated from the CLOB order books. Do not fall
        # back to Gamma here: outcomePrices can be cached and non-executable.
        prices = market.get("_liveAskPrices", [])
        for outcome, raw_price in zip(outcomes, prices):
            if up_down_only and str(outcome).strip().lower() not in {"up", "down"}:
                continue
            try:
                price = Decimal(str(raw_price))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if price <= threshold:
                continue
            rows.append({
                "expiry": expiry,
                "minutes": max(0, int((expiry - now).total_seconds() / 60)),
                "question": str(market.get("question") or market.get("slug") or "--"),
                "outcome": str(outcome),
                "price": price,
                "liquidity": Decimal(str(market.get("liquidityNum") or 0)),
                "slug": str(market.get("slug") or ""),
            })
    return sorted(rows, key=lambda row: (row["expiry"], -row["price"]))


def add_live_ask_prices(client, markets, up_down_only=True):
    """Attach current executable best asks from batched CLOB books."""
    wanted_tokens = []
    market_tokens = []
    for market in markets:
        outcomes = json_list(market.get("outcomes"))
        tokens = [str(token) for token in json_list(market.get("clobTokenIds"))]
        aligned = []
        for index, outcome in enumerate(outcomes):
            token = tokens[index] if index < len(tokens) else None
            wanted = (not up_down_only or
                      str(outcome).strip().lower() in {"up", "down"})
            aligned.append(token if wanted else None)
            if wanted and token:
                wanted_tokens.append(token)
        market_tokens.append((market, aligned))

    best_asks = {}
    unique_tokens = list(dict.fromkeys(wanted_tokens))
    for start in range(0, len(unique_tokens), CLOB_BATCH_SIZE):
        batch = unique_tokens[start:start + CLOB_BATCH_SIZE]
        response = client.post(CLOB_BOOKS,
                               json=[{"token_id": token} for token in batch])
        response.raise_for_status()
        books = response.json()
        if not isinstance(books, list):
            raise ValueError("unexpected CLOB books response")
        for book in books:
            token = str(book.get("asset_id") or "")
            asks = []
            for level in book.get("asks", []):
                try:
                    price = Decimal(str(level.get("price")))
                    size = Decimal(str(level.get("size")))
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if price > 0 and size > 0:
                    asks.append(price)
            if token and asks:
                best_asks[token] = min(asks)

    for market, tokens in market_tokens:
        market["_liveAskPrices"] = [best_asks.get(token) if token else None
                                    for token in tokens]
    return markets


def fetch_markets(client, now, deadline):
    results = []
    offset = 0
    while True:
        response = client.get(GAMMA_MARKETS, params={
            "active": "true",
            "closed": "false",
            "end_date_min": now.isoformat().replace("+00:00", "Z"),
            "end_date_max": deadline.isoformat().replace("+00:00", "Z"),
            "limit": PAGE_SIZE,
            "offset": offset,
        })
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError("unexpected Gamma API response")
        results.extend(page)
        if len(page) < PAGE_SIZE:
            return results
        offset += PAGE_SIZE


def render(rows, hours, threshold, checked_at, error=None):
    title = (f"Polymarket expiring within {hours:g}h | "
             f"outcome price > {threshold:.2f} | "
             f"checked {checked_at.astimezone():%H:%M:%S}")
    table = Table(title=title, expand=True)
    table.add_column("Expires", no_wrap=True)
    table.add_column("Left", justify="right", no_wrap=True)
    table.add_column("Outcome", style="bold cyan", no_wrap=True)
    table.add_column("Live ask", justify="right", style="bold green", no_wrap=True)
    table.add_column("Liquidity", justify="right", no_wrap=True)
    table.add_column("Market", overflow="fold")
    if error:
        table.add_row("--", "--", "ERROR", "--", "--", error, style="red")
    elif not rows:
        table.add_row("--", "--", "--", "--", "--",
                      "No qualifying markets found.")
    else:
        for row in rows:
            table.add_row(
                row["expiry"].astimezone().strftime("%Y-%m-%d %H:%M"),
                f"{row['minutes']}m", row["outcome"],
                f"{row['price']:.3f}", f"${row['liquidity']:,.0f}",
                row["question"])
    return table


def scan(client, hours, threshold, up_down_only=True):
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=hours)
    markets = fetch_markets(client, now, deadline)
    add_live_ask_prices(client, markets, up_down_only)
    return qualifying_rows(markets, now, deadline, threshold, up_down_only), now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1.0,
                        help="expiry horizon in hours (default: 1)")
    parser.add_argument("--threshold", type=Decimal, default=Decimal("0.90"),
                        help="show prices strictly above this value (default: 0.90)")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="refresh interval in seconds (default: 5)")
    parser.add_argument("--once", action="store_true",
                        help="scan once and exit")
    parser.add_argument("--all-outcomes", action="store_true",
                        help="include Yes/No and other outcome names")
    args = parser.parse_args()
    if args.hours <= 0 or args.refresh <= 0 or not Decimal("0") < args.threshold < Decimal("1"):
        parser.error("hours/refresh must be positive and threshold must be between 0 and 1")

    console = Console()
    with httpx.Client(timeout=20, headers={"User-Agent": "polymarket-expiry-scanner/1.0"}) as client:
        if args.once:
            try:
                rows, checked = scan(client, args.hours, args.threshold,
                                     not args.all_outcomes)
                console.print(render(rows, args.hours, args.threshold, checked))
            except Exception as exc:
                checked = datetime.now(timezone.utc)
                console.print(render([], args.hours, args.threshold, checked,
                                     f"{type(exc).__name__}: {exc}"))
                raise SystemExit(1)
            return
        try:
            while True:
                try:
                    rows, checked = scan(client, args.hours, args.threshold,
                                         not args.all_outcomes)
                    view = render(rows, args.hours, args.threshold, checked)
                except Exception as exc:
                    checked = datetime.now(timezone.utc)
                    view = render([], args.hours, args.threshold, checked,
                                  f"{type(exc).__name__}: {exc}")
                console.clear()
                console.print(view)
                console.print(
                    f"[dim]Refreshing every {args.refresh:g}s — Ctrl+C to stop[/dim]")
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            console.print("[yellow]Scanner stopped.[/yellow]")


if __name__ == "__main__":
    main()
