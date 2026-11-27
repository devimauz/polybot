"""Research non-momentum BTC 5m pricing rules on recorded Polymarket books.

Research only: this file never submits orders.  All combined snapshots are
assigned the latest component source timestamp, entries sweep the recorded
book for $2, and execution occurs after a seeded 300--800 ms delay subject to
the decision quote's worst price plus a $0.05 FOK allowance.  Settlement P&L
uses the current crypto taker-fee formula exposed by the shared strategy.

The main candidate is a no-drift fair-value model.  It estimates short-horizon
BTC volatility from causally available one-second Coinbase observations and
computes the probability that BTC finishes above its opening level.  Unlike a
momentum rule, it does not inspect whether recent returns are rising or falling.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from btc5m_momentum_strategy import fee_for


STAKE = 2.0
DELAY_MIN = 0.300
DELAY_MAX = 0.800
FOK_ALLOWANCE = 0.05


@dataclass(frozen=True)
class Row:
    t: float
    coinbase: float
    up_bids: tuple[tuple[float, float], ...]
    up_asks: tuple[tuple[float, float], ...]
    down_bids: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    rows: tuple[Row, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    entry_second: float = 150.0
    minimum_edge: float = 0.0
    ask_min: float = 0.05
    ask_max: float = 0.82
    imbalance: float = 0.0


def epoch(value):
    try:
        number = float(value)
        return number / 1000.0 if number > 10_000_000_000 else number
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def levels(book, side):
    result = []
    for item in book.get(side) or []:
        try:
            price, size = float(item["price"]), float(item["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            result.append((price, size))
    if not result:
        prefix = "bid" if side == "bids" else "ask"
        try:
            price, size = float(book[prefix]), float(book[prefix + "_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if 0 < price < 1 and size > 0:
            result.append((price, size))
    return tuple(sorted(result, reverse=side == "bids"))


def load_window(path):
    rows, winner, start = [], None, None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in ("Up", "Down"):
                winner = raw["winner"]
                continue
            if raw.get("event") != "snapshot":
                continue
            try:
                start = int(raw["window_start"])
                cb = float((raw.get("coinbase") or {})["price"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            component_times = (
                epoch(up.get("book_timestamp")),
                epoch(down.get("book_timestamp")),
                epoch((raw.get("coinbase") or {}).get("exchange_ts")),
            )
            component_times = [item for item in component_times if item is not None]
            if not component_times:
                continue
            t = max(component_times) - start
            if -2 <= t <= 301:
                rows.append(Row(t, cb, levels(up, "bids"), levels(up, "asks"),
                                levels(down, "bids"), levels(down, "asks")))
    rows.sort(key=lambda row: row.t)
    if winner not in ("Up", "Down") or start is None or not rows or rows[0].t > 5:
        return None
    packed = tuple(rows)
    return Window(path.stem, start, winner, packed, tuple(row.t for row in packed))


def load_all(directory):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        window = load_window(path)
        if window is not None:
            windows.append(window)
    return windows


def sweep_buy(book, dollars, max_price=1.0):
    remaining, shares, fees, worst = dollars, 0.0, 0.0, None
    for price, size in book:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        qty = used / price
        shares += qty
        fees += fee_for(price, qty)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return {"shares": shares, "cost": dollars + fees,
                    "worst": worst, "average": dollars / shares}
    return None


def side_books(row, side):
    return ((row.up_bids, row.up_asks) if side == "Up"
            else (row.down_bids, row.down_asks))


def sampled_volatility(window, decision_index, step=1.0):
    """Estimate log-return sigma per sqrt(second), without future samples."""
    decision = window.rows[decision_index]
    start_t = max(0.0, window.rows[0].t)
    sample_times = []
    point = math.ceil(start_t / step) * step
    while point <= decision.t + 1e-12:
        sample_times.append(point)
        point += step
    prices = []
    for target in sample_times:
        index = bisect.bisect_right(window.times, target, 0, decision_index + 1) - 1
        if index >= 0:
            price = window.rows[index].coinbase
            if not prices or price != prices[-1]:
                prices.append(price)
            else:
                # Retain unchanged one-second observations: zero is information.
                prices.append(price)
    if len(prices) < 30:
        return None
    returns = [math.log(new / old) for old, new in zip(prices, prices[1:])]
    if not returns:
        return None
    # Root mean squared one-second return.  The small floor prevents a frozen
    # feed from creating unjustified probabilities of exactly zero or one.
    return max(1e-7, math.sqrt(sum(value * value for value in returns) / len(returns)))


def normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def fair_probability(window, index):
    row = window.rows[index]
    sigma = sampled_volatility(window, index)
    if sigma is None:
        return None
    opening = window.rows[0].coinbase
    remaining = max(0.25, 300.0 - row.t)
    z = math.log(row.coinbase / opening) / (sigma * math.sqrt(remaining))
    return min(0.999, max(0.001, normal_cdf(z)))


def best_market_side(row):
    candidates = []
    for side in ("Up", "Down"):
        bids, asks = side_books(row, side)
        if bids and asks:
            candidates.append(((bids[0][0] + asks[0][0]) / 2.0, side))
    return max(candidates)[1] if len(candidates) == 2 else None


def imbalance_side(row, threshold):
    """Choose from top-level microprice pressure, independent of BTC trend."""
    if not row.up_bids or not row.up_asks:
        return None
    bid_price, bid_size = row.up_bids[0]
    ask_price, ask_size = row.up_asks[0]
    total = bid_size + ask_size
    if total <= 0:
        return None
    imbalance = (bid_size - ask_size) / total
    if abs(imbalance) < threshold:
        return None
    return "Up" if imbalance > 0 else "Down"


def decision_for(window, rule):
    index = bisect.bisect_left(window.times, rule.entry_second)
    if index >= len(window.rows):
        return None
    row = window.rows[index]
    probability = None
    if rule.kind == "fair":
        probability = fair_probability(window, index)
        if probability is None:
            return None
        scored = []
        for side, chance in (("Up", probability), ("Down", 1.0 - probability)):
            _, asks = side_books(row, side)
            quoted = sweep_buy(asks, STAKE)
            if quoted is None or not rule.ask_min <= asks[0][0] <= rule.ask_max:
                continue
            expected_roi = (chance * quoted["shares"] - quoted["cost"]) / quoted["cost"]
            scored.append((expected_roi, side, quoted))
        if not scored:
            return None
        expected_roi, side, quote = max(scored)
        if expected_roi < rule.minimum_edge:
            return None
    elif rule.kind == "favorite":
        side = best_market_side(row)
        if side is None:
            return None
        _, asks = side_books(row, side)
        if not asks or not rule.ask_min <= asks[0][0] <= rule.ask_max:
            return None
        quote = sweep_buy(asks, STAKE)
        if quote is None:
            return None
        expected_roi = None
    elif rule.kind == "imbalance":
        side = imbalance_side(row, rule.imbalance)
        if side is None:
            return None
        _, asks = side_books(row, side)
        if not asks or not rule.ask_min <= asks[0][0] <= rule.ask_max:
            return None
        quote = sweep_buy(asks, STAKE)
        if quote is None:
            return None
        expected_roi = None
    else:
        raise ValueError(rule.kind)
    return {"index": index, "t": row.t, "side": side, "quote": quote,
            "fair_up": probability, "expected_roi": expected_roi}


def execute(window, rule, run):
    decision = decision_for(window, rule)
    if decision is None:
        return None
    rng = random.Random(f"nonmomentum:{run}:{rule.name}:{window.slug}")
    target = decision["t"] + rng.uniform(DELAY_MIN, DELAY_MAX)
    index = bisect.bisect_left(window.times, target, decision["index"] + 1)
    if index >= len(window.rows) or window.rows[index].t >= 300:
        return None
    _, asks = side_books(window.rows[index], decision["side"])
    limit = min(0.99, decision["quote"]["worst"] + FOK_ALLOWANCE)
    fill = sweep_buy(asks, STAKE, limit)
    if fill is None:
        return None
    won = decision["side"] == window.winner
    pnl = (fill["shares"] if won else 0.0) - fill["cost"]
    return {**decision, **fill, "slug": window.slug, "start": window.start,
            "won": won, "pnl": pnl, "execution_t": window.rows[index].t}


def simulate(windows, rule, run):
    trades = [trade for window in windows if (trade := execute(window, rule, run))]
    equity = peak = drawdown = 0.0
    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {"trades": len(trades), "wins": sum(t["won"] for t in trades),
            "pnl": sum(t["pnl"] for t in trades), "drawdown": drawdown,
            "cost": sum(t["cost"] for t in trades), "rows": trades}


def aggregate(windows, rule, runs):
    samples = [simulate(windows, rule, run) for run in range(runs)]
    return {
        "trades": statistics.median(row["trades"] for row in samples),
        "wins": statistics.median(row["wins"] for row in samples),
        "pnl": statistics.median(row["pnl"] for row in samples),
        "pnl_min": min(row["pnl"] for row in samples),
        "pnl_max": max(row["pnl"] for row in samples),
        "drawdown": statistics.median(row["drawdown"] for row in samples),
        "roi": statistics.median(100 * row["pnl"] / row["cost"]
                                 if row["cost"] else 0 for row in samples),
    }


def format_metric(metric):
    return (f"n={metric['trades']:>4.0f} W={metric['wins']:>4.0f} "
            f"P=${metric['pnl']:+6.2f} [{metric['pnl_min']:+.2f},{metric['pnl_max']:+.2f}] "
            f"ROI={metric['roi']:>6.1f}% DD=${metric['drawdown']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "price_logs" / "polymarket")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    windows = load_all(args.directory)
    if len(windows) < 10:
        raise SystemExit("not enough resolved, complete windows")

    # The last five complete windows are never used to choose a rule.  They are
    # a small pseudo-prospective tail because the earlier logs had been examined.
    development, prospective = windows[:-5], windows[-5:]
    train_cut = int(len(development) * .60)
    valid_cut = int(len(development) * .80)
    train = development[:train_cut]
    validation = development[train_cut:valid_cut]
    holdout = development[valid_cut:]

    rules = [
        Rule("fair_edge_0", "fair", minimum_edge=0.00),
        Rule("fair_edge_5", "fair", minimum_edge=0.05),
        Rule("fair_edge_10", "fair", minimum_edge=0.10),
        Rule("market_favorite", "favorite"),
        Rule("book_imbalance_20", "imbalance", imbalance=0.20),
        Rule("book_imbalance_40", "imbalance", imbalance=0.40),
    ]
    groups = (("train", train), ("validation", validation),
              ("old_holdout", holdout), ("prospective5", prospective),
              ("all", windows))
    print(f"source-aligned complete windows={len(windows)}; development={len(development)}; "
          f"split={len(train)}/{len(validation)}/{len(holdout)}; prospective={len(prospective)}")
    print(f"stake=${STAKE:.0f}; delay={DELAY_MIN*1000:.0f}-{DELAY_MAX*1000:.0f}ms; "
          f"FOK allowance=${FOK_ALLOWANCE:.2f}; ask=.05-.82; current crypto fees")
    print("fair model: no drift, causal 1-second realized volatility, t=150")

    results = {}
    for rule in rules:
        print(f"\n{rule.name}")
        results[rule.name] = {}
        for name, group in groups:
            metric = aggregate(group, rule, args.runs)
            results[rule.name][name] = metric
            print(f"  {name:<12} {format_metric(metric)}")

    # Selection is restricted to the three predeclared fair-value edge gates,
    # and uses train P&L only with a minimum of five median fills.
    eligible = [rule for rule in rules[:3]
                if results[rule.name]["train"]["trades"] >= 5]
    selected = max(eligible, key=lambda rule: results[rule.name]["train"]["pnl"])
    print(f"\nTRAIN-SELECTED FAIR RULE: {selected.name}")
    for name, _ in groups:
        print(f"  {name:<12} {format_metric(results[selected.name][name])}")

    example = simulate(windows, selected, 0)
    print("\nselected run-0 trades")
    for trade in example["rows"]:
        fair = trade["fair_up"] if trade["side"] == "Up" else 1 - trade["fair_up"]
        print(f"  {trade['slug']} {trade['side']:<4} fair={fair:.3f} "
              f"avg={trade['average']:.3f} {'WIN ' if trade['won'] else 'LOSS'} "
              f"pnl=${trade['pnl']:+.3f}")


if __name__ == "__main__":
    main()
