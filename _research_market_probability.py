"""Research-only Polymarket-price strategies for BTC five-minute markets.

These rules do not use Coinbase direction or momentum.  At a fixed time they
buy either Polymarket's favorite or its underdog when the selected outcome is
inside a predeclared price band.  Replays use source-aligned timestamps, full
recorded depth, a fixed $2 stake, current crypto taker fees, a $0.05 FOK
allowance, and 300--800 ms execution latency.  No orders are submitted.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
import random
import statistics

from _research_value_stop import load_source_aligned
from _exit_strategy_oos import Row, Window


STAKE = 2.0
FOK_ALLOWANCE = 0.05


@dataclass(frozen=True)
class Rule:
    side_rule: str
    entry_second: float
    ask_min: float
    ask_max: float

    @property
    def name(self) -> str:
        return (
            f"{self.side_rule}_t{self.entry_second:g}_"
            f"ask{self.ask_min:.2f}-{self.ask_max:.2f}"
        )


def fee(price: float, shares: float) -> float:
    return round(shares * 0.07 * price * (1.0 - price), 5)


def book_sides(row: Row, side: str):
    if side == "Up":
        return row.up_bids, row.up_asks
    return row.down_bids, row.down_asks


def midpoint(row: Row, side: str) -> float | None:
    bids, asks = book_sides(row, side)
    if not bids or not asks:
        return None
    return (bids[0][0] + asks[0][0]) / 2.0


def sweep(levels, dollars: float, maximum_price: float = 0.99):
    remaining = dollars
    shares = charge = 0.0
    worst = None
    for price, size in levels:
        if price > maximum_price + 1e-12:
            continue
        used = min(remaining, price * size)
        quantity = used / price
        shares += quantity
        charge += fee(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + charge, worst
    return None


def execute(window: Window, rule: Rule, seed: int):
    decision_i = bisect.bisect_left(window.times, rule.entry_second)
    if decision_i >= len(window.rows):
        return None
    decision = window.rows[decision_i]
    up_mid = midpoint(decision, "Up")
    down_mid = midpoint(decision, "Down")
    if up_mid is None or down_mid is None or up_mid == down_mid:
        return None
    favorite = "Up" if up_mid > down_mid else "Down"
    side = favorite if rule.side_rule == "favorite" else (
        "Down" if favorite == "Up" else "Up"
    )
    _, decision_asks = book_sides(decision, side)
    if not decision_asks:
        return None
    best_ask = decision_asks[0][0]
    if not rule.ask_min <= best_ask <= rule.ask_max:
        return None
    quoted = sweep(decision_asks, STAKE)
    if quoted is None:
        return None
    limit = min(0.99, quoted[2] + FOK_ALLOWANCE)
    rng = random.Random(f"market-probability:{seed}:{rule.name}:{window.slug}")
    target = decision.t + rng.uniform(0.300, 0.800)
    execution_i = bisect.bisect_left(window.times, target, lo=decision_i + 1)
    if execution_i >= len(window.rows) or window.rows[execution_i].t >= 300:
        return None
    _, execution_asks = book_sides(window.rows[execution_i], side)
    filled = sweep(execution_asks, STAKE, limit)
    if filled is None:
        return None
    shares, cost, _ = filled
    return {
        "slug": window.slug,
        "side": side,
        "winner": window.winner,
        "won": side == window.winner,
        "ask": best_ask,
        "pnl": (shares if side == window.winner else 0.0) - cost,
        "cost": cost,
    }


def metrics(windows, rule: Rule, seeds):
    runs = []
    for seed in seeds:
        trades = [
            trade for window in windows
            if (trade := execute(window, rule, seed)) is not None
        ]
        equity = peak = drawdown = 0.0
        for trade in trades:
            equity += trade["pnl"]
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        pnl = sum(trade["pnl"] for trade in trades)
        cost = sum(trade["cost"] for trade in trades)
        runs.append({
            "trades": len(trades),
            "wins": sum(trade["won"] for trade in trades),
            "pnl": pnl,
            "roi": 100.0 * pnl / cost if cost else 0.0,
            "drawdown": drawdown,
        })
    result = {
        key: statistics.median(run[key] for run in runs)
        for key in ("trades", "wins", "pnl", "roi", "drawdown")
    }
    result["pnl_min"] = min(run["pnl"] for run in runs)
    result["pnl_max"] = max(run["pnl"] for run in runs)
    return result


def rules():
    result = []
    for entry in (120.0, 150.0, 180.0, 210.0, 240.0, 270.0):
        result.append(Rule("favorite", entry, 0.55, 0.82))
        result.append(Rule("favorite", entry, 0.65, 0.90))
        result.append(Rule("underdog", entry, 0.18, 0.45))
        result.append(Rule("underdog", entry, 0.30, 0.49))
    return result


def show(value):
    return (
        f"n={value['trades']:.1f} w={value['wins']:.1f} "
        f"pnl=${value['pnl']:+.2f} "
        f"range=${value['pnl_min']:+.2f}..${value['pnl_max']:+.2f} "
        f"roi={value['roi']:+.1f}% dd=${value['drawdown']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "price_logs" / "polymarket"
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--through", type=int)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("btc-updown-5m-*.jsonl"))
    windows = []
    for path in paths:
        try:
            epoch = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if args.through is not None and epoch > args.through:
            continue
        window = load_source_aligned(path)
        if window is not None:
            windows.append(window)
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    train_mid = len(train) // 2
    seeds = range(args.seeds)
    print(
        f"source-aligned resolved={len(windows)} split="
        f"{len(train)}/{len(validation)}/{len(holdout)}; rules={len(rules())}"
    )
    if windows:
        print(f"through={windows[-1].slug.rsplit('-', 1)[-1]}")
    print(
        "no Coinbase signal; stake=$2; full depth; delay=300-800ms; "
        "FOK allowance=$0.05; current fees; hold to settlement"
    )

    screened = []
    for rule in rules():
        first = metrics(train[:train_mid], rule, seeds)
        second = metrics(train[train_mid:], rule, seeds)
        combined = metrics(train, rule, seeds)
        if (
            first["trades"] >= 5 and second["trades"] >= 5
            and first["pnl_min"] > 0 and second["pnl_min"] > 0
        ):
            screened.append((min(first["roi"], second["roi"]), rule, combined))
    screened.sort(key=lambda item: item[0], reverse=True)
    print("\nTRAIN-ROBUST RULES")
    if not screened:
        print("  none")
    for score, rule, value in screened:
        print(f"  {rule.name:<34} {show(value)} worst-half-roi={score:+.1f}%")

    validated = []
    print("\nVALIDATION")
    for item in screened[:5]:
        score, rule, train_value = item
        value = metrics(validation, rule, seeds)
        print(f"  {rule.name:<34} {show(value)}")
        if value["trades"] >= 3 and value["pnl_min"] > 0:
            validated.append((value["roi"], rule, train_value, value))
    if not validated:
        print("\nNo rule passed validation. Holdout was not used to rescue one.")
        return

    validated.sort(key=lambda item: item[0], reverse=True)
    _, selected, train_value, validation_value = validated[0]
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    for label, group, value in (
        ("train", train, train_value),
        ("validation", validation, validation_value),
        ("holdout", holdout, metrics(holdout, selected, seeds)),
        ("last5", windows[-5:], metrics(windows[-5:], selected, seeds)),
        ("all", windows, metrics(windows, selected, seeds)),
    ):
        print(f"  {label:<10} {show(value)}")


if __name__ == "__main__":
    main()
