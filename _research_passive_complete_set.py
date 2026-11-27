"""Research a conservative passive complete-set strategy.

The strategy posts maker BUY orders for equal Up and Down shares at (or below)
the displayed best bids.  The two limit prices sum to less than $1, so filling
both legs locks a settlement profit.  A maker order is counted as filled only
after the opposite ask trades strictly through its limit, avoiding optimistic
touch/queue assumptions.  If only one leg fills, the position is hedged or
unwound at a fixed cutoff using full depth, current taker fees, a $0.05 FOK
allowance, and 300--800 ms latency.  Maker rebates are ignored.

This file is research-only and never submits an order.
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


TOTAL_PLANNED_CASH = 2.0
FOK_ALLOWANCE = 0.05


@dataclass(frozen=True)
class Rule:
    start: float
    offset_ticks: int
    cutoff: float

    @property
    def name(self):
        return f"maker_t{self.start:g}_off{self.offset_ticks}_cut{self.cutoff:g}"


def fee(price, shares):
    return round(shares * 0.07 * price * (1.0 - price), 5)


def sides(row: Row, side: str):
    if side == "Up":
        return row.up_bids, row.up_asks
    return row.down_bids, row.down_asks


def quote_buy(levels, shares, maximum=0.99):
    remaining = shares
    cash = charge = 0.0
    worst = None
    for price, size in levels:
        if price > maximum + 1e-12:
            continue
        quantity = min(remaining, size)
        cash += price * quantity
        charge += fee(price, quantity)
        remaining -= quantity
        worst = price
        if remaining <= 1e-9:
            return cash + charge, worst
    return None


def quote_sell(levels, shares, minimum=0.01):
    remaining = shares
    proceeds = charge = 0.0
    worst = None
    for price, size in levels:
        if price + 1e-12 < minimum:
            continue
        quantity = min(remaining, size)
        proceeds += price * quantity
        charge += fee(price, quantity)
        remaining -= quantity
        worst = price
        if remaining <= 1e-9:
            return proceeds - charge, worst
    return None


def frame_after(window: Window, target: float, lower: int = 0):
    index = bisect.bisect_left(window.times, target, lo=lower)
    if index >= len(window.rows) or window.rows[index].t >= 300:
        return None, index
    return window.rows[index], index


def simulate(window: Window, rule: Rule, seed: int):
    rng = random.Random(f"passive-complete:{seed}:{rule.name}:{window.slug}")
    decision_i = bisect.bisect_left(window.times, rule.start)
    if decision_i >= len(window.rows):
        return None

    # The initial 300--800 ms models fetching the placement book and getting
    # both post-only orders accepted.  Limits are derived only at that frame.
    active, active_i = frame_after(
        window,
        window.rows[decision_i].t + rng.uniform(0.300, 0.800),
        decision_i + 1,
    )
    if active is None or active.t >= rule.cutoff:
        return None
    limits = {}
    for side in ("Up", "Down"):
        bids, asks = sides(active, side)
        if not bids or not asks:
            return None
        limit = round(bids[0][0] - 0.01 * rule.offset_ticks, 2)
        if limit <= 0 or limit >= asks[0][0] - 1e-12:
            return None
        limits[side] = limit
    if limits["Up"] + limits["Down"] >= 1.0:
        return None
    shares = TOTAL_PLANNED_CASH / (limits["Up"] + limits["Down"])

    filled = {}
    for index in range(active_i + 1, len(window.rows)):
        row = window.rows[index]
        if row.t >= rule.cutoff:
            break
        for side in ("Up", "Down"):
            if side in filled:
                continue
            _bids, asks = sides(row, side)
            # Strict trade-through is deliberately stronger than a touch and
            # is our conservative evidence that queue ahead was exhausted.
            if asks and asks[0][0] < limits[side] - 1e-12:
                filled[side] = row.t
        if len(filled) == 2:
            break

    if not filled:
        return {
            "slug": window.slug, "traded": False, "both": False,
            "one": False, "action": "none", "pnl": 0.0,
        }
    maker_cost = sum(limits[side] * shares for side in filled)
    if len(filled) == 2:
        return {
            "slug": window.slug, "traded": True, "both": True,
            "one": False, "action": "complete", "pnl": shares - maker_cost,
        }

    held = next(iter(filled))
    other = "Down" if held == "Up" else "Up"
    cutoff_i = bisect.bisect_left(window.times, rule.cutoff)
    if cutoff_i >= len(window.rows):
        payout = shares if held == window.winner else 0.0
        return {
            "slug": window.slug, "traded": True, "both": False,
            "one": True, "action": "naked", "pnl": payout - maker_cost,
        }
    decision = window.rows[cutoff_i]
    held_bids, _held_asks = sides(decision, held)
    _other_bids, other_asks = sides(decision, other)
    hedge_quote = quote_buy(other_asks, shares)
    unwind_quote = quote_sell(held_bids, shares)
    hedge_locked = (
        shares - maker_cost - hedge_quote[0] if hedge_quote is not None else -1e9
    )
    unwind_locked = (
        unwind_quote[0] - maker_cost if unwind_quote is not None else -1e9
    )
    action = "hedge" if hedge_locked >= unwind_locked else "unwind"
    quote = hedge_quote if action == "hedge" else unwind_quote
    if quote is None:
        payout = shares if held == window.winner else 0.0
        return {
            "slug": window.slug, "traded": True, "both": False,
            "one": True, "action": "naked", "pnl": payout - maker_cost,
        }
    execution, _ = frame_after(
        window,
        decision.t + rng.uniform(0.300, 0.800),
        cutoff_i + 1,
    )
    if execution is None:
        payout = shares if held == window.winner else 0.0
        return {
            "slug": window.slug, "traded": True, "both": False,
            "one": True, "action": "naked", "pnl": payout - maker_cost,
        }
    if action == "hedge":
        _bids, asks = sides(execution, other)
        executed = quote_buy(asks, shares, min(0.99, quote[1] + FOK_ALLOWANCE))
        if executed is not None:
            pnl = shares - maker_cost - executed[0]
            return {
                "slug": window.slug, "traded": True, "both": False,
                "one": True, "action": "hedge", "pnl": pnl,
            }
    else:
        bids, _asks = sides(execution, held)
        executed = quote_sell(bids, shares, max(0.01, quote[1] - FOK_ALLOWANCE))
        if executed is not None:
            pnl = executed[0] - maker_cost
            return {
                "slug": window.slug, "traded": True, "both": False,
                "one": True, "action": "unwind", "pnl": pnl,
            }

    # A failed risk-off FOK leaves the real position naked; settle it rather
    # than silently dropping the loss from the replay.
    payout = shares if held == window.winner else 0.0
    return {
        "slug": window.slug, "traded": True, "both": False,
        "one": True, "action": "naked", "pnl": payout - maker_cost,
    }


def metrics(windows, rule: Rule, seeds):
    runs = []
    for seed in seeds:
        rows = [simulate(window, rule, seed) for window in windows]
        trades = [row for row in rows if row and row["traded"]]
        equity = peak = drawdown = 0.0
        for trade in trades:
            equity += trade["pnl"]
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        runs.append({
            "trades": len(trades),
            "both": sum(row["both"] for row in trades),
            "one": sum(row["one"] for row in trades),
            "naked": sum(row["action"] == "naked" for row in trades),
            "pnl": sum(row["pnl"] for row in trades),
            "drawdown": drawdown,
        })
    result = {
        key: statistics.median(run[key] for run in runs)
        for key in ("trades", "both", "one", "naked", "pnl", "drawdown")
    }
    result["pnl_min"] = min(run["pnl"] for run in runs)
    result["pnl_max"] = max(run["pnl"] for run in runs)
    return result


def rules():
    return [
        Rule(start, offset, cutoff)
        for start in (15.0, 30.0, 60.0)
        for offset in (0, 1, 2)
        for cutoff in (180.0, 240.0, 270.0)
    ]


def show(value):
    return (
        f"n={value['trades']:.1f} pair={value['both']:.1f} "
        f"one={value['one']:.1f} naked={value['naked']:.1f} "
        f"pnl=${value['pnl']:+.2f} "
        f"range=${value['pnl_min']:+.2f}..${value['pnl_max']:+.2f} "
        f"dd=${value['drawdown']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "price_logs" / "polymarket",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--through", type=int)
    args = parser.parse_args()
    windows = []
    for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
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
    midpoint = len(train) // 2
    seeds = range(args.seeds)
    print(
        f"source-aligned resolved={len(windows)} split="
        f"{len(train)}/{len(validation)}/{len(holdout)}; configs={len(rules())}"
    )
    if windows:
        print(f"through={windows[-1].slug.rsplit('-', 1)[-1]}")
    print(
        "maker rebates ignored; strict trade-through fills; planned cash=$2; "
        "risk-off delay=300-800ms; taker allowance=$0.05"
    )

    screened = []
    for rule in rules():
        first = metrics(train[:midpoint], rule, seeds)
        second = metrics(train[midpoint:], rule, seeds)
        total = metrics(train, rule, seeds)
        if (
            first["trades"] >= 5 and second["trades"] >= 5
            and first["pnl_min"] > 0 and second["pnl_min"] > 0
        ):
            screened.append((min(first["pnl"], second["pnl"]), rule, total))
    screened.sort(key=lambda item: item[0], reverse=True)
    print("\nTRAIN-ROBUST RULES")
    if not screened:
        print("  none")
    for score, rule, result in screened[:8]:
        print(f"  {rule.name:<27} {show(result)} worst-half=${score:+.2f}")

    validated = []
    print("\nVALIDATION")
    for item in screened[:5]:
        _score, rule, train_result = item
        result = metrics(validation, rule, seeds)
        print(f"  {rule.name:<27} {show(result)}")
        if result["trades"] >= 3 and result["pnl_min"] > 0:
            validated.append((result["pnl"], rule, train_result, result))
    if not validated:
        print("\nNo rule passed validation. Holdout was not used to rescue one.")
        return
    validated.sort(key=lambda item: item[0], reverse=True)
    _score, selected, train_result, validation_result = validated[0]
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    for label, group, result in (
        ("train", train, train_result),
        ("validation", validation, validation_result),
        ("holdout", holdout, metrics(holdout, selected, seeds)),
        ("last5", windows[-5:], metrics(windows[-5:], selected, seeds)),
        ("all", windows, metrics(windows, selected, seeds)),
    ):
        print(f"  {label:<10} {show(result)}")


if __name__ == "__main__":
    main()
