"""Research-only non-momentum strategies for BTC 5-minute Polymarket logs.

The script deliberately keeps the rule set small and interpretable.  It uses
source-aligned timestamps, full recorded depth, a fixed $2 stake, the current
crypto taker-fee formula, a $0.05 adverse FOK allowance, and independently
seeded 300--800 ms execution latency.  It never submits an order.

Families:

* disagreement: buy the side indicated by Coinbase only when Polymarket still
  prices that side below 50 cents (a cross-market dislocation, not a trend).
* failed_breakout: after BTC starts reversing toward its window open, buy the
  currently losing outcome in anticipation of an opening-price recross.
* unsupported_favorite: fade a strong Polymarket favorite when Coinbase no
  longer supports that direction and has started moving against it.

Rules are screened on the first 60% only.  A short list is checked on the next
20%, and the last 20% is printed only as a final untouched audit.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import random
import statistics
from pathlib import Path

from _research_value_stop import load_source_aligned
from _exit_strategy_oos import Row, Window


STAKE = 2.0
FOK_ALLOWANCE = 0.05


@dataclass(frozen=True)
class Rule:
    family: str
    entry_second: float
    minimum_move: float = 0.0
    ask_max: float = 0.49
    lookback: float = 20.0
    reversal_bps: float = 0.0
    maximum_distance: float = 99.0
    favorite_mid: float = 0.0
    maximum_support: float = 99.0

    @property
    def name(self) -> str:
        if self.family == "disagreement":
            return (
                f"disagree_t{self.entry_second:g}_move{self.minimum_move:g}"
                f"_ask{self.ask_max:.2f}"
            )
        if self.family == "failed_breakout":
            return (
                f"failed_t{self.entry_second:g}_lb{self.lookback:g}"
                f"_rev{self.reversal_bps:g}_dist{self.maximum_distance:g}"
            )
        return (
            f"fade_t{self.entry_second:g}_fav{self.favorite_mid:.2f}"
            f"_support{self.maximum_support:g}_rev{self.reversal_bps:g}"
        )


@dataclass(frozen=True)
class Trade:
    slug: str
    side: str
    winner: str
    decision_t: float
    execution_t: float
    move_bps: float
    recent_bps: float
    ask: float
    average_price: float
    shares: float
    cost: float
    pnl: float


def official_fee(price: float, shares: float) -> float:
    """Polymarket crypto taker fee for one price level."""
    return round(shares * 0.07 * price * (1.0 - price), 5)


def midpoint(row: Row, side: str) -> float | None:
    bids = row.up_bids if side == "Up" else row.down_bids
    asks = row.up_asks if side == "Up" else row.down_asks
    if not bids or not asks:
        return None
    return (bids[0][0] + asks[0][0]) / 2.0


def asks_for(row: Row, side: str):
    return row.up_asks if side == "Up" else row.down_asks


def sweep_buy(levels, dollars: float, max_price: float = 1.0):
    remaining = dollars
    shares = fees = 0.0
    worst = None
    for price, size in levels:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        quantity = used / price
        shares += quantity
        fees += official_fee(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, worst
    return None


def reference_row(window: Window, decision_i: int, seconds_ago: float) -> Row:
    target = window.rows[decision_i].t - seconds_ago
    index = bisect.bisect_right(window.times, target, 0, decision_i + 1) - 1
    return window.rows[max(0, index)]


def signal(window: Window, rule: Rule):
    i = bisect.bisect_left(window.times, rule.entry_second)
    if i >= len(window.rows):
        return None
    row = window.rows[i]
    opening = window.rows[0].coinbase
    move = (row.coinbase / opening - 1.0) * 10_000.0
    if move == 0:
        return None

    if rule.family == "disagreement":
        if abs(move) < rule.minimum_move:
            return None
        side = "Up" if move > 0 else "Down"
        selected_mid = midpoint(row, side)
        asks = asks_for(row, side)
        if selected_mid is None or not asks:
            return None
        # The defining condition is actual disagreement: Coinbase says the
        # side is currently ahead while the contract remains an underdog.
        if selected_mid >= 0.50 or asks[0][0] > rule.ask_max:
            return None
        old = reference_row(window, i, 10.0)
        recent = (row.coinbase / old.coinbase - 1.0) * 10_000.0
        return i, side, move, recent

    if rule.family == "failed_breakout":
        current_side = "Up" if move > 0 else "Down"
        direction = 1.0 if current_side == "Up" else -1.0
        old = reference_row(window, i, rule.lookback)
        recent = direction * (row.coinbase / old.coinbase - 1.0) * 10_000.0
        # BTC remains on one side of the strike, but is moving back toward it.
        if abs(move) > rule.maximum_distance or recent > -rule.reversal_bps:
            return None
        side = "Down" if current_side == "Up" else "Up"
        asks = asks_for(row, side)
        selected_mid = midpoint(row, side)
        if not asks or selected_mid is None or selected_mid >= 0.50:
            return None
        if asks[0][0] > rule.ask_max:
            return None
        return i, side, move, recent

    # Fade a book favorite only when Coinbase has little/no support in the
    # favorite's direction and the most recent move is against that favorite.
    up_mid = midpoint(row, "Up")
    down_mid = midpoint(row, "Down")
    if up_mid is None or down_mid is None:
        return None
    favorite = "Up" if up_mid >= down_mid else "Down"
    favorite_probability = max(up_mid, down_mid)
    direction = 1.0 if favorite == "Up" else -1.0
    support = direction * move
    old = reference_row(window, i, rule.lookback)
    recent = direction * (row.coinbase / old.coinbase - 1.0) * 10_000.0
    if favorite_probability < rule.favorite_mid:
        return None
    if support > rule.maximum_support or recent > -rule.reversal_bps:
        return None
    side = "Down" if favorite == "Up" else "Up"
    asks = asks_for(row, side)
    if not asks or asks[0][0] > rule.ask_max:
        return None
    return i, side, move, recent


def execute(window: Window, rule: Rule, seed: int) -> Trade | None:
    found = signal(window, rule)
    if found is None:
        return None
    decision_i, side, move, recent = found
    decision = window.rows[decision_i]
    decision_asks = asks_for(decision, side)
    quoted = sweep_buy(decision_asks, STAKE)
    if quoted is None:
        return None
    order_limit = min(0.99, quoted[2] + FOK_ALLOWANCE)
    rng = random.Random(f"nonmomentum:{seed}:{rule.name}:{window.slug}")
    target = decision.t + rng.uniform(0.300, 0.800)
    execution_i = bisect.bisect_left(window.times, target, lo=decision_i + 1)
    if execution_i >= len(window.rows) or window.rows[execution_i].t >= 300.0:
        return None
    execution = window.rows[execution_i]
    filled = sweep_buy(asks_for(execution, side), STAKE, order_limit)
    if filled is None:
        return None
    shares, cost, _ = filled
    pnl = (shares if side == window.winner else 0.0) - cost
    return Trade(
        window.slug,
        side,
        window.winner,
        decision.t,
        execution.t,
        move,
        recent,
        decision_asks[0][0],
        STAKE / shares,
        shares,
        cost,
        pnl,
    )


def run(windows, rule: Rule, seed: int):
    return [trade for window in windows if (trade := execute(window, rule, seed))]


def metrics(trades):
    pnl = sum(trade.pnl for trade in trades)
    cost = sum(trade.cost for trade in trades)
    equity = peak = max_dd = 0.0
    for trade in trades:
        equity += trade.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "trades": len(trades),
        "wins": sum(trade.pnl > 0 for trade in trades),
        "pnl": pnl,
        "roi": 100.0 * pnl / cost if cost else 0.0,
        "max_dd": max_dd,
    }


def aggregate(windows, rule: Rule, seeds):
    rows = [metrics(run(windows, rule, seed)) for seed in seeds]
    result = {
        key: statistics.median(row[key] for row in rows)
        for key in ("trades", "wins", "pnl", "roi", "max_dd")
    }
    result["pnl_min"] = min(row["pnl"] for row in rows)
    result["pnl_max"] = max(row["pnl"] for row in rows)
    return result


def rules():
    result = []
    # 12 cross-market disagreement rules.
    for entry in (120.0, 150.0, 180.0):
        for minimum_move in (0.5, 1.0):
            for ask_max in (0.45, 0.49):
                result.append(
                    Rule("disagreement", entry, minimum_move=minimum_move, ask_max=ask_max)
                )
    # 12 failed-breakout rules; 20 seconds is fixed to limit flexibility.
    for entry in (120.0, 150.0, 180.0):
        for reversal in (1.0, 2.0):
            for distance in (4.0, 6.0):
                result.append(
                    Rule(
                        "failed_breakout",
                        entry,
                        lookback=20.0,
                        reversal_bps=reversal,
                        maximum_distance=distance,
                    )
                )
    # 12 unsupported-favorite rules; recent lookback and underdog cap fixed.
    for entry in (120.0, 150.0, 180.0):
        for favorite in (0.65, 0.75):
            for maximum_support in (0.0, 2.0):
                result.append(
                    Rule(
                        "unsupported_favorite",
                        entry,
                        lookback=20.0,
                        reversal_bps=1.0,
                        favorite_mid=favorite,
                        maximum_support=maximum_support,
                        ask_max=0.49,
                    )
                )
    return result


def format_result(value):
    return (
        f"n={value['trades']:.1f} w={value['wins']:.1f} "
        f"pnl=${value['pnl']:+.2f} "
        f"range=${value['pnl_min']:+.2f}..${value['pnl_max']:+.2f} "
        f"roi={value['roi']:+.1f}% dd=${value['max_dd']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
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

    cut60 = int(len(windows) * 0.60)
    cut80 = int(len(windows) * 0.80)
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    train_mid = len(train) // 2
    seeds = range(args.seeds)
    configs = rules()

    print(
        f"source-aligned resolved={len(windows)}; chronological split="
        f"{len(train)}/{len(validation)}/{len(holdout)}; rules={len(configs)}"
    )
    if windows:
        print(f"through={windows[-1].slug.rsplit('-', 1)[-1]}")
    print(
        "stake=$2; full depth; delay=300-800ms; FOK allowance=$0.05; "
        "fee=shares*0.07*p*(1-p); hold to settlement"
    )

    # Screen without touching either later segment.  A rule must produce at
    # least two fills and positive median PnL in both chronological train halves.
    screened = []
    for rule in configs:
        first = aggregate(train[:train_mid], rule, seeds)
        second = aggregate(train[train_mid:], rule, seeds)
        total = aggregate(train, rule, seeds)
        if (
            first["trades"] >= 2
            and second["trades"] >= 2
            and first["pnl_min"] > 0
            and second["pnl_min"] > 0
        ):
            score = min(first["roi"], second["roi"])
            screened.append((score, total["trades"], rule, first, second, total))

    screened.sort(key=lambda item: (item[0], item[1]), reverse=True)
    print("\nTRAIN-ROBUST SHORTLIST (positive under every latency seed in both halves)")
    if not screened:
        print("  none")
    for score, _, rule, first, second, total in screened[:8]:
        print(f"  {rule.name}")
        print(f"    train A: {format_result(first)}")
        print(f"    train B: {format_result(second)}")
        print(f"    train:   {format_result(total)}; worst-half ROI={score:+.1f}%")

    # Only the top five train-robust rules get a validation look.  Select by
    # validation PnL subject to at least two validation fills and PnL > 0 for
    # every delay seed.  Holdout remains untouched until selection is frozen.
    validated = []
    print("\nVALIDATION OF TOP FIVE TRAIN RULES")
    for item in screened[:5]:
        rule = item[2]
        value = aggregate(validation, rule, seeds)
        print(f"  {rule.name}: {format_result(value)}")
        if value["trades"] >= 2 and value["pnl_min"] > 0:
            validated.append((value["pnl"], item, value))

    if not validated:
        print("\nNO CANDIDATE PASSED VALIDATION; holdout was not used to rescue a rule.")
        print("\nFAMILY DIAGNOSTIC (all windows; descriptive only)")
        for family in ("disagreement", "failed_breakout", "unsupported_favorite"):
            family_rules = [item for item in screened if item[2].family == family]
            if not family_rules:
                print(f"  {family}: no train-robust rule")
                continue
            rule = family_rules[0][2]
            print(f"  {family}: {rule.name}: {format_result(aggregate(windows, rule, seeds))}")
        return

    validated.sort(key=lambda item: item[0], reverse=True)
    _, selected_train, selected_validation = validated[0]
    selected = selected_train[2]
    print(f"\nFROZEN SELECTION BEFORE HOLDOUT: {selected.name}")
    print(f"  validation: {format_result(selected_validation)}")
    print("\nFINAL AUDIT")
    for label, group in (
        ("train", train),
        ("validation", validation),
        ("holdout", holdout),
        ("all", windows),
    ):
        print(f"  {label:<10} {format_result(aggregate(group, selected, seeds))}")

    # Show actual fills for one deterministic seed, which makes it possible to
    # inspect whether a tiny number of trades is driving the result.
    print("\nSELECTED RULE TRADES (seed 0)")
    for trade in run(windows, selected, 0):
        outcome = "W" if trade.pnl > 0 else "L"
        print(
            f"  {trade.slug} {trade.side:<4} {outcome} "
            f"ask={trade.ask:.2f} avg={trade.average_price:.3f} "
            f"move={trade.move_bps:+.2f}bps recent={trade.recent_bps:+.2f}bps "
            f"pnl=${trade.pnl:+.2f}"
        )


if __name__ == "__main__":
    main()
