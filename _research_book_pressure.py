"""Causal order-book-pressure research for recorded BTC five-minute markets.

Research only: this module never places an order and does not modify a live
bot.  It tests a small, predeclared collection of interpretable Polymarket
microstructure signals.  Combined snapshots are timestamped at the latest
component *source* timestamp, rather than at the logger's early row timestamp.

The audit is deliberately staged.  Configurations are ranked on two
chronological halves of the 60% training slice.  At most one configuration is
then evaluated on the following 20% validation slice.  The final 20% holdout
is evaluated only when that candidate is profitable for every common-random
latency seed in both training halves and validation.  A failed gate therefore
leaves every later slice untouched by model selection.

Execution assumptions match the requested live constraints: full recorded
depth, a fixed $2 stake, current crypto taker fees, a $0.05 adverse FOK price
allowance, and a 300--800 ms delay.  The execution book must be a later
observation than the decision book.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import math
from pathlib import Path
import random
import statistics

from _exit_strategy_oos import Row, Window
from _research_value_stop import load_source_aligned
from btc5m_momentum_strategy import fee_for


ROOT = Path(__file__).resolve().parent
FROZEN_THROUGH = 1792066500
STAKE = 2.0
FOK_ALLOWANCE = 0.05
DELAY_MIN = 0.300
DELAY_MAX = 0.800
ASK_MIN = 0.20
ASK_MAX = 0.82
MIN_SPLIT_TRADES = 5
MIN_VALIDATION_TRADES = 6


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    entry_second: float
    metric: str
    threshold: float
    lookback: float = 0.0
    minimum_mid_move: float = 0.0


@dataclass(frozen=True)
class Decision:
    index: int
    t: float
    side: str
    signal: float
    paired_mid: float
    best_ask: float
    quoted_worst: float


@dataclass(frozen=True)
class Trade:
    slug: str
    side: str
    winner: str
    won: bool
    signal: float
    decision_t: float
    execution_t: float
    average_price: float
    shares: float
    cost: float
    pnl: float


def asks_for(row: Row, side: str):
    return row.up_asks if side == "Up" else row.down_asks


def side_books(row: Row, side: str):
    if side == "Up":
        return row.up_bids, row.up_asks
    return row.down_bids, row.down_asks


def sweep_buy(levels, dollars: float, max_price: float = 1.0):
    """Buy a fixed dollar amount across all displayed levels, including fees."""
    remaining = dollars
    shares = 0.0
    fees = 0.0
    worst = None
    for price, size in levels:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        quantity = used / price
        shares += quantity
        fees += fee_for(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, worst, dollars / shares
    return None


def top_mid(row: Row, side: str):
    bids, asks = side_books(row, side)
    if not bids or not asks:
        return None
    return (bids[0][0] + asks[0][0]) / 2.0


def paired_mid(row: Row):
    """One Up-probability quote using both complementary outcome books."""
    up = top_mid(row, "Up")
    down = top_mid(row, "Down")
    if up is None or down is None:
        return None
    return (up + (1.0 - down)) / 2.0


def side_pressure(row: Row, side: str, metric: str):
    """Return a normalized [-1, 1] bid-versus-ask depth pressure.

    ``top`` is also the normalized standard microprice displacement: twice
    (microprice - midpoint) divided by spread.  ``depth3`` extends the same
    idea to three price levels with linearly declining weights.  Sizes are in
    shares on both sides, so they can be compared directly within a token.
    """
    bids, asks = side_books(row, side)
    if not bids or not asks:
        return None
    if bids[0][0] >= asks[0][0] or asks[0][0] - bids[0][0] > 0.06:
        return None
    if metric == "top":
        bid_depth = bids[0][1]
        ask_depth = asks[0][1]
    elif metric == "depth3":
        weights = (1.0, 0.60, 0.30)
        bid_depth = sum(size * weights[i] for i, (_, size) in enumerate(bids[:3]))
        ask_depth = sum(size * weights[i] for i, (_, size) in enumerate(asks[:3]))
    else:
        raise ValueError(f"unknown pressure metric: {metric}")
    total = bid_depth + ask_depth
    if total <= 0:
        return None
    return (bid_depth - ask_depth) / total


def paired_pressure(row: Row, metric: str):
    """Positive means pressure toward Up, negative means pressure toward Down."""
    up = side_pressure(row, "Up", metric)
    down = side_pressure(row, "Down", metric)
    if up is None or down is None:
        return None
    return (up - down) / 2.0


def reference_index(window: Window, current_index: int, lookback: float):
    target = window.rows[current_index].t - lookback
    index = bisect.bisect_right(window.times, target, 0, current_index) - 1
    return index if index >= 0 else None


def signal_for(window: Window, config: Config):
    index = bisect.bisect_left(window.times, config.entry_second)
    if index >= len(window.rows):
        return None
    row = window.rows[index]
    current = paired_pressure(row, config.metric)
    current_mid = paired_mid(row)
    if current is None or current_mid is None:
        return None

    signal = current
    if config.family == "current":
        if abs(current) < config.threshold:
            return None
    else:
        old_index = reference_index(window, index, config.lookback)
        if old_index is None:
            return None
        old = window.rows[old_index]
        previous = paired_pressure(old, config.metric)
        previous_mid = paired_mid(old)
        if previous is None or previous_mid is None:
            return None
        pressure_change = current - previous
        mid_change = current_mid - previous_mid

        if config.family == "persistent":
            if (current * previous <= 0
                    or min(abs(current), abs(previous)) < config.threshold):
                return None
            signal = current
        elif config.family == "flow":
            if (current * pressure_change <= 0
                    or abs(pressure_change) < config.threshold):
                return None
            signal = pressure_change
        elif config.family == "confirm":
            if (abs(current) < config.threshold
                    or abs(mid_change) < config.minimum_mid_move
                    or current * mid_change <= 0):
                return None
            signal = mid_change
        elif config.family == "reversal":
            if (abs(current) < config.threshold
                    or abs(mid_change) < config.minimum_mid_move
                    or current * mid_change >= 0):
                return None
            signal = current
        else:
            raise ValueError(f"unknown signal family: {config.family}")

    side = "Up" if signal > 0 else "Down"
    asks = asks_for(row, side)
    if not asks or not ASK_MIN <= asks[0][0] <= ASK_MAX:
        return None
    quote = sweep_buy(asks, STAKE)
    if quote is None:
        return None
    _, _, quoted_worst, _ = quote
    return Decision(
        index=index,
        t=row.t,
        side=side,
        signal=signal,
        paired_mid=current_mid,
        best_ask=asks[0][0],
        quoted_worst=quoted_worst,
    )


def latency(seed: int, slug: str):
    """Common random number: every configuration gets the same window delay."""
    return random.Random(f"book-pressure:{seed}:{slug}").uniform(DELAY_MIN, DELAY_MAX)


def execute(window: Window, decision: Decision | None, seed: int):
    if decision is None:
        return None
    target = decision.t + latency(seed, window.slug)
    index = bisect.bisect_left(
        window.times, target, lo=decision.index + 1
    )
    if index >= len(window.rows) or window.rows[index].t >= 300.0:
        return None
    execution = window.rows[index]
    limit = min(0.99, decision.quoted_worst + FOK_ALLOWANCE)
    fill = sweep_buy(asks_for(execution, decision.side), STAKE, limit)
    if fill is None:
        return None
    shares, cost, _, average = fill
    won = decision.side == window.winner
    pnl = (shares if won else 0.0) - cost
    return Trade(
        slug=window.slug,
        side=decision.side,
        winner=window.winner,
        won=won,
        signal=decision.signal,
        decision_t=decision.t,
        execution_t=execution.t,
        average_price=average,
        shares=shares,
        cost=cost,
        pnl=pnl,
    )


def run_once(windows, decisions, seed: int):
    trades = []
    for window, decision in zip(windows, decisions):
        trade = execute(window, decision, seed)
        if trade is not None:
            trades.append(trade)
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for trade in trades:
        equity += trade.pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    cost = sum(trade.cost for trade in trades)
    return {
        "trades": len(trades),
        "wins": sum(trade.won for trade in trades),
        "pnl": sum(trade.pnl for trade in trades),
        "cost": cost,
        "roi": 100.0 * sum(trade.pnl for trade in trades) / cost if cost else 0.0,
        "drawdown": drawdown,
        "rows": trades,
    }


def aggregate(windows, config: Config, seeds: int):
    decisions = [signal_for(window, config) for window in windows]
    samples = [run_once(windows, decisions, seed) for seed in range(seeds)]
    return {
        "attempts": sum(decision is not None for decision in decisions),
        "trades": statistics.median(sample["trades"] for sample in samples),
        "trades_min": min(sample["trades"] for sample in samples),
        "trades_max": max(sample["trades"] for sample in samples),
        "wins": statistics.median(sample["wins"] for sample in samples),
        "pnl": statistics.median(sample["pnl"] for sample in samples),
        "pnl_min": min(sample["pnl"] for sample in samples),
        "pnl_max": max(sample["pnl"] for sample in samples),
        "roi": statistics.median(sample["roi"] for sample in samples),
        "drawdown": statistics.median(sample["drawdown"] for sample in samples),
        "drawdown_max": max(sample["drawdown"] for sample in samples),
        "seed0": samples[0],
    }


def configurations():
    """Small grid declared before validation or holdout is inspected (30 total)."""
    result = []
    # Static top-level microprice and three-level depth pressure.
    for entry in (150.0, 180.0):
        for threshold in (0.35, 0.55):
            result.append(Config(
                f"top_current_t{entry:g}_p{threshold:.2f}",
                "current", entry, "top", threshold,
            ))
    for entry in (120.0, 150.0, 180.0):
        for threshold in (0.25, 0.45):
            result.append(Config(
                f"depth3_current_t{entry:g}_p{threshold:.2f}",
                "current", entry, "depth3", threshold,
            ))

    # Pressure must persist, or newly arrive, over a short causal lookback.
    for entry in (150.0, 180.0):
        for lookback in (5.0, 15.0):
            result.append(Config(
                f"depth3_persist_t{entry:g}_lb{lookback:g}",
                "persistent", entry, "depth3", 0.20, lookback,
            ))
            result.append(Config(
                f"depth3_flow_t{entry:g}_lb{lookback:g}",
                "flow", entry, "depth3", 0.25, lookback,
            ))

    # Pressure that confirms a one-cent move versus pressure that absorbs it.
    for entry in (150.0, 180.0):
        for lookback in (10.0, 30.0):
            result.append(Config(
                f"depth3_confirm_t{entry:g}_lb{lookback:g}",
                "confirm", entry, "depth3", 0.20, lookback, 0.01,
            ))
            result.append(Config(
                f"depth3_reversal_t{entry:g}_lb{lookback:g}",
                "reversal", entry, "depth3", 0.20, lookback, 0.01,
            ))

    # Top-level microprice changes catch more transient pressure than depth3.
    for entry in (150.0, 180.0):
        for lookback in (5.0, 15.0):
            result.append(Config(
                f"top_flow_t{entry:g}_lb{lookback:g}",
                "flow", entry, "top", 0.30, lookback,
            ))
    return tuple(result)


def metric_line(metric):
    return (
        f"signals={metric['attempts']:2d} "
        f"fills={metric['trades']:4.1f}({metric['trades_min']}-{metric['trades_max']}) "
        f"wins={metric['wins']:4.1f} "
        f"P=${metric['pnl']:+7.3f} "
        f"range=${metric['pnl_min']:+.3f}..${metric['pnl_max']:+.3f} "
        f"ROI={metric['roi']:+6.2f}% "
        f"DD=${metric['drawdown']:.3f}(max {metric['drawdown_max']:.3f})"
    )


def training_pass(first, second):
    return (
        first["trades"] >= MIN_SPLIT_TRADES
        and second["trades"] >= MIN_SPLIT_TRADES
        and first["pnl_min"] > 0.0
        and second["pnl_min"] > 0.0
    )


def validation_pass(metric):
    return metric["trades"] >= MIN_VALIDATION_TRADES and metric["pnl_min"] > 0.0


def load_frozen(directory: Path, through: int):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if start > through:
            continue
        window = load_source_aligned(path)
        if window is not None:
            windows.append(window)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--through", type=int, default=FROZEN_THROUGH)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--train-only", action="store_true",
        help="stop before the validation gate even if a candidate passes training",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    windows = load_frozen(args.directory, args.through)
    if len(windows) < 30:
        raise SystemExit("not enough source-aligned resolved windows")
    cut60 = int(len(windows) * 0.60)
    cut80 = int(len(windows) * 0.80)
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    half = len(train) // 2
    train_a = train[:half]
    train_b = train[half:]

    print(
        f"BOOK PRESSURE AUDIT frozen_through={args.through}; "
        f"resolved={len(windows)}; split={len(train)}/{len(validation)}/{len(holdout)}"
    )
    print(
        f"source-aligned; full depth; stake=${STAKE:.2f}; current crypto fees; "
        f"delay={DELAY_MIN*1000:.0f}-{DELAY_MAX*1000:.0f}ms; "
        f"FOK allowance=${FOK_ALLOWANCE:.2f}; ask=${ASK_MIN:.2f}-${ASK_MAX:.2f}; "
        f"common seeds={args.seeds}"
    )
    print(
        f"training halves={len(train_a)}/{len(train_b)}; "
        f"gate: >=${0.0:.2f} worst-seed P&L and >= {MIN_SPLIT_TRADES} median fills "
        "in each half"
    )

    rows = []
    for config in configurations():
        first = aggregate(train_a, config, args.seeds)
        second = aggregate(train_b, config, args.seeds)
        score = (min(first["pnl"], second["pnl"]), first["pnl"] + second["pnl"])
        rows.append((config, first, second, score))

    rows.sort(key=lambda item: item[3], reverse=True)
    print(f"\nTRAINING LEADERBOARD (all {len(rows)} predeclared configurations)")
    for config, first, second, _ in rows:
        passed = "PASS" if training_pass(first, second) else "fail"
        print(f"{passed:4} {config.name:<36} A {metric_line(first)}")
        print(f"{'':41} B {metric_line(second)}")

    eligible = [row for row in rows if training_pass(row[1], row[2])]
    print(f"\ntraining-pass candidates={len(eligible)}")
    if not eligible:
        print("REJECT: no configuration cleared both training halves.")
        print("VALIDATION NOT EVALUATED; FINAL HOLDOUT NOT EVALUATED.")
        return

    selected, first, second, _ = eligible[0]
    print(f"TRAIN-SELECTED: {selected.name}")
    print(f"  train-A {metric_line(first)}")
    print(f"  train-B {metric_line(second)}")
    if args.train_only:
        print("Stopped by --train-only; validation and final holdout not evaluated.")
        return

    valid_metric = aggregate(validation, selected, args.seeds)
    print(f"  validation {metric_line(valid_metric)}")
    if not validation_pass(valid_metric):
        print("REJECT: train-selected configuration failed validation.")
        print("FINAL HOLDOUT NOT EVALUATED.")
        return

    print("VALIDATION PASS: unlocking the untouched final holdout once.")
    holdout_metric = aggregate(holdout, selected, args.seeds)
    print(f"  FINAL holdout {metric_line(holdout_metric)}")
    if not validation_pass(holdout_metric):
        print("REJECT: configuration failed the final holdout.")
        return

    all_metric = aggregate(windows, selected, args.seeds)
    print(f"  all          {metric_line(all_metric)}")
    print("ROBUST HISTORICAL PASS under every declared latency seed.")
    print("Seed-0 final-holdout trades:")
    for trade in holdout_metric["seed0"]["rows"]:
        outcome = "WIN" if trade.won else "LOSS"
        print(
            f"  {trade.slug} {trade.side:<4} {outcome:<4} "
            f"signal={trade.signal:+.3f} avg={trade.average_price:.3f} "
            f"t={trade.decision_t:.3f}->{trade.execution_t:.3f} "
            f"pnl=${trade.pnl:+.3f}"
        )


if __name__ == "__main__":
    main()
