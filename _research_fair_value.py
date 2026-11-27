"""Research-only causal fair-value strategies for BTC five-minute markets.

The model estimates the probability that Coinbase BTC finishes above its
price at the start of the window.  Its inputs are deliberately small:

* log distance from the observed window-open Coinbase price;
* seconds remaining in the five-minute window; and
* one-second realized volatility observed before the decision.

Two probability mappings are audited: the zero-drift Brownian probability
``Phi(distance / (volatility * sqrt(time_remaining)))`` and a one-parameter
logistic calibration.  The logistic slope is fitted on the chronological
training set only and shrunk toward the standard normal/logistic approximation.

Entries compare model probability with the complete $2 executable ask sweep,
including the current crypto taker fee.  Replays use source-aligned timestamps,
full recorded depth, a $0.05 FOK allowance, and independently seeded 300--800
ms latency.  Positions are held to settlement.  This file never submits an
order and does not modify the live bot.
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


ROOT = Path(__file__).resolve().parent
FROZEN_THROUGH = 1792065900
STAKE = 2.0
FOK_ALLOWANCE = 0.05
ENTRY_GRID = (120.0, 150.0, 180.0, 210.0, 240.0)
EDGE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)
LOGISTIC_PRIOR_SLOPE = 1.702
LOGISTIC_RIDGE = 4.0


@dataclass(frozen=True)
class Feature:
    index: int
    decision_t: float
    distance: float
    sigma: float
    remaining: float
    z: float


@dataclass(frozen=True)
class Config:
    model: str
    schedule: str
    edge: float

    @property
    def name(self) -> str:
        return f"{self.model}_{self.schedule}_edge{self.edge:.2f}"


@dataclass(frozen=True)
class Attempt:
    slug: str
    filled: bool
    side: str
    winner: str
    decision_t: float
    execution_t: float | None
    probability: float
    quoted_break_even: float
    quoted_edge: float
    average_price: float | None
    shares: float
    cost: float
    pnl: float


def official_fee(price: float, shares: float) -> float:
    """Current crypto taker fee for one price level."""
    return round(shares * 0.07 * price * (1.0 - price), 5)


def asks_for(row: Row, side: str):
    return row.up_asks if side == "Up" else row.down_asks


def sweep_buy(levels, dollars: float, max_price: float = 1.0):
    """Sweep displayed asks and include taker fees; return None without depth."""
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


def one_second_prices(window: Window, end_index: int):
    """Causal last-observation-carried-forward one-second Coinbase series."""
    end_t = window.rows[end_index].t
    first_second = math.ceil(window.rows[0].t)
    last_second = math.floor(end_t)
    if last_second - first_second < 30:
        return ()
    result = []
    for second in range(first_second, last_second + 1):
        index = bisect.bisect_right(window.times, second, 0, end_index + 1) - 1
        if index >= 0:
            price = window.rows[index].coinbase
            if price > 0:
                result.append(price)
    return tuple(result)


def feature_at(window: Window, target: float) -> Feature | None:
    index = bisect.bisect_left(window.times, target)
    if index >= len(window.rows):
        return None
    row = window.rows[index]
    remaining = 300.0 - row.t
    opening = window.rows[0].coinbase
    if opening <= 0 or row.coinbase <= 0 or remaining <= 0:
        return None
    prices = one_second_prices(window, index)
    if len(prices) < 31:
        return None
    returns = [
        math.log(current / previous)
        for previous, current in zip(prices, prices[1:])
        if previous > 0 and current > 0
    ]
    if len(returns) < 30:
        return None
    # Quadratic variation per second.  Do not remove the in-window drift: the
    # short sample is used only to estimate diffusion, and demeaning it would
    # make strong directional windows spuriously overconfident.
    sigma = math.sqrt(sum(value * value for value in returns) / len(returns))
    if not math.isfinite(sigma) or sigma <= 1e-8:
        return None
    distance = math.log(row.coinbase / opening)
    z = max(-8.0, min(8.0, distance / (sigma * math.sqrt(remaining))))
    return Feature(index, row.t, distance, sigma, remaining, z)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def logistic(value: float) -> float:
    if value >= 0:
        e = math.exp(-value)
        return 1.0 / (1.0 + e)
    e = math.exp(value)
    return e / (1.0 + e)


def fit_logistic_slope(train: list[Window]) -> tuple[float, int]:
    """Fit one slope at t=150 only; intercept stays zero by construction."""
    observations = []
    for window in train:
        feature = feature_at(window, 150.0)
        if feature is not None:
            observations.append((feature.z, 1.0 if window.winner == "Up" else 0.0))
    slope = LOGISTIC_PRIOR_SLOPE
    for _ in range(50):
        gradient = -LOGISTIC_RIDGE * (slope - LOGISTIC_PRIOR_SLOPE)
        information = LOGISTIC_RIDGE
        for z, outcome in observations:
            probability = logistic(slope * z)
            gradient += z * (outcome - probability)
            information += z * z * probability * (1.0 - probability)
        if information <= 0:
            break
        step = gradient / information
        slope = max(0.25, min(3.5, slope + step))
        if abs(step) < 1e-9:
            break
    return slope, len(observations)


def probability_up(feature: Feature, model: str, logistic_slope: float) -> float:
    if model == "brownian":
        return normal_cdf(feature.z)
    if model == "logistic":
        return logistic(logistic_slope * feature.z)
    raise ValueError(f"unknown model: {model}")


def candidate_at(
    window: Window,
    target: float,
    config: Config,
    logistic_slope: float,
):
    feature = feature_at(window, target)
    if feature is None:
        return None
    row = window.rows[feature.index]
    up_probability = probability_up(feature, config.model, logistic_slope)
    candidates = []
    for side, probability in (("Up", up_probability), ("Down", 1.0 - up_probability)):
        asks = asks_for(row, side)
        if not asks:
            continue
        quote = sweep_buy(asks, STAKE)
        if quote is None:
            continue
        shares, cost, worst = quote
        break_even = cost / shares
        edge = probability - break_even
        if edge >= config.edge:
            candidates.append((edge, side, probability, break_even, worst, feature))
    return max(candidates, default=None, key=lambda item: item[0])


def decision_for(window: Window, config: Config, logistic_slope: float):
    targets = (150.0,) if config.schedule == "fixed150" else ENTRY_GRID
    for target in targets:
        candidate = candidate_at(window, target, config, logistic_slope)
        if candidate is not None:
            return candidate
    return None


def execute(
    window: Window,
    config: Config,
    logistic_slope: float,
    seed: int,
) -> Attempt | None:
    decision = decision_for(window, config, logistic_slope)
    if decision is None:
        return None
    edge, side, probability, break_even, quoted_worst, feature = decision
    limit = min(0.99, quoted_worst + FOK_ALLOWANCE)
    rng = random.Random(f"fair-value:{seed}:{config.name}:{window.slug}")
    target = feature.decision_t + rng.uniform(0.300, 0.800)
    execution_index = bisect.bisect_left(
        window.times, target, lo=feature.index + 1
    )
    if execution_index >= len(window.rows) or window.rows[execution_index].t >= 300:
        return Attempt(
            window.slug, False, side, window.winner, feature.decision_t,
            None, probability, break_even, edge, None, 0.0, 0.0, 0.0,
        )
    execution = window.rows[execution_index]
    fill = sweep_buy(asks_for(execution, side), STAKE, limit)
    if fill is None:
        return Attempt(
            window.slug, False, side, window.winner, feature.decision_t,
            execution.t, probability, break_even, edge, None, 0.0, 0.0, 0.0,
        )
    shares, cost, _ = fill
    pnl = (shares if side == window.winner else 0.0) - cost
    return Attempt(
        window.slug, True, side, window.winner, feature.decision_t,
        execution.t, probability, break_even, edge, STAKE / shares,
        shares, cost, pnl,
    )


def one_run(windows, config: Config, logistic_slope: float, seed: int):
    attempts = [
        attempt
        for window in windows
        if (attempt := execute(window, config, logistic_slope, seed)) is not None
    ]
    trades = [attempt for attempt in attempts if attempt.filled]
    equity = peak = max_drawdown = 0.0
    for trade in trades:
        equity += trade.pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    pnl = sum(trade.pnl for trade in trades)
    cost = sum(trade.cost for trade in trades)
    return {
        "signals": len(attempts),
        "trades": len(trades),
        "wins": sum(trade.pnl > 0 for trade in trades),
        "fok_failures": len(attempts) - len(trades),
        "pnl": pnl,
        "roi": 100.0 * pnl / cost if cost else 0.0,
        "max_drawdown": max_drawdown,
    }


def aggregate(windows, config: Config, logistic_slope: float, seeds):
    rows = [one_run(windows, config, logistic_slope, seed) for seed in seeds]
    result = {
        key: statistics.median(row[key] for row in rows)
        for key in (
            "signals", "trades", "wins", "fok_failures", "pnl", "roi",
            "max_drawdown",
        )
    }
    result["pnl_min"] = min(row["pnl"] for row in rows)
    result["pnl_max"] = max(row["pnl"] for row in rows)
    return result


def configs():
    return [
        Config(model, schedule, edge)
        for model in ("brownian", "logistic")
        for schedule in ("fixed150", "grid120-240")
        for edge in EDGE_THRESHOLDS
    ]


def format_result(result):
    return (
        f"sig={result['signals']:4.1f} fill={result['trades']:4.1f} "
        f"w={result['wins']:4.1f} fok={result['fok_failures']:3.1f} "
        f"pnl=${result['pnl']:+6.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:+6.1f}% dd=${result['max_drawdown']:.2f}"
    )


def load_windows(directory: Path, through: int):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            epoch = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if epoch > through:
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
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--through", type=int, default=FROZEN_THROUGH,
        help="last included window start (default: frozen research cutoff)",
    )
    args = parser.parse_args()
    windows = load_windows(args.directory, args.through)
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    half = len(train) // 2
    seeds = range(args.seeds)
    slope, observations = fit_logistic_slope(train)

    print(
        f"FAIR VALUE RESEARCH; frozen_through={args.through}; "
        f"source-aligned resolved={len(windows)}; split="
        f"{len(train)}/{len(validation)}/{len(holdout)}"
    )
    print(
        "features=Coinbase distance/open + seconds remaining + causal 1s RV; "
        f"logistic_slope={slope:.4f} fitted_on={observations} train windows"
    )
    print(
        "stake=$2; full depth; delay=300-800ms; FOK allowance=$0.05; "
        "current crypto fees; hold to settlement"
    )

    # Screen using training only. Requiring both chronological halves to be
    # positive prevents one local regime from dominating the selection.
    screened = []
    print("\nTRAIN SCREEN (positive in both halves, >=8 median fills)")
    for config in configs():
        first = aggregate(train[:half], config, slope, seeds)
        second = aggregate(train[half:], config, slope, seeds)
        total = aggregate(train, config, slope, seeds)
        passed = (
            total["trades"] >= 8
            and first["pnl_min"] > 0
            and second["pnl_min"] > 0
        )
        marker = "PASS" if passed else "fail"
        print(f"  {marker} {config.name:<36} {format_result(total)}")
        if passed:
            screened.append((config, total))

    print("\nVALIDATION (holdout still untouched for selection)")
    validated = []
    if not screened:
        print("  no configuration passed the training stability gate")
    for config, train_result in screened:
        result = aggregate(validation, config, slope, seeds)
        print(f"  {config.name:<41} {format_result(result)}")
        if result["trades"] >= 4 and result["pnl_min"] > 0:
            validated.append((result["pnl_min"], config, train_result, result))

    if not validated:
        print("\nNo strategy passed validation; holdout was not used to rescue one.")
        return

    # Selection is finalized using validation worst-delay PnL, then the last
    # 20% is opened once as an audit rather than fed back into parameters.
    validated.sort(key=lambda item: item[0], reverse=True)
    _, selected, train_result, validation_result = validated[0]
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    groups = (
        ("train", train, train_result),
        ("validation", validation, validation_result),
        ("holdout", holdout, aggregate(holdout, selected, slope, seeds)),
        ("all", windows, aggregate(windows, selected, slope, seeds)),
        ("last5", windows[-5:], aggregate(windows[-5:], selected, slope, seeds)),
    )
    for label, group, result in groups:
        del group
        print(f"  {label:<10} {format_result(result)}")


if __name__ == "__main__":
    main()
