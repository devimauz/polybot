"""Research a Coinbase proxy for Polymarket's 60-second BTC TWAP.

The BTC five-minute contract is resolved from a 60-second Chainlink BTC/USD
TWAP, while these historical files contain Coinbase spot observations.  This
script therefore stitches the Coinbase observations across adjacent files and
constructs a *causal*, piecewise-constant 60-second time-weighted mean from
Coinbase exchange timestamps.  The trading signal compares that rolling mean
with the proxy TWAP at the exact market-open timestamp; it never compares with
the first in-window spot sample.

Only a small, declared grid of fixed decision times, minimum TWAP margins and
maximum token asks is screened.  Selection uses the two chronological halves
of the first 60% and then the next 20%.  The final 20% is opened exactly once
only if a configuration clears both gates.  Orders use recorded full depth, a
$2 stake, current crypto taker fees, a $0.05 FOK allowance, and common-random
300--800 ms latency over 100 seeds.  This is research-only and cannot submit
orders or modify the live bot.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics

from _exit_strategy_oos import Row, Window, levels
from btc5m_momentum_strategy import fee_for


ROOT = Path(__file__).resolve().parent
FROZEN_THROUGH = 1792066800
STAKE = 2.0
FOK_ALLOWANCE = 0.05
TWAP_SECONDS = 60.0
MAX_SOURCE_STALENESS = 2.0

# Deliberately compact and fixed before validation is inspected.
ENTRY_SECONDS = (150.0, 210.0, 240.0, 270.0, 285.0)
MARGIN_BPS = (1.0, 2.0, 4.0)
ASK_CAPS = (0.70, 0.82, 0.90)
ASK_MIN = 0.45


@dataclass(frozen=True)
class Config:
    entry_second: float
    margin_bps: float
    ask_cap: float

    @property
    def name(self) -> str:
        return (
            f"t{self.entry_second:g}_margin{self.margin_bps:g}bps_"
            f"ask{self.ask_cap:.2f}"
        )


@dataclass(frozen=True)
class Signal:
    index: int
    decision_t: float
    side: str
    move_bps: float
    best_ask: float
    quoted_worst: float


@dataclass(frozen=True)
class Attempt:
    slug: str
    filled: bool
    side: str
    winner: str
    decision_t: float
    execution_t: float | None
    move_bps: float
    best_ask: float
    average_price: float | None
    shares: float
    cost: float
    pnl: float


class PiecewisePrice:
    """Piecewise-constant source-price series with exact interval integrals."""

    def __init__(self, observations: list[tuple[float, float]]):
        # Cached Coinbase observations repeat.  One value per exchange
        # timestamp prevents snapshots from accidentally reweighting a trade.
        by_time: dict[float, float] = {}
        for timestamp, price in observations:
            if math.isfinite(timestamp) and math.isfinite(price) and price > 0:
                by_time[timestamp] = price
        packed = sorted(by_time.items())
        self.times = tuple(item[0] for item in packed)
        self.prices = tuple(item[1] for item in packed)
        prefix = [0.0] * len(packed)
        for index in range(1, len(packed)):
            prefix[index] = (
                prefix[index - 1]
                + self.prices[index - 1]
                * (self.times[index] - self.times[index - 1])
            )
        self.prefix = tuple(prefix)

    def _integral_to(self, timestamp: float) -> float | None:
        index = bisect.bisect_right(self.times, timestamp) - 1
        if index < 0:
            return None
        return (
            self.prefix[index]
            + self.prices[index] * (timestamp - self.times[index])
        )

    def twap(
        self,
        timestamp: float,
        seconds: float = TWAP_SECONDS,
        max_staleness: float = MAX_SOURCE_STALENESS,
    ) -> float | None:
        """Return a backward-looking TWAP using no sample after ``timestamp``."""
        if seconds <= 0 or not self.times:
            return None
        left = timestamp - seconds
        left_index = bisect.bisect_right(self.times, left) - 1
        right_index = bisect.bisect_right(self.times, timestamp) - 1
        if left_index < 0 or right_index < 0:
            return None
        # BTC trades frequently; a long tail gap here indicates logger/feed
        # loss, not a trustworthy flat market.  Fail closed in that case.
        if timestamp - self.times[right_index] > max_staleness:
            return None
        start = self._integral_to(left)
        end = self._integral_to(timestamp)
        if start is None or end is None:
            return None
        return (end - start) / seconds


def source_epoch(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and math.isfinite(numeric):
        # CLOB book timestamps are milliseconds; tolerate second timestamps.
        return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def parse_file(path: Path) -> tuple[Window | None, list[tuple[float, float]]]:
    rows: list[Row] = []
    coinbase: list[tuple[float, float]] = []
    winner = None
    window_start = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in (
                "Up", "Down"
            ):
                winner = raw["winner"]
                continue
            if raw.get("event") != "snapshot":
                continue
            cb = raw.get("coinbase") or {}
            exchange_time = source_epoch(cb.get("exchange_ts"))
            try:
                price = float(cb["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if exchange_time is None or not math.isfinite(price) or price <= 0:
                continue
            coinbase.append((exchange_time, price))
            try:
                window_start = float(raw["window_start"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            component_times = (
                source_epoch(up.get("book_timestamp")),
                source_epoch(down.get("book_timestamp")),
                exchange_time,
            )
            if any(value is None for value in component_times):
                continue
            causal_time = max(component_times) - window_start
            rows.append(
                Row(
                    causal_time,
                    price,
                    levels(up, "bids"),
                    levels(up, "asks"),
                    levels(down, "bids"),
                    levels(down, "asks"),
                )
            )
    rows.sort(key=lambda row: row.t)
    if (
        winner not in ("Up", "Down")
        or window_start is None
        or not rows
        or rows[0].t > 5.0
    ):
        return None, coinbase
    packed = tuple(rows)
    return (
        Window(path.stem, winner, packed, tuple(row.t for row in packed)),
        coinbase,
    )


def window_epoch(window: Window) -> int:
    return int(window.slug.rsplit("-", 1)[-1])


def sweep_buy(
    asks: tuple[tuple[float, float], ...],
    dollars: float,
    max_price: float = 1.0,
):
    remaining = dollars
    shares = fees = 0.0
    worst = None
    for price, size in asks:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        quantity = used / price
        shares += quantity
        fees += fee_for(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, worst
    return None


def asks_for(row: Row, side: str):
    return row.up_asks if side == "Up" else row.down_asks


def signal_for(
    window: Window,
    series: PiecewisePrice,
    opening_twap: float,
    config: Config,
) -> Signal | None:
    index = bisect.bisect_left(window.times, config.entry_second)
    if index >= len(window.rows):
        return None
    row = window.rows[index]
    absolute_time = window_epoch(window) + row.t
    current_twap = series.twap(absolute_time)
    if current_twap is None or opening_twap <= 0:
        return None
    move_bps = (current_twap / opening_twap - 1.0) * 10_000.0
    if abs(move_bps) < config.margin_bps:
        return None
    side = "Up" if move_bps > 0 else "Down"
    asks = asks_for(row, side)
    if not asks or not ASK_MIN <= asks[0][0] <= config.ask_cap:
        return None
    quoted = sweep_buy(asks, STAKE)
    if quoted is None:
        return None
    return Signal(index, row.t, side, move_bps, asks[0][0], quoted[2])


def execute(
    window: Window,
    series: PiecewisePrice,
    opening_twap: float,
    config: Config,
    seed: int,
) -> Attempt | None:
    signal = signal_for(window, series, opening_twap, config)
    if signal is None:
        return None
    # The seed deliberately excludes config: all candidates get the same
    # latency quantile for a given seed/window (common random numbers).
    latency = random.Random(f"twap-proxy:{seed}:{window.slug}").uniform(
        0.300, 0.800
    )
    target = signal.decision_t + latency
    execution_index = bisect.bisect_left(
        window.times, target, lo=signal.index + 1
    )
    if execution_index >= len(window.rows):
        return Attempt(
            window.slug, False, signal.side, window.winner,
            signal.decision_t, None, signal.move_bps, signal.best_ask,
            None, 0.0, 0.0, 0.0,
        )
    execution = window.rows[execution_index]
    if execution.t >= 300.0:
        return Attempt(
            window.slug, False, signal.side, window.winner,
            signal.decision_t, execution.t, signal.move_bps, signal.best_ask,
            None, 0.0, 0.0, 0.0,
        )
    order_limit = min(0.99, signal.quoted_worst + FOK_ALLOWANCE)
    fill = sweep_buy(asks_for(execution, signal.side), STAKE, order_limit)
    if fill is None:
        return Attempt(
            window.slug, False, signal.side, window.winner,
            signal.decision_t, execution.t, signal.move_bps, signal.best_ask,
            None, 0.0, 0.0, 0.0,
        )
    shares, cost, _ = fill
    pnl = (shares if signal.side == window.winner else 0.0) - cost
    return Attempt(
        window.slug, True, signal.side, window.winner,
        signal.decision_t, execution.t, signal.move_bps, signal.best_ask,
        STAKE / shares, shares, cost, pnl,
    )


def one_run(
    windows: list[Window],
    series: PiecewisePrice,
    opening_twaps: dict[str, float],
    config: Config,
    seed: int,
):
    attempts = []
    for window in windows:
        opening = opening_twaps.get(window.slug)
        if opening is None:
            continue
        attempt = execute(window, series, opening, config, seed)
        if attempt is not None:
            attempts.append(attempt)
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


def aggregate(
    windows: list[Window],
    series: PiecewisePrice,
    opening_twaps: dict[str, float],
    config: Config,
    seeds,
):
    runs = [
        one_run(windows, series, opening_twaps, config, seed)
        for seed in seeds
    ]
    result = {
        key: statistics.median(run[key] for run in runs)
        for key in (
            "signals", "trades", "wins", "fok_failures", "pnl", "roi",
            "max_drawdown",
        )
    }
    result["pnl_min"] = min(run["pnl"] for run in runs)
    result["pnl_max"] = max(run["pnl"] for run in runs)
    return result


def configs():
    return [
        Config(entry, margin, cap)
        for entry in ENTRY_SECONDS
        for margin in MARGIN_BPS
        for cap in ASK_CAPS
    ]


def format_result(result) -> str:
    return (
        f"sig={result['signals']:4.1f} fill={result['trades']:4.1f} "
        f"w={result['wins']:4.1f} fok={result['fok_failures']:3.1f} "
        f"pnl=${result['pnl']:+6.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:+6.1f}% dd=${result['max_drawdown']:.2f}"
    )


def direction_match(
    windows: list[Window],
    series: PiecewisePrice,
    opening_twaps: dict[str, float],
):
    comparable = matches = ties = 0
    for window in windows:
        opening = opening_twaps.get(window.slug)
        if opening is None:
            continue
        closing = series.twap(window_epoch(window) + 300.0)
        if closing is None:
            continue
        comparable += 1
        if math.isclose(closing, opening, rel_tol=0.0, abs_tol=1e-12):
            ties += 1
            proxy = "Down"
        else:
            proxy = "Up" if closing > opening else "Down"
        matches += int(proxy == window.winner)
    return comparable, matches, ties


def show_match(
    label: str,
    windows: list[Window],
    series: PiecewisePrice,
    opening_twaps: dict[str, float],
):
    total, matched, ties = direction_match(windows, series, opening_twaps)
    rate = 100.0 * matched / total if total else 0.0
    print(
        f"  {label:<12} {matched}/{total} = {rate:.1f}% "
        f"(exact proxy ties={ties})"
    )


def load_dataset(directory: Path, through: int):
    windows: list[Window] = []
    observations: list[tuple[float, float]] = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            epoch = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if epoch > through:
            continue
        window, events = parse_file(path)
        # Do not let a malformed file contribute source samples beyond the
        # frozen dataset's final market boundary.
        observations.extend(
            (timestamp, price)
            for timestamp, price in events
            if timestamp <= through + 300.0
        )
        if window is not None:
            windows.append(window)
    series = PiecewisePrice(observations)
    opening_twaps = {
        window.slug: opening
        for window in windows
        if (opening := series.twap(float(window_epoch(window)))) is not None
    }
    # A usable window must have a causal 60-second baseline.  Decision-time
    # TWAP availability is checked separately for each fixed entry time.
    usable = [window for window in windows if window.slug in opening_twaps]
    return usable, series, opening_twaps, len(observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--through", type=int, default=FROZEN_THROUGH,
        help="last included market start (default: frozen research cutoff)",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    windows, series, opening_twaps, raw_observations = load_dataset(
        args.directory, args.through
    )
    if not windows:
        raise SystemExit("no resolved source-aligned windows with TWAP history")
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    half = len(train) // 2
    seeds = range(args.seeds)

    print(
        f"CHAINLINK-TWAP PROXY RESEARCH; frozen_through={args.through}; "
        f"source-aligned resolved={len(windows)}; split="
        f"{len(train)}/{len(validation)}/{len(holdout)}"
    )
    print(
        f"Coinbase source observations={raw_observations}; "
        f"unique exchange timestamps={len(series.times)}; "
        "60s causal time-weighted LOCF; max endpoint staleness=2s"
    )
    print(
        "stake=$2; full depth; delay=300-800ms common-random; "
        "FOK allowance=$0.05; current crypto fees; hold to settlement"
    )

    print("\nPROXY FINAL-DIRECTION MATCH (development data only)")
    show_match("train half 1", train[:half], series, opening_twaps)
    show_match("train half 2", train[half:], series, opening_twaps)
    show_match("validation", validation, series, opening_twaps)
    print("  holdout      sealed unless a trading configuration validates")

    screened = []
    print("\nTRAIN SCREEN (positive worst seed in both halves; >=8 fills)")
    for config in configs():
        first = aggregate(
            train[:half], series, opening_twaps, config, seeds
        )
        second = aggregate(
            train[half:], series, opening_twaps, config, seeds
        )
        total = aggregate(train, series, opening_twaps, config, seeds)
        passed = (
            total["trades"] >= 8
            and first["pnl_min"] > 0
            and second["pnl_min"] > 0
        )
        marker = "PASS" if passed else "fail"
        print(f"  {marker} {config.name:<34} {format_result(total)}")
        if passed:
            screened.append((config, total))

    print("\nVALIDATION (holdout remains sealed during selection)")
    validated = []
    if not screened:
        print("  no configuration passed the two-half training gate")
    for config, train_result in screened:
        result = aggregate(
            validation, series, opening_twaps, config, seeds
        )
        print(f"  {config.name:<39} {format_result(result)}")
        if result["trades"] >= 4 and result["pnl_min"] > 0:
            validated.append((result["pnl_min"], config, train_result, result))

    if not validated:
        print("\nNo strategy passed validation; final holdout was not opened.")
        return

    # The rule is frozen here.  Validation worst-latency PnL is the only
    # ranking criterion; the holdout cannot affect parameters or selection.
    validated.sort(key=lambda item: item[0], reverse=True)
    _, selected, train_result, validation_result = validated[0]
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    groups = (
        ("train", train, train_result),
        ("validation", validation, validation_result),
        (
            "holdout", holdout,
            aggregate(holdout, series, opening_twaps, selected, seeds),
        ),
        (
            "all", windows,
            aggregate(windows, series, opening_twaps, selected, seeds),
        ),
        (
            "last5", windows[-5:],
            aggregate(windows[-5:], series, opening_twaps, selected, seeds),
        ),
    )
    for label, _group, result in groups:
        print(f"  {label:<10} {format_result(result)}")

    print("\nPROXY FINAL-DIRECTION MATCH (frozen audit)")
    show_match("train", train, series, opening_twaps)
    show_match("validation", validation, series, opening_twaps)
    show_match("holdout", holdout, series, opening_twaps)
    show_match("all", windows, series, opening_twaps)


if __name__ == "__main__":
    main()
