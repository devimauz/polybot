"""Research-only probability trading for the BTC five-minute contract.

This experiment is deliberately isolated from every live bot.  The logs do
not historically contain Polymarket's Chainlink 60-second TWAP, so Coinbase
prices received by the logger are used as a clearly labelled *proxy*.  The
contract outcome in the P&L calculation is still the recorded Polymarket
resolution.

Let ``X_s = log(price_s)`` and model future log price as driftless Brownian
motion, ``dX_s = sigma dW_s``.  The opening proxy is the already-known log
average over ``[-60, 0]`` and the final proxy is the log average over
``[240, 300]``.  Conditional on information at second ``t``, the final
average is exactly Gaussian under this model:

* for ``t <= 240``, mean ``X_t`` and variance
  ``sigma^2 * ((240 - t) + 20)``;
* for ``240 < t < 300``, mean
  ``(integral_240^t X_s ds + (300-t) X_t) / 60`` and variance
  ``sigma^2 * (300-t)^3 / 10800``.

The ``20`` in the first expression is ``60^3 / (3 * 60^2)``.  It is the
variance contribution of averaging Brownian motion, and is why treating the
last spot price as the settlement price gives the wrong probability.  The
real oracle averages ordinary prices rather than log prices; combined with
the Coinbase/Chainlink source difference, this makes the probability a model
proxy, not the true contract probability.

Only logger receive times are used for Coinbase.  A REST book snapshot is
considered available at the beginning of the following logger cycle, which
is a conservative receive-time proxy and prevents same-cycle look-ahead.
Volatility is estimated causally from 180 one-second log returns.  A compact
12-rule grid (four fixed decision times by three minimum model edges) is
declared below.  Selection uses chronological train/validation data; the
last 20 percent is opened once only if a rule validates.  Execution uses the
recorded full ask depth, a fixed $2 stake, current crypto taker fees, a $0.05
FOK allowance, and common-random 300--800 ms latency.
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
from statistics import NormalDist

from btc5m_momentum_strategy import fee_for


ROOT = Path(__file__).resolve().parent
LOG_DIRECTORY = ROOT / "price_logs" / "polymarket"
STAKE = 2.0
FOK_ALLOWANCE = 0.05
DELAY_MIN = 0.300
DELAY_MAX = 0.800
VOLATILITY_LOOKBACK = 180
MAX_PRICE_STALENESS = 2.0

# Fixed before any train/validation/holdout result is inspected.
ENTRY_SECONDS = (180.0, 225.0, 255.0, 275.0)
MINIMUM_EDGES = (0.02, 0.05, 0.08)


@dataclass(frozen=True)
class Frame:
    available: float
    t: float
    up_asks: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]

    def asks(self, side: str):
        return self.up_asks if side == "Up" else self.down_asks


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    frames: tuple[Frame, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Rule:
    entry_second: float
    minimum_edge: float

    @property
    def name(self) -> str:
        return f"t{self.entry_second:g}_edge{100*self.minimum_edge:g}pct"


@dataclass(frozen=True)
class Candidate:
    index: int
    decision_time: float
    side: str
    probability: float
    sigma_bps_sqrt_second: float
    edge: float
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
    probability: float
    edge: float
    cost: float
    pnl: float


def epoch(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and math.isfinite(number):
        return number / 1000.0 if number > 10_000_000_000 else number
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).timestamp()
    except (TypeError, ValueError):
        return None


def ask_prefix(book: dict, dollars: float = STAKE):
    """Return the exact full-depth prefix needed for a ``dollars`` sweep."""
    clean = []
    for level in book.get("asks") or ():
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(price)
            and math.isfinite(size)
            and 0.0 < price < 1.0
            and size > 0.0
        ):
            clean.append((price, size))
    result = []
    notional = 0.0
    for price, size in sorted(clean):
        result.append((price, size))
        notional += price * size
        if notional + 1e-12 >= dollars:
            break
    return tuple(result)


def sweep_buy(
    asks: tuple[tuple[float, float], ...],
    dollars: float,
    maximum_price: float = 1.0,
):
    remaining = dollars
    shares = fees = 0.0
    worst = None
    for price, size in asks:
        if price > maximum_price + 1e-12:
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


class LogPriceSeries:
    """Piecewise-constant log prices indexed only by logger receive time."""

    def __init__(self, observations: list[tuple[float, float]]):
        by_time: dict[float, float] = {}
        for timestamp, price in observations:
            if (
                math.isfinite(timestamp)
                and math.isfinite(price)
                and timestamp > 0.0
                and price > 0.0
            ):
                by_time[timestamp] = math.log(price)
        packed = sorted(by_time.items())
        self.times = tuple(item[0] for item in packed)
        self.logs = tuple(item[1] for item in packed)
        prefix = [0.0] * len(packed)
        for index in range(1, len(packed)):
            prefix[index] = (
                prefix[index - 1]
                + self.logs[index - 1]
                * (self.times[index] - self.times[index - 1])
            )
        self.prefix = tuple(prefix)

    def _index(self, timestamp: float) -> int:
        return bisect.bisect_right(self.times, timestamp) - 1

    def log_at(
        self,
        timestamp: float,
        maximum_staleness: float | None = None,
    ) -> float | None:
        index = self._index(timestamp)
        if index < 0:
            return None
        if (
            maximum_staleness is not None
            and timestamp - self.times[index] > maximum_staleness
        ):
            return None
        return self.logs[index]

    def _integral_to(self, timestamp: float) -> float | None:
        index = self._index(timestamp)
        if index < 0:
            return None
        return (
            self.prefix[index]
            + self.logs[index] * (timestamp - self.times[index])
        )

    def integral(
        self,
        left: float,
        right: float,
        maximum_endpoint_staleness: float | None = MAX_PRICE_STALENESS,
    ) -> float | None:
        if right < left:
            return None
        if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            return 0.0
        if self.log_at(left) is None:
            return None
        if self.log_at(right, maximum_endpoint_staleness) is None:
            return None
        first = self._integral_to(left)
        last = self._integral_to(right)
        if first is None or last is None:
            return None
        return last - first

    def average(self, left: float, right: float) -> float | None:
        total = self.integral(left, right)
        if total is None or right <= left:
            return None
        return total / (right - left)

    def volatility(
        self,
        timestamp: float,
        seconds: int = VOLATILITY_LOOKBACK,
    ) -> float | None:
        """Causal realized volatility per sqrt(second), sampled each second."""
        if seconds < 2:
            return None
        values = [
            self.log_at(timestamp - seconds + step)
            for step in range(seconds + 1)
        ]
        if any(value is None for value in values):
            return None
        if self.log_at(timestamp, MAX_PRICE_STALENESS) is None:
            return None
        quadratic_variation = sum(
            (right - left) ** 2
            for left, right in zip(values, values[1:])
        )
        if not math.isfinite(quadratic_variation) or quadratic_variation <= 0:
            return None
        return math.sqrt(quadratic_variation / seconds)


@dataclass(frozen=True)
class _RawFrame:
    logger_time: float
    coinbase_received: float | None
    up_asks: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


def load_file(path: Path):
    try:
        start = int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return None, []
    winner = None
    frames: list[Frame] = []
    observations: list[tuple[float, float]] = []
    previous: _RawFrame | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"event":"resolution"' in line:
                try:
                    raw = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if raw.get("winner") in ("Up", "Down"):
                    winner = raw["winner"]
                continue
            if '"event":"snapshot"' not in line:
                continue
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            logger_time = epoch(raw.get("ts_epoch")) or epoch(raw.get("ts"))
            if logger_time is None:
                continue
            coinbase = raw.get("coinbase") or {}
            received = epoch(coinbase.get("received_ts"))
            try:
                price = float(coinbase["price"])
            except (KeyError, TypeError, ValueError):
                price = math.nan
            if (
                received is not None
                and math.isfinite(price)
                and price > 0.0
            ):
                observations.append((received, price))

            current = _RawFrame(
                logger_time,
                received,
                ask_prefix(raw.get("Up") or {}),
                ask_prefix(raw.get("Down") or {}),
            )
            if previous is not None:
                # The prior JSON record and both REST responses must have
                # completed before the logger begins this cycle.  This is a
                # conservative receive-time upper bound on availability.
                available = logger_time
                if previous.coinbase_received is not None:
                    available = max(available, previous.coinbase_received)
                relative = available - start
                if -2.0 <= relative < 300.0:
                    frames.append(
                        Frame(
                            available,
                            relative,
                            previous.up_asks,
                            previous.down_asks,
                        )
                    )
            previous = current

    frames.sort(key=lambda frame: frame.available)
    if (
        winner not in ("Up", "Down")
        or not frames
        or frames[0].t > 5.0
    ):
        return None, observations
    packed = tuple(frames)
    return (
        Window(
            path.stem,
            start,
            winner,
            packed,
            tuple(frame.available for frame in packed),
        ),
        observations,
    )


def load_dataset(directory: Path, through: int | None):
    paths = []
    for path in directory.glob("btc-updown-5m-*.jsonl"):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if through is None or start <= through:
            paths.append((start, path))
    paths.sort()
    windows: list[Window] = []
    observations: list[tuple[float, float]] = []
    for _start, path in paths:
        window, received = load_file(path)
        observations.extend(received)
        if window is not None:
            windows.append(window)
    series = LogPriceSeries(observations)
    openings = {
        window.slug: opening
        for window in windows
        if (
            opening := series.average(
                window.start - 60.0,
                float(window.start),
            )
        )
        is not None
    }
    usable = [window for window in windows if window.slug in openings]
    return usable, series, openings, len(observations), len(paths)


def conditional_up_probability(
    series: LogPriceSeries,
    window_start: int,
    decision_time: float,
    opening_average: float,
    sigma: float,
) -> float | None:
    """Exact conditional probability for the log-average Brownian proxy."""
    t = decision_time - window_start
    if not 0.0 <= t < 300.0 or sigma <= 0.0:
        return None
    current = series.log_at(decision_time, MAX_PRICE_STALENESS)
    if current is None:
        return None
    if t <= 240.0:
        mean = current
        variance = sigma * sigma * ((240.0 - t) + 20.0)
    else:
        known = series.integral(window_start + 240.0, decision_time)
        if known is None:
            return None
        remaining = 300.0 - t
        mean = (known + remaining * current) / 60.0
        variance = sigma * sigma * remaining ** 3 / 10_800.0
    if variance <= 0.0 or not math.isfinite(variance):
        return None
    z_score = (mean - opening_average) / math.sqrt(variance)
    return NormalDist().cdf(z_score)


def candidate_for(
    window: Window,
    series: LogPriceSeries,
    opening_average: float,
    entry_second: float,
) -> Candidate | None:
    target = window.start + entry_second
    index = bisect.bisect_left(window.times, target)
    if index >= len(window.frames):
        return None
    frame = window.frames[index]
    if frame.t >= 295.0:
        return None
    sigma = series.volatility(frame.available)
    if sigma is None:
        return None
    up_probability = conditional_up_probability(
        series,
        window.start,
        frame.available,
        opening_average,
        sigma,
    )
    if up_probability is None:
        return None

    choices = []
    for side, probability in (
        ("Up", up_probability),
        ("Down", 1.0 - up_probability),
    ):
        asks = frame.asks(side)
        quote = sweep_buy(asks, STAKE)
        if quote is None:
            continue
        shares, cost, worst = quote
        edge = probability - cost / shares
        choices.append(
            (edge, side, probability, asks[0][0], worst)
        )
    if not choices:
        return None
    edge, side, probability, best_ask, worst = max(choices)
    return Candidate(
        index,
        frame.available,
        side,
        probability,
        sigma * 10_000.0,
        edge,
        best_ask,
        worst,
    )


def execute(
    window: Window,
    candidate: Candidate,
    seed: int,
) -> Attempt:
    delay = random.Random(
        f"twap-probability:{seed}:{window.slug}"
    ).uniform(DELAY_MIN, DELAY_MAX)
    target = candidate.decision_time + delay
    index = bisect.bisect_left(
        window.times,
        target,
        lo=candidate.index + 1,
    )
    if index >= len(window.frames):
        return Attempt(
            window.slug,
            False,
            candidate.side,
            window.winner,
            candidate.decision_time - window.start,
            None,
            candidate.probability,
            candidate.edge,
            0.0,
            0.0,
        )
    execution = window.frames[index]
    if execution.t >= 300.0:
        return Attempt(
            window.slug,
            False,
            candidate.side,
            window.winner,
            candidate.decision_time - window.start,
            execution.t,
            candidate.probability,
            candidate.edge,
            0.0,
            0.0,
        )
    order_limit = min(0.99, candidate.quoted_worst + FOK_ALLOWANCE)
    fill = sweep_buy(
        execution.asks(candidate.side),
        STAKE,
        order_limit,
    )
    if fill is None:
        return Attempt(
            window.slug,
            False,
            candidate.side,
            window.winner,
            candidate.decision_time - window.start,
            execution.t,
            candidate.probability,
            candidate.edge,
            0.0,
            0.0,
        )
    shares, cost, _worst = fill
    payout = shares if candidate.side == window.winner else 0.0
    return Attempt(
        window.slug,
        True,
        candidate.side,
        window.winner,
        candidate.decision_time - window.start,
        execution.t,
        candidate.probability,
        candidate.edge,
        cost,
        payout - cost,
    )


def one_run(
    windows: list[Window],
    candidates: list[Candidate | None],
    rule: Rule,
    seed: int,
):
    attempts = []
    for window, candidate in zip(windows, candidates):
        if candidate is None or candidate.edge + 1e-12 < rule.minimum_edge:
            continue
        attempts.append(execute(window, candidate, seed))
    trades = [attempt for attempt in attempts if attempt.filled]
    equity = peak = drawdown = 0.0
    for trade in trades:
        equity += trade.pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    pnl = sum(trade.pnl for trade in trades)
    cost = sum(trade.cost for trade in trades)
    return {
        "signals": len(attempts),
        "trades": len(trades),
        "wins": sum(trade.pnl > 0.0 for trade in trades),
        "fok": len(attempts) - len(trades),
        "pnl": pnl,
        "roi": 100.0 * pnl / cost if cost else 0.0,
        "drawdown": drawdown,
    }


def aggregate(
    windows: list[Window],
    candidates: list[Candidate | None],
    rule: Rule,
    seeds,
):
    runs = [one_run(windows, candidates, rule, seed) for seed in seeds]
    result = {
        key: statistics.median(run[key] for run in runs)
        for key in (
            "signals",
            "trades",
            "wins",
            "fok",
            "pnl",
            "roi",
            "drawdown",
        )
    }
    result["pnl_min"] = min(run["pnl"] for run in runs)
    result["pnl_max"] = max(run["pnl"] for run in runs)
    result["trades_min"] = min(run["trades"] for run in runs)
    result["trades_max"] = max(run["trades"] for run in runs)
    return result


def format_result(result) -> str:
    return (
        f"sig={result['signals']:4.1f} "
        f"fill={result['trades']:4.1f}"
        f"[{result['trades_min']:.0f}-{result['trades_max']:.0f}] "
        f"w={result['wins']:4.1f} fok={result['fok']:3.1f} "
        f"pnl=${result['pnl']:+7.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:+6.1f}% dd=${result['drawdown']:.2f}"
    )


def proxy_direction_match(
    windows: list[Window],
    series: LogPriceSeries,
    openings: dict[str, float],
):
    matches = comparable = 0
    for window in windows:
        opening = openings.get(window.slug)
        final = series.average(window.start + 240.0, window.start + 300.0)
        if opening is None or final is None:
            continue
        predicted = "Up" if final > opening else "Down"
        matches += int(predicted == window.winner)
        comparable += 1
    return matches, comparable


def rules():
    return [
        Rule(entry, edge)
        for entry in ENTRY_SECONDS
        for edge in MINIMUM_EDGES
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=LOG_DIRECTORY,
    )
    parser.add_argument(
        "--through",
        type=int,
        default=None,
        help="optional last market epoch; default reads every log",
    )
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least one")

    windows, series, openings, raw_count, file_count = load_dataset(
        args.directory,
        args.through,
    )
    if len(windows) < 30:
        raise SystemExit("not enough resolved, source-aligned proxy windows")
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    half = len(train) // 2
    seeds = range(args.seeds)

    print(
        "COINBASE LOG-TWAP PROBABILITY PROXY (RESEARCH ONLY); "
        f"files={file_count}; resolved_usable={len(windows)}; "
        f"range={windows[0].start}-{windows[-1].start}; "
        f"split={len(train)}/{len(validation)}/{len(holdout)}"
    )
    print(
        f"Coinbase receive-time observations={raw_count}; "
        f"unique={len(series.times)}; volatility={VOLATILITY_LOOKBACK}s "
        "causal one-second QV"
    )
    print(
        "model=driftless Brownian log-price; target=log averages "
        "[-60,0] vs [240,300]; this is NOT the Chainlink arithmetic TWAP"
    )
    print(
        "stake=$2; full recorded depth; current fees; delay=300-800ms; "
        "FOK allowance=$0.05; hold to recorded resolution"
    )

    print("\nCOINBASE-PROXY FINAL-DIRECTION MATCH (pre-holdout diagnostic)")
    for label, group in (
        ("train", train),
        ("validation", validation),
    ):
        matched, total = proxy_direction_match(group, series, openings)
        print(
            f"  {label:<15} {matched}/{total} "
            f"({100.0*matched/max(1,total):.1f}%)"
        )

    by_entry = {}
    for entry in ENTRY_SECONDS:
        by_entry[entry] = [
            candidate_for(window, series, openings[window.slug], entry)
            for window in windows[:cut80]
        ]

    screened = []
    print("\nTRAIN SCREEN (both chronological halves worst-seed positive)")
    for rule in rules():
        candidates = by_entry[rule.entry_second]
        first = aggregate(
            train[:half],
            candidates[:half],
            rule,
            seeds,
        )
        second = aggregate(
            train[half:],
            candidates[half:cut60],
            rule,
            seeds,
        )
        total = aggregate(
            train,
            candidates[:cut60],
            rule,
            seeds,
        )
        passed = (
            total["trades_min"] >= 8
            and first["trades_min"] >= 2
            and second["trades_min"] >= 2
            and first["pnl_min"] > 0.0
            and second["pnl_min"] > 0.0
        )
        print(
            f"  {'PASS' if passed else 'fail'} {rule.name:<18} "
            f"{format_result(total)}"
        )
        if passed:
            screened.append((rule, total))

    print("\nVALIDATION (holdout is not used for selection)")
    validated = []
    if not screened:
        print("  no rule passed the chronological training gate")
    for rule, train_result in screened:
        candidates = by_entry[rule.entry_second]
        result = aggregate(
            validation,
            candidates[cut60:cut80],
            rule,
            seeds,
        )
        print(f"  {rule.name:<23} {format_result(result)}")
        if result["trades_min"] >= 4 and result["pnl_min"] > 0.0:
            validated.append((result["pnl_min"], rule, train_result, result))

    if not validated:
        print("\nREJECTED: no probability-proxy rule passed validation.")
        print("The holdout was not inspected to rescue or tune a rule.")
        return

    validated.sort(
        key=lambda item: (item[0], item[2]["pnl_min"]),
        reverse=True,
    )
    _score, selected, train_result, validation_result = validated[0]
    # Only now, after the rule is frozen, calculate any holdout signals.
    candidates = by_entry[selected.entry_second] + [
        candidate_for(
            window,
            series,
            openings[window.slug],
            selected.entry_second,
        )
        for window in holdout
    ]
    holdout_result = aggregate(
        holdout,
        candidates[cut80:],
        selected,
        seeds,
    )
    all_result = aggregate(windows, candidates, selected, seeds)
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    for label, result in (
        ("train", train_result),
        ("validation", validation_result),
        ("holdout", holdout_result),
        ("all", all_result),
    ):
        print(f"  {label:<10} {format_result(result)}")
    matched, total = proxy_direction_match(holdout, series, openings)
    print(
        f"  holdout Coinbase-proxy direction={matched}/{total} "
        f"({100.0*matched/max(1,total):.1f}%)"
    )
    survived = (
        holdout_result["trades_min"] >= 4
        and holdout_result["pnl_min"] > 0.0
    )
    print(
        "\nSURVIVED honest holdout"
        if survived
        else "\nREJECTED: the frozen rule failed the honest holdout gate."
    )


if __name__ == "__main__":
    main()
