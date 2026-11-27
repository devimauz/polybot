"""Pre-registered prospective evaluation of an exact Chainlink-TWAP rule.

This file is deliberately research-only.  It reads the per-window JSONL logs,
does not contact Polymarket, and cannot place an order.

The rule and data-quality gates are frozen here before the required sample
exists:

* only windows strictly after ``btc-updown-5m-1792081500``;
* compare Polymarket's actual Chainlink 60-second BTC/USD TWAP at source-time
  second 285 with the opening TWAP (legacy RTDS or authenticated PolyBolt);
* require that direction at every source-time second through 290 agrees;
* require the absolute second-285 margin to exceed 1.5 times the largest
  rolling 15-second TWAP change in the preceding 120 seconds;
* quote a full-depth $2 buy after confirmation, require at least a 10% return
  if it wins after taker fees, and cap a FOK at quoted worst price + $0.05;
* replay the six fixed 300/400/500/600/700/800 ms minimum execution delays
  separately, using the first logged snapshot no more than 250 ms later and
  reporting that observed delay explicitly;
* make at most one settlement-held trade per window and per delay scenario.

TWAP values are selected only by source timestamp.  Their local receive
timestamps determine when the signal was knowable.  Books are selected by
snapshot availability, with their WebSocket ``book_received_ts`` used to
detect missing or stale state.  Compact books and proxy/backfilled TWAP fields
are rejected.

The cohort starts at the first future file and is consecutive through the last
resolved file. Missing pre-trade data are zero-P&L skips, not silently removed;
their rate must stay at or below 5%, and any unknown post-signal execution makes
the study fail. No outcome metric or bootstrap interval is released until every
delay scenario has at least 200 resolved cohort windows, 30 fills, and three
distinct UTC days. Once that gate is met, a deterministic UTC-hour-block
bootstrap gives one-sided 95% lower bounds for P&L per cohort window and ROI.
Deployment passes only if both lower bounds are positive at all six fixed
delays; the best delay is never selected after seeing the results.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT / "price_logs" / "polymarket"

# Frozen strategy constants.  There are intentionally no CLI switches for
# these values: changing one creates a different, newly pre-registered test.
PROSPECTIVE_AFTER = 1_792_081_500
DECISION_SECOND = 285
PERSISTENCE_SECONDS = 5
ENVELOPE_LOOKBACK_SECONDS = 120
ENVELOPE_CHANGE_SECONDS = 15
ENVELOPE_MULTIPLIER = 1.5
STAKE = 2.0
MIN_WIN_RETURN = 0.10
FOK_ALLOWANCE = 0.05
EXECUTION_DELAYS_MS = (300, 400, 500, 600, 700, 800)
FEE_RATE = 0.07

# Frozen observability gates.  RTDS is expected roughly once a second and the
# logger snapshot rate is 100 ms.  These tolerances fail closed around gaps;
# they are not strategy parameters and must not be optimized on results.
# One distinct observation must fall in each source-time second bin.  A point
# exactly one second old belongs to the previous bin and cannot be reused.
MAX_TWAP_SOURCE_AGE_SECONDS = 1.0
MAX_TWAP_RECEIVE_LAG_SECONDS = 3.0
MAX_TWAP_NEGATIVE_LAG_SECONDS = 0.5
MAX_BOOK_AGE_SECONDS = 2.0
MAX_SNAPSHOT_WAIT_SECONDS = 0.250
TIMESTAMP_EPSILON = 1e-6
ORDER_PRICE_INCREMENT = 0.01
# At $0.90, a worst-case single-price fill still returns about 10.34% after
# the current crypto fee. At $0.91 it returns only about 9.15%.
MAX_PROFITABLE_ENTRY_PRICE = 0.90

MIN_COHORT_WINDOWS = 200
MIN_FILLS = 30
MIN_DISTINCT_UTC_DAYS = 3
MAX_PRETRADE_DATA_SKIP_RATE = 0.05
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_261_017

LEGACY_TWAP_IDENTITY = (
    "polymarket_rtds", "crypto_prices_twap_sixty", "btc/usd"
)
POLYBOLT_TWAP_IDENTITY = (
    "polymarket_polybolt", "price.crypto.twap", "btcusd"
)
EXACT_TWAP_IDENTITIES = {LEGACY_TWAP_IDENTITY, POLYBOLT_TWAP_IDENTITY}


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def start_from_slug(value: str) -> int | None:
    try:
        prefix, raw = value.rsplit("-", 1)
        start = int(raw)
    except (AttributeError, TypeError, ValueError):
        return None
    return start if prefix == "btc-updown-5m" else None


def fee_for(price: float, shares: float) -> float:
    """Polymarket's current crypto taker-fee model used by this repository."""
    return round(shares * FEE_RATE * price * (1.0 - price), 5)


@dataclass(frozen=True)
class Fill:
    shares: float
    cost: float
    average_price: float
    worst_price: float
    fee: float

    @property
    def winning_return(self) -> float:
        return (self.shares - self.cost) / self.cost


def clean_full_asks(raw: object) -> tuple[tuple[float, float], ...] | None:
    """Return a complete sorted ask side, or None for compact/corrupt data."""
    if not isinstance(raw, list):
        return None
    clean: list[tuple[float, float]] = []
    for level in raw:
        if not isinstance(level, dict):
            return None
        price = finite(level.get("price"))
        size = finite(level.get("size"))
        if price is None or size is None or not 0.0 < price < 1.0 or size <= 0.0:
            return None
        clean.append((price, size))
    clean.sort()
    return tuple(clean)


def sweep_buy(
    asks: tuple[tuple[float, float], ...],
    dollars: float = STAKE,
    limit: float = 1.0,
) -> Fill | None:
    """Sweep recorded depth without inventing liquidity beyond ``limit``."""
    remaining = dollars
    shares = fees = notional = 0.0
    worst = None
    for price, size in asks:
        if price > limit + TIMESTAMP_EPSILON:
            break
        used = min(remaining, price * size)
        if used <= 0.0:
            continue
        quantity = used / price
        shares += quantity
        notional += used
        fees += fee_for(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return Fill(
                shares=shares,
                cost=notional + fees,
                average_price=notional / shares,
                worst_price=float(worst),
                fee=fees,
            )
    return None


def fok_limit_for_quote(worst_price: float) -> float:
    """Conservative valid limit that preserves allowance and the 10% floor."""
    raw_limit = min(
        MAX_PROFITABLE_ENTRY_PRICE, worst_price + FOK_ALLOWANCE
    )
    return (
        math.floor((raw_limit + TIMESTAMP_EPSILON) / ORDER_PRICE_INCREMENT)
        * ORDER_PRICE_INCREMENT
    )


@dataclass(frozen=True)
class TwapPoint:
    source_ts: float
    received_ts: float
    price: float

    @property
    def available_ts(self) -> float:
        """Earliest defensible availability under source/local clock skew."""
        return max(self.source_ts, self.received_ts)


@dataclass(frozen=True)
class BookSide:
    received_ts: float | None
    asks: tuple[tuple[float, float], ...] | None
    ready: bool


@dataclass(frozen=True)
class Snapshot:
    snapshot_ts: float
    up: BookSide
    down: BookSide

    def side(self, name: str) -> BookSide:
        return self.up if name == "Up" else self.down


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    twap: tuple[TwapPoint, ...]
    twap_source_times: tuple[float, ...]
    snapshots: tuple[Snapshot, ...]
    snapshot_times: tuple[float, ...]


@dataclass(frozen=True)
class Signal:
    side: str
    opening_twap: float
    decision_twap: float
    margin: float
    envelope: float
    available_ts: float


@dataclass(frozen=True)
class Replay:
    start: int
    delay_ms: int
    usable: bool
    signaled: bool
    filled: bool
    won: bool
    pnl: float
    cost: float
    status: str
    actual_delay_ms: float | None = None


@dataclass(frozen=True)
class WindowEvaluation:
    start: int
    exact_twap: bool
    exact_reason: str
    signal_reason: str
    replays: tuple[Replay, ...]


@dataclass(frozen=True)
class ReportAnalysis:
    """One evaluation pass shared by the text and machine-readable reports."""

    evaluations: list[WindowEvaluation]
    load_reasons: Counter
    exact_count: int
    exact_rejects: Counter
    signal_reasons: Counter
    rows_by_delay: dict[int, list[Replay]]
    metric_by_delay: dict[int, dict[str, float]]
    days_by_delay: dict[int, int]
    sample_gate_reasons: tuple[str, ...]
    bootstrap_by_delay: dict[int, dict[str, float]] | None
    status: str
    exit_code: int


def parse_book_side(raw: object) -> BookSide:
    if not isinstance(raw, dict):
        return BookSide(None, None, False)
    received = finite(raw.get("book_received_ts"))
    ready = raw.get("book_ready") is True
    asks = clean_full_asks(raw.get("asks"))
    return BookSide(received, asks, ready)


def parse_actual_twap(raw: object, snapshot_ts: float) -> TwapPoint | None:
    """Accept only an actual Polymarket Chainlink 60-second feed."""
    if not isinstance(raw, dict):
        return None
    identity = (
        str(raw.get("source") or ""),
        str(raw.get("topic") or ""),
        str(raw.get("symbol") or "").lower(),
    )
    if identity not in EXACT_TWAP_IDENTITIES:
        return None
    try:
        if int(raw.get("window_seconds")) != 60:
            return None
    except (TypeError, ValueError):
        return None
    source = finite(raw.get("source_ts_epoch"))
    received = finite(raw.get("received_ts"))
    price = finite(raw.get("price"))
    if source is None or received is None or price is None or price <= 0.0:
        return None
    # A row can take a few milliseconds to assemble after snapshot_ts was
    # captured.  Anything materially "received" after the containing row is
    # inconsistent and therefore unusable.
    if received > snapshot_ts + 0.250:
        return None
    lag = received - source
    if lag > MAX_TWAP_RECEIVE_LAG_SECONDS + TIMESTAMP_EPSILON:
        return None
    if lag < -MAX_TWAP_NEGATIVE_LAG_SECONDS - TIMESTAMP_EPSILON:
        return None
    return TwapPoint(source, received, price)


def load_window(path: Path) -> tuple[Window | None, str]:
    start = start_from_slug(path.stem)
    if start is None:
        return None, "invalid_slug"
    if start <= PROSPECTIVE_AFTER:
        return None, "not_prospective"

    winner = None
    # A source timestamp should name one immutable RTDS observation.  Keep the
    # earliest locally received copy because later snapshots repeat it.
    points: dict[float, TwapPoint] = {}
    snapshots: list[Snapshot] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError:
        return None, "unreadable"
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if row.get("event") == "resolution":
                if row.get("winner") in ("Up", "Down"):
                    winner = row["winner"]
                continue
            if row.get("event") != "snapshot":
                continue
            if row.get("slug") != path.stem:
                continue
            snapshot_ts = finite(row.get("ts_epoch"))
            if snapshot_ts is None:
                continue
            point = parse_actual_twap(row.get("chainlink_twap_60s"), snapshot_ts)
            if point is not None:
                previous = points.get(point.source_ts)
                if previous is None or point.received_ts < previous.received_ts:
                    points[point.source_ts] = point
            snapshots.append(
                Snapshot(
                    snapshot_ts=snapshot_ts,
                    up=parse_book_side(row.get("Up")),
                    down=parse_book_side(row.get("Down")),
                )
            )

    if winner not in ("Up", "Down"):
        return None, "unresolved"
    ordered_points = tuple(sorted(points.values(), key=lambda item: item.source_ts))
    ordered_snapshots = tuple(sorted(snapshots, key=lambda item: item.snapshot_ts))
    return (
        Window(
            slug=path.stem,
            start=start,
            winner=winner,
            twap=ordered_points,
            twap_source_times=tuple(item.source_ts for item in ordered_points),
            snapshots=ordered_snapshots,
            snapshot_times=tuple(item.snapshot_ts for item in ordered_snapshots),
        ),
        "loaded",
    )


def asof_twap(window: Window, second: int) -> TwapPoint | None:
    """Latest source observation no later than the requested market second."""
    target = window.start + second
    index = bisect.bisect_right(
        window.twap_source_times, target + TIMESTAMP_EPSILON
    ) - 1
    if index < 0:
        return None
    point = window.twap[index]
    source_age = target - point.source_ts
    if source_age < -TIMESTAMP_EPSILON:
        return None
    if source_age >= MAX_TWAP_SOURCE_AGE_SECONDS - TIMESTAMP_EPSILON:
        return None
    return point


def frozen_signal(window: Window) -> tuple[Signal | None, str]:
    """Evaluate the frozen TWAP rule without looking at its outcome/book."""
    opening = asof_twap(window, 0)
    # The one-second grid makes the rolling 15-second maximum literal and
    # verifies continuous source-time coverage of the full 120-second history.
    needed_seconds = range(
        DECISION_SECOND - ENVELOPE_LOOKBACK_SECONDS,
        DECISION_SECOND + PERSISTENCE_SECONDS + 1,
    )
    samples = {second: asof_twap(window, second) for second in needed_seconds}
    if opening is None or any(point is None for point in samples.values()):
        return None, "missing_or_stale_exact_twap"
    # The same stale observation must never stand in for adjacent source-time
    # seconds; that could hide a short reversal or understate the envelope.
    source_times = [
        point.source_ts for point in samples.values() if point is not None
    ]
    if len(set(source_times)) != len(source_times):
        return None, "missing_or_stale_exact_twap"

    opening_price = opening.price
    decision_price = samples[DECISION_SECOND].price  # type: ignore[union-attr]
    signed_margin = decision_price - opening_price
    if abs(signed_margin) <= 1e-12:
        return None, "zero_direction"
    direction = 1 if signed_margin > 0.0 else -1
    side = "Up" if direction > 0 else "Down"

    for second in range(DECISION_SECOND, DECISION_SECOND + PERSISTENCE_SECONDS + 1):
        price = samples[second].price  # type: ignore[union-attr]
        if direction * (price - opening_price) <= 0.0:
            return None, "direction_not_persistent"

    changes = []
    first_change_second = (
        DECISION_SECOND - ENVELOPE_LOOKBACK_SECONDS + ENVELOPE_CHANGE_SECONDS
    )
    for second in range(first_change_second, DECISION_SECOND + 1):
        current = samples[second].price  # type: ignore[union-attr]
        prior = samples[second - ENVELOPE_CHANGE_SECONDS].price  # type: ignore[union-attr]
        changes.append(abs(current - prior))
    envelope = max(changes)
    margin = abs(signed_margin)
    if not margin > ENVELOPE_MULTIPLIER * envelope:
        return None, "safety_envelope_failed"

    used = [opening, *samples.values()]
    available = max(point.available_ts for point in used if point is not None)
    if available >= window.start + 300.0:
        return None, "twap_arrived_after_window"
    return (
        Signal(
            side=side,
            opening_twap=opening_price,
            decision_twap=decision_price,
            margin=margin,
            envelope=envelope,
            available_ts=available,
        ),
        "signal",
    )


def book_at_or_after(
    window: Window, side: str, target_ts: float
) -> tuple[Snapshot | None, BookSide | None, str]:
    """First observed book after a minimum target time.

    The selected snapshot's own causal availability is the simulated execution
    time.  It is never mislabeled as the exact target and must arrive within
    the frozen 250 ms sampling tolerance.
    """
    index = bisect.bisect_left(
        window.snapshot_times, target_ts - TIMESTAMP_EPSILON
    )
    if index >= len(window.snapshots):
        return None, None, "missing_snapshot"
    snapshot = window.snapshots[index]
    book = snapshot.side(side)
    if not book.ready or book.received_ts is None or book.asks is None:
        return snapshot, None, "missing_full_book_or_receive_ts"
    available = max(snapshot.snapshot_ts, book.received_ts)
    if available - target_ts > MAX_SNAPSHOT_WAIT_SECONDS + TIMESTAMP_EPSILON:
        return snapshot, None, "snapshot_too_late"
    age = available - book.received_ts
    if age > MAX_BOOK_AGE_SECONDS + TIMESTAMP_EPSILON:
        return snapshot, None, "stale_book"
    if available >= window.start + 300.0:
        return snapshot, None, "book_after_window"
    return snapshot, book, "book"


def evaluate_window(window: Window) -> WindowEvaluation:
    signal, reason = frozen_signal(window)
    if reason == "missing_or_stale_exact_twap":
        replays = tuple(
            Replay(window.start, delay, True, False, False, False, 0.0, 0.0,
                   reason)
            for delay in EXECUTION_DELAYS_MS
        )
        return WindowEvaluation(window.start, False, reason, reason, replays)
    # A complete exact TWAP record is a valid prospective observation even
    # when the frozen strategy intentionally makes no trade.
    if signal is None:
        replays = tuple(
            Replay(window.start, delay, True, False, False, False, 0.0, 0.0, reason)
            for delay in EXECUTION_DELAYS_MS
        )
        return WindowEvaluation(window.start, True, "exact", reason, replays)

    quote_snapshot, quote_book, quote_reason = book_at_or_after(
        window, signal.side, signal.available_ts
    )
    if quote_book is None or quote_snapshot is None:
        replays = tuple(
            Replay(window.start, delay, True, True, False, False, 0.0, 0.0,
                   f"quote_{quote_reason}")
            for delay in EXECUTION_DELAYS_MS
        )
        return WindowEvaluation(window.start, True, "exact", quote_reason, replays)

    quote = sweep_buy(
        quote_book.asks or (), limit=MAX_PROFITABLE_ENTRY_PRICE
    )
    if quote is None:
        replays = tuple(
            Replay(window.start, delay, True, False, False, False, 0.0, 0.0,
                   "quote_insufficient_profit_safe_depth")
            for delay in EXECUTION_DELAYS_MS
        )
        return WindowEvaluation(
            window.start, True, "exact",
            "quote_insufficient_profit_safe_depth", replays
        )
    if quote.winning_return + 1e-12 < MIN_WIN_RETURN:
        replays = tuple(
            Replay(window.start, delay, True, False, False, False, 0.0, 0.0,
                   "winning_return_below_10pct")
            for delay in EXECUTION_DELAYS_MS
        )
        return WindowEvaluation(
            window.start, True, "exact", "winning_return_below_10pct", replays
        )

    quote_available = max(quote_snapshot.snapshot_ts, quote_book.received_ts or 0.0)
    # Round down, never up, on a conservative one-cent grid.  BTC5m's finer
    # ticks also accept this price while the $0.05 allowance is never exceeded.
    order_limit = fok_limit_for_quote(quote.worst_price)
    replays: list[Replay] = []
    for delay in EXECUTION_DELAYS_MS:
        target = quote_available + delay / 1000.0
        execution_snapshot, execution_book, execution_reason = book_at_or_after(
            window, signal.side, target
        )
        if execution_book is None or execution_snapshot is None:
            replays.append(
                Replay(window.start, delay, False, True, False, False, 0.0, 0.0,
                       f"execution_{execution_reason}")
            )
            continue
        execution_available = max(
            execution_snapshot.snapshot_ts,
            execution_book.received_ts or 0.0,
        )
        observed_delay_ms = round(
            (execution_available - quote_available) * 1000.0, 3
        )
        fill = sweep_buy(execution_book.asks or (), limit=order_limit)
        if fill is None:
            replays.append(
                Replay(window.start, delay, True, True, False, False, 0.0, 0.0,
                       "fok_killed", actual_delay_ms=observed_delay_ms)
            )
            continue
        if fill.winning_return + 1e-12 < MIN_WIN_RETURN:
            # This should be unreachable because the FOK limit itself is
            # profit-safe. Keep the invariant explicit against future fee or
            # accounting changes.
            replays.append(
                Replay(window.start, delay, False, True, False, False, 0.0, 0.0,
                       "execution_profit_floor_failed",
                       actual_delay_ms=observed_delay_ms)
            )
            continue
        won = signal.side == window.winner
        pnl = (fill.shares if won else 0.0) - fill.cost
        replays.append(
            Replay(
                window.start, delay, True, True, True, won, pnl, fill.cost,
                "filled",
                actual_delay_ms=observed_delay_ms,
            )
        )
    return WindowEvaluation(window.start, True, "exact", "signal", tuple(replays))


def metrics(rows: list[Replay]) -> dict[str, float]:
    usable = [row for row in rows if row.usable]
    fills = [row for row in usable if row.filled]
    equity = peak = drawdown = 0.0
    for row in usable:
        equity += row.pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    pnl = sum(row.pnl for row in fills)
    cost = sum(row.cost for row in fills)
    return {
        "windows": float(len(rows)),
        "usable": float(len(usable)),
        "postsignal_unknown": float(len(rows) - len(usable)),
        "pretrade_data_skips": float(sum(is_pretrade_data_skip(row) for row in rows)),
        "signals": float(sum(row.signaled for row in rows)),
        "fills": float(len(fills)),
        "wins": float(sum(row.won for row in fills)),
        "fok_kills": float(sum(row.status == "fok_killed" for row in usable)),
        "pnl": pnl,
        "cost": cost,
        "roi": 100.0 * pnl / cost if cost else 0.0,
        "max_drawdown": drawdown,
    }


def is_pretrade_data_skip(row: Replay) -> bool:
    if row.status in {
        "missing_file", "unreadable", "invalid_slug",
        "missing_or_stale_exact_twap", "twap_arrived_after_window",
    }:
        return True
    return row.status in {
        "quote_missing_snapshot",
        "quote_missing_full_book_or_receive_ts",
        "quote_snapshot_too_late",
        "quote_stale_book",
        "quote_book_after_window",
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def hourly_block_bootstrap(rows: list[Replay], delay_ms: int) -> dict[str, float]:
    """Deterministic UTC-hour cluster bootstrap for one fixed delay."""
    usable = [row for row in rows if row.usable]
    blocks: dict[int, list[Replay]] = {}
    for row in usable:
        blocks.setdefault(row.start // 3600, []).append(row)
    packed = list(blocks.values())
    if not packed:
        raise ValueError("bootstrap requires usable rows")
    rng = random.Random(f"{BOOTSTRAP_SEED}:{delay_ms}")
    pnl_per_window: list[float] = []
    rois: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sample = [packed[rng.randrange(len(packed))] for _ in packed]
        selected = [row for block in sample for row in block]
        pnl = sum(row.pnl for row in selected if row.filled)
        cost = sum(row.cost for row in selected if row.filled)
        pnl_per_window.append(pnl / len(selected))
        if cost > 0.0:
            rois.append(100.0 * pnl / cost)
    observed_pnl = sum(row.pnl for row in usable if row.filled)
    observed_cost = sum(row.cost for row in usable if row.filled)
    return {
        "hours": float(len(packed)),
        "pnl_per_window": observed_pnl / len(usable),
        "pnl_per_window_lb95": percentile(pnl_per_window, 0.05),
        "roi": 100.0 * observed_pnl / observed_cost if observed_cost else 0.0,
        "roi_lb95": percentile(rois, 0.05) if rois else 0.0,
    }


def prospective_paths(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        start = start_from_slug(path.stem)
        if start is not None and start > PROSPECTIVE_AFTER:
            yield path


def evaluate_directory(directory: Path) -> tuple[list[WindowEvaluation], Counter]:
    paths = {
        start_from_slug(path.stem): path
        for path in prospective_paths(directory)
    }
    paths = {start: path for start, path in paths.items() if start is not None}
    loaded: dict[int, Window] = {}
    reasons_by_start: dict[int, str] = {}
    load_reasons: Counter = Counter()
    for start, path in sorted(paths.items()):
        window, reason = load_window(path)
        load_reasons[reason] += 1
        reasons_by_start[start] = reason
        if window is not None:
            loaded[start] = window
    # The frozen cohort begins with the first future file and ends at the last
    # resolved file. Every intervening five-minute slot is represented, so a
    # logging outage cannot selectively disappear difficult markets.
    if not paths or not loaded:
        return [], load_reasons
    cohort_start = min(paths)
    cohort_end = max(loaded)
    evaluations: list[WindowEvaluation] = []
    for start in range(cohort_start, cohort_end + 1, 300):
        window = loaded.get(start)
        if window is not None:
            evaluations.append(evaluate_window(window))
            continue
        reason = reasons_by_start.get(start, "missing_file")
        outcome_unknown = reason == "unresolved"
        status = "unresolved_outcome" if outcome_unknown else reason
        replays = tuple(
            Replay(
                start, delay, not outcome_unknown, False, False, False,
                0.0, 0.0, status,
            )
            for delay in EXECUTION_DELAYS_MS
        )
        evaluations.append(
            WindowEvaluation(start, False, reason, status, replays)
        )
        if reason == "missing_file":
            load_reasons[reason] += 1
    return evaluations, load_reasons


def _sample_gate_reasons(
    evaluations: list[WindowEvaluation],
    metric_by_delay: dict[int, dict[str, float]],
    days_by_delay: dict[int, int],
) -> tuple[str, ...]:
    """Return stable machine-readable reasons for a closed sample gate."""
    reasons: list[str] = []
    if not evaluations:
        reasons.append("no_evaluable_cohort")
    for delay in EXECUTION_DELAYS_MS:
        result = metric_by_delay[delay]
        windows = int(result["windows"])
        if windows < MIN_COHORT_WINDOWS:
            reasons.append(f"delay_{delay}_insufficient_cohort_windows")
        if int(result["fills"]) < MIN_FILLS:
            reasons.append(f"delay_{delay}_insufficient_fills")
        if days_by_delay[delay] < MIN_DISTINCT_UTC_DAYS:
            reasons.append(f"delay_{delay}_insufficient_utc_days")
        if int(result["postsignal_unknown"]) > 0:
            reasons.append(f"delay_{delay}_postsignal_unknowns_present")
        skip_rate = result["pretrade_data_skips"] / windows if windows else 0.0
        if skip_rate > MAX_PRETRADE_DATA_SKIP_RATE:
            reasons.append(f"delay_{delay}_pretrade_data_skip_rate_exceeded")
    return tuple(reasons)


def analyze_report(directory: Path) -> ReportAnalysis:
    """Evaluate once and classify the frozen sample/deployment gates."""
    evaluations, load_reasons = evaluate_directory(directory)
    exact = [item for item in evaluations if item.exact_twap]
    exact_rejects = Counter(
        item.exact_reason for item in evaluations if not item.exact_twap
    )
    signal_reasons = Counter(item.signal_reason for item in exact)
    rows_by_delay = {
        delay: [
            replay
            for item in evaluations
            for replay in item.replays
            if replay.delay_ms == delay
        ]
        for delay in EXECUTION_DELAYS_MS
    }
    metric_by_delay: dict[int, dict[str, float]] = {}
    for delay in EXECUTION_DELAYS_MS:
        metric_by_delay[delay] = metrics(rows_by_delay[delay])
    days_by_delay = {
        delay: len({row.start // 86_400 for row in rows_by_delay[delay]})
        for delay in EXECUTION_DELAYS_MS
    }
    sample_gate_reasons = _sample_gate_reasons(
        evaluations, metric_by_delay, days_by_delay
    )
    if sample_gate_reasons:
        return ReportAnalysis(
            evaluations=evaluations,
            load_reasons=load_reasons,
            exact_count=len(exact),
            exact_rejects=exact_rejects,
            signal_reasons=signal_reasons,
            rows_by_delay=rows_by_delay,
            metric_by_delay=metric_by_delay,
            days_by_delay=days_by_delay,
            sample_gate_reasons=sample_gate_reasons,
            bootstrap_by_delay=None,
            status="insufficient",
            exit_code=3,
        )

    bootstrap_by_delay = {
        delay: hourly_block_bootstrap(rows_by_delay[delay], delay)
        for delay in EXECUTION_DELAYS_MS
    }
    robust = all(
        result["pnl_per_window_lb95"] > 0.0 and result["roi_lb95"] > 0.0
        for result in bootstrap_by_delay.values()
    )
    return ReportAnalysis(
        evaluations=evaluations,
        load_reasons=load_reasons,
        exact_count=len(exact),
        exact_rejects=exact_rejects,
        signal_reasons=signal_reasons,
        rows_by_delay=rows_by_delay,
        metric_by_delay=metric_by_delay,
        days_by_delay=days_by_delay,
        sample_gate_reasons=(),
        bootstrap_by_delay=bootstrap_by_delay,
        status="passed" if robust else "rejected",
        exit_code=0 if robust else 2,
    )


def _gate_current(analysis: ReportAnalysis) -> dict[str, float | int]:
    min_windows = min(
        (int(analysis.metric_by_delay[delay]["windows"])
         for delay in EXECUTION_DELAYS_MS),
        default=0,
    )
    min_fills = min(
        (int(analysis.metric_by_delay[delay]["fills"])
         for delay in EXECUTION_DELAYS_MS),
        default=0,
    )
    min_days = min(analysis.days_by_delay.values(), default=0)
    max_skip_rate = max((
        analysis.metric_by_delay[delay]["pretrade_data_skips"]
        / analysis.metric_by_delay[delay]["windows"]
        if analysis.metric_by_delay[delay]["windows"] else 0.0
        for delay in EXECUTION_DELAYS_MS
    ), default=0.0)
    max_unknown = max((
        int(analysis.metric_by_delay[delay]["postsignal_unknown"])
        for delay in EXECUTION_DELAYS_MS
    ), default=0)
    return {
        "minimum_windows": min_windows,
        "minimum_fills": min_fills,
        "minimum_utc_days": min_days,
        "maximum_pretrade_data_skip_rate": max_skip_rate,
        "maximum_postsignal_unknowns": max_unknown,
    }


def _acquisition_by_delay(analysis: ReportAnalysis) -> dict[str, dict]:
    acquisition: dict[str, dict] = {}
    for delay in EXECUTION_DELAYS_MS:
        result = analysis.metric_by_delay[delay]
        windows = int(result["windows"])
        observed_delays = [
            row.actual_delay_ms for row in analysis.rows_by_delay[delay]
            if row.actual_delay_ms is not None
        ]
        acquisition[str(delay)] = {
            "minimum_delay_ms": delay,
            "cohort_windows": windows,
            "utc_days": analysis.days_by_delay[delay],
            "pretrade_data_skips": int(result["pretrade_data_skips"]),
            "pretrade_data_skip_rate": (
                result["pretrade_data_skips"] / windows if windows else 0.0
            ),
            "postsignal_unknowns": int(result["postsignal_unknown"]),
            "signals": int(result["signals"]),
            "fills": int(result["fills"]),
            "fok_kills": int(result["fok_kills"]),
            "observed_delay_ms": (
                {
                    "minimum": min(observed_delays),
                    "maximum": max(observed_delays),
                }
                if observed_delays else None
            ),
        }
    return acquisition


def json_report(directory: Path, analysis: ReportAnalysis | None = None) -> dict:
    """Build the stable JSON report, withholding outcomes until sample-ready."""
    if analysis is None:
        analysis = analyze_report(directory)
    future_files = sum(
        count for reason, count in analysis.load_reasons.items()
        if reason != "missing_file"
    )
    cohort_starts = [item.start for item in analysis.evaluations]
    deployment_reasons: list[str] = []
    if analysis.status == "rejected":
        assert analysis.bootstrap_by_delay is not None
        for delay in EXECUTION_DELAYS_MS:
            result = analysis.bootstrap_by_delay[delay]
            if result["pnl_per_window_lb95"] <= 0.0:
                deployment_reasons.append(
                    f"delay_{delay}_nonpositive_pnl_per_window_lb95"
                )
            if result["roi_lb95"] <= 0.0:
                deployment_reasons.append(f"delay_{delay}_nonpositive_roi_lb95")

    payload: dict = {
        "schema_version": 1,
        "status": analysis.status,
        "exit_code": analysis.exit_code,
        "directory": str(directory.resolve()),
        "cutoff": {
            "operator": "strictly_after",
            "slug_start": PROSPECTIVE_AFTER,
            "first_eligible_slug_start": PROSPECTIVE_AFTER + 300,
        },
        "strategy": {
            "decision_second": DECISION_SECOND,
            "persistence_seconds": PERSISTENCE_SECONDS,
            "stake_usd": STAKE,
            "fok_allowance_usd": FOK_ALLOWANCE,
            "fok_allowance_rule": "up_to_allowance_capped_at_max_entry",
            "maximum_entry_price": MAX_PROFITABLE_ENTRY_PRICE,
            "minimum_win_return": MIN_WIN_RETURN,
            "execution_minimum_delays_ms": list(EXECUTION_DELAYS_MS),
        },
        "cohort": {
            "future_files": future_files,
            "windows": len(analysis.evaluations),
            "start": min(cohort_starts) if cohort_starts else None,
            "end": max(cohort_starts) if cohort_starts else None,
            "exact_twap_resolved": analysis.exact_count,
        },
        "acquisition": {
            "load_status": dict(sorted(analysis.load_reasons.items())),
            "exact_rejects": dict(sorted(analysis.exact_rejects.items())),
            "frozen_rule_status": dict(sorted(analysis.signal_reasons.items())),
            "by_delay_ms": _acquisition_by_delay(analysis),
        },
        "gate": {
            "sample": {
                "passed": not analysis.sample_gate_reasons,
                "outcomes_withheld": bool(analysis.sample_gate_reasons),
                "requirements": {
                    "minimum_cohort_windows_per_delay": MIN_COHORT_WINDOWS,
                    "minimum_fills_per_delay": MIN_FILLS,
                    "minimum_distinct_utc_days_per_delay": MIN_DISTINCT_UTC_DAYS,
                    "maximum_pretrade_data_skip_rate": MAX_PRETRADE_DATA_SKIP_RATE,
                    "maximum_postsignal_unknowns": 0,
                },
                "current": _gate_current(analysis),
                "reasons": list(analysis.sample_gate_reasons),
            },
            "deployment": {
                "evaluated": analysis.bootstrap_by_delay is not None,
                "passed": (
                    analysis.status == "passed"
                    if analysis.bootstrap_by_delay is not None else None
                ),
                "reasons": deployment_reasons,
            },
        },
    }
    # This key is deliberately absent, rather than populated with nulls or
    # zeroes, while the frozen sample/data gate is closed.
    if analysis.bootstrap_by_delay is not None:
        delay_metrics: dict[str, dict] = {}
        for delay in EXECUTION_DELAYS_MS:
            result = analysis.metric_by_delay[delay]
            bootstrap = analysis.bootstrap_by_delay[delay]
            fills = int(result["fills"])
            delay_metrics[str(delay)] = {
                "minimum_delay_ms": delay,
                "fills": fills,
                "wins": int(result["wins"]),
                "win_rate_pct": 100.0 * result["wins"] / fills if fills else 0.0,
                "pnl_usd": result["pnl"],
                "cost_usd": result["cost"],
                "roi_pct": result["roi"],
                "maximum_drawdown_usd": result["max_drawdown"],
                "bootstrap": {
                    "utc_hour_blocks": int(bootstrap["hours"]),
                    "pnl_per_window_usd": bootstrap["pnl_per_window"],
                    "pnl_per_window_lb95_usd": bootstrap["pnl_per_window_lb95"],
                    "roi_pct": bootstrap["roi"],
                    "roi_lb95_pct": bootstrap["roi_lb95"],
                },
            }
        payload["delay_metrics"] = delay_metrics
    return payload


def print_report(directory: Path) -> int:
    analysis = analyze_report(directory)

    print("PRE-REGISTERED EXACT CHAINLINK-TWAP PROSPECTIVE EVALUATION")
    print(f"directory={directory.resolve()}")
    print(
        f"strict_start>{PROSPECTIVE_AFTER}; decision={DECISION_SECOND}s; "
        f"persistence={PERSISTENCE_SECONDS}s; stake=${STAKE:.2f}; "
        f"FOK allowance=up to ${FOK_ALLOWANCE:.2f}, "
        f"capped at ${MAX_PROFITABLE_ENTRY_PRICE:.2f}"
    )
    print(
        "delays_ms=" + ",".join(map(str, EXECUTION_DELAYS_MS))
        + f"; inference_gate={MIN_COHORT_WINDOWS} consecutive cohort windows and "
        f"{MIN_FILLS} fills over {MIN_DISTINCT_UTC_DAYS} UTC days for every delay"
    )
    print(
        f"future_files={sum(count for reason, count in analysis.load_reasons.items() if reason != 'missing_file')}; "
        f"cohort_windows={len(analysis.evaluations)}; "
        f"exact_twap_resolved={analysis.exact_count}"
    )
    if analysis.load_reasons:
        print("load_status=" + json.dumps(
            dict(sorted(analysis.load_reasons.items())), sort_keys=True
        ))
    if analysis.exact_rejects:
        print("exact_rejects=" + json.dumps(
            dict(sorted(analysis.exact_rejects.items()))
        ))
    if analysis.signal_reasons:
        print("frozen_rule_status=" + json.dumps(
            dict(sorted(analysis.signal_reasons.items()))
        ))

    print("\nACQUISITION STATUS (outcomes withheld until every gate passes)")
    for delay in EXECUTION_DELAYS_MS:
        result = analysis.metric_by_delay[delay]
        fill_count = int(result["fills"])
        cohort_count = int(result["windows"])
        data_skip_rate = (
            result["pretrade_data_skips"] / cohort_count if cohort_count else 0.0
        )
        observed_delays = [
            row.actual_delay_ms for row in analysis.rows_by_delay[delay]
            if row.actual_delay_ms is not None
        ]
        observed = (
            f" observed={min(observed_delays):.0f}-{max(observed_delays):.0f}ms"
            if observed_delays else ""
        )
        print(
            f"  min {delay:>3}ms cohort={cohort_count:3d} "
            f"pretrade_data_skips={int(result['pretrade_data_skips']):3d} "
            f"({100.0 * data_skip_rate:4.1f}%) "
            f"postsignal_unknown={int(result['postsignal_unknown']):3d} "
            f"signals={int(result['signals']):3d} fills={fill_count:3d} "
            f"FOK_kills={int(result['fok_kills']):3d}{observed}"
        )

    if analysis.sample_gate_reasons:
        current = _gate_current(analysis)
        print(
            "\nINSUFFICIENT/INVALID DATA: no profitability conclusion. "
            f"Need >= {MIN_COHORT_WINDOWS} consecutive future cohort windows and "
            f">= {MIN_FILLS} fills across >= {MIN_DISTINCT_UTC_DAYS} UTC days "
            f"for every delay, <= {100 * MAX_PRETRADE_DATA_SKIP_RATE:.1f}% "
            f"pre-trade data skips, and zero post-signal unknowns; currently "
            f"minimum windows={current['minimum_windows']}, "
            f"minimum fills={current['minimum_fills']}, "
            f"minimum days={current['minimum_utc_days']}, maximum data-skip rate="
            f"{100 * current['maximum_pretrade_data_skip_rate']:.1f}%, "
            f"maximum unknowns={current['maximum_postsignal_unknowns']}."
        )
        print("Hourly-block bootstrap withheld by the pre-registered sample gate.")
        return analysis.exit_code

    print("\nOUTCOME METRICS (sample/data gates passed)")
    for delay in EXECUTION_DELAYS_MS:
        result = analysis.metric_by_delay[delay]
        fill_count = int(result["fills"])
        win_rate = 100.0 * result["wins"] / fill_count if fill_count else 0.0
        print(
            f"  min {delay:>3}ms fills={fill_count:3d} "
            f"wins={int(result['wins']):3d} win_rate={win_rate:5.1f}% "
            f"pnl=${result['pnl']:+.2f} roi={result['roi']:+.2f}% "
            f"max_dd=${result['max_drawdown']:.2f}"
        )

    print("\nUTC-HOUR-BLOCK BOOTSTRAP (one-sided 95% lower bounds)")
    assert analysis.bootstrap_by_delay is not None
    for delay in EXECUTION_DELAYS_MS:
        result = analysis.bootstrap_by_delay[delay]
        print(
            f"  {delay:>3}ms hours={int(result['hours']):3d} "
            f"pnl/window=${result['pnl_per_window']:+.4f} "
            f"LB95=${result['pnl_per_window_lb95']:+.4f} "
            f"roi={result['roi']:+.2f}% LB95={result['roi_lb95']:+.2f}%"
        )
    if analysis.status == "passed":
        print(
            "\nDEPLOYMENT GATE: PASS. Both one-sided lower bounds are positive "
            "at every fixed delay."
        )
        return analysis.exit_code
    print(
        "\nDEPLOYMENT GATE: FAIL. At least one fixed delay has a non-positive "
        "P&L/window or ROI lower bound; do not deploy this rule."
    )
    return analysis.exit_code


def _synthetic_row(start: int, second: int, price: float, *, missing_book_ts=False):
    snapshot_ts = start + second + 0.10
    received_ts = start + second + 0.05
    side = {
        "book_ready": True,
        "book_received_ts": None if missing_book_ts else start + second + 0.05,
        "asks": [{"price": 0.50, "size": 20.0}],
    }
    return {
        "event": "snapshot",
        "slug": f"btc-updown-5m-{start}",
        "ts_epoch": snapshot_ts,
        "Up": side,
        "Down": dict(side),
        "chainlink_twap_60s": {
            "source": LEGACY_TWAP_IDENTITY[0],
            "topic": LEGACY_TWAP_IDENTITY[1],
            "symbol": LEGACY_TWAP_IDENTITY[2],
            "window_seconds": 60,
            "price": price,
            "source_ts_epoch": start + second,
            "received_ts": received_ts,
        },
    }


def run_self_tests() -> None:
    """Small dependency-free regression tests for causality and execution."""
    assert start_from_slug("btc-updown-5m-1792081800") == 1_792_081_800
    assert start_from_slug("eth-updown-5m-1792081800") is None
    cheap = sweep_buy(((0.50, 10.0),))
    assert cheap is not None and cheap.winning_return > MIN_WIN_RETURN
    expensive = sweep_buy(((0.91, 10.0),))
    assert expensive is not None and expensive.winning_return < MIN_WIN_RETURN
    safe_edge = sweep_buy(((MAX_PROFITABLE_ENTRY_PRICE, 10.0),))
    assert safe_edge is not None and safe_edge.winning_return >= MIN_WIN_RETURN
    assert fok_limit_for_quote(0.89) == MAX_PROFITABLE_ENTRY_PRICE
    assert fok_limit_for_quote(0.50) <= 0.50 + FOK_ALLOWANCE
    assert sweep_buy(
        ((0.94, 10.0),), limit=fok_limit_for_quote(0.89)
    ) is None
    assert clean_full_asks(None) is None
    assert clean_full_asks([]) == ()

    polybolt = {
        "source": POLYBOLT_TWAP_IDENTITY[0],
        "topic": POLYBOLT_TWAP_IDENTITY[1],
        "symbol": POLYBOLT_TWAP_IDENTITY[2],
        "window_seconds": 60,
        "price": 70_123.5,
        "source_ts_epoch": PROSPECTIVE_AFTER + 300,
        "received_ts": PROSPECTIVE_AFTER + 300.05,
    }
    parsed_polybolt = parse_actual_twap(polybolt, PROSPECTIVE_AFTER + 300.10)
    assert parsed_polybolt is not None and parsed_polybolt.price == 70_123.5
    impossible_early = dict(polybolt)
    impossible_early["source_ts_epoch"] = PROSPECTIVE_AFTER + 302.0
    impossible_early["received_ts"] = PROSPECTIVE_AFTER + 300.0
    assert parse_actual_twap(impossible_early, PROSPECTIVE_AFTER + 300.1) is None

    start = PROSPECTIVE_AFTER + 300
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / f"btc-updown-5m-{start}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            # Slow one-direction move: second-285 margin is much larger than
            # every rolling 15-second change, and it remains positive to 290.
            for second in range(0, 293):
                price = 100.0 + max(0, second - 164) * 0.10
                stream.write(json.dumps(_synthetic_row(start, second, price)) + "\n")
            # The real logger samples at 100 ms. Add causal book-only rows so
            # each minimum-delay scenario has an observed execution no more
            # than 250 ms later without inventing an exact target-time book.
            for tenth in range(2, 13):
                second = 290.0 + tenth / 10.0
                row = _synthetic_row(start, second, 112.0)
                row.pop("chainlink_twap_60s")
                stream.write(json.dumps(row) + "\n")
            stream.write(json.dumps({
                "event": "resolution",
                "slug": path.stem,
                "winner": "Up",
            }) + "\n")
        window, reason = load_window(path)
        assert reason == "loaded" and window is not None
        assert asof_twap(window, 285) is not None
        signal, signal_reason = frozen_signal(window)
        assert signal_reason == "signal" and signal is not None and signal.side == "Up"
        evaluation = evaluate_window(window)
        assert evaluation.exact_twap
        assert len(evaluation.replays) == len(EXECUTION_DELAYS_MS)
        assert all(row.usable and row.filled and row.won for row in evaluation.replays)
        assert all(
            row.actual_delay_ms is not None
            and row.actual_delay_ms + 0.01 >= row.delay_ms
            and row.actual_delay_ms <= row.delay_ms + 250.0 + TIMESTAMP_EPSILON
            for row in evaluation.replays
        )

        # No source look-ahead: deleting the opening point cannot be repaired
        # by the source-second-1 observation.
        without_open = Window(
            window.slug,
            window.start,
            window.winner,
            window.twap[1:],
            window.twap_source_times[1:],
            window.snapshots,
            window.snapshot_times,
        )
        assert asof_twap(without_open, 0) is None

        without_one_second = tuple(
            point for point in window.twap
            if abs(point.source_ts - (start + 200)) > TIMESTAMP_EPSILON
        )
        gapped = Window(
            window.slug,
            window.start,
            window.winner,
            without_one_second,
            tuple(point.source_ts for point in without_one_second),
            window.snapshots,
            window.snapshot_times,
        )
        assert frozen_signal(gapped)[1] == "missing_or_stale_exact_twap"

        bad = _synthetic_row(start, 291, 110.0, missing_book_ts=True)
        parsed = parse_book_side(bad["Up"])
        assert parsed.received_ts is None
        temporary = Window(
            window.slug,
            window.start,
            window.winner,
            window.twap,
            window.twap_source_times,
            (Snapshot(bad["ts_epoch"], parsed, parsed),),
            (bad["ts_epoch"],),
        )
        _, selected, why = book_at_or_after(temporary, "Up", start + 290.5)
        assert selected is None and why == "missing_full_book_or_receive_ts"

        ready_side = BookSide(
            start + 290.75, ((0.50, 20.0),), True
        )
        too_late = Window(
            window.slug,
            window.start,
            window.winner,
            window.twap,
            window.twap_source_times,
            (Snapshot(start + 290.80, ready_side, ready_side),),
            (start + 290.80,),
        )
        _, selected, why = book_at_or_after(too_late, "Up", start + 290.5)
        assert selected is None and why == "snapshot_too_late"

        # A missing file inside the observed prospective span stays in the
        # consecutive cohort as a zero-P&L pre-trade data skip.
        unresolved_start = start + 600
        unresolved_path = Path(temp) / f"btc-updown-5m-{unresolved_start}.jsonl"
        unresolved_path.write_text("", encoding="utf-8")
        later_start = start + 900
        later_path = Path(temp) / f"btc-updown-5m-{later_start}.jsonl"
        later_path.write_text(json.dumps({
            "event": "resolution", "slug": later_path.stem, "winner": "Down",
        }) + "\n", encoding="utf-8")
        cohort, reasons = evaluate_directory(Path(temp))
        assert [item.start for item in cohort] == [
            start, start + 300, unresolved_start, later_start,
        ]
        missing = cohort[1].replays
        assert missing and all(row.status == "missing_file" for row in missing)
        unresolved = cohort[2].replays
        assert unresolved and all(
            not row.usable and row.status == "unresolved_outcome"
            for row in unresolved
        )
        assert reasons["missing_file"] == 1
        assert reasons["unresolved"] == 1
        unresolved_metrics = metrics(list(unresolved))
        assert unresolved_metrics["postsignal_unknown"] == len(EXECUTION_DELAYS_MS)
        assert unresolved_metrics["pretrade_data_skips"] == 0

        # Machine-readable reports must expose acquisition failures without
        # leaking any outcome metrics before the frozen sample gate opens.
        insufficient_analysis = analyze_report(Path(temp))
        insufficient_payload = json_report(Path(temp), insufficient_analysis)
        assert insufficient_payload["status"] == "insufficient"
        assert insufficient_payload["exit_code"] == 3
        assert insufficient_payload["gate"]["sample"]["outcomes_withheld"]
        assert insufficient_payload["gate"]["sample"]["reasons"]
        assert "delay_metrics" not in insufficient_payload
        json.dumps(insufficient_payload, allow_nan=False)

        # Focused serializer fixtures cover both post-gate statuses without
        # weakening the real 200-window gate or running a bootstrap in tests.
        ready_metrics = {
            delay: {
                "windows": 200.0,
                "usable": 200.0,
                "postsignal_unknown": 0.0,
                "pretrade_data_skips": 0.0,
                "signals": 40.0,
                "fills": 40.0,
                "wins": 20.0,
                "fok_kills": 0.0,
                "pnl": -1.0,
                "cost": 80.0,
                "roi": -1.25,
                "max_drawdown": 2.0,
            }
            for delay in EXECUTION_DELAYS_MS
        }
        rejected_bootstrap = {
            delay: {
                "hours": 3.0,
                "pnl_per_window": -0.005,
                "pnl_per_window_lb95": -0.01,
                "roi": -1.25,
                "roi_lb95": -2.0,
            }
            for delay in EXECUTION_DELAYS_MS
        }
        rejected_analysis = ReportAnalysis(
            evaluations=insufficient_analysis.evaluations,
            load_reasons=insufficient_analysis.load_reasons,
            exact_count=insufficient_analysis.exact_count,
            exact_rejects=insufficient_analysis.exact_rejects,
            signal_reasons=insufficient_analysis.signal_reasons,
            rows_by_delay=insufficient_analysis.rows_by_delay,
            metric_by_delay=ready_metrics,
            days_by_delay={delay: 3 for delay in EXECUTION_DELAYS_MS},
            sample_gate_reasons=(),
            bootstrap_by_delay=rejected_bootstrap,
            status="rejected",
            exit_code=2,
        )
        rejected_payload = json_report(Path(temp), rejected_analysis)
        assert rejected_payload["status"] == "rejected"
        assert rejected_payload["exit_code"] == 2
        assert "delay_metrics" in rejected_payload
        assert rejected_payload["gate"]["deployment"]["reasons"]
        json.dumps(rejected_payload, allow_nan=False)

        passed_bootstrap = {
            delay: {
                "hours": 3.0,
                "pnl_per_window": 0.01,
                "pnl_per_window_lb95": 0.005,
                "roi": 2.0,
                "roi_lb95": 0.5,
            }
            for delay in EXECUTION_DELAYS_MS
        }
        passed_analysis = replace(
            rejected_analysis,
            bootstrap_by_delay=passed_bootstrap,
            status="passed",
            exit_code=0,
        )
        passed_payload = json_report(Path(temp), passed_analysis)
        assert passed_payload["status"] == "passed"
        assert passed_payload["exit_code"] == 0
        assert passed_payload["gate"]["deployment"]["passed"] is True
        assert not passed_payload["gate"]["deployment"]["reasons"]
        assert "delay_metrics" in passed_payload
        json.dumps(passed_payload, allow_nan=False)

    print(
        "self-test: PASS (exact TWAP, no-lookahead, full-depth/FOK, "
        "missing-data gates, JSON outcome withholding)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path, default=DEFAULT_LOG_DIR,
        help="directory containing one btc-updown-5m-START.jsonl file per window",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-test", action="store_true",
        help="run dependency-free synthetic regression tests and exit",
    )
    mode.add_argument(
        "--json", action="store_true",
        help="emit one machine-readable JSON object and no human-readable text",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    if args.json:
        analysis = analyze_report(args.directory)
        print(json.dumps(
            json_report(args.directory, analysis),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
        return analysis.exit_code
    return print_report(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
