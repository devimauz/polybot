"""Causal walk-forward research for BTC 5-minute strategy selection.

This file is deliberately research-only: it never submits an order and does
not import or modify the live bot.  The primary rule is declared before the
final holdout is evaluated:

* At the first source-aligned frame at/after t=150, score a small fixed set of
  experts by their counterfactual PnL over the last 24 *already resolved*
  windows.
* Select the expert with the largest positive 24-window PnL; otherwise stay in
  cash.  The first 24 available resolutions are a warm-up.
* A prior window becomes usable only at the timestamp of its recorded
  ``resolution`` event.  This preserves the real resolution lag.

All entries use recorded full ask depth, a fixed $2 stake, the current crypto
taker-fee formula, a $0.05 adverse FOK allowance, and a deterministic-per-seed
uniform 300--800 ms order delay.  Logger row labels are not used for market
timing: a frame becomes available at max(Up book timestamp, Down book
timestamp, Coinbase exchange timestamp).

The expert family and thresholds are intentionally small and fixed to avoid a
large search over this short data set.  Results are shown on chronological
60/20/20 train/validation/final-holdout slices, while adaptive state is carried
forward continuously and is never reset at a split boundary.
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


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "price_logs" / "polymarket"

# Frozen when this experiment was created.  Pass --through explicitly to test
# a later data freeze without silently changing the original experiment.
FROZEN_THROUGH = 1792066200
STAKE = 2.0
ENTRY_SECOND = 150.0
FOK_ALLOWANCE = 0.05
MIN_DELAY = 0.300
MAX_DELAY = 0.800
LOOKBACK = 24


@dataclass(frozen=True)
class Frame:
    t: float
    coinbase: float
    up_asks: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Features:
    move_bps: float
    recent_30_bps: float
    oriented_recent_30_bps: float
    efficiency: float


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    resolution_epoch: float
    opening: float
    frames: tuple[Frame, ...]
    times: tuple[float, ...]
    features: Features


@dataclass(frozen=True)
class Expert:
    name: str
    direction: str
    minimum_move_bps: float
    ask_min: float
    ask_max: float
    minimum_efficiency: float | None = None
    maximum_efficiency: float | None = None
    minimum_oriented_recent_bps: float | None = None
    maximum_oriented_recent_bps: float | None = None


@dataclass(frozen=True)
class Trade:
    window_index: int
    expert: str
    side: str
    pnl: float
    cost: float
    won: bool
    decision_t: float
    execution_t: float
    best_ask: float
    average_price: float


# Declaration order is the deterministic tie-break order.  ``value82`` comes
# first because it takes a strict subset of the old strategy's high-price risk.
EXPERTS = (
    Expert("value82", "momentum", 2.0, 0.45, 0.82),
    Expert("old_momentum90", "momentum", 2.0, 0.45, 0.90),
    Expert(
        "efficient_momentum90",
        "momentum",
        2.0,
        0.45,
        0.90,
        minimum_efficiency=0.20,
        minimum_oriented_recent_bps=0.0,
    ),
    Expert(
        "failed_trend_contrarian",
        "contrarian",
        5.0,
        0.20,
        0.65,
        maximum_efficiency=0.20,
        maximum_oriented_recent_bps=-2.0,
    ),
)


def epoch(value) -> float | None:
    try:
        # Book timestamps are integer milliseconds.  Guard ISO strings from
        # the numeric conversion rather than assuming every number is ms.
        numeric = float(value)
        return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def clean_asks(book: dict) -> tuple[tuple[float, float], ...]:
    result = []
    for level in book.get("asks") or []:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            result.append((price, size))
    if not result:
        try:
            price = float(book["ask"])
            size = float(book["ask_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            result.append((price, size))
    return tuple(sorted(result))


def source_time(raw: dict, start: int) -> float | None:
    values = (
        epoch((raw.get("Up") or {}).get("book_timestamp")),
        epoch((raw.get("Down") or {}).get("book_timestamp")),
        epoch((raw.get("coinbase") or {}).get("exchange_ts")),
    )
    usable = [value for value in values if value is not None]
    return max(usable) - start if usable else None


def compute_features(points: list[tuple[float, float]], decision: Frame) -> tuple[float, Features]:
    history = sorted((t, price) for t, price in points if t <= decision.t)
    opening = history[0][1]
    move = (decision.coinbase / opening - 1.0) * 10_000.0
    old_i = bisect.bisect_right([row[0] for row in history], decision.t - 30.0) - 1
    old_price = history[max(0, old_i)][1]
    recent = (decision.coinbase / old_price - 1.0) * 10_000.0
    direction = 1.0 if move >= 0 else -1.0

    path = 0.0
    prior = opening
    for _, price in history[1:]:
        if price != prior:
            path += abs(math.log(price / prior)) * 10_000.0
            prior = price
    displacement = abs(math.log(decision.coinbase / opening)) * 10_000.0
    efficiency = min(1.0, displacement / path) if path > 0 else 0.0
    return opening, Features(move, recent, direction * recent, efficiency)


def load_window(path: Path) -> tuple[Window | None, int, int]:
    try:
        start = int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return None, 0, 0

    winner = None
    resolution_epoch = None
    points: list[tuple[float, float]] = []
    frames: list[Frame] = []
    decision_seen = None
    snapshots = full_depth = 0

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in ("Up", "Down"):
                winner = raw["winner"]
                resolution_epoch = epoch(raw.get("ts"))
                continue
            if raw.get("event") != "snapshot":
                continue
            snapshots += 1
            up = raw.get("Up") or {}
            down = raw.get("Down") or {}
            full_depth += int(bool(up.get("asks") and down.get("asks")))
            try:
                coinbase = float((raw.get("coinbase") or {})["price"])
            except (KeyError, TypeError, ValueError):
                continue
            t = source_time(raw, start)
            if t is None or not math.isfinite(t) or not math.isfinite(coinbase):
                continue

            # Price-only history is enough for the causal regime features.
            # Once the execution horizon has been captured, later book arrays
            # would add substantial memory without affecting an entry.
            if decision_seen is None or t <= decision_seen + MAX_DELAY + 1.0:
                points.append((t, coinbase))
            if t < ENTRY_SECOND - 2.0:
                continue
            if decision_seen is None and t >= ENTRY_SECOND:
                decision_seen = t
            if decision_seen is None or t <= decision_seen + MAX_DELAY + 1.0:
                frames.append(Frame(t, coinbase, clean_asks(up), clean_asks(down)))

    frames.sort(key=lambda row: row.t)
    points.sort(key=lambda row: row[0])
    if (
        winner not in ("Up", "Down")
        or resolution_epoch is None
        or not points
        or points[0][0] > 5.0
        or not frames
    ):
        return None, snapshots, full_depth
    decision_i = bisect.bisect_left([row.t for row in frames], ENTRY_SECOND)
    if decision_i >= len(frames):
        return None, snapshots, full_depth
    decision = frames[decision_i]
    opening, features = compute_features(points, decision)
    packed = tuple(frames)
    return (
        Window(
            path.stem,
            start,
            winner,
            resolution_epoch,
            opening,
            packed,
            tuple(row.t for row in packed),
            features,
        ),
        snapshots,
        full_depth,
    )


def fee(price: float, shares: float) -> float:
    return shares * 0.07 * price * (1.0 - price)


def sweep_buy(
    asks: tuple[tuple[float, float], ...], dollars: float, max_price: float = 1.0
) -> tuple[float, float, float, float] | None:
    remaining = dollars
    shares = fees = 0.0
    worst = None
    for price, size in asks:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        quantity = used / price
        shares += quantity
        fees += fee(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, float(worst), dollars / shares
    return None


def expert_side(expert: Expert, features: Features) -> str | None:
    if abs(features.move_bps) < expert.minimum_move_bps:
        return None
    if expert.minimum_efficiency is not None and features.efficiency < expert.minimum_efficiency:
        return None
    if expert.maximum_efficiency is not None and features.efficiency > expert.maximum_efficiency:
        return None
    if (
        expert.minimum_oriented_recent_bps is not None
        and features.oriented_recent_30_bps < expert.minimum_oriented_recent_bps
    ):
        return None
    if (
        expert.maximum_oriented_recent_bps is not None
        and features.oriented_recent_30_bps > expert.maximum_oriented_recent_bps
    ):
        return None
    momentum_side = "Up" if features.move_bps > 0 else "Down"
    if expert.direction == "momentum":
        return momentum_side
    return "Down" if momentum_side == "Up" else "Up"


def trade_for(window: Window, window_index: int, expert: Expert, seed: int) -> Trade | None:
    decision_i = bisect.bisect_left(window.times, ENTRY_SECOND)
    if decision_i >= len(window.frames):
        return None
    decision = window.frames[decision_i]
    side = expert_side(expert, window.features)
    if side is None:
        return None
    asks = decision.up_asks if side == "Up" else decision.down_asks
    if not asks or not expert.ask_min <= asks[0][0] <= expert.ask_max:
        return None
    quoted = sweep_buy(asks, STAKE)
    if quoted is None:
        return None
    order_limit = min(0.99, quoted[2] + FOK_ALLOWANCE)

    # Common random numbers: experts acting in the same window experience the
    # same latency draw, avoiding noise-driven expert ranking.
    rng = random.Random(f"adaptive-experts:{seed}:{window.slug}:entry")
    target = decision.t + rng.uniform(MIN_DELAY, MAX_DELAY)
    execution_i = bisect.bisect_left(window.times, target, lo=decision_i + 1)
    if execution_i >= len(window.frames):
        return None
    execution = window.frames[execution_i]
    execution_asks = execution.up_asks if side == "Up" else execution.down_asks
    filled = sweep_buy(execution_asks, STAKE, order_limit)
    if filled is None:
        return None
    shares, cost, _, average_price = filled
    won = side == window.winner
    pnl = (shares if won else 0.0) - cost
    return Trade(
        window_index,
        expert.name,
        side,
        pnl,
        cost,
        won,
        decision.t,
        execution.t,
        asks[0][0],
        average_price,
    )


def simulate_seed(windows: list[Window], seed: int):
    expert_trades: dict[str, list[Trade | None]] = {
        expert.name: [trade_for(window, i, expert, seed) for i, window in enumerate(windows)]
        for expert in EXPERTS
    }
    adaptive: list[Trade | None] = []
    selections: list[str] = []

    for i, window in enumerate(windows):
        decision_i = bisect.bisect_left(window.times, ENTRY_SECOND)
        decision_epoch = window.start + window.frames[decision_i].t
        known = [
            j
            for j in range(i)
            if windows[j].resolution_epoch <= decision_epoch
        ]
        history = known[-LOOKBACK:]
        if len(history) < LOOKBACK:
            selections.append("cash_warmup")
            adaptive.append(None)
            continue
        scores = {
            expert.name: sum(
                expert_trades[expert.name][j].pnl
                if expert_trades[expert.name][j] is not None
                else 0.0
                for j in history
            )
            for expert in EXPERTS
        }
        selected = max(EXPERTS, key=lambda expert: scores[expert.name])
        if scores[selected.name] <= 0.0:
            selections.append("cash_nonpositive")
            adaptive.append(None)
            continue
        selections.append(selected.name)
        adaptive.append(expert_trades[selected.name][i])
    return expert_trades, adaptive, selections


def metrics(trades: list[Trade | None], start: int, end: int) -> dict[str, float]:
    actual = [trade for trade in trades[start:end] if trade is not None]
    pnl = sum(trade.pnl for trade in actual)
    cost = sum(trade.cost for trade in actual)
    equity = peak = drawdown = 0.0
    for trade in actual:
        equity += trade.pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": float(len(actual)),
        "wins": float(sum(trade.won for trade in actual)),
        "pnl": pnl,
        "roi": pnl / cost * 100.0 if cost else 0.0,
        "max_dd": drawdown,
    }


def aggregate(seed_rows: list[dict[str, float]]) -> dict[str, float]:
    result = {
        key: statistics.median(row[key] for row in seed_rows)
        for key in ("trades", "wins", "pnl", "roi", "max_dd")
    }
    result["pnl_min"] = min(row["pnl"] for row in seed_rows)
    result["pnl_max"] = max(row["pnl"] for row in seed_rows)
    return result


def show(label: str, result: dict[str, float]) -> None:
    print(
        f"{label:<25} n={result['trades']:5.1f} W={result['wins']:5.1f} "
        f"pnl=${result['pnl']:+7.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:7.2f}% dd=${result['max_dd']:5.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=LOG_DIR)
    parser.add_argument("--through", type=int, default=FROZEN_THROUGH)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    windows = []
    snapshots = full_depth = 0
    paths = []
    for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if start <= args.through:
            paths.append(path)
    for position, path in enumerate(paths, 1):
        window, count, full_count = load_window(path)
        snapshots += count
        full_depth += full_count
        if window is not None:
            windows.append(window)
        if position % 50 == 0:
            print(f"loaded {position}/{len(paths)} files", flush=True)

    windows.sort(key=lambda window: window.start)
    cut60 = int(len(windows) * 0.60)
    cut80 = int(len(windows) * 0.80)
    splits = (
        ("train60", 0, cut60),
        ("validation20", cut60, cut80),
        ("FINAL holdout20", cut80, len(windows)),
        ("all", 0, len(windows)),
    )
    print(
        f"\nFROZEN through={args.through}; usable resolved={len(windows)}; "
        f"split={cut60}/{cut80-cut60}/{len(windows)-cut80}"
    )
    print(
        f"snapshots={snapshots}; both-side full ask depth={full_depth} "
        f"({full_depth / max(1, snapshots) * 100:.2f}%)"
    )
    print(
        "source timestamps; t=150; $2; full-depth FOK; allowance=$0.05; "
        "delay=300-800ms; current crypto fees; hold to settlement"
    )
    print(
        "PRIMARY (predeclared): last-24-resolved-window PnL leader, "
        "cash when no positive leader"
    )

    runs = [simulate_seed(windows, seed) for seed in range(args.seeds)]
    for split_name, start, end in splits:
        print(f"\n{split_name} windows [{start}:{end}]")
        for expert in EXPERTS:
            show(
                expert.name,
                aggregate(
                    [metrics(run[0][expert.name], start, end) for run in runs]
                ),
            )
        show(
            "ADAPTIVE24 primary",
            aggregate([metrics(run[1], start, end) for run in runs]),
        )

    print("\nADAPTIVE SELECTION COUNTS (median across delay seeds)")
    labels = ["cash_warmup", "cash_nonpositive"] + [expert.name for expert in EXPERTS]
    for split_name, start, end in splits[:3]:
        cells = []
        for label in labels:
            count = statistics.median(
                sum(value == label for value in run[2][start:end]) for run in runs
            )
            cells.append(f"{label}={count:g}")
        print(f"  {split_name:<16} " + "; ".join(cells))

    efficiencies = [window.features.efficiency for window in windows]
    print(
        "\nEFFICIENCY DIAGNOSTIC (not used to retune thresholds): "
        f"median={statistics.median(efficiencies):.3f}; "
        f"p25={statistics.quantiles(efficiencies, n=4)[0]:.3f}; "
        f"p75={statistics.quantiles(efficiencies, n=4)[2]:.3f}"
    )


if __name__ == "__main__":
    main()
