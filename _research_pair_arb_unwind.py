"""Research-only BTC5m complete-set arbitrage with naked-leg unwind.

This replay never submits an order and never changes the live bot.  It tests
whether requiring a displayed Up+Down discount to persist makes a two-FOK
complete-set trade executable.  If exactly one buy fills, it submits an
immediate sell FOK for that leg after both buy results are known.

Execution assumptions are deliberately explicit:

* $2 combined buy principal, split into equal Up/Down shares;
* current crypto taker fee: shares * 0.07 * price * (1-price);
* 300--800 ms independent delay for every buy or sell action;
* buy and sell FOK limits allow $0.05 adverse movement;
* full recorded depth is required for every fill;
* at most one pair attempt per five-minute window.

Both source-time and the more conservative next-logger-cycle clocks are
reported.  The input set is frozen to files already resolved when the program
starts.  Parameters printed by default are a small, predeclared sensitivity
set, not an optimizer.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


FEE_RATE = 0.07
TOTAL_PRINCIPAL = 2.0
FOK_ALLOWANCE = 0.05
DEFAULT_THRESHOLD = 0.98
DEFAULT_PERSISTENCE = (0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Frame:
    t: float
    up_bids: tuple[tuple[float, float], ...]
    up_asks: tuple[tuple[float, float], ...]
    down_bids: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    frames: tuple[Frame, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Fill:
    shares: float
    cash: float
    fee: float
    worst: float


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
    for level in book.get(side) or []:
        try:
            price, size = float(level["price"]), float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < price < 1 and size > 0:
            result.append((price, size))
    if not result:
        prefix = "bid" if side == "bids" else "ask"
        try:
            price = float(book[prefix])
            size = float(book[f"{prefix}_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if 0 < price < 1 and size > 0:
            result.append((price, size))
    return tuple(sorted(result, reverse=(side == "bids")))


def fee(price, shares):
    return shares * FEE_RATE * price * (1.0 - price)


def sweep_buy(asks, shares, max_price=0.99):
    remaining = shares
    cash = charge = 0.0
    worst = None
    for price, size in asks:
        if price > max_price + 1e-12:
            continue
        quantity = min(remaining, size)
        cash += price * quantity
        charge += fee(price, quantity)
        remaining -= quantity
        worst = price
        if remaining <= 1e-9:
            return Fill(shares, cash, charge, worst)
    return None


def sweep_sell(bids, shares, min_price=0.01):
    remaining = shares
    cash = charge = 0.0
    worst = None
    for price, size in bids:
        if price + 1e-12 < min_price:
            continue
        quantity = min(remaining, size)
        cash += price * quantity
        charge += fee(price, quantity)
        remaining -= quantity
        worst = price
        if remaining <= 1e-9:
            return Fill(shares, cash, charge, worst)
    return None


def has_resolution(path):
    """Return the recorded winner without trusting the filename alone."""
    winner = None
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if '"event":"resolution"' not in line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    continue
                if raw.get("winner") in ("Up", "Down"):
                    winner = raw["winner"]
    except (OSError, UnicodeDecodeError):
        return None
    return winner


def load_window(path, clock, frozen_winner):
    snapshots = []
    start = None
    try:
        stream = path.open(encoding="utf-8")
    except OSError:
        return None
    with stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if raw.get("event") != "snapshot":
                continue
            try:
                start = int(float(raw["window_start"]))
                label_t = float(raw["seconds_into_window"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            up_ts, down_ts = epoch(up.get("book_timestamp")), epoch(down.get("book_timestamp"))
            if up_ts is None or down_ts is None:
                continue
            snapshots.append((
                label_t,
                max(up_ts, down_ts) - start,
                levels(up, "bids"), levels(up, "asks"),
                levels(down, "bids"), levels(down, "asks"),
            ))
    if frozen_winner not in ("Up", "Down") or start is None or len(snapshots) < 2:
        return None
    frames = []
    for index, row in enumerate(snapshots):
        label_t, source_t, up_bids, up_asks, down_bids, down_asks = row
        if clock == "next_cycle":
            if index + 1 >= len(snapshots):
                break
            available_t = max(source_t, snapshots[index + 1][0])
        elif clock == "source":
            available_t = source_t
        else:
            raise ValueError(clock)
        if available_t < 0 or available_t >= 300:
            continue
        frames.append(Frame(
            available_t, up_bids, up_asks, down_bids, down_asks
        ))
    frames.sort(key=lambda frame: frame.t)
    if not frames or frames[0].t > 10:
        return None
    packed = tuple(frames)
    return Window(path.stem, start, frozen_winner, packed,
                  tuple(frame.t for frame in packed))


def pair_quote(frame):
    """Quote equal shares with $2 combined cash principal before fees."""
    if not frame.up_asks or not frame.down_asks:
        return None
    # At this stake the top size normally suffices.  Re-quote against all
    # recorded depth and reject any exceptional case that needs >$2 cash.
    shares = TOTAL_PRINCIPAL / (frame.up_asks[0][0] + frame.down_asks[0][0])
    up = sweep_buy(frame.up_asks, shares)
    down = sweep_buy(frame.down_asks, shares)
    if up is None or down is None:
        return None
    cash = up.cash + down.cash
    if cash > TOTAL_PRINCIPAL + 1e-8:
        # Deeper levels changed the cash requirement. Scale once, then verify.
        shares *= TOTAL_PRINCIPAL / cash
        up = sweep_buy(frame.up_asks, shares)
        down = sweep_buy(frame.down_asks, shares)
        if up is None or down is None:
            return None
    unit_cost = (up.cash + up.fee + down.cash + down.fee) / shares
    return shares, up, down, unit_cost


def first_persistent_signal(window, threshold, persistence,
                            earliest=5.0, latest=297.0):
    streak_start = None
    previous_t = None
    for index, frame in enumerate(window.frames):
        if frame.t < earliest:
            continue
        if frame.t > latest:
            break
        quoted = pair_quote(frame)
        qualifies = quoted is not None and quoted[3] <= threshold + 1e-12
        # A logger outage is unknown market state, not proof that the quote
        # persisted.  One second is intentionally generous versus the normal
        # ~0.3-second REST cycle.
        if previous_t is not None and frame.t - previous_t > 1.0:
            streak_start = None
        previous_t = frame.t
        if not qualifies:
            streak_start = None
            continue
        if streak_start is None:
            streak_start = frame.t
        if frame.t - streak_start + 1e-12 >= persistence:
            return index, frame, quoted
    return None


def later_frame(window, target, lower=0):
    index = bisect.bisect_left(window.times, target, lo=lower)
    if index >= len(window.frames) or window.frames[index].t >= 300:
        return None, index
    return window.frames[index], index


def side_levels(frame, side, kind):
    return getattr(frame, f"{side.lower()}_{kind}")


def buy_action(window, side, shares, limit, decision_i, target):
    frame, index = later_frame(window, target, decision_i + 1)
    if frame is None:
        return None, target, index
    fill = sweep_buy(side_levels(frame, side, "asks"), shares, limit)
    return fill, frame.t, index


def unwind_action(window, side, shares, known_t, lower, rng):
    """Quote after naked exposure is known, then send one delayed sell FOK."""
    quote_frame, quote_i = later_frame(window, known_t, lower)
    if quote_frame is None:
        return None, "no_unwind_quote", None
    quoted = sweep_sell(side_levels(quote_frame, side, "bids"), shares)
    if quoted is None:
        return None, "no_unwind_depth", quote_frame.t
    limit = max(0.01, quoted.worst - FOK_ALLOWANCE)
    target = quote_frame.t + rng.uniform(0.300, 0.800)
    execution, _ = later_frame(window, target, quote_i + 1)
    if execution is None:
        return None, "unwind_after_close", target
    fill = sweep_sell(side_levels(execution, side, "bids"), shares, limit)
    if fill is None:
        return None, "unwind_fok_failed", execution.t
    return fill, "unwound", execution.t


def settle_naked(window, side, buy):
    payout = buy.shares if side == window.winner else 0.0
    return payout - buy.cash - buy.fee


def simulate(window, threshold, persistence, seed, mode):
    signal = first_persistent_signal(window, threshold, persistence)
    if signal is None:
        return None
    decision_i, decision, (shares, up_quote, down_quote, displayed_unit) = signal
    decision_quotes = {"Up": up_quote, "Down": down_quote}
    limits = {
        side: min(0.99, decision_quotes[side].worst + FOK_ALLOWANCE)
        for side in ("Up", "Down")
    }
    rng = random.Random(
        f"pair-unwind:{seed}:{window.slug}:{threshold}:{persistence}:{mode}"
    )
    buys = []
    completions = []
    lower = decision_i + 1

    if mode == "concurrent":
        for side in ("Up", "Down"):
            target = decision.t + rng.uniform(0.300, 0.800)
            fill, completed_t, execution_i = buy_action(
                window, side, shares, limits[side], decision_i, target
            )
            completions.append(completed_t)
            lower = max(lower, min(execution_i, len(window.frames)))
            if fill is not None:
                buys.append((side, fill))
    elif mode == "sequential":
        # Buy the cheaper leg first; do not expose capital if it already fails.
        sides = ("Up", "Down") if up_quote.cash <= down_quote.cash else ("Down", "Up")
        target = decision.t + rng.uniform(0.300, 0.800)
        first, completed_t, execution_i = buy_action(
            window, sides[0], shares, limits[sides[0]], decision_i, target
        )
        completions.append(completed_t)
        lower = max(lower, min(execution_i, len(window.frames)))
        if first is not None:
            buys.append((sides[0], first))
            target = completed_t + rng.uniform(0.300, 0.800)
            second, completed_t, execution_i = buy_action(
                window, sides[1], shares, limits[sides[1]], lower - 1, target
            )
            completions.append(completed_t)
            lower = max(lower, min(execution_i, len(window.frames)))
            if second is not None:
                buys.append((sides[1], second))
    else:
        raise ValueError(mode)

    result = {
        "slug": window.slug,
        "signal_t": decision.t,
        "displayed_unit": displayed_unit,
        "buy_fills": len(buys),
        "complete": len(buys) == 2,
        "initial_naked": len(buys) == 1,
        "unwound": False,
        "unwind_failed": False,
        "unwind_reason": None,
        "pnl": 0.0,
    }
    if len(buys) == 2:
        cost = sum(fill.cash + fill.fee for _side, fill in buys)
        result["pnl"] = shares - cost
        return result
    if not buys:
        return result

    side, buy = buys[0]
    known_t = max(completions) if completions else decision.t
    sold, reason, _ = unwind_action(window, side, buy.shares, known_t, lower, rng)
    result["unwind_reason"] = reason
    if sold is not None:
        result["unwound"] = True
        result["pnl"] = sold.cash - sold.fee - buy.cash - buy.fee
    else:
        result["unwind_failed"] = True
        result["pnl"] = settle_naked(window, side, buy)
    return result


def one_run(windows, threshold, persistence, seed, mode):
    results = [
        result for window in windows
        if (result := simulate(window, threshold, persistence, seed, mode)) is not None
    ]
    equity = peak = drawdown = 0.0
    for result in results:
        equity += result["pnl"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "signals": len(results),
        "attempts": sum(result["buy_fills"] > 0 for result in results),
        "no_fill": sum(result["buy_fills"] == 0 for result in results),
        "pairs": sum(result["complete"] for result in results),
        "naked": sum(result["initial_naked"] for result in results),
        "unwound": sum(result["unwound"] for result in results),
        "unwind_failed": sum(result["unwind_failed"] for result in results),
        "pnl": equity,
        "dd": drawdown,
    }


def metrics(windows, threshold, persistence, seeds, mode):
    runs = [one_run(windows, threshold, persistence, seed, mode) for seed in seeds]
    keys = ("signals", "attempts", "no_fill", "pairs", "naked",
            "unwound", "unwind_failed", "pnl", "dd")
    answer = {key: statistics.median(run[key] for run in runs) for key in keys}
    answer["pnl_min"] = min(run["pnl"] for run in runs)
    answer["pnl_max"] = max(run["pnl"] for run in runs)
    return answer


def show(label, result):
    print(
        f"{label:<20} sig={result['signals']:4.1f} try={result['attempts']:4.1f} "
        f"pair={result['pairs']:4.1f} naked={result['naked']:4.1f} "
        f"exit={result['unwound']:4.1f} exitFail={result['unwind_failed']:3.1f} "
        f"pnl=${result['pnl']:+6.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"dd=${result['dd']:.2f}"
    )


def frozen_paths(directory, through=None):
    selected = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if through is not None and start > through:
            continue
        winner = has_resolution(path)
        if winner is not None:
            selected.append((path, winner))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "price_logs" / "polymarket",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--through", type=int,
                        help="freeze at or before this window epoch")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--persistence", type=float, nargs="*", default=list(DEFAULT_PERSISTENCE),
        help="seconds the fee-adjusted discount must remain continuously visible",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if any(value < 0 for value in args.persistence):
        parser.error("--persistence cannot be negative")

    # This exact path/winner list is the frozen data universe for the run.
    paths = frozen_paths(args.directory, args.through)
    cutoff = max((int(path.stem.rsplit("-", 1)[-1]) for path, _ in paths), default=None)
    print(
        f"frozen resolved files={len(paths)} through={cutoff}; "
        f"principal=${TOTAL_PRINCIPAL:.2f}; fee=.07*p*(1-p); "
        f"delay=300-800ms/action; buy/sell allowance=${FOK_ALLOWANCE:.2f}"
    )
    print(
        f"fee-adjusted pair threshold<={args.threshold:.3f}; "
        "one attempt/window; naked leg immediately sold by FOK"
    )
    seeds = range(args.seeds)

    for clock in ("next_cycle", "source"):
        windows = [
            window for path, winner in paths
            if (window := load_window(path, clock, winner)) is not None
        ]
        cut60, cut80 = int(0.60 * len(windows)), int(0.80 * len(windows))
        folds = (
            ("train60", windows[:cut60]),
            ("valid20", windows[cut60:cut80]),
            ("holdout20", windows[cut80:]),
            ("all", windows),
        )
        print(
            f"\nCLOCK={clock} usable={len(windows)} split="
            f"{cut60}/{cut80-cut60}/{len(windows)-cut80}"
        )
        for persistence in args.persistence:
            print(f"\npersistence={persistence:.1f}s")
            for fold_name, fold in folds:
                concurrent = metrics(
                    fold, args.threshold, persistence, seeds, "concurrent"
                )
                sequential = metrics(
                    fold, args.threshold, persistence, seeds, "sequential"
                )
                show(f"{fold_name} concurrent", concurrent)
                show(f"{fold_name} sequential", sequential)


if __name__ == "__main__":
    main()
