"""Research-only BTC5m complete-set arbitrage replay.

The strategy is market neutral: buy equal shares of Up and Down when their
combined displayed cost, including current crypto taker fees, is below the
$1 settlement value.  Nothing in this file submits an order or changes the
live bot.

Two execution models are reported:

* concurrent: both FOKs are sent together, each with an independent
  300--800 ms delay;
* sequential: the cheaper leg is filled first and the hedge is sent only
  after that fill, adding another independent 300--800 ms delay.  A failed
  hedge remains a naked position through settlement.

The logger labels a row before fetching both REST books.  The default
``next_cycle`` clock therefore makes a row usable no earlier than the next
logger cycle and no earlier than either book's exchange timestamp.  The
``source`` clock is also printed as a less-conservative sensitivity check.
Each window may make at most one attempt: its first qualifying observation.
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


@dataclass(frozen=True)
class Frame:
    t: float
    up_asks: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    frames: tuple[Frame, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Quote:
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


def levels(book):
    clean = []
    for level in book.get("asks") or []:
        try:
            price, size = float(level["price"]), float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < price < 1 and size > 0:
            clean.append((price, size))
    if not clean:
        try:
            price, size = float(book["ask"]), float(book["ask_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if 0 < price < 1 and size > 0:
            clean.append((price, size))
    return tuple(sorted(clean))


def fee(price, shares):
    return shares * FEE_RATE * price * (1.0 - price)


def quote_shares(asks, shares, max_price=0.99):
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
            return Quote(shares, cash, charge, worst)
    return None


def load_window(path, clock):
    snapshots = []
    winner = None
    start = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in ("Up", "Down"):
                winner = raw["winner"]
                continue
            if raw.get("event") != "snapshot":
                continue
            try:
                start = int(float(raw["window_start"]))
                label_t = float(raw["seconds_into_window"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            up_asks, down_asks = levels(up), levels(down)
            if not up_asks or not down_asks:
                continue
            source = [epoch(up.get("book_timestamp")), epoch(down.get("book_timestamp"))]
            source = [value - start for value in source if value is not None]
            if len(source) != 2:
                continue
            snapshots.append((label_t, max(source), up_asks, down_asks))
    if winner not in ("Up", "Down") or start is None or len(snapshots) < 2:
        return None
    frames = []
    for index, (_label, source_t, up_asks, down_asks) in enumerate(snapshots):
        if clock == "next_cycle":
            if index + 1 >= len(snapshots):
                break
            available_t = max(source_t, snapshots[index + 1][0])
        else:
            available_t = source_t
        frames.append(Frame(available_t, up_asks, down_asks))
    frames.sort(key=lambda frame: frame.t)
    if not frames or frames[0].t > 10:
        return None
    packed = tuple(frames)
    return Window(path.stem, start, winner, packed, tuple(frame.t for frame in packed))


def decision_quote(frame, principal=TOTAL_PRINCIPAL):
    # At BTC5m depth, a $2 pair normally fits at the top.  Re-quote full
    # depth once so unusual thin top levels still get treated correctly.
    shares = principal / (frame.up_asks[0][0] + frame.down_asks[0][0])
    up = quote_shares(frame.up_asks, shares)
    down = quote_shares(frame.down_asks, shares)
    if up is None or down is None:
        return None
    total = up.cash + up.fee + down.cash + down.fee
    return shares, up, down, total / shares


def later_frame(window, target, lower=0):
    index = bisect.bisect_left(window.times, target, lo=lower)
    if index >= len(window.frames) or window.frames[index].t >= 300:
        return None, index
    return window.frames[index], index


def settle(window, fills):
    cost = sum(quote.cash + quote.fee for _side, quote in fills)
    payout = sum(quote.shares for side, quote in fills if side == window.winner)
    return payout - cost


def first_signal(window, threshold, earliest=5.0, latest=297.5):
    for index, frame in enumerate(window.frames):
        if frame.t < earliest:
            continue
        if frame.t > latest:
            break
        quoted = decision_quote(frame)
        if quoted is not None and quoted[3] <= threshold + 1e-12:
            return index, frame, quoted
    return None


def simulate(window, threshold, seed, mode, order="cheap_first"):
    signal = first_signal(window, threshold)
    if signal is None:
        return None
    index, frame, (shares, up_decision, down_decision, displayed_unit) = signal
    limits = {
        "Up": min(0.99, up_decision.worst + FOK_ALLOWANCE),
        "Down": min(0.99, down_decision.worst + FOK_ALLOWANCE),
    }
    rng = random.Random(f"pair-arb:{seed}:{window.slug}:{mode}:{threshold}:{order}")
    fills = []
    failures = 0
    if mode == "concurrent":
        for side in ("Up", "Down"):
            target = frame.t + rng.uniform(0.300, 0.800)
            execution, _ = later_frame(window, target, index + 1)
            asks = () if execution is None else (
                execution.up_asks if side == "Up" else execution.down_asks
            )
            fill = quote_shares(asks, shares, limits[side])
            if fill is None:
                failures += 1
            else:
                fills.append((side, fill))
    elif mode == "sequential":
        if order == "cheap_first":
            sides = ("Up", "Down") if up_decision.cash <= down_decision.cash else ("Down", "Up")
        else:
            sides = ("Up", "Down") if up_decision.cash > down_decision.cash else ("Down", "Up")
        target = frame.t + rng.uniform(0.300, 0.800)
        execution, execution_i = later_frame(window, target, index + 1)
        side = sides[0]
        asks = () if execution is None else (
            execution.up_asks if side == "Up" else execution.down_asks
        )
        fill = quote_shares(asks, shares, limits[side])
        if fill is None:
            failures += 1
        else:
            fills.append((side, fill))
            target = execution.t + rng.uniform(0.300, 0.800)
            execution, _ = later_frame(window, target, execution_i + 1)
            side = sides[1]
            asks = () if execution is None else (
                execution.up_asks if side == "Up" else execution.down_asks
            )
            fill = quote_shares(asks, shares, limits[side])
            if fill is None:
                failures += 1
            else:
                fills.append((side, fill))
    else:
        raise ValueError(mode)
    return {
        "slug": window.slug,
        "signal_t": frame.t,
        "displayed_unit": displayed_unit,
        "fills": len(fills),
        "failures": failures,
        "naked": len(fills) == 1,
        "complete": len(fills) == 2,
        "pnl": settle(window, fills),
    }


def metrics(windows, threshold, seeds, mode, order="cheap_first"):
    runs = []
    for seed in seeds:
        results = [result for window in windows if (result := simulate(
            window, threshold, seed, mode, order
        )) is not None]
        traded = [result for result in results if result["fills"]]
        equity = drawdown = peak = 0.0
        for result in traded:
            equity += result["pnl"]
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        runs.append({
            "signals": len(results),
            "trades": len(traded),
            "complete": sum(result["complete"] for result in results),
            "naked": sum(result["naked"] for result in results),
            "failures": sum(result["failures"] for result in results),
            "pnl": sum(result["pnl"] for result in results),
            "dd": drawdown,
        })
    answer = {
        key: statistics.median(run[key] for run in runs)
        for key in ("signals", "trades", "complete", "naked", "failures", "pnl", "dd")
    }
    answer["pnl_min"] = min(run["pnl"] for run in runs)
    answer["pnl_max"] = max(run["pnl"] for run in runs)
    return answer


def show(label, value):
    print(
        f"{label:<25} sig={value['signals']:4.1f} trade={value['trades']:4.1f} "
        f"pair={value['complete']:4.1f} naked={value['naked']:4.1f} "
        f"fail={value['failures']:4.1f} pnl=${value['pnl']:+6.2f} "
        f"range=${value['pnl_min']:+.2f}..${value['pnl_max']:+.2f} "
        f"dd=${value['dd']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "price_logs" / "polymarket")
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    seeds = range(args.seeds)
    paths = sorted(args.directory.glob("btc-updown-5m-*.jsonl"))
    print(f"files={len(paths)} stake=${TOTAL_PRINCIPAL:.2f} total, fee=0.07*p*(1-p), "
          f"delay=300-800ms/leg, allowance=${FOK_ALLOWANCE:.2f}")
    for clock in ("next_cycle", "source"):
        windows = [window for path in paths if (window := load_window(path, clock))]
        cut60, cut80 = int(.6 * len(windows)), int(.8 * len(windows))
        folds = (
            ("train60", windows[:cut60]),
            ("valid20", windows[cut60:cut80]),
            ("holdout20", windows[cut80:]),
            ("last5", windows[-5:]),
            ("all", windows),
        )
        print(f"\nCLOCK={clock} resolved={len(windows)} split={cut60}/"
              f"{cut80-cut60}/{len(windows)-cut80}")
        for threshold in (1.000, .990, .980, .970, .960, .950):
            print(f"\ntrigger displayed fee-adjusted pair <= {threshold:.3f}")
            for fold_name, fold in folds:
                concurrent = metrics(fold, threshold, seeds, "concurrent")
                sequential = metrics(fold, threshold, seeds, "sequential", "cheap_first")
                show(f"{fold_name} concurrent", concurrent)
                show(f"{fold_name} sequential", sequential)


if __name__ == "__main__":
    main()
