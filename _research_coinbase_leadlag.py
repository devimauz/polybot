"""Causal Coinbase-to-Polymarket lead/lag research on BTC five-minute logs.

This is a research-only replay: it never places an order and does not modify
the live bot.  A signal is the first fresh Coinbase price shock in a window
whose matching Polymarket outcome has moved by no more than a small number of
ticks over the same lookback.  The final 20% of windows is not opened unless
a rule first survives the chronological training and validation gates.

Timing is deliberately conservative.  A combined logger frame becomes
available at max(Up book timestamp, Down book timestamp, Coinbase exchange
timestamp).  A $2 FOK buy then reaches the first later recorded book after a
seeded 300--800 ms delay.  The limit is the decision-time full-depth sweep's
worst price plus $0.05.  Filled positions pay the current crypto taker fee and
are held to settlement.

The logger records Coinbase price and timestamps but not Coinbase trade
volume, so the experiment cannot test a volume filter.
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

from btc5m_momentum_strategy import fee_for


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "price_logs" / "polymarket"
FROZEN_THROUGH = 1792066500
STAKE = 2.0
FOK_ALLOWANCE = 0.05
DELAY_MIN = 0.300
DELAY_MAX = 0.800
FIRST_SIGNAL_SECOND = 45.0
LAST_SIGNAL_SECOND = 255.0
ASK_MIN = 0.15
ASK_MAX = 0.85


@dataclass(frozen=True)
class Frame:
    t: float
    cb_t: float
    coinbase: float
    up_bid: float | None
    down_bid: float | None
    up_asks: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]

    def asks(self, side: str):
        return self.up_asks if side == "Up" else self.down_asks

    def midpoint(self, side: str) -> float | None:
        asks = self.asks(side)
        bid = self.up_bid if side == "Up" else self.down_bid
        if bid is None or not asks:
            return None
        return (bid + asks[0][0]) / 2.0


@dataclass(frozen=True)
class Window:
    slug: str
    start: int
    winner: str
    frames: tuple[Frame, ...]
    times: tuple[float, ...]
    opening: float


@dataclass(frozen=True)
class Rule:
    lookback: float
    jump_bps: float
    lag_cap: float
    context: str

    @property
    def name(self) -> str:
        cap_ticks = int(round(self.lag_cap * 100))
        return (
            f"shock{self.lookback:g}s_{self.jump_bps:g}bps_"
            f"lag{cap_ticks}tick_{self.context}"
        )


@dataclass(frozen=True)
class Decision:
    index: int
    t: float
    side: str
    shock_bps: float
    prior_bps: float
    market_move: float
    ask: float
    quoted_worst: float


@dataclass(frozen=True)
class Attempt:
    slug: str
    signaled: bool
    filled: bool
    side: str
    winner: str
    decision_t: float
    execution_t: float | None
    pnl: float
    cost: float


def epoch(value) -> float | None:
    try:
        numeric = float(value)
        return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def clean_best_bid(book: dict) -> float | None:
    bids = []
    for level in book.get("bids") or []:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            bids.append(price)
    if bids:
        return max(bids)
    try:
        price = float(book["bid"])
        size = float(book["bid_size"])
    except (KeyError, TypeError, ValueError):
        return None
    return price if 0 < price < 1 and size > 0 else None


def buy_depth_prefix(book: dict, dollars: float = STAKE):
    """Keep the exact full-book prefix needed to decide a $2 sweep.

    Asks are sorted low-to-high.  Once their cumulative notional reaches the
    stake, all omitted levels are worse and therefore cannot affect either an
    unconstrained sweep or one capped by a FOK limit price.
    """
    clean = []
    for level in book.get("asks") or []:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0:
            clean.append((price, size))
    if not clean:
        try:
            price = float(book["ask"])
            size = float(book["ask_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if 0 < price < 1 and size > 0:
            clean.append((price, size))
    result = []
    notional = 0.0
    for price, size in sorted(clean):
        result.append((price, size))
        notional += price * size
        if notional + 1e-12 >= dollars:
            break
    return tuple(result)


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
        fees += fee_for(price, quantity)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, worst
    return None


def load_window(path: Path) -> Window | None:
    try:
        start = int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return None
    winner = None
    frames: list[Frame] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in ("Up", "Down"):
                winner = raw["winner"]
                continue
            if raw.get("event") != "snapshot":
                continue
            up = raw.get("Up") or {}
            down = raw.get("Down") or {}
            coinbase = raw.get("coinbase") or {}
            up_t = epoch(up.get("book_timestamp"))
            down_t = epoch(down.get("book_timestamp"))
            cb_t = epoch(coinbase.get("exchange_ts"))
            source_times = [value for value in (up_t, down_t, cb_t) if value is not None]
            try:
                price = float(coinbase["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if not source_times or cb_t is None or not math.isfinite(price) or price <= 0:
                continue
            available_t = max(source_times) - start
            relative_cb_t = cb_t - start
            if not -2.0 <= available_t <= 301.5:
                continue
            frames.append(
                Frame(
                    available_t,
                    relative_cb_t,
                    price,
                    clean_best_bid(up),
                    clean_best_bid(down),
                    buy_depth_prefix(up),
                    buy_depth_prefix(down),
                )
            )
    frames.sort(key=lambda frame: frame.t)
    if winner not in ("Up", "Down") or not frames or frames[0].t > 5.0:
        return None
    packed = tuple(frames)
    return Window(
        path.stem,
        start,
        winner,
        packed,
        tuple(frame.t for frame in packed),
        packed[0].coinbase,
    )


def load_windows(directory: Path, through: int):
    result = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if start > through:
            continue
        window = load_window(path)
        if window is not None:
            result.append(window)
    return result


def prior_index(window: Window, index: int, seconds: float) -> int | None:
    target = window.frames[index].t - seconds
    found = bisect.bisect_right(window.times, target, 0, index) - 1
    return found if found >= 0 else None


def decision_for(window: Window, rule: Rule) -> Decision | None:
    start_i = bisect.bisect_left(window.times, FIRST_SIGNAL_SECOND)
    # Frames are ordered by combined availability rather than Coinbase's own
    # timestamp.  Seed this with every already-known Coinbase timestamp so the
    # first scanned row cannot masquerade as a fresh exchange update.
    last_cb_t = max(
        (frame.cb_t for frame in window.frames[:start_i]), default=-math.inf
    )
    for index in range(start_i, len(window.frames)):
        current = window.frames[index]
        if current.t > LAST_SIGNAL_SECOND:
            break
        # Only a newly observed Coinbase exchange update can create an event.
        if current.cb_t <= last_cb_t + 1e-9:
            continue
        last_cb_t = current.cb_t
        old_i = prior_index(window, index, rule.lookback)
        older_i = prior_index(window, index, 2.0 * rule.lookback)
        if old_i is None or older_i is None:
            continue
        old = window.frames[old_i]
        older = window.frames[older_i]
        shock = (current.coinbase / old.coinbase - 1.0) * 10_000.0
        if abs(shock) + 1e-12 < rule.jump_bps:
            continue
        side = "Up" if shock > 0 else "Down"
        direction = 1.0 if shock > 0 else -1.0
        distance = direction * (current.coinbase / window.opening - 1.0) * 10_000.0
        if rule.context == "aligned" and distance < 0.0:
            continue

        old_mid = old.midpoint(side)
        current_mid = current.midpoint(side)
        asks = current.asks(side)
        if old_mid is None or current_mid is None or not asks:
            continue
        market_move = current_mid - old_mid
        if market_move > rule.lag_cap + 1e-12:
            continue
        if not ASK_MIN <= asks[0][0] <= ASK_MAX:
            continue
        quote = sweep_buy(asks, STAKE)
        if quote is None:
            continue
        prior = direction * (old.coinbase / older.coinbase - 1.0) * 10_000.0
        return Decision(
            index,
            current.t,
            side,
            direction * shock,
            prior,
            market_move,
            asks[0][0],
            quote[2],
        )
    return None


def execute(window: Window, rule: Rule, seed: int, decision: Decision | None = None):
    if decision is None:
        decision = decision_for(window, rule)
    if decision is None:
        return None
    # The random variate is common across configurations for a given
    # window/seed.  This makes latency comparisons paired rather than noisy.
    rng = random.Random(f"coinbase-leadlag:{seed}:{window.slug}:entry")
    target = decision.t + rng.uniform(DELAY_MIN, DELAY_MAX)
    execution_i = bisect.bisect_left(window.times, target, decision.index + 1)
    if execution_i >= len(window.frames) or window.frames[execution_i].t >= 300.0:
        return Attempt(
            window.slug, True, False, decision.side, window.winner,
            decision.t, None, 0.0, 0.0,
        )
    execution = window.frames[execution_i]
    limit = min(0.99, decision.quoted_worst + FOK_ALLOWANCE)
    fill = sweep_buy(execution.asks(decision.side), STAKE, limit)
    if fill is None:
        return Attempt(
            window.slug, True, False, decision.side, window.winner,
            decision.t, execution.t, 0.0, 0.0,
        )
    shares, cost, _ = fill
    pnl = (shares if decision.side == window.winner else 0.0) - cost
    return Attempt(
        window.slug, True, True, decision.side, window.winner,
        decision.t, execution.t, pnl, cost,
    )


def decisions_for(windows, rule):
    return [decision_for(window, rule) for window in windows]


def one_run(windows, decisions, rule, seed: int):
    attempts = []
    for window, decision in zip(windows, decisions):
        if decision is not None:
            attempts.append(execute(window, rule, seed, decision))
    trades = [attempt for attempt in attempts if attempt is not None and attempt.filled]
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
        "wins": sum(trade.pnl > 0 for trade in trades),
        "fok_failures": len(attempts) - len(trades),
        "pnl": pnl,
        "roi": 100.0 * pnl / cost if cost else 0.0,
        "drawdown": drawdown,
    }


def aggregate(windows, decisions, rule, seeds):
    rows = [one_run(windows, decisions, rule, seed) for seed in seeds]
    result = {
        key: statistics.median(row[key] for row in rows)
        for key in (
            "signals", "trades", "wins", "fok_failures", "pnl", "roi", "drawdown"
        )
    }
    result["pnl_min"] = min(row["pnl"] for row in rows)
    result["pnl_max"] = max(row["pnl"] for row in rows)
    result["trades_min"] = min(row["trades"] for row in rows)
    result["trades_max"] = max(row["trades"] for row in rows)
    result["drawdown_min"] = min(row["drawdown"] for row in rows)
    result["drawdown_max"] = max(row["drawdown"] for row in rows)
    return result


def rules():
    # A small, declared-before-holdout grid.  One tick is $0.01.
    return [
        Rule(lookback, jump, lag, context)
        for lookback in (1.0, 2.0, 3.0)
        for jump in (2.0, 4.0)
        for lag in (0.00, 0.01)
        for context in ("aligned", "any")
    ]


def fmt(row):
    return (
        f"sig={row['signals']:4.1f} fill={row['trades']:4.1f}"
        f"[{row['trades_min']:.0f}-{row['trades_max']:.0f}] "
        f"w={row['wins']:4.1f} fok={row['fok_failures']:3.1f} "
        f"pnl=${row['pnl']:+7.2f} "
        f"range=${row['pnl_min']:+.2f}..${row['pnl_max']:+.2f} "
        f"roi={row['roi']:+6.1f}% dd=${row['drawdown']:.2f}"
        f"[{row['drawdown_min']:.2f}-{row['drawdown_max']:.2f}]"
    )


def subset_decisions(all_decisions, start: int, stop: int):
    return all_decisions[start:stop]


def main():
    global ASK_MIN, ASK_MAX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=LOG_DIR)
    parser.add_argument("--through", type=int, default=FROZEN_THROUGH)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--ask-min", type=float, default=ASK_MIN)
    parser.add_argument("--ask-max", type=float, default=ASK_MAX)
    parser.add_argument(
        "--details", action="store_true",
        help="print seed-0 filled trades for the rule frozen before holdout",
    )
    args = parser.parse_args()
    if not 0.0 < args.ask_min <= args.ask_max < 1.0:
        raise SystemExit("require 0 < --ask-min <= --ask-max < 1")
    ASK_MIN, ASK_MAX = args.ask_min, args.ask_max

    windows = load_windows(args.directory, args.through)
    if len(windows) < 30:
        raise SystemExit("not enough fully resolved source-aligned windows")
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    train = windows[:cut60]
    validation = windows[cut60:cut80]
    holdout = windows[cut80:]
    half = len(train) // 2
    seeds = range(args.seeds)

    print(
        f"COINBASE LEAD/LAG; frozen_through={args.through}; "
        f"source-aligned resolved={len(windows)}; split="
        f"{len(train)}/{len(validation)}/{len(holdout)}"
    )
    print(
        "stake=$2; full-depth buys; delay=300-800ms; FOK allowance=$0.05; "
        "current crypto fees; hold to settlement"
    )
    print(
        f"event window={FIRST_SIGNAL_SECOND:g}-{LAST_SIGNAL_SECOND:g}s; "
        f"ask=${ASK_MIN:.2f}-${ASK_MAX:.2f}; Coinbase volume unavailable"
    )

    cached = {}
    screened = []
    print("\nTRAIN SCREEN (both chronological halves worst-seed positive)")
    for rule in rules():
        all_decisions = decisions_for(windows, rule)
        cached[rule.name] = all_decisions
        first = aggregate(
            train[:half], subset_decisions(all_decisions, 0, half), rule, seeds
        )
        second = aggregate(
            train[half:], subset_decisions(all_decisions, half, cut60), rule, seeds
        )
        total = aggregate(
            train, subset_decisions(all_decisions, 0, cut60), rule, seeds
        )
        passed = (
            total["trades"] >= 10
            and first["pnl_min"] > 0.0
            and second["pnl_min"] > 0.0
        )
        print(f"  {'PASS' if passed else 'fail'} {rule.name:<40} {fmt(total)}")
        if passed:
            screened.append((rule, total))

    print("\nVALIDATION (final holdout remains untouched for selection)")
    validated = []
    if not screened:
        print("  no rule passed the training stability gate")
    for rule, train_result in screened:
        all_decisions = cached[rule.name]
        result = aggregate(
            validation,
            subset_decisions(all_decisions, cut60, cut80),
            rule,
            seeds,
        )
        print(f"  {rule.name:<45} {fmt(result)}")
        if result["trades"] >= 4 and result["pnl_min"] > 0.0:
            validated.append((result["pnl_min"], rule, train_result, result))

    if not validated:
        print("\nREJECTED: no Coinbase lead/lag rule passed validation.")
        print("The final holdout was not inspected to rescue or tune a rule.")
        return

    validated.sort(key=lambda item: (item[0], item[2]["pnl_min"]), reverse=True)
    _, selected, train_result, validation_result = validated[0]
    decisions = cached[selected.name]
    print(f"\nFROZEN BEFORE HOLDOUT: {selected.name}")
    groups = (
        ("train", train, decisions[:cut60], train_result),
        ("validation", validation, decisions[cut60:cut80], validation_result),
        (
            "holdout",
            holdout,
            decisions[cut80:],
            aggregate(holdout, decisions[cut80:], selected, seeds),
        ),
        (
            "all",
            windows,
            decisions,
            aggregate(windows, decisions, selected, seeds),
        ),
        (
            "last5",
            windows[-5:],
            decisions[-5:],
            aggregate(windows[-5:], decisions[-5:], selected, seeds),
        ),
    )
    for label, group, group_decisions, result in groups:
        del group, group_decisions
        print(f"  {label:<10} {fmt(result)}")

    if args.details:
        print("\nSELECTED SEED-0 FILLS")
        for window, decision in zip(windows, decisions):
            if decision is None:
                continue
            attempt = execute(window, selected, 0, decision)
            if attempt is None or not attempt.filled:
                continue
            print(
                f"  {window.start} t={decision.t:6.2f}->{attempt.execution_t:6.2f} "
                f"{decision.side:<4} shock={decision.shock_bps:5.2f}bps "
                f"poly={decision.market_move:+.3f} ask={decision.ask:.2f} "
                f"winner={window.winner:<4} pnl=${attempt.pnl:+.3f}"
            )


if __name__ == "__main__":
    main()
