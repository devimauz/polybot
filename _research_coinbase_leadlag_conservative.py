"""Conservative audit of the frozen Coinbase/Polymarket lead-lag rule.

The candidate was selected before this audit by ``_research_coinbase_leadlag``:
a >=2 bps Coinbase move over three seconds, aligned with Coinbase's direction
from the five-minute open, while the corresponding Polymarket midpoint moved
no more than one tick.  Entry asks may be $0.15--$0.85.

This audit removes a possible within-logger-cycle timing advantage.  A signal
must remain valid in the next source-ordered combined frame.  That confirming
frame is treated as unavailable until the following logger cycle starts; only
then does the normal 300--800 ms order delay begin.  Thus no book paired with
either Coinbase observation is treated as immediately executable.  All other
assumptions remain unchanged: $2 fixed
stake, full recorded depth, current crypto taker fees, $0.05 FOK allowance,
100 paired latency seeds, and settlement exits.  It never places an order.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import math
from pathlib import Path
import random
import statistics

import _research_coinbase_leadlag as base
from _research_coinbase_leadlag import Decision, Rule, Window


ROOT = Path(__file__).resolve().parent
FROZEN_THROUGH = 1792066500
RULE = Rule(3.0, 2.0, 0.01, "aligned")
ASK_MIN = 0.15
ASK_MAX = 0.85


def condition_at(window: Window, index: int) -> Decision | None:
    """Evaluate the already-frozen shock/lag rule at one causal frame."""
    current = window.frames[index]
    if not base.FIRST_SIGNAL_SECOND <= current.t <= base.LAST_SIGNAL_SECOND:
        return None
    old_i = base.prior_index(window, index, RULE.lookback)
    older_i = base.prior_index(window, index, 2.0 * RULE.lookback)
    if old_i is None or older_i is None:
        return None
    old = window.frames[old_i]
    older = window.frames[older_i]
    shock = (current.coinbase / old.coinbase - 1.0) * 10_000.0
    if abs(shock) + 1e-12 < RULE.jump_bps:
        return None
    side = "Up" if shock > 0 else "Down"
    direction = 1.0 if shock > 0 else -1.0
    distance = direction * (current.coinbase / window.opening - 1.0) * 10_000.0
    if distance < 0.0:
        return None
    old_mid = old.midpoint(side)
    current_mid = current.midpoint(side)
    asks = current.asks(side)
    if old_mid is None or current_mid is None or not asks:
        return None
    market_move = current_mid - old_mid
    if market_move > RULE.lag_cap + 1e-12:
        return None
    if not ASK_MIN <= asks[0][0] <= ASK_MAX:
        return None
    quote = base.sweep_buy(asks, base.STAKE)
    if quote is None:
        return None
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


def confirmed_decision_for(window: Window) -> Decision | None:
    """Require persistence, then delay availability to another logger cycle."""
    start_i = bisect.bisect_left(window.times, base.FIRST_SIGNAL_SECOND)
    last_cb_t = max(
        (frame.cb_t for frame in window.frames[:start_i]), default=-math.inf
    )
    for index in range(start_i, len(window.frames) - 2):
        current = window.frames[index]
        if current.t > base.LAST_SIGNAL_SECOND:
            break
        if current.cb_t <= last_cb_t + 1e-9:
            continue
        last_cb_t = current.cb_t
        initial = condition_at(window, index)
        if initial is None:
            continue
        # The successor confirms persistence.  Because the logger labels a
        # row before doing its two REST calls, even that row is not considered
        # available until the following cycle begins.  Keep its observed FOK
        # quote, but move decision time/index forward one more full frame.
        confirmation = condition_at(window, index + 1)
        if confirmation is not None and confirmation.side == initial.side:
            available = window.frames[index + 2]
            return Decision(
                index + 2,
                available.t,
                confirmation.side,
                confirmation.shock_bps,
                confirmation.prior_bps,
                confirmation.market_move,
                confirmation.ask,
                confirmation.quoted_worst,
            )
    return None


def execute_min5_sensitivity(window, decision, seed: int):
    """Sensitivity only: reject an otherwise valid fill below five shares.

    This is deliberately not the primary model.  The user's matched live $2
    BUY journal demonstrates that sub-five-share market/FOK buys can execute.
    """
    rng = random.Random(f"coinbase-leadlag:{seed}:{window.slug}:entry")
    target = decision.t + rng.uniform(base.DELAY_MIN, base.DELAY_MAX)
    execution_i = bisect.bisect_left(
        window.times, target, decision.index + 1
    )
    if execution_i >= len(window.frames) or window.frames[execution_i].t >= 300.0:
        return base.Attempt(
            window.slug, True, False, decision.side, window.winner,
            decision.t, None, 0.0, 0.0,
        )
    execution = window.frames[execution_i]
    limit = min(0.99, decision.quoted_worst + base.FOK_ALLOWANCE)
    fill = base.sweep_buy(execution.asks(decision.side), base.STAKE, limit)
    if fill is None or fill[0] + 1e-12 < 5.0:
        return base.Attempt(
            window.slug, True, False, decision.side, window.winner,
            decision.t, execution.t, 0.0, 0.0,
        )
    shares, cost, _ = fill
    pnl = (shares if decision.side == window.winner else 0.0) - cost
    return base.Attempt(
        window.slug, True, True, decision.side, window.winner,
        decision.t, execution.t, pnl, cost,
    )


def attempts_for(windows, decisions, seed: int, executor=base.execute):
    attempts = []
    for window, decision in zip(windows, decisions):
        if decision is None:
            continue
        if executor is base.execute:
            attempt = executor(window, RULE, seed, decision)
        else:
            attempt = executor(window, decision, seed)
        if attempt is not None:
            attempts.append(attempt)
    return attempts


def concentration(windows, decisions, seeds, executor=base.execute):
    samples = []
    for seed in seeds:
        trades = [
            row for row in attempts_for(windows, decisions, seed, executor)
            if row.filled
        ]
        profits = sorted((row.pnl for row in trades), reverse=True)
        total = sum(profits)
        record = {"total": total, "fills": len(profits)}
        for count in (1, 3, 5):
            top = sum(profits[:count])
            record[f"top{count}"] = top
            record[f"without{count}"] = total - top
            record[f"share{count}"] = 100.0 * top / total if total > 0 else math.inf
        samples.append(record)
    result = {}
    for key in ("fills", "total", "top1", "top3", "top5", "without1", "without3", "without5"):
        values = [row[key] for row in samples]
        result[key] = statistics.median(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    for count in (1, 3, 5):
        finite = [row[f"share{count}"] for row in samples if math.isfinite(row[f"share{count}"])]
        result[f"share{count}"] = statistics.median(finite) if finite else math.inf
    return result


def show_concentration(result):
    print("\nP&L CONCENTRATION (all windows; 100-seed median)")
    for count in (1, 3, 5):
        print(
            f"  largest {count}: ${result[f'top{count}']:+.2f} "
            f"({result[f'share{count}']:.1f}% of total); "
            f"without=${result[f'without{count}']:+.2f} "
            f"range=${result[f'without{count}_min']:+.2f}.."
            f"${result[f'without{count}_max']:+.2f}"
        )


def group_metric(windows, decisions, seeds):
    return base.aggregate(windows, decisions, RULE, seeds)


def sensitivity_metric(windows, decisions, seeds):
    rows = []
    for seed in seeds:
        attempts = attempts_for(
            windows, decisions, seed, execute_min5_sensitivity
        )
        trades = [attempt for attempt in attempts if attempt.filled]
        equity = peak = drawdown = 0.0
        for trade in trades:
            equity += trade.pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        pnl = sum(trade.pnl for trade in trades)
        cost = sum(trade.cost for trade in trades)
        rows.append({
            "signals": len(attempts),
            "trades": len(trades),
            "wins": sum(trade.pnl > 0 for trade in trades),
            "fok_failures": len(attempts) - len(trades),
            "pnl": pnl,
            "roi": 100.0 * pnl / cost if cost else 0.0,
            "drawdown": drawdown,
        })
    result = {
        key: statistics.median(row[key] for row in rows)
        for key in (
            "signals", "trades", "wins", "fok_failures", "pnl", "roi", "drawdown"
        )
    }
    for key in ("pnl", "trades", "drawdown"):
        values = [row[key] for row in rows]
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--through", type=int, default=FROZEN_THROUGH)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    loaded = base.load_windows(args.directory, args.through)
    prospective = [window for window in loaded if window.start > FROZEN_THROUGH]
    windows = [window for window in loaded if window.start <= FROZEN_THROUGH]
    if len(windows) < 30:
        raise SystemExit("not enough resolved source-aligned windows")
    decisions = [confirmed_decision_for(window) for window in windows]
    cut60 = int(0.60 * len(windows))
    cut80 = int(0.80 * len(windows))
    seeds = range(args.seeds)

    print(
        f"CONSERVATIVE LEAD/LAG AUDIT; frozen_through={args.through}; "
        f"resolved={len(windows)}; split={cut60}/{cut80-cut60}/{len(windows)-cut80}"
    )
    print(
        f"frozen_rule={RULE.name}; persistence=next combined frame; "
        "decision clock=following-cycle completion proxy"
    )
    print(
        "stake=$2; full depth; delay=300-800ms after confirmation; "
        "FOK allowance=$0.05; current fees; hold to settlement"
    )

    groups = (
        ("train", 0, cut60),
        ("validation", cut60, cut80),
        ("holdout", cut80, len(windows)),
        ("all", 0, len(windows)),
        ("last5", len(windows) - 5, len(windows)),
    )
    print("\nCHRONOLOGICAL SPLITS")
    for label, start, stop in groups:
        result = group_metric(windows[start:stop], decisions[start:stop], seeds)
        print(f"  {label:<10} {base.fmt(result)}")

    print("\nCHRONOLOGICAL QUARTILES")
    bounds = [0, len(windows) // 4, len(windows) // 2, 3 * len(windows) // 4, len(windows)]
    for number, (start, stop) in enumerate(zip(bounds, bounds[1:]), 1):
        result = group_metric(windows[start:stop], decisions[start:stop], seeds)
        starts = f"{windows[start].start}-{windows[stop-1].start}"
        print(f"  Q{number} {starts:<22} {base.fmt(result)}")

    print("\nDECISION-ASK BINS")
    bins = (
        (0.15, 0.30),
        (0.30, 0.45),
        (0.45, 0.60),
        (0.60, 0.75),
        (0.75, 0.851),
    )
    for low, high in bins:
        chosen = [
            (window, decision)
            for window, decision in zip(windows, decisions)
            if decision is not None and low <= decision.ask < high
        ]
        group_windows = [item[0] for item in chosen]
        group_decisions = [item[1] for item in chosen]
        result = group_metric(group_windows, group_decisions, seeds)
        upper = 0.85 if high > 0.85 else high
        print(f"  ${low:.2f}-${upper:.2f} {base.fmt(result)}")

    show_concentration(concentration(windows, decisions, seeds))

    print("\nMINIMUM-5-SHARES SENSITIVITY ONLY")
    print("  (not the primary model: the live journal contains matched sub-5-share $2 BUYs)")
    for label, start, stop in groups[:4]:
        result = sensitivity_metric(
            windows[start:stop], decisions[start:stop], seeds
        )
        print(f"  {label:<10} {base.fmt(result)}")
    show_concentration(
        concentration(
            windows, decisions, seeds, execute_min5_sensitivity
        )
    )

    if prospective:
        future_decisions = [confirmed_decision_for(window) for window in prospective]
        result = group_metric(prospective, future_decisions, seeds)
        print(
            f"\nPROSPECTIVE AFTER {FROZEN_THROUGH} "
            f"({prospective[0].start}-{prospective[-1].start})"
        )
        print(f"  new        {base.fmt(result)}")


if __name__ == "__main__":
    main()
