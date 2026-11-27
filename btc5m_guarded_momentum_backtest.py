"""Compare the frozen trend-guard candidate with the BTC5m momentum baseline.

The guard keeps the normal 150-second momentum entry only when:
  * signed Coinbase return over the prior 10 seconds is at least -2 bps; and
  * the selected outcome's bid fell no more than $0.05 over the prior 20 seconds.

Execution uses the decision-time ask plus the configured FOK allowance, the
recorded post-delay top ask/size, fees, and deterministic 300--800 ms delays.
"""

from __future__ import annotations

import argparse
import bisect
import statistics
from pathlib import Path

from _research_entry_confirmation import execution_result, extract, load_windows


ROOT = Path(__file__).resolve().parent


def add_guard_feature(trade):
    rows = trade["rows"]
    index = trade["decision_index"]
    times = [row["t"] for row in rows]
    prior_index = max(0, bisect.bisect_right(
        times, rows[index]["t"] - 20, 0, index + 1) - 1)
    side = trade["side"]
    current = rows[index].get(f"{side}_bid")
    prior = rows[prior_index].get(f"{side}_bid")
    trade["bid_change_20"] = (
        current - prior if current is not None and prior is not None else None
    )
    return trade


def guarded(trade):
    bid_change = trade["bid_change_20"]
    return (trade["features"]["recent_10"] >= -2
            and bid_change is not None and bid_change >= -0.05)


def simulate(trades, predicate, run):
    pnls = []
    for trade in trades:
        if not predicate(trade):
            continue
        pnl = execution_result(trade, run)
        if pnl is not None:
            pnls.append(pnl)
    equity = peak = drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "pnl": sum(pnls), "trades": len(pnls),
        "wins": sum(pnl > 0 for pnl in pnls), "drawdown": drawdown,
    }


def report(label, trades, predicate, runs):
    samples = [simulate(trades, predicate, run) for run in range(runs)]
    first = samples[0]
    print(
        f"{label:<18} run0={first['trades']:3d} trades "
        f"{first['wins']:3d} wins pnl=${first['pnl']:7.2f} | "
        f"{runs}-delay median=${statistics.median(x['pnl'] for x in samples):7.2f} "
        f"range=${min(x['pnl'] for x in samples):7.2f}.."
        f"${max(x['pnl'] for x in samples):7.2f} "
        f"median_dd=${statistics.median(x['drawdown'] for x in samples):6.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=ROOT / "price_logs" / "polymarket")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    windows = load_windows(args.directory)
    trades = [add_guard_feature(trade) for window in windows
              if (trade := extract(window))]
    split_slug = windows[int(len(windows) * .60)]["slug"]
    train = [trade for trade in trades if trade["slug"] < split_slug]
    holdout = [trade for trade in trades if trade["slug"] >= split_slug]
    print(f"Resolved windows: {len(windows)} | baseline signals: {len(trades)}")
    for group_name, group in (("First 60%", train), ("Holdout 40%", holdout),
                              ("All", trades)):
        print(group_name)
        report("  Baseline", group, lambda _: True, args.runs)
        report("  Trend guard", group, guarded, args.runs)


if __name__ == "__main__":
    main()
