"""Research-only confirmation filters for the BTC 5-minute momentum strategy.

This script never places orders and does not change the live bot.  It extracts
only information available at the decision tick, then models the existing
300--800 ms FOK execution against later recorded snapshots.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
from pathlib import Path

from btc5m_momentum_strategy import fee_for, momentum_signal, sweep_asks


ROOT = Path(__file__).resolve().parent


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_windows(directory: Path, entry_second: float = 150.0):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        rows, winner = [], None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row.get("event") == "resolution" and row.get("winner") in ("Up", "Down"):
                    winner = row["winner"]
                    continue
                if row.get("event") != "snapshot":
                    continue
                elapsed = finite(row.get("seconds_into_window"))
                coinbase = finite((row.get("coinbase") or {}).get("price"))
                # Keep enough post-decision rows even when the first available
                # decision snapshot is a little late because of a feed gap.
                if elapsed is None or coinbase is None or elapsed > entry_second + 5:
                    continue
                values = {"t": elapsed, "coinbase": coinbase}
                for side in ("Up", "Down"):
                    book = row.get(side) or {}
                    for key in ("bid", "bid_size", "ask", "ask_size", "spread"):
                        values[f"{side}_{key}"] = finite(book.get(key))
                rows.append(values)
        if winner in ("Up", "Down") and rows and rows[0]["t"] <= 5:
            windows.append({"slug": path.stem, "winner": winner, "rows": rows})
    return windows


def prior_row(rows, target, upper_index):
    times = [r["t"] for r in rows]
    index = bisect.bisect_right(times, target, 0, upper_index + 1) - 1
    return rows[max(0, index)]


def signed(value, side):
    return value if side == "Up" else -value


def extract(window, stake=2.0, entry_second=150.0):
    rows = window["rows"]
    times = [r["t"] for r in rows]
    decision_index = bisect.bisect_left(times, entry_second)
    if decision_index >= len(rows):
        return None
    decision = rows[decision_index]
    first = rows[0]
    books = {
        side: {key: decision.get(f"{side}_{key}") for key in
               ("bid", "bid_size", "ask", "ask_size", "spread")}
        for side in ("Up", "Down")
    }
    signal = momentum_signal(first["coinbase"], decision["coinbase"], books,
                             stake, 2, .45, .90)
    if not signal:
        return None
    side = signal["side"]
    direction = 1 if side == "Up" else -1
    open_price = first["coinbase"]
    features = {
        "abs_move": abs(signal["move_bps"]),
        "ask": signal["best_ask"],
        "bid": decision.get(f"{side}_bid"),
        "spread": decision.get(f"{side}_spread"),
        "ask_size_dollars": (decision.get(f"{side}_ask_size") or 0) * signal["best_ask"],
        "book_imbalance": None,
    }
    bid_size = decision.get(f"{side}_bid_size")
    ask_size = decision.get(f"{side}_ask_size")
    if bid_size is not None and ask_size is not None and bid_size + ask_size > 0:
        features["book_imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)

    # Direction-aligned Coinbase path features.  Positive means continuation.
    for lookback in (1, 2, 3, 5, 10, 20, 30, 60, 90, 120):
        old = prior_row(rows, decision["t"] - lookback, decision_index)
        recent = (decision["coinbase"] / old["coinbase"] - 1) * 10_000
        features[f"recent_{lookback}"] = direction * recent

    # Direction-aligned market-price confirmation and its recent change.
    current_mid = None
    bid, ask = decision.get(f"{side}_bid"), decision.get(f"{side}_ask")
    if bid is not None and ask is not None:
        current_mid = (bid + ask) / 2
    features["mid"] = current_mid
    for lookback in (3, 5, 10, 20, 30, 60):
        old = prior_row(rows, decision["t"] - lookback, decision_index)
        old_bid, old_ask = old.get(f"{side}_bid"), old.get(f"{side}_ask")
        old_mid = (old_bid + old_ask) / 2 if old_bid is not None and old_ask is not None else None
        features[f"mid_change_{lookback}"] = (
            current_mid - old_mid if current_mid is not None and old_mid is not None else None
        )

    # Sample the path at one-second intervals, never beyond decision time.
    samples = [prior_row(rows, decision["t"] - age, decision_index)
               for age in range(0, 121)]
    aligned_moves = [direction * (r["coinbase"] / open_price - 1) * 10_000 for r in samples]
    features["same_side_fraction_30"] = sum(x > 0 for x in aligned_moves[:31]) / 31
    features["same_side_fraction_60"] = sum(x > 0 for x in aligned_moves[:61]) / 61
    features["same_side_fraction_120"] = sum(x > 0 for x in aligned_moves) / len(aligned_moves)
    features["pullback_30"] = max(aligned_moves[:31]) - aligned_moves[0]
    features["pullback_60"] = max(aligned_moves[:61]) - aligned_moves[0]
    increments = [aligned_moves[i] - aligned_moves[i + 1] for i in range(len(aligned_moves) - 1)]
    features["positive_step_fraction_30"] = sum(x > 0 for x in increments[:30]) / 30
    features["positive_step_fraction_60"] = sum(x > 0 for x in increments[:60]) / 60
    features["volatility_30"] = statistics.pstdev(increments[:30])
    features["volatility_60"] = statistics.pstdev(increments[:60])

    return {
        "slug": window["slug"], "winner": window["winner"], "won": side == window["winner"],
        "side": side, "decision_index": decision_index, "decision_t": decision["t"],
        "decision_worst": signal["worst_price"], "rows": rows, "features": features,
    }


def execution_result(trade, run=0, stake=2.0, slippage=.03):
    # Match btc5m_momentum_backtest.py exactly: seed 42, run number, slug.
    rng = random.Random(f"42:{run}:{trade['slug']}")
    target = trade["decision_t"] + rng.uniform(.3, .8)
    rows = trade["rows"]
    times = [r["t"] for r in rows]
    index = bisect.bisect_left(times, target, trade["decision_index"] + 1)
    if index >= len(rows):
        return None
    row, side = rows[index], trade["side"]
    ask, size = row.get(f"{side}_ask"), row.get(f"{side}_ask_size")
    limit = min(.99, trade["decision_worst"] + slippage)
    if ask is None or size is None or ask > limit + 1e-12:
        return None
    fill = sweep_asks({"ask": ask, "ask_size": size}, stake)
    if not fill:
        return None
    return (fill["shares"] if trade["won"] else 0) - fill["cost"]


def summarize(trades, predicate=lambda _: True, runs=range(100)):
    results = []
    for run in runs:
        pnls = []
        for trade in trades:
            if not predicate(trade["features"]):
                continue
            pnl = execution_result(trade, run)
            if pnl is not None:
                pnls.append(pnl)
        equity = peak = drawdown = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        results.append((sum(pnls), len(pnls), drawdown))
    return {
        "pnl": statistics.median(r[0] for r in results),
        "pnl_min": min(r[0] for r in results),
        "pnl_max": max(r[0] for r in results),
        "trades": statistics.mean(r[1] for r in results),
        "max_dd": statistics.median(r[2] for r in results),
    }


def print_metric(label, metric):
    print(f"{label:<28} trades={metric['trades']:6.1f} pnl=${metric['pnl']:7.2f} "
          f"range=${metric['pnl_min']:7.2f}..${metric['pnl_max']:7.2f} "
          f"median_dd=${metric['max_dd']:6.2f}")


def candidate_report(folds):
    candidates = []
    for key in ("recent_3", "recent_5", "recent_10", "recent_20", "recent_30"):
        for threshold in (-2, -1.5, -1, -.75, -.5, -.25, 0, .25, .5, 1):
            candidates.append((f"{key}>={threshold:g}",
                               lambda f, k=key, x=threshold: f[k] >= x))
    for key in ("mid_change_3", "mid_change_5", "mid_change_10", "mid_change_20", "mid_change_30"):
        for threshold in (-.05, -.03, -.02, -.01, 0, .01, .02, .03, .05):
            candidates.append((f"{key}>={threshold:g}",
                               lambda f, k=key, x=threshold: f[k] is not None and f[k] >= x))
    for threshold in (.5, .55, .6, .65, .7, .75, .8, .85):
        candidates.append((f"step30>={threshold:g}",
                           lambda f, x=threshold: f["positive_step_fraction_30"] >= x))
    # A reversal veto is deliberately softer than requiring both feeds to be
    # positive: skip only when Coinbase and the contract price both weaken.
    for cb in (-1, -.5, 0, .5):
        for market in (-.03, -.02, -.01, 0, .01):
            candidates.append((f"veto cb10<{cb:g}&mid10<{market:g}",
                               lambda f, c=cb, m=market:
                               not (f["recent_10"] < c and f["mid_change_10"] < m)))

    # Rank solely on the first 60%, with a minimum of half the baseline trades.
    train_name, train = folds[0]
    ranked = []
    for label, predicate in candidates:
        m = summarize(train, predicate, runs=range(30))
        if m["trades"] >= 50:
            ranked.append((m["pnl"], label, predicate, m))
    ranked.sort(reverse=True, key=lambda row: row[0])
    print("\nTop single rules ranked on train60 only:")
    for _, label, predicate, train_metric in ranked[:20]:
        cells = [f"{name} ${summarize(data, predicate, runs=range(30))['pnl']:+.2f}"
                 for name, data in folds[1:]]
        print(f"  {label:<34} train ${train_metric['pnl']:+.2f} n={train_metric['trades']:.0f}; "
              + "; ".join(cells))

    print("\nEconomically motivated confirmations:")
    selected = [
        ("baseline", lambda f: True),
        ("coinbase 10s not reversing", lambda f: f["recent_10"] >= 0),
        ("coinbase 20s not reversing", lambda f: f["recent_20"] >= 0),
        ("market 10s not weakening", lambda f: f["mid_change_10"] >= 0),
        ("market 20s not weakening", lambda f: f["mid_change_20"] >= 0),
        ("veto both weakening",
         lambda f: not (f["recent_10"] < 0 and f["mid_change_10"] < 0)),
        ("veto Coinbase<0 & market<-1c",
         lambda f: not (f["recent_10"] < 0 and f["mid_change_10"] < -.01)),
        ("veto strong reversal",
         lambda f: not (f["recent_10"] < -.5 and f["mid_change_10"] < -.01)),
        ("confirm either feed",
         lambda f: f["recent_10"] >= 0 or f["mid_change_10"] >= 0),
        ("confirm both feeds",
         lambda f: f["recent_10"] >= 0 and f["mid_change_10"] >= 0),
    ]
    for label, predicate in selected:
        print(f"\n  {label}")
        for name, data in folds + [("all", sum((d for _, d in folds), []))]:
            print_metric("    " + name, summarize(data, predicate))

    rule = lambda f: not (f["recent_10"] < 0 and f["mid_change_10"] < -.01)
    print("\nSix chronological block check (baseline -> reversal veto):")
    all_trades = sum((data for _, data in folds), [])
    for index in range(6):
        lo = index * len(all_trades) // 6
        hi = (index + 1) * len(all_trades) // 6
        block = all_trades[lo:hi]
        base, filtered = summarize(block), summarize(block, rule)
        print(f"  block {index + 1}: ${base['pnl']:+.2f} -> ${filtered['pnl']:+.2f} "
              f"(delta ${filtered['pnl'] - base['pnl']:+.2f}, "
              f"trades {base['trades']:.0f}->{filtered['trades']:.0f})")

    print("\nRun-0 fill outcomes (for audit, not the delay aggregate):")
    for label, predicate in (("baseline", lambda f: True), ("veto", rule),
                             ("excluded", lambda f: not rule(f))):
        executed = [(trade, execution_result(trade, 0)) for trade in all_trades
                    if predicate(trade["features"])]
        executed = [(trade, pnl) for trade, pnl in executed if pnl is not None]
        wins = sum(trade["won"] for trade, _ in executed)
        pnl = sum(value for _, value in executed)
        cost = sum(2 + fee_for(trade["features"]["ask"], 2 / trade["features"]["ask"])
                   for trade, _ in executed)
        print(f"  {label:<8} n={len(executed):3d} W/L={wins}/{len(executed)-wins} "
              f"win_rate={wins/len(executed)*100:5.1f}% pnl=${pnl:+7.2f} "
              f"roi={pnl/cost*100:5.1f}%")


def quantile_bins(trades, feature, folds):
    values = sorted(t["features"][feature] for t in trades if t["features"].get(feature) is not None)
    if not values:
        return
    cuts = [values[int((len(values) - 1) * q / 5)] for q in range(1, 5)]
    print(f"\n{feature} cuts: " + ", ".join(f"{x:.4g}" for x in cuts))
    bounds = [-math.inf, *cuts, math.inf]
    for name, data in folds:
        cells = []
        for lo, hi in zip(bounds, bounds[1:]):
            subset = [t for t in data if (v := t["features"].get(feature)) is not None and lo <= v <= hi]
            m = summarize(subset, runs=range(20))
            cells.append(f"n={m['trades']:.0f} p={m['pnl']:+.1f}")
        print(f"  {name:<10} " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "price_logs" / "polymarket")
    parser.add_argument("--features", action="store_true", help="print feature quintile diagnostics")
    parser.add_argument("--candidates", action="store_true", help="evaluate predeclared confirmation rules")
    args = parser.parse_args()
    windows = load_windows(args.directory)
    trades = [trade for window in windows if (trade := extract(window))]
    cut1, cut2 = int(len(windows) * .60), int(len(windows) * .80)
    slug1, slug2 = windows[cut1]["slug"], windows[cut2]["slug"]
    train = [t for t in trades if t["slug"] < slug1]
    validation = [t for t in trades if slug1 <= t["slug"] < slug2]
    holdout = [t for t in trades if t["slug"] >= slug2]
    folds = [("train60", train), ("valid20", validation), ("holdout20", holdout)]
    print(f"Resolved windows={len(windows)}, baseline signals={len(trades)}, split slugs={slug1}, {slug2}")
    for name, data in folds + [("all", trades)]:
        print_metric(name, summarize(data))
    if args.features:
        keys = list(trades[0]["features"])
        for key in keys:
            quantile_bins(train, key, folds)
    if args.candidates:
        candidate_report(folds)


if __name__ == "__main__":
    main()
