"""Causal research for a late BTC 5-minute safety-margin entry.

This is research-only.  It never submits orders and never changes the live
bot.  A signal follows the sign of Coinbase's move from the first recorded
price, but only near expiry and only when the absolute move clears a fixed
margin.  The entry is filled against the first later recorded Polymarket book
after a deterministic 300--800 ms delay, using full displayed depth, a $0.05
FOK allowance, a fixed $2 stake, and the current crypto taker fee.

The small parameter grid is ranked on two halves of training.  Validation may
reject or select among the five training finalists.  The final chronological
holdout is not evaluated unless a candidate is positive in both training
halves and validation.  This avoids using the last period to rescue a rule.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
import random
import statistics

from _exit_strategy_oos import sweep_buy
from _research_value_stop import load_source_aligned


ROOT = Path(__file__).resolve().parent
FROZEN_THROUGH = 1792066800
STAKE = 2.0
ALLOWANCE = 0.05


@dataclass(frozen=True)
class Config:
    second: float
    move_bps: float
    ask_cap: float

    @property
    def name(self) -> str:
        return f"t{self.second:g}_move{self.move_bps:g}_cap{self.ask_cap:.2f}"


def one(window, cfg: Config, seed: int):
    """Return an attempted-trade record, or None when there is no signal."""
    decision_i = bisect.bisect_left(window.times, cfg.second)
    if decision_i >= len(window.rows):
        return None
    decision = window.rows[decision_i]
    opening = window.rows[0].coinbase
    move = (decision.coinbase / opening - 1.0) * 10_000.0
    if abs(move) < cfg.move_bps:
        return None
    side = "Up" if move > 0 else "Down"
    asks = decision.up_asks if side == "Up" else decision.down_asks
    if not asks or not 0.45 <= asks[0][0] <= cfg.ask_cap:
        return None
    quote = sweep_buy(asks, STAKE)
    if quote is None:
        return {"attempt": 1, "fill": 0, "pnl": 0.0, "cost": 0.0,
                "win": 0, "slug": window.slug}
    order_limit = min(0.99, quote[2] + ALLOWANCE)
    rng = random.Random(f"near-expiry:{seed}:{window.slug}")
    target = decision.t + rng.uniform(0.300, 0.800)
    execution_i = bisect.bisect_left(
        window.times, target, lo=decision_i + 1
    )
    if execution_i >= len(window.rows) or window.rows[execution_i].t >= 300.0:
        return {"attempt": 1, "fill": 0, "pnl": 0.0, "cost": 0.0,
                "win": 0, "slug": window.slug}
    execution = window.rows[execution_i]
    exec_asks = execution.up_asks if side == "Up" else execution.down_asks
    fill = sweep_buy(exec_asks, STAKE, order_limit)
    if fill is None:
        return {"attempt": 1, "fill": 0, "pnl": 0.0, "cost": 0.0,
                "win": 0, "slug": window.slug}
    shares, cost, worst = fill
    won = side == window.winner
    return {
        "attempt": 1,
        "fill": 1,
        "pnl": (shares if won else 0.0) - cost,
        "cost": cost,
        "win": int(won),
        "slug": window.slug,
        "side": side,
        "decision_t": decision.t,
        "execution_t": execution.t,
        "move_bps": move,
        "avg_price": STAKE / shares,
        "worst_price": worst,
    }


def run(windows, cfg: Config, seed: int):
    attempts = [row for window in windows if (row := one(window, cfg, seed))]
    fills = [row for row in attempts if row["fill"]]
    equity = peak = max_dd = 0.0
    for row in fills:
        equity += row["pnl"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    cost = sum(row["cost"] for row in fills)
    return {
        "attempts": len(attempts),
        "trades": len(fills),
        "fok": len(attempts) - len(fills),
        "wins": sum(row["win"] for row in fills),
        "pnl": sum(row["pnl"] for row in fills),
        "roi": 100.0 * sum(row["pnl"] for row in fills) / cost if cost else 0.0,
        "dd": max_dd,
    }


def aggregate(windows, cfg: Config, seeds):
    rows = [run(windows, cfg, seed) for seed in seeds]
    result = {
        key: statistics.median(row[key] for row in rows)
        for key in ("attempts", "trades", "fok", "wins", "pnl", "roi", "dd")
    }
    result["pnl_min"] = min(row["pnl"] for row in rows)
    result["pnl_max"] = max(row["pnl"] for row in rows)
    return result


def show(label, cfg, result):
    print(
        f"{label:<14} {cfg.name:<24} n={result['trades']:5.1f} "
        f"win={result['wins']:5.1f} fok={result['fok']:4.1f} "
        f"pnl=${result['pnl']:+7.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:+6.2f}% dd=${result['dd']:.2f}"
    )


def load_windows(directory: Path, through: int):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if start > through:
            continue
        window = load_source_aligned(path)
        if window is not None:
            windows.append(window)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=ROOT / "price_logs" / "polymarket")
    parser.add_argument("--through", type=int, default=FROZEN_THROUGH)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--search-seeds", type=int, default=20)
    args = parser.parse_args()

    windows = load_windows(args.directory, args.through)
    cut50 = len(windows) // 2
    cut75 = len(windows) * 3 // 4
    train_a = windows[:cut50 // 2]
    train_b = windows[cut50 // 2:cut50]
    validation = windows[cut50:cut75]
    holdout = windows[cut75:]
    print(
        f"NEAR-EXPIRY RESEARCH frozen_through={args.through}; "
        f"source-aligned resolved={len(windows)}; "
        f"train halves={len(train_a)}/{len(train_b)} "
        f"validation={len(validation)} untouched_holdout={len(holdout)}"
    )
    print("$2; full depth; 300-800ms; FOK allowance=$0.05; current taker fee")
    if not windows:
        return

    search_seeds = range(args.search_seeds)
    configs = [
        Config(second, move, cap)
        for second in (220.0, 240.0, 260.0, 275.0, 285.0, 290.0)
        for move in (3.0, 5.0, 8.0, 12.0, 20.0)
        for cap in (0.90, 0.95, 0.99)
    ]
    ranked = []
    for cfg in configs:
        a = aggregate(train_a, cfg, search_seeds)
        b = aggregate(train_b, cfg, search_seeds)
        if min(a["trades"], b["trades"]) < 3:
            continue
        stability = min(a["pnl"], b["pnl"])
        ranked.append((stability, a["pnl"] + b["pnl"], cfg, a, b))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    finalists = ranked[:5]
    print("\nTraining finalists (grid ranked without validation/holdout):")
    if not finalists:
        print("  none with at least three trades in each training half")
        return
    for _, _, cfg, a, b in finalists:
        show("train-A", cfg, a)
        show("train-B", cfg, b)

    passing = []
    print("\nValidation of training finalists:")
    for _, _, cfg, a, b in finalists:
        valid = aggregate(validation, cfg, search_seeds)
        show("validation", cfg, valid)
        if a["pnl"] > 0 and b["pnl"] > 0 and valid["pnl"] > 0:
            passing.append((min(a["pnl"], b["pnl"], valid["pnl"]), cfg))
    if not passing:
        print("\nREJECT: no training finalist passed validation; holdout not opened.")
        return

    passing.sort(reverse=True, key=lambda row: row[0])
    selected = passing[0][1]
    seeds = range(args.seeds)
    print(f"\nSelected before holdout: {selected.name}")
    for label, fold in (
        ("train-A", train_a),
        ("train-B", train_b),
        ("validation", validation),
        ("HOLDOUT", holdout),
        ("all", windows),
    ):
        show(label, selected, aggregate(fold, selected, seeds))


if __name__ == "__main__":
    main()
