"""Research-only test of value-capped momentum entries and a late stop.

This script imports the executable full-depth entry/exit model from
``_exit_strategy_oos.py``.  It never submits orders and does not modify the
live bot. Every entry and exit has an independently seeded 300--800 ms delay.
Entries use a $0.05 adverse FOK allowance, pay Polymarket's current crypto
taker-fee formula, and require enough displayed depth for the complete $2
position. Experimental exits retain their original $0.03 allowance.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import statistics
from pathlib import Path

import _exit_strategy_oos as exit_model
from _exit_strategy_oos import Config, entry_for, load, metrics, simulate_exit
from _exit_strategy_oos import Row, Window, levels
from btc5m_momentum_strategy import DEFAULT_FOK_ALLOWANCE


ROOT = Path(__file__).resolve().parent
def legacy_repo_fee(price: float, shares: float) -> float:
    """Fee approximation used by the repository before the crypto-fee update."""
    return 0.05 * min(price, 1.0 - price) * shares


LEGACY_FEE = legacy_repo_fee


def official_crypto_fee(price: float, shares: float) -> float:
    """Current Polymarket crypto taker fee: C * 0.07 * p * (1-p)."""
    return shares * 0.07 * price * (1.0 - price)


def _epoch(value):
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def load_source_aligned(path: Path):
    """Load a window using source timestamps, not the early row label.

    The logger stamps each row before its two REST calls.  A combined frame
    cannot exist until every component in it exists, so max(Up book time,
    Down book time, Coinbase exchange time) is the earliest causal timestamp
    that can be assigned to that frame.  It remains a lower bound because the
    logger did not record the final response-receive time.
    """
    rows = []
    winner = None
    window_start = None
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
            try:
                window_start = float(raw["window_start"])
                cb = float((raw.get("coinbase") or {})["price"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            source_times = [
                _epoch(up.get("book_timestamp")),
                _epoch(down.get("book_timestamp")),
                _epoch((raw.get("coinbase") or {}).get("exchange_ts")),
            ]
            source_times = [value for value in source_times if value is not None]
            if not source_times:
                continue
            t = max(source_times) - window_start
            rows.append(
                Row(
                    t,
                    cb,
                    levels(up, "bids"),
                    levels(up, "asks"),
                    levels(down, "bids"),
                    levels(down, "asks"),
                )
            )
    rows.sort(key=lambda row: row.t)
    if winner not in ("Up", "Down") or not rows or rows[0].t > 5:
        return None
    packed = tuple(rows)
    return Window(path.stem, winner, packed, tuple(row.t for row in packed))


def one_run(windows, *, cap: float, stop: float | None, seed: int):
    entries = [
        entry
        for window in windows
        if (
            entry := entry_for(
                window,
                seed,
                stake=2.0,
                entry_second=150.0,
                minimum_move_bps=2.0,
                ask_min=0.45,
                ask_max=cap,
                slippage=DEFAULT_FOK_ALLOWANCE,
            )
        )
        is not None
    ]
    cfg = (
        Config("hold")
        if stop is None
        else Config(f"sl{stop:g}_after240", sl=stop, sl_after=240.0)
    )
    return metrics([simulate_exit(entry, cfg, seed, sell_slippage=0.03) for entry in entries])


def aggregate(
    windows,
    *,
    cap: float,
    stop: float | None,
    seeds,
    fee_model=official_crypto_fee,
):
    # The imported execution model resolves fee_for from its module globals.
    # Swap it explicitly so the legacy approximation can be shown as a
    # sensitivity check without changing the shared module.
    exit_model.fee_for = fee_model
    rows = [one_run(windows, cap=cap, stop=stop, seed=seed) for seed in seeds]
    keys = (
        "trades",
        "pnl",
        "roi",
        "max_dd",
        "losses",
        "gross_loss",
        "exits",
        "exit_winners",
        "fok_failures",
    )
    result = {key: statistics.median(row[key] for row in rows) for key in keys}
    result["pnl_min"] = min(row["pnl"] for row in rows)
    result["pnl_max"] = max(row["pnl"] for row in rows)
    return result


def show(label, result):
    print(
        f"{label:<18} n={result['trades']:4.1f} "
        f"pnl=${result['pnl']:+7.2f} "
        f"range=${result['pnl_min']:+.2f}..${result['pnl_max']:+.2f} "
        f"roi={result['roi']:6.2f}% dd=${result['max_dd']:5.2f} "
        f"losses={result['losses']:3.1f} exits={result['exits']:3.1f} "
        f"exitW={result['exit_winners']:3.1f} FOKfail={result['fok_failures']:3.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--through",
        type=int,
        help="include only window starts at or before this epoch (freeze a growing log set)",
    )
    parser.add_argument(
        "--old-momentum-only",
        action="store_true",
        help="quick source-aligned replay of the original ask<=.90 hold strategy",
    )
    args = parser.parse_args()

    if args.old_momentum_only:
        aligned_windows = []
        for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
            try:
                window_epoch = int(path.stem.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if args.through is not None and window_epoch > args.through:
                continue
            aligned = load_source_aligned(path)
            if aligned is not None:
                aligned_windows.append(aligned)
        cut_60 = int(len(aligned_windows) * 0.60)
        cut_80 = int(len(aligned_windows) * 0.80)
        folds = (
            ("train60", aligned_windows[:cut_60]),
            ("valid20", aligned_windows[cut_60:cut_80]),
            ("holdout20", aligned_windows[cut_80:]),
            ("last5", aligned_windows[-5:]),
            ("all", aligned_windows),
        )
        seeds = range(args.seeds)
        print(
            f"OLD SIMPLE MOMENTUM; source-aligned resolved={len(aligned_windows)} "
            f"split={cut_60}/{cut_80-cut_60}/{len(aligned_windows)-cut_80}"
        )
        if aligned_windows:
            print(f"through={aligned_windows[-1].slug.rsplit('-', 1)[-1]}")
        print(
            "entry=t150; |Coinbase open move|>=2bps; ask=$.45-.90; "
            "stake=$2; hold to settlement"
        )
        print(
            "full depth; delay=300-800ms; "
            f"FOK allowance=${DEFAULT_FOK_ALLOWANCE:.2f}; current crypto fees"
        )
        for fold_name, fold_windows in folds:
            show(
                fold_name,
                aggregate(fold_windows, cap=0.90, stop=None, seeds=seeds),
            )
        return

    windows = []
    aligned_windows = []
    snapshots = full = 0
    for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
        try:
            window_epoch = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if args.through is not None and window_epoch > args.through:
            continue
        window, row_count, full_count = load(path)
        snapshots += row_count
        full += full_count
        if window is not None:
            windows.append(window)
        aligned = load_source_aligned(path)
        if aligned is not None:
            aligned_windows.append(aligned)

    cut_60 = int(len(windows) * 0.60)
    cut_80 = int(len(windows) * 0.80)
    folds = (
        ("train60", windows[:cut_60]),
        ("valid20", windows[cut_60:cut_80]),
        ("holdout20", windows[cut_80:]),
        ("holdout40", windows[cut_60:]),
        ("all", windows),
    )
    seeds = range(args.seeds)
    print(
        f"resolved={len(windows)} split={cut_60}/{cut_80-cut_60}/{len(windows)-cut_80}; "
        f"snapshots={snapshots}; full_depth={full} ({full/max(1, snapshots)*100:.1f}%)"
    )
    print("entry=t150, |Coinbase open move|>=2bps, ask>=.45, stake=$2")
    print(
        "buy/sell delays=uniform 300-800ms; "
        f"entry FOK allowance=${DEFAULT_FOK_ALLOWANCE:.2f}"
    )
    print("primary fee=shares*0.07*p*(1-p); legacy repo fee shown separately")

    print("\nA-PRIORI CANDIDATE: ask<=.82, hold vs -40% stop at/after t=240")
    for fold_name, fold_windows in folds:
        hold = aggregate(fold_windows, cap=0.82, stop=None, seeds=seeds)
        stopped = aggregate(fold_windows, cap=0.82, stop=40.0, seeds=seeds)
        show(f"{fold_name} hold", hold)
        show(f"{fold_name} stop", stopped)
        print(f"  stop delta=${stopped['pnl']-hold['pnl']:+.2f}\n")

    print("LEGACY-FEE SENSITIVITY FOR A-PRIORI CANDIDATE")
    for fold_name, fold_windows in folds:
        hold = aggregate(
            fold_windows, cap=0.82, stop=None, seeds=seeds, fee_model=LEGACY_FEE
        )
        stopped = aggregate(
            fold_windows, cap=0.82, stop=40.0, seeds=seeds, fee_model=LEGACY_FEE
        )
        print(
            f"  {fold_name:<9} hold ${hold['pnl']:+.2f}; "
            f"stop ${stopped['pnl']:+.2f}; delta ${stopped['pnl']-hold['pnl']:+.2f}"
        )

    print("NEIGHBORHOOD STABILITY (median PnL; research diagnostic)")
    caps = (0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.90)
    stops = (None, 30.0, 40.0, 50.0, 60.0)
    for cap in caps:
        print(f"\nask cap {cap:.2f}")
        for stop in stops:
            label = "hold" if stop is None else f"late SL {stop:g}%"
            cells = []
            for fold_name, fold_windows in folds[:3]:
                value = aggregate(fold_windows, cap=cap, stop=stop, seeds=seeds)
                cells.append(
                    f"{fold_name} ${value['pnl']:+.2f}/n{value['trades']:.0f}/dd{value['max_dd']:.2f}"
                )
            print(f"  {label:<11} " + "; ".join(cells))

    print("\nSOURCE-TIMESTAMP SENSITIVITY (earliest causal frame time)")
    aligned_cut_60 = int(len(aligned_windows) * 0.60)
    aligned_cut_80 = int(len(aligned_windows) * 0.80)
    aligned_folds = (
        ("train60", aligned_windows[:aligned_cut_60]),
        ("valid20", aligned_windows[aligned_cut_60:aligned_cut_80]),
        ("holdout20", aligned_windows[aligned_cut_80:]),
        ("all", aligned_windows),
    )
    print(
        f"aligned resolved={len(aligned_windows)} split="
        f"{aligned_cut_60}/{aligned_cut_80-aligned_cut_60}/"
        f"{len(aligned_windows)-aligned_cut_80}"
    )
    for cap, stop, name in (
        (0.90, None, "baseline hold"),
        (0.82, None, "cap82 hold"),
        (0.82, 40.0, "cap82 lateSL40"),
        (0.85, None, "cap85 hold"),
        (0.85, 40.0, "cap85 lateSL40"),
    ):
        print(f"  {name}")
        for fold_name, fold_windows in aligned_folds:
            value = aggregate(fold_windows, cap=cap, stop=stop, seeds=seeds)
            show("    " + fold_name, value)


if __name__ == "__main__":
    main()
