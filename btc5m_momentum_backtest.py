"""Backtest the Coinbase-momentum strategy on direct Polymarket window logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from btc5m_momentum_strategy import momentum_signal

ROOT = Path(__file__).resolve().parent


def load_window(path, entry_second):
    first = entry = winner = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if row.get("event") == "resolution" and row.get("winner") in ("Up", "Down"):
                winner = row["winner"]
            if row.get("event") != "snapshot":
                continue
            elapsed = row.get("seconds_into_window")
            price = (row.get("coinbase") or {}).get("price")
            if elapsed is None or price is None:
                continue
            if first is None:
                first = row
            if entry is None and float(elapsed) >= entry_second:
                entry = row
    # A bot launched mid-window cannot know the live Coinbase opening tick.
    if not first or float(first["seconds_into_window"]) > 5:
        return None
    return {"slug": path.stem, "first": first, "entry": entry, "winner": winner}


def backtest(windows, args):
    trades = []
    for window in windows:
        if not window or not window["entry"] or not window["winner"]:
            continue
        first, entry = window["first"], window["entry"]
        signal = momentum_signal((first.get("coinbase") or {}).get("price"),
                                 (entry.get("coinbase") or {}).get("price"),
                                 {"Up": entry.get("Up") or {}, "Down": entry.get("Down") or {}},
                                 args.stake, args.minimum_move_bps, args.ask_min, args.ask_max)
        if not signal:
            continue
        won = signal["side"] == window["winner"]
        pnl = (signal["shares"] if won else 0.0) - signal["cost"]
        trades.append({"slug": window["slug"], "entry_second": entry["seconds_into_window"],
                       "side": signal["side"], "winner": window["winner"], "won": won,
                       "move_bps": signal["move_bps"], "best_ask": signal["best_ask"],
                       "avg_price": signal["avg_price"], "worst_price": signal["worst_price"],
                       "shares": signal["shares"], "fee": signal["fee"], "pnl": pnl})
    return trades


def metrics(trades):
    equity = peak = max_drawdown = 0.0
    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    pnl = sum(t["pnl"] for t in trades)
    return {"trades": len(trades), "wins": sum(t["won"] for t in trades), "pnl": pnl,
            "roi": pnl / sum(t["shares"] * t["avg_price"] for t in trades) * 100 if trades else 0,
            "max_drawdown": max_drawdown}


def show(label, trades):
    m = metrics(trades)
    rate = m["wins"] / m["trades"] * 100 if m["trades"] else 0
    print(f"{label:<14} trades={m['trades']:3d} wins={m['wins']:3d} win_rate={rate:5.1f}% "
          f"pnl=${m['pnl']:8.2f} roi={m['roi']:6.2f}% max_dd=${m['max_drawdown']:7.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=ROOT / "price_logs" / "polymarket")
    parser.add_argument("--entry-second", type=float, default=150)
    parser.add_argument("--minimum-move-bps", type=float, default=2)
    parser.add_argument("--ask-min", type=float, default=.45)
    parser.add_argument("--ask-max", type=float, default=.90)
    parser.add_argument("--stake", type=float, default=100)
    parser.add_argument("--csv", type=Path, help="optional per-trade CSV output")
    args = parser.parse_args()
    paths = sorted(args.directory.glob("btc-updown-5m-*.jsonl"))
    windows = [load_window(path, args.entry_second) for path in paths]
    resolved = [window for window in windows if window and window["winner"]]
    cut = int(len(resolved) * .60)
    train, test = resolved[:cut], resolved[cut:]
    train_trades, test_trades = backtest(train, args), backtest(test, args)
    print(f"Resolved windows: {len(resolved)} (chronological split: {len(train)} / {len(test)})")
    show("First 60%", train_trades)
    show("Holdout 40%", test_trades)
    show("All", train_trades + test_trades)
    if args.csv:
        trades = train_trades + test_trades
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trades[0]) if trades else ["slug"])
            writer.writeheader(); writer.writerows(trades)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

