"""Backtest the Coinbase-momentum strategy on direct Polymarket window logs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from btc5m_momentum_strategy import (
    DEFAULT_ASK_MAX,
    DEFAULT_ASK_MIN,
    DEFAULT_ENTRY_SECOND,
    DEFAULT_FOK_ALLOWANCE,
    DEFAULT_MINIMUM_MOVE_BPS,
    DEFAULT_STAKE,
    momentum_signal,
    sweep_asks,
)

ROOT = Path(__file__).resolve().parent


def load_window(path, entry_second, max_delay_seconds):
    first = decision = winner = None
    execution_rows = []
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
            elapsed = float(elapsed)
            if decision is None and elapsed >= entry_second:
                decision = row
            if (decision is not None
                    and float(decision["seconds_into_window"]) <= elapsed
                    <= float(decision["seconds_into_window"]) + max_delay_seconds + 1):
                execution_rows.append(row)
    # A bot launched mid-window cannot know the live Coinbase opening tick.
    if not first or float(first["seconds_into_window"]) > 5:
        return None
    return {"slug": path.stem, "first": first, "decision": decision,
            "execution_rows": execution_rows, "winner": winner}


def delay_ms_for(slug, args, run):
    rng = random.Random(f"{args.seed}:{run}:{slug}")
    return rng.uniform(args.delay_min_ms, args.delay_max_ms)


def backtest(windows, args, run=0):
    trades = []
    for window in windows:
        if not window or not window["decision"] or not window["winner"]:
            continue
        first, decision = window["first"], window["decision"]
        decision_signal = momentum_signal((first.get("coinbase") or {}).get("price"),
                                          (decision.get("coinbase") or {}).get("price"),
                                          {"Up": decision.get("Up") or {},
                                           "Down": decision.get("Down") or {}},
                                          args.stake, args.minimum_move_bps,
                                          args.ask_min, args.ask_max)
        if not decision_signal:
            continue
        delay_ms = delay_ms_for(window["slug"], args, run)
        execution_target = float(decision["seconds_into_window"]) + delay_ms / 1000
        entry = next((row for row in window["execution_rows"]
                      if float(row["seconds_into_window"]) >= execution_target), None)
        if entry is None:
            continue
        # A real FOK order is signed with the decision-time worst fill as its
        # limit. Later liquidity above that limit cannot be chased.
        side = decision_signal["side"]
        book = dict(entry.get(side) or {})
        order_limit = min(.99, decision_signal["worst_price"] + args.slippage)
        recorded_asks = book.get("asks") or []
        if recorded_asks:
            book["asks"] = [level for level in recorded_asks
                            if float(level["price"]) <= order_limit + 1e-12]
            # Do not let sweep_asks fall back to an unfiltered top-of-book ask
            # when every recorded level is above the signed FOK limit.
            if not book["asks"]:
                book["ask"] = book["ask_size"] = None
        elif (book.get("ask") is not None
              and float(book["ask"]) > order_limit + 1e-12):
            # Compact logger rows contain only the top level. A real FOK would
            # be killed here rather than chase that ask above its limit.
            book["ask"] = book["ask_size"] = None
        fill = sweep_asks(book, args.stake)
        if not fill:
            continue
        signal = {**decision_signal, **fill, "side": side, "order_limit": order_limit}
        won = signal["side"] == window["winner"]
        pnl = (signal["shares"] if won else 0.0) - signal["cost"]
        trades.append({"slug": window["slug"],
                       "signal_second": decision["seconds_into_window"],
                       "delay_ms": delay_ms, "entry_second": entry["seconds_into_window"],
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
    parser.add_argument("--entry-second", type=float, default=DEFAULT_ENTRY_SECOND)
    parser.add_argument("--minimum-move-bps", type=float,
                        default=DEFAULT_MINIMUM_MOVE_BPS)
    parser.add_argument("--ask-min", type=float, default=DEFAULT_ASK_MIN)
    parser.add_argument("--ask-max", type=float, default=DEFAULT_ASK_MAX)
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE)
    parser.add_argument("--slippage", type=float, default=DEFAULT_FOK_ALLOWANCE,
                        help="extra dollars/share allowed above the decision-time FOK limit")
    parser.add_argument("--delay-min-ms", type=float, default=300)
    parser.add_argument("--delay-max-ms", type=float, default=800)
    parser.add_argument("--runs", type=int, default=25,
                        help="number of deterministic random-delay simulations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, help="optional per-trade CSV output")
    args = parser.parse_args()
    if not 0 <= args.delay_min_ms <= args.delay_max_ms or args.runs < 1 or not 0 <= args.slippage < 1:
        raise SystemExit("require 0 <= delay-min-ms <= delay-max-ms and runs >= 1")
    paths = sorted(args.directory.glob("btc-updown-5m-*.jsonl"))
    windows = [load_window(path, args.entry_second, args.delay_max_ms / 1000) for path in paths]
    resolved = [window for window in windows if window and window["winner"]]
    cut = int(len(resolved) * .60)
    train, test = resolved[:cut], resolved[cut:]
    train_trades, test_trades = backtest(train, args), backtest(test, args)
    print(f"Resolved windows: {len(resolved)} (chronological split: {len(train)} / {len(test)})")
    show("First 60%", train_trades)
    show("Holdout 40%", test_trades)
    show("All", train_trades + test_trades)
    print(f"Execution delay: uniform {args.delay_min_ms:.0f}-{args.delay_max_ms:.0f} ms "
          f"({args.runs} deterministic simulations, seed {args.seed})")
    if args.runs > 1:
        train_pnls, test_pnls, all_pnls = [], [], []
        for run in range(args.runs):
            a, b = backtest(train, args, run), backtest(test, args, run)
            train_pnls.append(metrics(a)["pnl"]); test_pnls.append(metrics(b)["pnl"])
            all_pnls.append(metrics(a + b)["pnl"])
        for label, values in (("First 60%", train_pnls), ("Holdout 40%", test_pnls),
                              ("All", all_pnls)):
            print(f"{label:<14} delay PnL median=${statistics.median(values):8.2f} "
                  f"range=${min(values):8.2f} to ${max(values):8.2f}")
    if args.csv:
        trades = train_trades + test_trades
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trades[0]) if trades else ["slug"])
            writer.writeheader(); writer.writerows(trades)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

