"""Backtest the experimental multi-entry BTC 5m take-profit strategy."""

import argparse, bisect, json, random, statistics
from pathlib import Path
from btc5m_momentum_strategy import fee_for

ROOT = Path(__file__).resolve().parent


def load(path):
    rows, winner = [], None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try: row = json.loads(line)
            except (ValueError, UnicodeDecodeError): continue
            if row.get("event") == "resolution": winner = row.get("winner"); continue
            if row.get("event") != "snapshot": continue
            cb = (row.get("coinbase") or {}).get("price")
            up, down = row.get("Up") or {}, row.get("Down") or {}
            if row.get("seconds_into_window") is None or cb is None: continue
            rows.append((float(row["seconds_into_window"]), float(cb),
                         up.get("bid"), up.get("bid_size"), up.get("ask"), up.get("ask_size"),
                         down.get("bid"), down.get("bid_size"), down.get("ask"), down.get("ask_size")))
    return (path.stem, winner, rows) if winner in ("Up", "Down") and rows and rows[0][0] <= 5 else None


def simulate(windows, args, seed):
    results = []
    for slug, winner, rows in windows:
        times = [row[0] for row in rows]; opening = rows[0][1]
        position = None; entries = 0; next_entry = args.entry_second; index = 0
        rng = random.Random(f"{args.seed + seed}:{slug}")
        while index < len(rows):
            row = rows[index]; elapsed, coinbase = row[0], row[1]
            if position:
                side, shares, cost, fill = position
                bid_i, size_i = (2, 3) if side == "Up" else (6, 7)
                bid, size = row[bid_i], row[size_i]
                # TP is net position return: entry fee is already in cost and
                # the prospective exit fee is deducted from proceeds.
                net_at_bid = (shares * bid - fee_for(bid, shares)) if bid is not None else None
                if net_at_bid is not None and net_at_bid >= cost * (1 + args.take_profit / 100):
                    delay = rng.uniform(args.delay_min_ms, args.delay_max_ms) / 1000
                    later = bisect.bisect_left(times, elapsed + delay, index + 1)
                    if later < len(rows):
                        delayed_bid, delayed_size = rows[later][bid_i], rows[later][size_i]
                        delayed_net = (shares * delayed_bid - fee_for(delayed_bid, shares)) if delayed_bid is not None else None
                        if delayed_net is not None and delayed_net >= cost * (1 + args.take_profit / 100) and delayed_size is not None and delayed_size >= shares:
                            pnl = delayed_net - cost
                            results.append({"slug": slug, "side": side, "exit": "take_profit", "pnl": pnl})
                            position = None; next_entry = rows[later][0] + args.cooldown
                            index = later; continue
            elif entries < args.max_trades and next_entry <= elapsed < 290:
                move = (coinbase / opening - 1) * 10_000
                if abs(move) >= args.minimum_move_bps:
                    side = "Up" if move > 0 else "Down"
                    ask_i, size_i = (4, 5) if side == "Up" else (8, 9)
                    ask, size = row[ask_i], row[size_i]
                    if ask is not None and args.ask_min <= ask <= args.ask_max and size is not None and ask * size >= args.stake:
                        delay = rng.uniform(args.delay_min_ms, args.delay_max_ms) / 1000
                        later = bisect.bisect_left(times, elapsed + delay, index + 1)
                        if later < len(rows):
                            delayed_ask, delayed_size = rows[later][ask_i], rows[later][size_i]
                            if delayed_ask is not None and delayed_ask <= ask and delayed_size is not None and delayed_ask * delayed_size >= args.stake:
                                shares = args.stake / delayed_ask
                                position = (side, shares, args.stake + fee_for(delayed_ask, shares), delayed_ask)
                                entries += 1; index = later; continue
            index += 1
        if position:
            side, shares, cost, _ = position
            results.append({"slug": slug, "side": side, "exit": "settlement",
                            "pnl": (shares if side == winner else 0) - cost})
    return results


def summary(results):
    pnl = sum(x["pnl"] for x in results); stake = len(results) * 2
    return pnl, len(results), sum(x["pnl"] > 0 for x in results), sum(x["exit"] == "take_profit" for x in results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", type=Path, default=ROOT/"price_logs"/"polymarket")
    p.add_argument("--stake", type=float, default=2); p.add_argument("--entry-second", type=float, default=90)
    p.add_argument("--minimum-move-bps", type=float, default=6); p.add_argument("--take-profit", type=float, default=10,
                    help="minimum net position return after entry and exit fees")
    p.add_argument("--cooldown", type=float, default=10); p.add_argument("--max-trades", type=int, default=2)
    p.add_argument("--ask-min", type=float, default=.45); p.add_argument("--ask-max", type=float, default=.90)
    p.add_argument("--delay-min-ms", type=float, default=300); p.add_argument("--delay-max-ms", type=float, default=800)
    p.add_argument("--runs", type=int, default=100); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    windows = [w for w in (load(x) for x in sorted(args.directory.glob("btc-updown-5m-*.jsonl"))) if w]
    cut = int(len(windows)*.6); groups = (("First 60%", windows[:cut]), ("Holdout 40%", windows[cut:]), ("All", windows))
    print(f"Resolved windows: {len(windows)} | stake ${args.stake:g} | max {args.max_trades} entries/window")
    for label, group in groups:
        runs = [summary(simulate(group, args, seed)) for seed in range(args.runs)]
        pnls = [x[0] for x in runs]; sample = runs[0]
        print(f"{label:<12} median=${statistics.median(pnls):6.2f} range=${min(pnls):6.2f} to ${max(pnls):6.2f} "
              f"trades~{statistics.mean(x[1] for x in runs):.1f} TP~{statistics.mean(x[3] for x in runs):.1f}")

if __name__ == "__main__": main()

