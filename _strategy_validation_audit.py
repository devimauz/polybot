"""Research-only robustness audit for BTC5m Polymarket logs.

This intentionally does not touch the live bot.  Candidate rules are a small,
predeclared family and are evaluated in chronological window blocks with the
same delayed FOK-style fill model as the momentum backtest.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from btc5m_momentum_strategy import fee_for


@dataclass(frozen=True)
class Config:
    name: str
    entry: float = 150.0
    min_move: float = 2.0
    ask_min: float = 0.45
    ask_max: float = 0.90
    agreement: str = "any"  # any, favorite, disagreement
    max_spread: float | None = None
    min_recent_10: float | None = None
    side_rule: str = "momentum"  # momentum, market_favorite, market_underdog


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clean_levels(book, key):
    result = []
    for level in book.get(key) or []:
        price = finite(level.get("price"))
        size = finite(level.get("size"))
        if price is not None and size is not None and 0 < price < 1 and size > 0:
            result.append((price, size))
    if not result:
        prefix = "ask" if key == "asks" else "bid"
        price, size = finite(book.get(prefix)), finite(book.get(prefix + "_size"))
        if price is not None and size is not None and 0 < price < 1 and size > 0:
            result.append((price, size))
    return sorted(result, reverse=key == "bids")


def load(directory):
    windows = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        rows, winner = [], None
        for line in path.open(encoding="utf-8"):
            try:
                raw = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if raw.get("event") == "resolution" and raw.get("winner") in ("Up", "Down"):
                winner = raw["winner"]
            if raw.get("event") != "snapshot":
                continue
            t = finite(raw.get("seconds_into_window"))
            cb = finite((raw.get("coinbase") or {}).get("price"))
            if t is None or cb is None:
                continue
            row = {"t": t, "cb": cb}
            for side in ("Up", "Down"):
                book = raw.get(side) or {}
                row[side] = {
                    "ask": finite(book.get("ask")),
                    "bid": finite(book.get("bid")),
                    "spread": finite(book.get("spread")),
                    "asks": clean_levels(book, "asks"),
                }
            rows.append(row)
        rows.sort(key=lambda row: row["t"])
        if winner in ("Up", "Down") and rows and rows[0]["t"] <= 5:
            windows.append({
                "slug": path.stem,
                "winner": winner,
                "rows": rows,
                "times": [row["t"] for row in rows],
            })
    return windows


def sweep(levels, dollars, limit):
    remaining, shares, fee = dollars, 0.0, 0.0
    worst = None
    for price, size in levels:
        if price > limit + 1e-12:
            continue
        used = min(remaining, price * size)
        qty = used / price
        shares += qty
        fee += fee_for(price, qty)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return {"shares": shares, "cost": dollars + fee, "worst": worst,
                    "avg": dollars / shares}
    return None


def prior(rows, times, target, upper):
    index = bisect.bisect_right(times, target, 0, upper + 1) - 1
    return rows[max(0, index)]


def decision_for(window, cfg):
    rows, times = window["rows"], window["times"]
    index = bisect.bisect_left(times, cfg.entry)
    if index >= len(rows):
        return None
    row, opening = rows[index], rows[0]["cb"]
    move = (row["cb"] / opening - 1) * 10_000
    if abs(move) < cfg.min_move:
        return None
    side = "Up" if move > 0 else "Down"
    if cfg.side_rule != "momentum":
        up_ask, down_ask = row["Up"]["ask"], row["Down"]["ask"]
        if up_ask is None or down_ask is None or up_ask == down_ask:
            return None
        favorite = "Up" if up_ask > down_ask else "Down"
        side = favorite if cfg.side_rule == "market_favorite" else (
            "Down" if favorite == "Up" else "Up")
    other = "Down" if side == "Up" else "Up"
    book = row[side]
    ask = book["ask"]
    if ask is None or not cfg.ask_min <= ask <= cfg.ask_max:
        return None
    other_ask = row[other]["ask"]
    favorite = other_ask is not None and ask >= other_ask
    if cfg.agreement == "favorite" and not favorite:
        return None
    if cfg.agreement == "disagreement" and favorite:
        return None
    if cfg.max_spread is not None:
        spread = book["spread"]
        if spread is None or spread > cfg.max_spread + 1e-12:
            return None
    direction = 1 if side == "Up" else -1
    old = prior(rows, times, row["t"] - 10, index)
    recent = direction * (row["cb"] / old["cb"] - 1) * 10_000
    if cfg.min_recent_10 is not None and recent < cfg.min_recent_10:
        return None
    quoted = sweep(book["asks"], 2.0, 1.0)
    if quoted is None:
        return None
    return {"index": index, "t": row["t"], "side": side, "move": move,
            "ask": ask, "recent": recent, "decision_worst": quoted["worst"],
            "won": side == window["winner"]}


def execute(window, decision, run):
    rng = random.Random(f"audit:42:{run}:{window['slug']}")
    target = decision["t"] + rng.uniform(0.300, 0.800)
    index = bisect.bisect_left(window["times"], target, decision["index"] + 1)
    if index >= len(window["rows"]):
        return None
    book = window["rows"][index][decision["side"]]
    limit = min(0.99, decision["decision_worst"] + 0.03)
    fill = sweep(book["asks"], 2.0, limit)
    if fill is None:
        return None
    pnl = (fill["shares"] if decision["won"] else 0.0) - fill["cost"]
    return {**decision, "pnl": pnl, "entry_avg": fill["avg"],
            "entry_t": window["rows"][index]["t"]}


def simulate(windows, cfg, run):
    trades = []
    for window in windows:
        decision = decision_for(window, cfg)
        if decision is None:
            continue
        trade = execute(window, decision, run)
        if trade is not None:
            trades.append(trade)
    equity = peak = drawdown = 0.0
    for trade in trades:
        equity += trade["pnl"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {"trades": len(trades), "wins": sum(t["won"] for t in trades),
            "pnl": sum(t["pnl"] for t in trades), "dd": drawdown,
            "rows": trades}


def aggregate(windows, cfg, runs=100):
    samples = [simulate(windows, cfg, run) for run in range(runs)]
    return {
        "trades": statistics.median(x["trades"] for x in samples),
        "wins": statistics.median(x["wins"] for x in samples),
        "pnl": statistics.median(x["pnl"] for x in samples),
        "pnl_min": min(x["pnl"] for x in samples),
        "pnl_max": max(x["pnl"] for x in samples),
        "dd": statistics.median(x["dd"] for x in samples),
    }


def configs():
    result = [Config("baseline")]
    result += [Config(f"entry{entry:g}", entry=entry)
               for entry in (90, 120, 180, 210, 240)]
    result += [Config(f"move{move:g}", min_move=move)
               for move in (1, 3, 5, 8, 10)]
    result += [Config(f"ask45-{cap:g}", ask_max=cap)
               for cap in (.70, .75, .80, .85)]
    result += [Config("ask55-80", ask_min=.55, ask_max=.80),
               Config("ask55-85", ask_min=.55, ask_max=.85),
               Config("market_agrees", agreement="favorite"),
               Config("market_disagrees", agreement="disagreement"),
               Config("spread<=1c", max_spread=.01),
               Config("recent10>=0", min_recent_10=0),
               Config("recent10>=-2", min_recent_10=-2),
               Config("favorite_only", min_move=0, side_rule="market_favorite"),
               Config("favorite_ask<=75", min_move=0, ask_max=.75,
                      side_rule="market_favorite"),
               Config("underdog_only", min_move=0, side_rule="market_underdog")]
    return result


def fmt(metric):
    return (f"n={metric['trades']:>4.0f} W={metric['wins']:>4.0f} "
            f"P=${metric['pnl']:+6.2f} DD=${metric['dd']:4.2f}")


def wilson(wins, total, z=1.959963984540054):
    if not total:
        return 0.0, 1.0
    p = wins / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return center - half, center + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "price_logs" / "polymarket")
    args = parser.parse_args()
    windows = load(args.directory)
    n = len(windows)
    cut1, cut2 = int(n * .60), int(n * .80)
    groups = [("train60", windows[:cut1]), ("valid20", windows[cut1:cut2]),
              ("holdout20", windows[cut2:])]
    print(f"resolved complete windows={n}; blocks={[len(g) for _, g in groups]}")
    print("candidate              train60                   valid20                   holdout20                  all")
    scored = []
    for cfg in configs():
        metrics = [aggregate(group, cfg) for _, group in groups]
        overall = aggregate(windows, cfg)
        print(f"{cfg.name:<21} " + " | ".join(fmt(m) for m in metrics + [overall]))
        scored.append((metrics[0]["pnl"], cfg, metrics, overall))

    eligible = [row for row in scored if row[2][0]["trades"] >= 5]
    best = max(eligible, key=lambda row: row[0])
    print(f"\ntrain-selected (>=5 train fills): {best[1].name}")
    print("  " + " | ".join(f"{name} {fmt(metric)}"
                            for (name, _), metric in zip(groups, best[2])))

    base = simulate(windows, configs()[0], 0)
    low, high = wilson(base["wins"], base["trades"])
    avg_ask = statistics.mean(t["entry_avg"] for t in base["rows"])
    # Fee makes this a slight underestimate of the true break-even rate.
    print(f"\nbaseline run0: {base['wins']}/{base['trades']} wins; "
          f"95% Wilson win-rate interval={low*100:.1f}%..{high*100:.1f}%; "
          f"mean executed price={avg_ask:.3f}")
    print("baseline trades:")
    for trade in base["rows"]:
        print(f"  {trade['side']:<4} ask={trade['entry_avg']:.3f} "
              f"move={trade['move']:+6.2f} recent10={trade['recent']:+6.2f} "
              f"{'WIN ' if trade['won'] else 'LOSS'} pnl=${trade['pnl']:+.3f}")


if __name__ == "__main__":
    main()
