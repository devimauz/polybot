"""Research executable exits for the BTC5m momentum strategy.

This is an analysis-only script.  It does not submit orders or modify the live
bot.  Entries reproduce btc5m_momentum_backtest.py.  Exits sell into recorded
bids, require enough displayed size, charge the same taker-fee model as the
entry backtest, and are delayed 300-800 ms.  A SELL FOK may execute no more
than $0.03 below the trigger-time worst bid.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from btc5m_momentum_strategy import fee_for


@dataclass(frozen=True)
class Row:
    t: float
    coinbase: float
    up_bids: tuple[tuple[float, float], ...]
    up_asks: tuple[tuple[float, float], ...]
    down_bids: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Window:
    slug: str
    winner: str
    rows: tuple[Row, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class Config:
    name: str
    tp: float | None = None
    tp_after: float = 0.0
    sl: float | None = None
    sl_after: float = 0.0
    trail_arm: float | None = None
    trail_draw: float | None = None
    reversal_bps: float | None = None
    reversal_confirm: float = 0.0
    reversal_after: float = 0.0
    reversal_return_cap: float | None = None


@dataclass
class Entry:
    slug: str
    winner: str
    side: str
    opening: float
    shares: float
    cost: float
    entry_t: float
    entry_avg: float
    rows: tuple[Row, ...]
    times: tuple[float, ...]


def levels(book: dict, side: str) -> tuple[tuple[float, float], ...]:
    raw = book.get(side) or []
    result = []
    for level in raw:
        try:
            price, size = float(level["price"]), float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < price < 1 and size > 0:
            result.append((price, size))
    if not result:
        prefix = "bid" if side == "bids" else "ask"
        try:
            price, size = float(book[prefix]), float(book[f"{prefix}_size"])
        except (KeyError, TypeError, ValueError):
            return ()
        if 0 < price < 1 and size > 0:
            result.append((price, size))
    return tuple(sorted(result, reverse=(side == "bids")))


def load(path: Path) -> tuple[Window | None, int, int]:
    rows = []
    winner = None
    snapshots = full = 0
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
                t = float(raw["seconds_into_window"])
                cb = float((raw.get("coinbase") or {})["price"])
            except (KeyError, TypeError, ValueError):
                continue
            up, down = raw.get("Up") or {}, raw.get("Down") or {}
            snapshots += 1
            full += int(bool(up.get("bids") or up.get("asks") or down.get("bids") or down.get("asks")))
            rows.append(Row(t, cb, levels(up, "bids"), levels(up, "asks"),
                            levels(down, "bids"), levels(down, "asks")))
    rows.sort(key=lambda row: row.t)
    if winner not in ("Up", "Down") or not rows or rows[0].t > 5:
        return None, snapshots, full
    packed = tuple(rows)
    return Window(path.stem, winner, packed, tuple(row.t for row in packed)), snapshots, full


def sweep_buy(levels_: tuple[tuple[float, float], ...], dollars: float,
              max_price: float = 1.0):
    remaining = dollars
    shares = fees = 0.0
    worst = None
    for price, size in levels_:
        if price > max_price + 1e-12:
            continue
        used = min(remaining, price * size)
        qty = used / price
        shares += qty
        fees += fee_for(price, qty)
        remaining -= used
        worst = price
        if remaining <= 1e-9:
            return shares, dollars + fees, worst
    return None


def sweep_sell(levels_: tuple[tuple[float, float], ...], shares: float,
               min_price: float = 0.0):
    remaining = shares
    proceeds = fees = 0.0
    worst = None
    for price, size in levels_:
        if price + 1e-12 < min_price:
            continue
        qty = min(remaining, size)
        proceeds += price * qty
        fees += fee_for(price, qty)
        remaining -= qty
        worst = price
        if remaining <= 1e-9:
            return proceeds - fees, worst
    return None


def entry_for(window: Window, seed: int, stake=2.0, entry_second=150.0,
              minimum_move_bps=2.0, ask_min=.45, ask_max=.90,
              slippage=.03) -> Entry | None:
    rows = window.rows
    times = window.times
    decision_i = bisect.bisect_left(times, entry_second)
    if decision_i >= len(rows):
        return None
    decision = rows[decision_i]
    opening = rows[0].coinbase
    move = (decision.coinbase / opening - 1) * 10_000
    if abs(move) < minimum_move_bps:
        return None
    side = "Up" if move > 0 else "Down"
    asks = decision.up_asks if side == "Up" else decision.down_asks
    if not asks or not ask_min <= asks[0][0] <= ask_max:
        return None
    quoted = sweep_buy(asks, stake)
    if quoted is None:
        return None
    order_limit = min(.99, quoted[2] + slippage)
    rng = random.Random(f"exit-oos:{seed}:{window.slug}:entry")
    target = decision.t + rng.uniform(.300, .800)
    execution_i = bisect.bisect_left(times, target, lo=decision_i + 1)
    if execution_i >= len(rows) or rows[execution_i].t >= 300:
        return None
    execution = rows[execution_i]
    exec_asks = execution.up_asks if side == "Up" else execution.down_asks
    fill = sweep_buy(exec_asks, stake, order_limit)
    if fill is None:
        return None
    shares, cost, _ = fill
    return Entry(window.slug, window.winner, side, opening, shares, cost,
                 execution.t, stake / shares, rows, times)


def passes_dual_feed_veto(window: Window, entry_second=150.0) -> bool:
    """The proposed entry veto: skip only if both 10-second feeds weaken."""
    i = bisect.bisect_left(window.times, entry_second)
    if i >= len(window.rows):
        return True
    row = window.rows[i]
    move = (row.coinbase / window.rows[0].coinbase - 1) * 10_000
    side = "Up" if move > 0 else "Down"
    direction = 1 if side == "Up" else -1
    old_i = max(0, bisect.bisect_right(window.times, row.t - 10, 0, i + 1) - 1)
    old = window.rows[old_i]
    recent_10 = direction * (row.coinbase / old.coinbase - 1) * 10_000
    current_bids = row.up_bids if side == "Up" else row.down_bids
    current_asks = row.up_asks if side == "Up" else row.down_asks
    old_bids = old.up_bids if side == "Up" else old.down_bids
    old_asks = old.up_asks if side == "Up" else old.down_asks
    if not current_bids or not current_asks or not old_bids or not old_asks:
        return True
    current_mid = (current_bids[0][0] + current_asks[0][0]) / 2
    old_mid = (old_bids[0][0] + old_asks[0][0]) / 2
    mid_change_10 = current_mid - old_mid
    return not (recent_10 < 0 and mid_change_10 < -.01)


def hold_result(entry: Entry):
    pnl = (entry.shares if entry.side == entry.winner else 0.0) - entry.cost
    return {"slug": entry.slug, "pnl": pnl, "cost": entry.cost,
            "exit": "settlement", "side_won": entry.side == entry.winner,
            "fok_failures": 0, "exit_return": None, "exit_t": 300.0,
            "trigger_return": None}


def simulate_exit(entry: Entry, cfg: Config, seed: int, sell_slippage=.03):
    if cfg.name == "hold":
        return hold_result(entry)
    start_i = bisect.bisect_right(entry.times, entry.entry_t)
    best_return = -10_000.0
    reverse_since = None
    i = start_i
    failures = 0
    attempt = 0
    while i < len(entry.rows):
        row = entry.rows[i]
        if row.t >= 299.0:
            break
        bids = row.up_bids if entry.side == "Up" else row.down_bids
        quote = sweep_sell(bids, entry.shares)
        if quote is None:
            i += 1
            continue
        net, worst = quote
        ret = (net / entry.cost - 1) * 100
        best_return = max(best_return, ret)
        move = (row.coinbase / entry.opening - 1) * 10_000
        oriented_move = move if entry.side == "Up" else -move

        reason = None
        if cfg.tp is not None and row.t >= cfg.tp_after and ret >= cfg.tp:
            reason = "take_profit"
        elif cfg.sl is not None and row.t >= cfg.sl_after and ret <= -cfg.sl:
            reason = "stop_loss"
        elif (cfg.trail_arm is not None and best_return >= cfg.trail_arm
              and ret <= best_return - float(cfg.trail_draw)):
            reason = "trailing"

        if cfg.reversal_bps is not None:
            reversal_now = (row.t >= cfg.reversal_after
                            and oriented_move <= -cfg.reversal_bps
                            and (cfg.reversal_return_cap is None
                                 or ret <= cfg.reversal_return_cap))
            if reversal_now:
                if reverse_since is None:
                    reverse_since = row.t
                if reason is None and row.t - reverse_since >= cfg.reversal_confirm:
                    reason = "reversal"
            else:
                reverse_since = None

        if reason is None:
            i += 1
            continue

        rng = random.Random(f"exit-oos:{seed}:{entry.slug}:exit:{i}:{attempt}")
        target = row.t + rng.uniform(.300, .800)
        execution_i = bisect.bisect_left(entry.times, target, lo=i + 1)
        if execution_i >= len(entry.rows) or entry.rows[execution_i].t >= 300:
            break
        execution = entry.rows[execution_i]
        exec_bids = execution.up_bids if entry.side == "Up" else execution.down_bids
        fill = sweep_sell(exec_bids, entry.shares, max(.01, worst - sell_slippage))
        attempt += 1
        if fill is not None:
            actual_net, _ = fill
            pnl = actual_net - entry.cost
            return {"slug": entry.slug, "pnl": pnl, "cost": entry.cost,
                    "exit": reason, "side_won": entry.side == entry.winner,
                    "fok_failures": failures,
                    "exit_return": pnl / entry.cost * 100,
                    "exit_t": execution.t, "trigger_return": ret}
        failures += 1
        i = execution_i + 1

    result = hold_result(entry)
    result["fok_failures"] = failures
    return result


def metrics(results):
    pnl = sum(row["pnl"] for row in results)
    cost = sum(row["cost"] for row in results)
    equity = peak = dd = 0.0
    for row in results:
        equity += row["pnl"]
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    losses = [row for row in results if row["pnl"] < 0]
    exits = [row for row in results if row["exit"] != "settlement"]
    return {
        "trades": len(results), "pnl": pnl, "roi": pnl / cost * 100 if cost else 0,
        "max_dd": dd, "losses": len(losses),
        "gross_loss": -sum(row["pnl"] for row in losses),
        "exits": len(exits), "exit_winners": sum(row["pnl"] > 0 for row in exits),
        "fok_failures": sum(row["fok_failures"] for row in results),
    }


def candidates():
    result = [Config("hold")]
    for tp in (10, 12.5, 15, 20, 25, 30, 40, 50, 75):
        result.append(Config(f"tp{tp:g}", tp=tp))
    for sl in (10, 12.5, 15, 20, 25, 30, 40, 50, 60, 75):
        result.append(Config(f"sl{sl:g}", sl=sl))
    for tp in (10, 15, 20, 25, 30, 40, 50):
        for sl in (15, 20, 25, 30, 40, 50, 60):
            result.append(Config(f"tp{tp:g}_sl{sl:g}", tp=tp, sl=sl))
    for arm in (15, 20, 25, 30, 40, 50, 75):
        for draw in (5, 10, 15, 20, 25, 30):
            if arm - draw >= 10:
                result.append(Config(f"trail{arm:g}_{draw:g}", trail_arm=arm, trail_draw=draw))
    for rev in (0, 1, 2, 3, 5, 8, 10):
        for confirm in (0, .5, 1, 2, 3, 5):
            result.append(Config(f"rev{rev:g}_{confirm:g}s", reversal_bps=rev,
                                 reversal_confirm=confirm))
    for tp in (10, 15, 20, 30, 40):
        for rev in (0, 2, 5):
            for confirm in (0, 1, 3):
                result.append(Config(f"tp{tp:g}_rev{rev:g}_{confirm:g}s", tp=tp,
                                     reversal_bps=rev, reversal_confirm=confirm))
    for sl in (15, 20, 30, 40, 50):
        for rev in (0, 2, 5):
            for confirm in (0, 1, 3):
                result.append(Config(f"sl{sl:g}_rev{rev:g}_{confirm:g}s", sl=sl,
                                     reversal_bps=rev, reversal_confirm=confirm))
    # Late exits avoid clipping the large upside of positions that are correct
    # early, while still testing whether the final 20-120 seconds can be
    # de-risked.  These ranges are declared before the holdout is evaluated.
    for after in (180, 210, 240, 255, 270, 280):
        for tp in (10, 15, 20, 30, 40, 50):
            result.append(Config(f"tp{tp:g}_after{after:g}", tp=tp, tp_after=after))
        for sl in (15, 20, 30, 40, 50, 60):
            result.append(Config(f"sl{sl:g}_after{after:g}", sl=sl, sl_after=after))
    for after in (180, 210, 240, 255, 270):
        for rev in (0, 2, 5):
            for confirm in (0, 1, 3):
                result.append(Config(f"rev{rev:g}_{confirm:g}s_after{after:g}",
                                     reversal_bps=rev, reversal_confirm=confirm,
                                     reversal_after=after))
    for after in (150, 210, 240, 270):
        for rev in (0, 2, 5):
            for confirm in (0, 1):
                for cap in (-10, -20, -30, -40):
                    result.append(Config(
                        f"rev{rev:g}_{confirm:g}s_after{after:g}_ret{cap:g}",
                        reversal_bps=rev, reversal_confirm=confirm,
                        reversal_after=after, reversal_return_cap=cap))
    return result


ENTRY_CACHE = {}


def cached_entries(windows, seed):
    key = (windows[0].slug if windows else "", windows[-1].slug if windows else "",
           len(windows), seed)
    if key not in ENTRY_CACHE:
        ENTRY_CACHE[key] = [entry for window in windows
                            if (entry := entry_for(window, seed)) is not None]
    return ENTRY_CACHE[key]


def run_group(windows, cfg, seeds):
    all_metrics = []
    for seed in seeds:
        entries = cached_entries(windows, seed)
        results = [simulate_exit(entry, cfg, seed) for entry in entries]
        all_metrics.append(metrics(results))
    return all_metrics


def median_metric(rows, key):
    return statistics.median(row[key] for row in rows)


def print_stats(label, rows):
    fields = {key: median_metric(rows, key) for key in
              ("trades", "pnl", "roi", "max_dd", "losses", "gross_loss",
               "exits", "exit_winners", "fok_failures")}
    pnls = [row["pnl"] for row in rows]
    print(f"{label:<22} trades={fields['trades']:5.1f} pnl=${fields['pnl']:7.2f} "
          f"range=${min(pnls):.2f}..${max(pnls):.2f} roi={fields['roi']:6.2f}% "
          f"dd=${fields['max_dd']:5.2f} losses={fields['losses']:4.1f} "
          f"gross_loss=${fields['gross_loss']:5.2f} exits={fields['exits']:4.1f} "
          f"exitW={fields['exit_winners']:4.1f} FOKfail={fields['fok_failures']:4.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "price_logs" / "polymarket")
    parser.add_argument("--screen-seeds", type=int, default=3)
    parser.add_argument("--final-seeds", type=int, default=100)
    args = parser.parse_args()

    windows = []
    snapshots = full = 0
    for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
        window, count, full_count = load(path)
        snapshots += count
        full += full_count
        if window:
            windows.append(window)
    cut = int(len(windows) * .60)
    train, test = windows[:cut], windows[cut:]
    train_mid = len(train) // 2
    seeds = range(args.screen_seeds)
    configs = candidates()
    print(f"Loaded {len(windows)} resolved windows; chronological 60/40={len(train)}/{len(test)}")
    print(f"Snapshots={snapshots}; snapshots containing depth arrays={full} ({full/max(1,snapshots)*100:.3f}%)")
    print(f"Screening {len(configs)} predeclared rules on first 60%, {args.screen_seeds} latency seeds")

    baseline_train = run_group(train, configs[0], seeds)
    baseline_a = run_group(train[:train_mid], configs[0], seeds)
    baseline_b = run_group(train[train_mid:], configs[0], seeds)
    base_pnl = median_metric(baseline_train, "pnl")
    base_a = median_metric(baseline_a, "pnl")
    base_b = median_metric(baseline_b, "pnl")
    screened = []
    for idx, cfg in enumerate(configs[1:], 1):
        overall = run_group(train, cfg, seeds)
        a = run_group(train[:train_mid], cfg, seeds)
        b = run_group(train[train_mid:], cfg, seeds)
        pnl = median_metric(overall, "pnl")
        delta_a = median_metric(a, "pnl") - base_a
        delta_b = median_metric(b, "pnl") - base_b
        screened.append((pnl, min(delta_a, delta_b), delta_a, delta_b, cfg, overall))
        if idx % 50 == 0:
            print(f"  screened {idx}/{len(configs)-1}")
    robust = [row for row in screened if row[2] >= 0 and row[3] >= 0]
    ranked = sorted(robust or screened, key=lambda row: (row[0], row[1]), reverse=True)
    selected = ranked[0][4]
    print("\nTRAIN SCREEN (rules improving both chronological train halves marked robust)")
    print_stats("hold", baseline_train)
    for pnl, worst, da, db, cfg, rows in ranked[:10]:
        print_stats(cfg.name, rows)
        print(f"  train delta halves: ${da:+.2f}, ${db:+.2f}; vs baseline total ${pnl-base_pnl:+.2f}")
    print(f"\nSelected before opening holdout: {selected.name}")

    final_seeds = range(args.final_seeds)
    print("\n100-SEED FINAL COMPARISON")
    for label, group in (("Train hold", train), (f"Train {selected.name}", train),
                         ("OOS hold", test), (f"OOS {selected.name}", test),
                         ("All hold", windows), (f"All {selected.name}", windows)):
        cfg = configs[0] if label.endswith("hold") else selected
        print_stats(label, run_group(group, cfg, final_seeds))

    print("\nCANONICAL EXIT CHECKS ON OOS (not used for selection)")
    wanted = {"tp10", "tp15", "tp20", "sl20", "sl30", "sl40",
              "trail20_10", "trail30_15", "rev0_1s", "rev2_1s", "rev5_1s"}
    for cfg in configs:
        if cfg.name in wanted:
            print_stats("OOS " + cfg.name, run_group(test, cfg, final_seeds))

    print("\nTOP TRAIN-ROBUST RULES: HOLDOUT DELTA")
    oos_hold_rows = run_group(test, configs[0], final_seeds)
    oos_hold_pnl = median_metric(oos_hold_rows, "pnl")
    for _, _, da, db, cfg, _ in ranked[:20]:
        rows = run_group(test, cfg, final_seeds)
        delta = median_metric(rows, "pnl") - oos_hold_pnl
        print(f"{cfg.name:<34} OOS delta=${delta:+6.2f} "
              f"train-halves=${da:+.2f}/${db:+.2f} "
              f"OOSdd=${median_metric(rows, 'max_dd'):.2f}")


if __name__ == "__main__":
    main()
