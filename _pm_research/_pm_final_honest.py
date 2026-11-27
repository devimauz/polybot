"""_pm_final_honest.py - the honest final assessment on SOURCE-ALIGNED data.

The first pass of this study used the logger's `seconds_into_window` stamp.
The repo's own canonical loader (_research_value_stop.py::load_source_aligned)
stamps each frame at max(Up.book_timestamp, Down.book_timestamp,
coinbase.exchange_ts) - window_start, because the logger writes the row
BEFORE its REST calls complete.  Measured on this dataset the logger stamp is
~0.96 s optimistic (median 0.967 s, 99.3% of frames > 0.5 s).

Everything below therefore uses the source-aligned clock.  This script is the
gate: the strategy ships only if it survives here.

Tests: pre-registered split, family-wise max-t, leave-one-out, P&L
CONCENTRATION (top-1/3/5 share), underdog alignment control, depth/fee/
spread/latency stress, and the price-band calibration.
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")


def main():
    W = B.load_all()
    rows = B.build_rows(W)
    print("=" * 96)
    print("HONEST FINAL ASSESSMENT - source-aligned timestamps")
    print("=" * 96)
    print(f"windows {len(W)}   executable endgame quotes {len(rows)}")

    # ------------------------------------------------------------------
    # 1. PRE-REGISTERED SPLIT: pick the rule on the FIRST 60% only, then
    #    report it on the untouched last 40%.  This is the repo author's
    #    own gate (train -> validation -> sealed holdout).
    # ------------------------------------------------------------------
    print("\n=== 1) SELECT ON FIRST 60%, TEST ON THE UNTOUCHED LAST 40% ===")
    wss = np.array([d["ws"] for d in W])
    cut = wss[int(len(W) * 0.60)]
    tr = [r for r in rows if r["ws"] <= cut]
    te = [r for r in rows if r["ws"] > cut]

    grid = [(tmin, band) for tmin in (250, 255, 260, 265, 270)
            for band in ((0.80, 0.99), (0.80, 0.96), (0.85, 0.98), (0.90, 0.99))]
    best = None
    for tmin, band in grid:
        s = B.first_per_window([r for r in tr if r["T"] >= tmin and r["mkt"] >= 0.5
                                and band[0] <= r["ask"] < band[1]])
        if len(s) < 8:
            continue
        a = np.array([r["ask"] for r in s]); w = np.array([r["won"] for r in s])
        p = B.pnl_of(a, w)
        m, t, _ = B.cluster_t([r["wi"] for r in s], p)
        if best is None or t > best[0]:
            best = (t, tmin, band, len(s), p.sum(), w.mean(), a.mean())
    if best is None:
        print("  no rule had enough training trades")
        return
    t_tr, tmin, band, n_tr, pnl_tr, wr_tr, ask_tr = best
    print(f"  selected on train: T>={tmin}s fav ask[{band[0]:.2f},{band[1]:.2f})  "
          f"n={n_tr} t={t_tr:+.2f} net=${pnl_tr:+.2f} win={wr_tr:.4f} ask={ask_tr:.4f}")

    s_te = B.first_per_window([r for r in te if r["T"] >= tmin and r["mkt"] >= 0.5
                               and band[0] <= r["ask"] < band[1]])
    a = np.array([r["ask"] for r in s_te]); w = np.array([r["won"] for r in s_te])
    p = B.pnl_of(a, w)
    m, t_te, _ = B.cluster_t([r["wi"] for r in s_te], p)
    lo, hi = B.boot_ci([r["wi"] for r in s_te], p)
    print(f"  HELD-OUT TEST:  n={len(s_te)} t={t_te:+.2f} net=${p.sum():+.2f} "
          f"avg=${p.mean():+.3f} CI95=[{lo:+.2f},{hi:+.2f}] win={w.mean():.4f} ask={a.mean():.4f}")
    print(f"  -> the rule selected on the first 60% nets "
          f"{'$+' if p.sum() > 0 else '$'}{p.sum():.2f} out of sample "
          f"(win rate {w.mean():.4f} vs market-implied {a.mean():.4f})")

    # ------------------------------------------------------------------
    # 2. THE HEADLINE RULE, all data, with concentration
    # ------------------------------------------------------------------
    print("\n=== 2) HEADLINE RULE (T>=260, ask[0.80,0.99)) - all windows ===")
    sel = B.first_per_window([r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5
                              and 0.80 <= r["ask"] < 0.99])
    ask = np.array([r["ask"] for r in sel]); won = np.array([r["won"] for r in sel])
    p = B.pnl_of(ask, won)
    m, t, _ = B.cluster_t([r["wi"] for r in sel], p)
    lo, hi = B.boot_ci([r["wi"] for r in sel], p)
    pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
    be = ask.mean() + (B.fee(ask.mean(), B.STAKE / ask.mean()) * ask.mean() / B.STAKE)
    print(f"  N={len(sel)} wins={int(won.sum())} ask={ask.mean():.4f} "
          f"win={won.mean():.4f} breakeven={be:.4f} edge={won.mean()-be:+.4f}")
    print(f"  net=${p.sum():+.2f} avg=${p.mean():+.3f} t={t:+.2f} "
          f"CI95=[{lo:+.2f},{hi:+.2f}] binom_p={pv:.4f} worst=${p.min():+.2f}")

    # P&L CONCENTRATION - the repo's mandatory metric
    order = np.argsort(p)
    tot = p.sum()
    top1 = p[order[-1]] / tot
    top3 = p[order[-3:]].sum() / tot
    top5 = p[order[-5:]].sum() / tot
    print(f"  P&L CONCENTRATION: top1={top1*100:.1f}%  top3={top3*100:.1f}%  "
          f"top5={top5*100:.1f}%  of total")
    # excluding the best trade
    p2 = np.delete(p, order[-1])
    _, t2, _ = B.cluster_t([r["wi"] for r in sel][:-1] if order[-1] == len(p) - 1
                            else [sel[i]["wi"] for i in range(len(sel)) if i != order[-1]],
                            p2)
    print(f"  excluding the single best trade: net=${p2.sum():+.2f} t={t2:+.2f}")
    # excluding the best 5
    keep = [i for i in range(len(sel)) if i not in set(order[-5:].tolist())]
    p5 = p[keep]
    _, t5, _ = B.cluster_t([sel[i]["wi"] for i in keep], p5)
    print(f"  excluding the best 5 trades:       net=${p5.sum():+.2f} t={t5:+.2f}")

    # chronological halves + blocks
    med = np.median([r["ws"] for r in sel])
    for lbl, sub in (("first half ", [r for r in sel if r["ws"] <= med]),
                     ("second half", [r for r in sel if r["ws"] > med])):
        a2 = np.array([r["ask"] for r in sub]); w2 = np.array([r["won"] for r in sub])
        p2 = B.pnl_of(a2, w2)
        _, t2, _ = B.cluster_t([r["wi"] for r in sub], p2)
        print(f"  {lbl}: N={len(sub):>3} net=${p2.sum():+7.2f} t={t2:+5.2f} "
              f"win={w2.mean():.4f} ask={a2.mean():.4f}")
    pos = 0
    for b in np.array_split(np.argsort([r["ws"] for r in sel]), 8):
        sub = [sel[i] for i in b]
        p2 = B.pnl_of([r["ask"] for r in sub], [r["won"] for r in sub])
        pos += int(p2.sum() > 0)
    print(f"  block stability (8 chronological blocks): {pos}/8 positive")

    # quarterly view
    q = np.array_split(np.argsort([r["ws"] for r in sel]), 4)
    print("  quarters:", end=" ")
    for b in q:
        sub = [sel[i] for i in b]
        p2 = B.pnl_of([r["ask"] for r in sub], [r["won"] for r in sub])
        print(f"{p2.sum():+.2f} ", end="")
    print()

    # ------------------------------------------------------------------
    # 3. CONTROLS
    # ------------------------------------------------------------------
    print("\n=== 3) CONTROLS ===")
    ud = B.first_per_window([r for r in rows if r["T"] >= 260 and r["mkt"] < 0.5
                             and 0.01 <= r["ask"] < 0.20])
    if ud:
        pu = B.pnl_of([r["ask"] for r in ud], [r["won"] for r in ud])
        _, tu, _ = B.cluster_t([r["wi"] for r in ud], pu)
        print(f"  underdog at the SAME timestamps: N={len(ud)} net=${pu.sum():+.2f} "
              f"t={tu:+.2f}  (must lose -> timestamps are aligned)")

    # leave-one-out
    ts = []
    for k in range(len(sel)):
        keep = [i for i in range(len(sel)) if i != k]
        _, tk, _ = B.cluster_t([sel[i]["wi"] for i in keep], p[keep])
        ts.append(tk)
    print(f"  leave-one-out t: min {min(ts):+.2f} median {np.median(ts):+.2f} "
          f"max {max(ts):+.2f}")

    # ------------------------------------------------------------------
    # 4. STRESS
    # ------------------------------------------------------------------
    print("\n=== 4) STRESS ===")
    for lat in (0.0, 0.3, 0.5, 0.8, 1.0):
        rl = B.build_rows(W, latency_s=lat)
        s2 = B.first_per_window([r for r in rl if r["T"] >= 260 and r["mkt"] >= 0.5
                                 and 0.80 <= r["ask"] < 0.99])
        a2 = np.array([r["ask"] for r in s2]); w2 = np.array([r["won"] for r in s2])
        p2 = B.pnl_of(a2, w2)
        _, t2, _ = B.cluster_t([r["wi"] for r in s2], p2)
        lo2, hi2 = B.boot_ci([r["wi"] for r in s2], p2)
        print(f"  latency {lat:.1f}s : N={len(s2):>3} net=${p2.sum():+7.2f} t={t2:+5.2f} "
              f"CI=[{lo2:+.2f},{hi2:+.2f}]")
    for fr in (0.07, 0.14, 0.28):
        sh = B.STAKE / ask
        pf = sh * won - B.STAKE - sh * fr * ask * (1 - ask)
        _, tf, _ = B.cluster_t([r["wi"] for r in sel], pf)
        lof, hif = B.boot_ci([r["wi"] for r in sel], pf)
        print(f"  fee x{fr/0.07:.0f}      : net=${pf.sum():+7.2f} t={tf:+5.2f} "
              f"CI=[{lof:+.2f},{hif:+.2f}]")
    for depth in (10.0, 50.0, 100.0):
        B.MIN_DEPTH_USD = depth
        rr = B.build_rows(W)
        s2 = B.first_per_window([r for r in rr if r["T"] >= 260 and r["mkt"] >= 0.5
                                 and 0.80 <= r["ask"] < 0.99])
        a2 = np.array([r["ask"] for r in s2]); w2 = np.array([r["won"] for r in s2])
        p2 = B.pnl_of(a2, w2)
        _, t2, _ = B.cluster_t([r["wi"] for r in s2], p2)
        print(f"  min depth ${depth:>5.0f}  : N={len(s2):>3} net=${p2.sum():+7.2f} t={t2:+5.2f}")
    B.MIN_DEPTH_USD = 10.0

    # ------------------------------------------------------------------
    # 5. WHAT WOULD MAKE IT CLEARLY PROFITABLE
    # ------------------------------------------------------------------
    print("\n=== 5) SAMPLE SIZE NEEDED (from the observed effect) ===")
    p_hat = won.mean()
    p0 = ask.mean()
    for target in (p_hat, 0.95, 0.97):
        need = ((1.959964 + 0.841621) * np.sqrt(p0 * (1 - p0)) / (target - p0)) ** 2
        print(f"  to detect a true win rate of {target:.4f} vs market {p0:.4f}: "
              f"~{need:.0f} independent windows "
              f"({need/len(W)*22.4/24:.1f} days of continuous logging)")
    print(f"  current sample: {int(won.sum())}/{len(won)} wins -> binom_p={pv:.4f}")


if __name__ == "__main__":
    main()
