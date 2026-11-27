"""_pm_selected_rule.py - assess the rule that was SELECTED on the first 60%
and then tested on the untouched last 40%.

The headline rule (T>=260) is NOT significant on the full source-aligned
sample: t=+1.83, CI95=[-0.12,+1.04].  The train-selected rule was T>=265.
This script gives that rule the full battery, and asks the uncomfortable
question the repo's own audit insists on: is the second half carrying the
whole effect?
"""
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

T_MIN = 265
BAND = (0.80, 0.99)


def rule(rows, ws_lo=None, ws_hi=None):
    sel = [r for r in rows if r["T"] >= T_MIN and r["mkt"] >= 0.5
           and BAND[0] <= r["ask"] < BAND[1]]
    if ws_lo is not None:
        sel = [r for r in sel if r["ws"] >= ws_lo]
    if ws_hi is not None:
        sel = [r for r in sel if r["ws"] < ws_hi]
    return B.first_per_window(sel)


def show(sel, label):
    if not sel:
        print(f"  {label}: no trades")
        return None
    a = np.array([r["ask"] for r in sel]); w = np.array([r["won"] for r in sel])
    p = B.pnl_of(a, w)
    m, t, _ = B.cluster_t([r["wi"] for r in sel], p)
    lo, hi = B.boot_ci([r["wi"] for r in sel], p)
    pv = float(binom.sf(int(w.sum()) - 1, len(w), a.mean()))
    be = a.mean() + (B.fee(a.mean(), B.STAKE / a.mean()) * a.mean() / B.STAKE)
    print(f"  {label:<34} N={len(sel):>3} w={int(w.sum()):>3} ask={a.mean():.4f} "
          f"win={w.mean():.4f} edge={w.mean()-be:+.4f} net=${p.sum():+7.2f} "
          f"t={t:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] bp={pv:.4f}")
    return dict(p=p, t=t, w=w, a=a)


def main():
    W = B.load_all()
    rows = B.build_rows(W)
    wss = np.array([d["ws"] for d in W])
    cut = wss[int(len(W) * 0.60)]

    print("=" * 96)
    print(f"SELECTED RULE: T>={T_MIN}s, favourite ask [{BAND[0]:.2f},{BAND[1]:.2f}), "
          f"hold to settlement  (source-aligned clock)")
    print("=" * 96)

    print("\n--- 1. TRAIN / TEST SPLIT (the rule was selected on train) ---")
    tr = show(rule(rows, ws_hi=cut + 1), "train (first 60%)")
    te = show(rule(rows, ws_lo=cut + 1), "TEST (untouched last 40%)")
    if tr and te:
        print(f"  => out-of-sample delta on the held-out 40%: "
              f"${te['p'].sum():+.2f}, win rate {te['w'].mean():.4f} vs "
              f"market-implied {te['a'].mean():.4f}")

    print("\n--- 2. ALL WINDOWS + concentration ---")
    allr = show(rule(rows), "all windows")
    p = allr["p"]
    order = np.argsort(p)
    tot = p.sum()
    print(f"  P&L concentration: top1={p[order[-1]]/tot*100:.1f}%  "
          f"top3={p[order[-3:]].sum()/tot*100:.1f}%  "
          f"top5={p[order[-5:]].sum()/tot*100:.1f}%")
    for k in (1, 5):
        keep = [i for i in range(len(p)) if i not in set(order[-k:].tolist())]
        pk = p[keep]
        _, tk, _ = B.cluster_t([rule(rows)[i]["wi"] for i in keep], pk)
        print(f"  excluding best {k}: net=${pk.sum():+.2f} t={tk:+.2f}")

    print("\n--- 3. IS THE SECOND HALF CARRYING EVERYTHING? ---")
    allsel = rule(rows)
    med = np.median([r["ws"] for r in allsel])
    show([r for r in allsel if r["ws"] <= med], "first half")
    show([r for r in allsel if r["ws"] > med], "second half")
    q = np.array_split(np.argsort([r["ws"] for r in allsel]), 4)
    print("  quarters:", end=" ")
    for b in q:
        sub = [allsel[i] for i in b]
        pp = B.pnl_of([r["ask"] for r in sub], [r["won"] for r in sub])
        print(f"{pp.sum():+7.2f} ", end="")
    print()
    pos = 0
    for b in np.array_split(np.argsort([r["ws"] for r in allsel]), 8):
        sub = [allsel[i] for i in b]
        pp = B.pnl_of([r["ask"] for r in sub], [r["won"] for r in sub])
        pos += int(pp.sum() > 0)
    print(f"  8 chronological blocks: {pos}/8 positive")

    print("\n--- 4. FAMILY-WISE MAX-t OVER THE 20-RULE GRID (outcomes permuted) ---")
    rng = np.random.default_rng(7)
    grid = [(tmin, band) for tmin in (250, 255, 260, 265, 270)
            for band in ((0.80, 0.99), (0.80, 0.96), (0.85, 0.98), (0.90, 0.99))]
    by_wi = {}
    for r in rows:
        if r["mkt"] >= 0.5:
            by_wi.setdefault(r["wi"], []).append(r)
    wi_list = sorted(by_wi)
    obs = []
    for tmin, band in grid:
        s = rule(rows)
        s = B.first_per_window([r for r in rows if r["T"] >= tmin and r["mkt"] >= 0.5
                                and band[0] <= r["ask"] < band[1]])
        if len(s) < 5:
            continue
        pp = B.pnl_of([r["ask"] for r in s], [r["won"] for r in s])
        _, tt, _ = B.cluster_t([r["wi"] for r in s], pp)
        obs.append(tt)
    obs_max = max(obs)
    null = []
    for _ in range(600):
        perm = list(wi_list)
        rng.shuffle(perm)
        wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
        ts = []
        for tmin, band in grid:
            c = []
            for src in wi_list:
                for r in by_wi[src]:
                    if r["T"] < tmin or not (band[0] <= r["ask"] < band[1]):
                        continue
                    r2 = dict(r)
                    r2["won"] = 1 if W[wperm[src]]["_winner"] == r["side"] else 0
                    c.append(r2)
            c = B.first_per_window(c)
            if len(c) < 5:
                continue
            pp = B.pnl_of([r["ask"] for r in c], [r["won"] for r in c])
            _, tt, _ = B.cluster_t([r["wi"] for r in c], pp)
            ts.append(tt)
        if ts:
            null.append(max(ts))
    p_fw = float(np.mean(np.array(null) >= obs_max))
    print(f"  observed max t {obs_max:+.2f}   null mean {np.mean(null):+.2f}  "
          f"95th pct {np.percentile(null, 95):+.2f}   family-wise p = {p_fw:.4f}")

    print("\n--- 5. STRESS on the selected rule ---")
    for lat in (0.0, 0.3, 0.5, 0.8, 1.0):
        rl = B.build_rows(W, latency_s=lat)
        s = B.first_per_window([r for r in rl if r["T"] >= T_MIN and r["mkt"] >= 0.5
                                and BAND[0] <= r["ask"] < BAND[1]])
        if len(s) < 5:
            print(f"  latency {lat:.1f}s: too few ({len(s)})")
            continue
        a2 = np.array([r["ask"] for r in s]); w2 = np.array([r["won"] for r in s])
        p2 = B.pnl_of(a2, w2)
        _, t2, _ = B.cluster_t([r["wi"] for r in s], p2)
        lo2, hi2 = B.boot_ci([r["wi"] for r in s], p2)
        print(f"  latency {lat:.1f}s : N={len(s):>3} net=${p2.sum():+7.2f} t={t2:+5.2f} "
              f"CI=[{lo2:+.2f},{hi2:+.2f}]")
    sel = rule(rows)
    a = np.array([r["ask"] for r in sel]); w = np.array([r["won"] for r in sel])
    for fr in (0.14, 0.28):
        sh = B.STAKE / a
        pf = sh * w - B.STAKE - sh * fr * a * (1 - a)
        _, tf, _ = B.cluster_t([r["wi"] for r in sel], pf)
        lof, hif = B.boot_ci([r["wi"] for r in sel], pf)
        print(f"  fee x{fr/0.07:.0f}      : net=${pf.sum():+7.2f} t={tf:+5.2f} "
              f"CI=[{lof:+.2f},{hif:+.2f}]")
    for depth in (25.0, 50.0, 100.0):
        B.MIN_DEPTH_USD = depth
        rr = B.build_rows(W)
        s = B.first_per_window([r for r in rr if r["T"] >= T_MIN and r["mkt"] >= 0.5
                                and BAND[0] <= r["ask"] < BAND[1]])
        a2 = np.array([r["ask"] for r in s]); w2 = np.array([r["won"] for r in s])
        p2 = B.pnl_of(a2, w2)
        _, t2, _ = B.cluster_t([r["wi"] for r in s], p2)
        print(f"  min depth ${depth:>5.0f}  : N={len(s):>3} net=${p2.sum():+7.2f} t={t2:+5.2f}")
    B.MIN_DEPTH_USD = 10.0

    print("\n--- 6. ALIGNMENT CONTROL ---")
    ud = B.first_per_window([r for r in rows if r["T"] >= T_MIN and r["mkt"] < 0.5
                             and 0.01 <= r["ask"] < 0.20])
    if ud:
        pu = B.pnl_of([r["ask"] for r in ud], [r["won"] for r in ud])
        _, tu, _ = B.cluster_t([r["wi"] for r in ud], pu)
        print(f"  underdog at the same timestamps: N={len(ud)} net=${pu.sum():+.2f} t={tu:+.2f}")


if __name__ == "__main__":
    main()
