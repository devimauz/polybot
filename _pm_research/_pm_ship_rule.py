"""_pm_ship_rule.py - pick the rule to ship, on source-aligned data, with the
repo author's own gate: positive in BOTH chronological halves, healthy P&L
concentration, survives latency/fee/depth, and a family-level sign test.
"""
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B


def main():
    W = B.load_all()
    rows = B.build_rows(W)
    wss = np.array([d["ws"] for d in W])
    cut = wss[int(len(W) * 0.60)]

    def sel(tmin, band, lo=None, hi=None):
        s = [r for r in rows if r["T"] >= tmin and r["mkt"] >= 0.5
             and band[0] <= r["ask"] < band[1]]
        if lo is not None:
            s = [r for r in s if r["ws"] >= lo]
        if hi is not None:
            s = [r for r in s if r["ws"] < hi]
        return B.first_per_window(s)

    def metrics(s):
        if len(s) < 5:
            return None
        a = np.array([r["ask"] for r in s]); w = np.array([r["won"] for r in s])
        p = B.pnl_of(a, w)
        m, t, _ = B.cluster_t([r["wi"] for r in s], p)
        lo, hi = B.boot_ci([r["wi"] for r in s], p)
        order = np.argsort(p)
        top5 = p[order[-5:]].sum() / p.sum() if p.sum() > 0 else np.nan
        return dict(n=len(s), wins=int(w.sum()), ask=a.mean(), win=w.mean(),
                    net=p.sum(), t=t, lo=lo, hi=hi, p=p, top5=top5,
                    bp=float(binom.sf(int(w.sum()) - 1, len(w), a.mean())))

    print("=" * 112)
    print("SHIPPING-RULE SELECTION (source-aligned clock)")
    print("=" * 112)
    print(f"{'rule':<32} {'N':>4} {'w':>4} {'ask':>6} {'win':>6} {'net$':>8} {'t':>6} "
          f"{'CI95':>16} {'top5%':>6} | {'h1 net':>8} {'h2 net':>8} {'OOS net':>8}")

    cands = []
    for tmin in (255, 258, 260, 262, 265, 268, 270):
        for band in ((0.80, 0.99), (0.80, 0.96), (0.82, 0.95), (0.85, 0.95)):
            allm = metrics(sel(tmin, band))
            if allm is None:
                continue
            med = np.median([r["ws"] for r in sel(tmin, band)])
            h1 = metrics(sel(tmin, band, hi=med + 1))
            h2 = metrics(sel(tmin, band, lo=med))
            oos = metrics(sel(tmin, band, lo=cut + 1))
            cands.append((tmin, band, allm, h1, h2, oos))
            print(f"T>={tmin} ask[{band[0]:.2f},{band[1]:.2f})".ljust(32)
                  + f" {allm['n']:>4} {allm['wins']:>4} {allm['ask']:>6.4f} "
                    f"{allm['win']:>6.4f} {allm['net']:>+8.2f} {allm['t']:>+6.2f} "
                    f"[{allm['lo']:>+6.2f},{allm['hi']:>+6.2f}] {allm['top5']*100:>5.0f}% | "
                    f"{(h1['net'] if h1 else float('nan')):>+8.2f} "
                    f"{(h2['net'] if h2 else float('nan')):>+8.2f} "
                    f"{(oos['net'] if oos else float('nan')):>+8.2f}")

    # ---- the shipping gate -------------------------------------------
    print("\n--- SHIPPING GATE ---")
    ok = []
    for tmin, band, allm, h1, h2, oos in cands:
        if allm is None or h1 is None or h2 is None:
            continue
        checks = {
            "net>0 all": allm["net"] > 0,
            "h1>0": h1["net"] > 0,
            "h2>0": h2["net"] > 0,
            "oos>0": (oos is None or oos["net"] > 0),
            "top5<50%": allm["top5"] < 0.50,
            "t>1.5": allm["t"] > 1.5,
            "n>=25": allm["n"] >= 25,
        }
        if all(checks.values()):
            ok.append((tmin, band, allm, oos))
            print(f"  PASS  T>={tmin} ask[{band[0]:.2f},{band[1]:.2f})  n={allm['n']} "
                  f"net=${allm['net']:+.2f} t={allm['t']:+.2f} top5={allm['top5']*100:.0f}%")
    if not ok:
        print("  NO RULE PASSES THE FULL GATE")
        return

    # pick the one with the best t among passers
    tmin, band, allm, oos = max(ok, key=lambda r: r[2]["t"])
    print(f"\n  -> SHIP: T>={tmin}s, favourite ask [{band[0]:.2f},{band[1]:.2f})")
    print(f"     n={allm['n']} wins={allm['wins']} ask={allm['ask']:.4f} "
          f"win={allm['win']:.4f} net=${allm['net']:+.2f} t={allm['t']:+.2f} "
          f"CI95=[{allm['lo']:+.2f},{allm['hi']:+.2f}]")

    # full stress on the shipping rule
    print("\n--- STRESS ON THE SHIPPING RULE ---")
    for lat in (0.0, 0.3, 0.5, 0.8, 1.0):
        rl = B.build_rows(W, latency_s=lat)
        s = B.first_per_window([r for r in rl if r["T"] >= tmin and r["mkt"] >= 0.5
                                 and band[0] <= r["ask"] < band[1]])
        m2 = metrics(s)
        if m2:
            print(f"  latency {lat:.1f}s : n={m2['n']:>3} net=${m2['net']:>+7.2f} "
                  f"t={m2['t']:>+5.2f} CI=[{m2['lo']:+.2f},{m2['hi']:+.2f}]")
    a = np.array([r["ask"] for r in sel(tmin, band)])
    w = np.array([r["won"] for r in sel(tmin, band)])
    for fr in (0.14, 0.28):
        sh = B.STAKE / a
        pf = sh * w - B.STAKE - sh * fr * a * (1 - a)
        _, tf, _ = B.cluster_t([r["wi"] for r in sel(tmin, band)], pf)
        lof, hif = B.boot_ci([r["wi"] for r in sel(tmin, band)], pf)
        print(f"  fee x{fr/0.07:.0f}      : net=${pf.sum():>+7.2f} t={tf:>+5.2f} "
              f"CI=[{lof:+.2f},{hif:+.2f}]")
    for depth in (25.0, 50.0, 100.0):
        B.MIN_DEPTH_USD = depth
        rr = B.build_rows(W)
        s = B.first_per_window([r for r in rr if r["T"] >= tmin and r["mkt"] >= 0.5
                                 and band[0] <= r["ask"] < band[1]])
        m2 = metrics(s)
        if m2:
            print(f"  min depth ${depth:>5.0f}  : n={m2['n']:>3} net=${m2['net']:>+7.2f} "
                  f"t={m2['t']:>+5.2f}")
    B.MIN_DEPTH_USD = 10.0

    # family-level sign test
    print("\n--- FAMILY-LEVEL SIGN TEST (how many of the 28 candidate rules")
    print("    are positive out of sample?) ---")
    pos = 0
    tot = 0
    for tmin2, band2, _, _, _, _ in cands:
        o = metrics(sel(tmin2, band2, lo=cut + 1))
        if o:
            tot += 1
            pos += int(o["net"] > 0)
    print(f"  positive OOS: {pos}/{tot} rules")
    from math import comb
    p_sign = sum(comb(tot, k) for k in range(pos, tot + 1)) / 2 ** tot
    print(f"  one-sided sign-test p = {p_sign:.6f}")


if __name__ == "__main__":
    main()
