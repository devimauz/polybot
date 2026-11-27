"""_pm_oos_sweep.py - is the 18/18 out-of-sample result special, or is it
just the luckiest of the 20 rules in the grid?

Protocol: for each of the 20 rules, select the rule's parameters on the
TRAIN half only, then measure it on the untouched TEST half.  Count how many
rules come out positive OOS.  If most do, the 18/18 is not evidence of a
real edge - it is the selection process.
"""
import os
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

GRID = [(tmin, band) for tmin in (250, 255, 260, 265, 270)
        for band in ((0.80, 0.99), (0.80, 0.96), (0.85, 0.98), (0.90, 0.99))]


def stat(sel):
    if len(sel) < 4:
        return None
    a = np.array([r["ask"] for r in sel]); w = np.array([r["won"] for r in sel])
    p = B.pnl_of(a, w)
    m, t, _ = B.cluster_t([r["wi"] for r in sel], p)
    pv = float(binom.sf(int(w.sum()) - 1, len(w), a.mean())) if w.sum() < len(w) else 1.0
    return dict(n=len(sel), wins=int(w.sum()), ask=a.mean(), win=w.mean(),
                net=p.sum(), t=t, p=pv, edge=w.mean() - (a.mean() +
                     (B.fee(a.mean(), B.STAKE / a.mean()) * a.mean() / B.STAKE)))


def main():
    W = B.load_all()
    rows = B.build_rows(W)
    wss = np.array([d["ws"] for d in W])
    cut = wss[int(len(W) * 0.60)]

    def sel_for(tmin, band, lo=None, hi=None):
        s = [r for r in rows if r["T"] >= tmin and r["mkt"] >= 0.5
             and band[0] <= r["ask"] < band[1]]
        if lo is not None:
            s = [r for r in s if r["ws"] >= lo]
        if hi is not None:
            s = [r for r in s if r["ws"] < hi]
        return B.first_per_window(s)

    print("=" * 100)
    print("OOS SWEEP - all 20 rules, each selected on TRAIN, each measured on the")
    print("untouched TEST half.  (source-aligned clock)")
    print("=" * 100)
    print(f"{'rule':<34} {'TRAIN n':>7} {'t':>6} | {'TEST n':>6} {'win':>6} "
          f"{'edge':>8} {'net$':>8} {'t':>6} {'bp':>7}")

    results = []
    for tmin, band in GRID:
        tr = stat(sel_for(tmin, band, hi=cut + 1))
        te = stat(sel_for(tmin, band, lo=cut + 1))
        if tr is None or te is None:
            continue
        lab = f"T>={tmin} ask[{band[0]:.2f},{band[1]:.2f})"
        results.append((tmin, band, tr, te))
        print(f"{lab:<34} {tr['n']:>7} {tr['t']:>+6.2f} | {te['n']:>6} "
              f"{te['win']:>6.4f} {te['edge']:>+8.4f} {te['net']:>+8.2f} "
              f"{te['t']:>+6.2f} {te['p']:>7.4f}")

    pos = sum(1 for _, _, _, te in results if te["net"] > 0)
    edge_pos = sum(1 for _, _, _, te in results if te["edge"] > 0)
    print(f"\n  rules with positive OOS P&L      : {pos}/{len(results)}")
    print(f"  rules with positive OOS EDGE     : {edge_pos}/{len(results)}")
    print(f"  rules with OOS win rate > market : {sum(1 for _, _, _, te in results if te['win'] > te['ask'])}/{len(results)}")
    print(f"  median OOS t across rules        : {np.median([te['t'] for _, _, _, te in results]):+.2f}")
    print(f"  best OOS t                       : {max(te['t'] for _, _, _, te in results):+.2f}")

    # the "did the grid-search pick the lucky one" check
    best_tr = max(results, key=lambda r: r[2]["t"])
    print(f"\n  rule with the best TRAIN t: T>={best_tr[0]} ask[{best_tr[1][0]:.2f},"
          f"{best_tr[1][1]:.2f})  train t={best_tr[2]['t']:+.2f}")
    print(f"  its OOS result: n={best_tr[3]['n']} win={best_tr[3]['win']:.4f} "
          f"net=${best_tr[3]['net']:+.2f} t={best_tr[3]['t']:+.2f}")

    # probability that a random rule with this win rate is that good
    print("\n  interpretation: the 18/18 OOS result is the MAXIMUM of the grid's")
    print("  OOS performance, so it is a selected maximum, not an unbiased estimate.")

    # Does the effect hold on BOTH halves for the top rules?
    print("\n--- rules positive in BOTH chronological halves (all data) ---")
    stable = 0
    for tmin, band, _, _ in results:
        allsel = sel_for(tmin, band)
        med = np.median([r["ws"] for r in allsel])
        h1 = stat([r for r in allsel if r["ws"] <= med])
        h2 = stat([r for r in allsel if r["ws"] > med])
        if h1 and h2 and h1["net"] > 0 and h2["net"] > 0:
            stable += 1
            print(f"  T>={tmin} ask[{band[0]:.2f},{band[1]:.2f}): "
                  f"h1 ${h1['net']:+.2f} (t{h1['t']:+.2f})  h2 ${h2['net']:+.2f} (t{h2['t']:+.2f})")
    print(f"  -> {stable}/{len(results)} rules are positive in BOTH halves")


if __name__ == "__main__":
    main()
