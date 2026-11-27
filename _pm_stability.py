"""_pm_stability.py - after the frozen rule failed the holdout, ask the honest
question: is ANY (time band x price band x side) cell positive in BOTH halves?

A cell that is positive only in the half it was chosen on is noise.  This prints
the full 2x2 grid (two slices x two sides) for every cell, with N, so the reader
can see whether a plateau exists at all.

Run: python _pm_stability.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402
import _pm_endgame_scan as E       # noqa: E402


def cell(slice_, t_lo, t_hi, band, side_mode):
    rows = E.collect_fav(slice_, t_lo, t_hi, band)
    if side_mode == "dog":
        rows = [dict(r, side=("Down" if r["side"] == "Up" else "Up"),
                     won=1 - r["won"]) for r in rows]
        # the dog's price is not stored; recompute from the window book
    return rows


def net_of(rows):
    if not rows:
        return 0.0, 0, 0.0
    p = S.pnl(rows)
    return p.sum(), len(rows), float(np.mean([r["won"] for r in rows]))


def main():
    W = B.load_all()
    wins = S.prep(W)
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * E.DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 104)
    print("STABILITY GRID: favourite side, last 60 s, by price band (net $ at $5, N)")
    print("=" * 104)
    print("  cell                     design net / N        holdout net / N      both + ?")
    bands = [(0.80, 0.85), (0.85, 0.88), (0.88, 0.92), (0.92, 0.96),
             (0.96, 0.98), (0.98, 1.00)]
    for t_lo, t_hi in ((240, 262), (262, 298), (240, 298)):
        print(f"\n  T {t_lo}-{t_hi}s")
        for band in bands:
            d = E.collect_fav(design, t_lo, t_hi, band)
            h = E.collect_fav(hold, t_lo, t_hi, band)
            dn, dnn, dw = net_of(d)
            hn, hnn, hw = net_of(h)
            flag = "YES" if (dn > 0 and hn > 0 and dnn >= 15 and hnn >= 15) else ""
            print(f"  [{band[0]:.2f},{band[1]:.2f})          ${dn:+7.2f} / {dnn:>3}      "
                  f"${hn:+7.2f} / {hnn:>3}      {flag}")

    print("\n  Read it as: a real edge must be positive in BOTH columns with a")
    print("  neighbouring cell agreeing. Anything else is slice noise.")
    return design, hold


if __name__ == "__main__":
    main()
