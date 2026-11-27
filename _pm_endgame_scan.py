"""_pm_endgame_scan.py - measure the edge surface of the LAST 40 SECONDS directly,
on the design slice, then test the frozen band once on the holdout.

Why this slice: the settlement is the mean of the last 60 s, so for T in [240,300]
a fraction (T-240)/60 of the settlement is already fixed.  The last 40 s is where
that lock-in is large.  No model is involved here - this measures the book's own
calibration: for each 1-cent ask bucket, did the favourite win more often than
the price implied, after the taker fee?

Protocol: design = earlier 70% of windows; pick a band with a PLATEAU (adjacent
buckets agree); freeze; run the holdout once with the full diagnostic battery.

Run: python _pm_endgame_scan.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402

FEE = S.FEE
STAKE = S.STAKE
DESIGN_FRAC = S.DESIGN_FRAC


def collect_fav(wins, t_lo, t_hi, band, one_per_window=True, latency=0.0):
    """Buy the favourite (higher ask) when its ask sits in `band`."""
    rows = []
    for w in wins:
        d = w["d"]
        t = np.asarray(d["t"], float)
        for fr in w["frames"]:
            T = fr["T"]
            if T < t_lo or T > t_hi:
                continue
            up_a, dn_a = fr["Up"], fr["Down"]
            if not (np.isfinite(up_a) and np.isfinite(dn_a)):
                continue
            side = "Up" if up_a >= dn_a else "Down"
            ask = up_a if side == "Up" else dn_a
            if not (band[0] <= ask < band[1]):
                continue
            if latency > 0:
                j = int(np.searchsorted(t, T + latency))
                if j >= len(t):
                    continue
                a2 = d[side + "_ask"][j]
                b2 = d[side + "_bid"][j]
                s2 = d[side + "_asksz"][j]
                if not (all(np.isfinite(v) for v in (a2, b2, s2))
                        and 0 < b2 <= a2 < 1 and (a2 - b2) <= 0.06
                        and s2 * a2 >= 5.0):
                    continue
                ask = float(a2)
            rows.append(dict(wi=w["ws"], T=T, side=side, ask=float(ask),
                             won=1 if d["_winner"] == side else 0))
            if one_per_window:
                break
    return rows


def edge_table(wins, t_lo, t_hi, bins):
    print(f"\n  favourite ask buckets, T in [{t_lo},{t_hi}]  (design slice)")
    print("  bucket        N   win%   paid   edge_gross  fee    edge_net   net$")
    out = []
    for lo in bins:
        hi = lo + 0.02
        rows = collect_fav(wins, t_lo, t_hi, (lo, hi))
        if len(rows) < 8:
            print(f"  [{lo:.2f},{hi:.2f})  {len(rows):>4}   (too few)")
            continue
        ask = np.array([r["ask"] for r in rows])
        won = np.array([r["won"] for r in rows], float)
        fee = FEE * ask * (1 - ask)
        p = S.pnl(rows)
        print(f"  [{lo:.2f},{hi:.2f})  {len(rows):>4}  {won.mean()*100:5.1f}  "
              f"{ask.mean():.3f}  {(won.mean()-ask.mean())*100:+6.2f}c  "
              f"{fee.mean()*100:5.2f}c  {(won.mean()-ask.mean()-fee.mean())*100:+6.2f}c  "
              f"{p.sum():+7.2f}")
        out.append((lo, hi, len(rows), won.mean(), ask.mean(), p.sum()))
    return out


def diagnostics(wins, t_lo, t_hi, band, label):
    print(f"\n=== {label} ===")
    r = collect_fav(wins, t_lo, t_hi, band)
    res = S.report(r, label)
    if not res:
        return None
    rows = res["rows"]
    p = S.pnl(rows)
    ws = [x["wi"] for x in rows]
    # latency sweep
    print("  fill delay:")
    for L in (0.0, 0.3, 0.5, 1.0):
        rr = collect_fav(wins, t_lo, t_hi, band, latency=L)
        S.report(rr, f"    latency {L:.1f}s")
    # Up vs Down
    for side in ("Up", "Down"):
        sub = [x for x in rows if x["side"] == side]
        S.report(sub, f"  side {side}")
    # outcome placebo: same bets, winners shuffled within the sample
    rng = np.random.default_rng(5)
    base = np.mean([x["won"] for x in rows])
    ask = np.array([x["ask"] for x in rows])
    obs = p.sum()
    sims = []
    for _ in range(4000):
        w = rng.binomial(1, base, len(rows)).astype(float)
        sh = STAKE / ask
        sims.append((sh * w - STAKE - sh * FEE * ask * (1 - ask)).sum())
    sims = np.array(sims)
    print(f"  placebo (outcomes reshuffled at the observed rate): p(>=obs)="
          f"{(sims >= obs).mean():.4f}  median ${np.median(sims):+.2f}")
    # time-band split inside the window
    print("  by entry time:")
    for a, b in ((t_lo, t_lo + 10), (t_lo + 10, t_lo + 20), (t_lo + 20, t_hi + 0.001)):
        sub = [x for x in rows if a <= x["T"] < b]
        S.report(sub, f"    T {a:.0f}-{b:.0f}s")
    return res


def main():
    W = B.load_all()
    wins = S.prep(W)
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 108)
    print("LAST-SECONDS FAVOURITE: EDGE SURFACE (DESIGN) -> FROZEN RULE ON HOLDOUT")
    print("=" * 108)
    print(f"windows {len(wins)}   design {len(design)}   holdout {len(hold)}")

    bins = [round(0.50 + 0.02 * i, 2) for i in range(0, 25)]
    # the lock-in grows with T, so scan three time bands
    for tlo, thi in ((240, 262), (262, 276), (276, 298)):
        edge_table(design, tlo, thi, bins)

    print("\n" + "-" * 108)
    print("Rule family: buy the favourite, T>=262, ask in a band (the lock-in argument)")
    for band in ((0.80, 0.96), (0.84, 0.96), (0.86, 0.98), (0.90, 0.98)):
        S.report(collect_fav(design, 262, 298, band), f"design band {band}")
    return wins, design, hold


if __name__ == "__main__":
    main()
