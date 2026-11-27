"""_pm_final_rule.py - the ONE rule frozen from the design scan, tested once on the
untouched holdout, with the full diagnostic battery.

FROZEN RULE (chosen on the design slice, `_pm_endgame_scan.py`):
    buy the FAVOURITE side, hold to settlement, when
      * T in [240, 262) seconds into the window, and
      * the favourite's ask is in [0.88, 0.96)
    one bet per window, $5, taker fee C*0.07*p*(1-p).

Why it was chosen: on the design slice those four adjacent 2-cent buckets are the
only region where the favourite won more often than its price, after the fee
(+4.60c, +4.92c, +2.18c, +2.70c per contract, N=49/51/93/81) - a plateau, not a
single lucky cell.  Everything above 0.96 is a no-edge zone, everything below
0.88 is noise, and the same band at T >= 262 does not work.

Diagnostics reported here: N, net, ROI, worst trade, cluster-t, bootstrap CI,
outcome placebo, Up vs Down split, entry-time split, top-5 P&L share, latency
sweep 0.3/0.5/1.0 s, and the same rule priced at the next best ask (a fill
penalty test).

Run: python _pm_final_rule.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402
import _pm_endgame_scan as E       # noqa: E402

FEE = S.FEE
STAKE = S.STAKE
T_LO, T_HI = 240.0, 262.0
BAND = (0.88, 0.96)


def full(rows, label, quiet=False):
    res = S.report(rows, label)
    if not res or not res["n"]:
        return res
    p = S.pnl(rows)
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    if quiet:
        return res
    print(f"      equity: final ${p.sum():+.2f}  best ${p.max():+.2f}  worst ${p.min():+.2f}  "
          f"staked ${STAKE*len(rows):.0f}")
    print(f"      entries: mean T={np.mean([r['T'] for r in rows]):.1f}s  "
          f"mean ask={ask.mean():.4f}  realised={won.mean():.4f}  "
          f"edge={((won-ask).mean())*100:+.2f}c gross, "
          f"{((won-ask-FEE*ask*(1-ask)).mean())*100:+.2f}c net")
    # drawdown
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq])) -
                np.concatenate([[0], eq])).max())
    streak = best = 0
    for x in p:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    print(f"      max drawdown ${dd:.2f}   longest losing streak {best}")
    return res


def placebo(rows, n=8000, seed=3):
    p = S.pnl(rows)
    ask = np.array([r["ask"] for r in rows])
    base = np.mean([r["won"] for r in rows])
    rng = np.random.default_rng(seed)
    sims = np.empty(n)
    for i in range(n):
        w = rng.binomial(1, base, len(rows)).astype(float)
        sh = STAKE / ask
        sims[i] = (sh * w - STAKE - sh * FEE * ask * (1 - ask)).sum()
    return float((sims >= p.sum()).mean()), float(np.median(sims)), float(np.percentile(sims, 95))


def main():
    W = B.load_all()
    wins = S.prep(W)
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * E.DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 112)
    print(f"FROZEN RULE: favourite, T in [{T_LO:.0f},{T_HI:.0f})s, ask in [{BAND[0]:.2f},{BAND[1]:.2f})")
    print("=" * 112)
    print(f"windows {len(wins)}   design {len(design)}   holdout {len(hold)}")

    def rows_for(slice_):
        return E.collect_fav(slice_, T_LO, T_HI, BAND)

    d = rows_for(design)
    h = rows_for(hold)
    print("\n--- DESIGN (this is where the band came from; not evidence) ---")
    full(d, "design")
    print("\n--- HOLDOUT (untouched, evaluated once) ---")
    res = full(h, "HOLDOUT")
    if not res:
        print("  no trades in the holdout")
        return

    rows = res["rows"]
    print("\n  holdout detail:")
    for L in (0.3, 0.5, 1.0):
        S.report(E.collect_fav(hold, T_LO, T_HI, BAND, latency=L), f"    fill {L:.1f}s later")
    for side in ("Up", "Down"):
        S.report([r for r in rows if r["side"] == side], f"    side {side}")
    for a, b in ((240, 250), (250, 258), (258, 262.001)):
        S.report([r for r in rows if a <= r["T"] < b], f"    entry T {a}-{b:.0f}s")
    pv = placebo(rows)
    print(f"    placebo: p(>=observed)={pv[0]:.4f}  shuffled median ${pv[1]:+.2f}  "
          f"95th pct ${pv[2]:+.2f}")
    print(f"    top-5 P&L share: {res['top5']*100:.1f}% of total")

    # ---- verdict -----------------------------------------------------------
    lo, hi = res["lo"], res["hi"]
    clears = (lo > 0)
    print("\n" + "=" * 112)
    print("VERDICT")
    print("=" * 112)
    print(f"  holdout N={res['n']}  net ${res['net']:+.2f}  "
          f"t={res['t']:+.2f}  bootstrap CI95 [{lo:+.2f}, {hi:+.2f}] $/trade")
    if clears:
        print("  CI excludes zero -> the rule clears the pre-registered bar on the holdout.")
    else:
        print("  CI includes zero -> the rule does NOT clear the bar. No edge is")
        print("  established on the holdout; the design-slice cell was inside the noise.")


if __name__ == "__main__":
    main()
