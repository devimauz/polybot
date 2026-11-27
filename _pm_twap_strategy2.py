"""_pm_twap_strategy2.py - same mechanism as _pm_twap_strategy.py, but the
probability is CALIBRATED before any edge is claimed.

The raw model is overconfident (Brier 0.191 vs the market's 0.165): it says 0.99
where the real hit rate is 0.88, and 0.01 where it is 0.15.  Trading an
uncalibrated probability against a price is meaningless, so the noise scale is
fitted on the DESIGN slice only (one parameter: an inflation factor k on sd).

    sd = k * sqrt(sigma_1s^2 * tau/3 + sigma_strike^2)

Then, and only then: edge = p_calibrated - ask - fee.  Rule selection on design,
one holdout run.

Run: python _pm_twap_strategy2.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S  # noqa: E402  (prep / pnl / report helpers)

FEE = S.FEE
STAKE = S.STAKE
DESIGN_FRAC = S.DESIGN_FRAC


def frames_with_z(wins):
    """Recompute frames keeping z0 = settle - S and sd0 so k is a free parameter."""
    out = []
    for w in wins:
        t, cb, cum, S0 = w["t"], w["cb"], None, w["S"]
        # rebuild the cumulative integral here (prep() keeps them only locally)
        cum = S.trapz_cum(t, cb)
        fr_list = []
        for fr in w["frames"]:
            T, i = fr["T"], fr["i"]
            P = float(np.interp(T, t, cb))
            elapsed = min(max(T - 240.0, 0.0), 60.0)
            locked = S.area(t, cum, 240.0, min(max(T, 240.0), 300.0))
            settle = (locked + P * (60.0 - elapsed)) / 60.0
            tau = 60.0 - elapsed
            sd0 = np.sqrt(S.SIGMA_1S ** 2 * tau / 3.0 + S.SIGMA_STRIKE ** 2)
            fr_list.append(dict(T=T, i=i, z=settle - S0, sd0=sd0,
                                Up=fr["Up"], Down=fr["Down"]))
        out.append(dict(d=w["d"], ws=w["ws"], frames=fr_list))
    return out


def p_up(fr, k):
    return float(norm.cdf(fr["z"] / (fr["sd0"] * k)))


def collect(wins, t_lo, t_hi, thr, k):
    rows = []
    for w in wins:
        d = w["d"]
        for fr in w["frames"]:
            if fr["T"] < t_lo or fr["T"] > t_hi:
                continue
            pu = p_up(fr, k)
            best = None
            for side, p, ask in (("Up", pu, fr["Up"]), ("Down", 1.0 - pu, fr["Down"])):
                if not np.isfinite(ask):
                    continue
                e = p - ask - FEE * ask * (1 - ask)
                if best is None or e > best[2]:
                    best = (side, ask, e, p)
            if best and best[2] >= thr:
                side, ask, e, p = best
                rows.append(dict(wi=w["ws"], T=fr["T"], side=side, ask=float(ask),
                                 p=float(p), edge=float(e),
                                 won=1 if d["_winner"] == side else 0))
                break
    return rows


def brier(wins, k, step=3):
    """Brier score of the model over all priced frames on these windows."""
    se, ns = 0.0, 0
    for w in wins:
        y = 1 if w["d"]["_winner"] == "Up" else 0
        for fr in w["frames"]:
            if not np.isfinite(fr["Up"]):
                continue
            se += (p_up(fr, k) - y) ** 2
            ns += 1
    return se / max(1, ns), ns


def main():
    W = B.load_all()
    wins = frames_with_z(S.prep(W))
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 116)
    print("CALIBRATED TWAP-SETTLEMENT MODEL -> TRADING RULE")
    print("=" * 116)
    print(f"windows {len(wins)}   design {len(design)}   holdout {len(hold)}")

    # ---- 1. fit the one free parameter on the design slice -----------------
    print("\n=== 1) CALIBRATION FIT ON DESIGN (k inflates the noise scale) ===")
    best = None
    for k in np.arange(1.0, 4.01, 0.1):
        b, n = brier(design, k)
        if best is None or b < best[1]:
            best = (k, b)
        if abs(k * 10 - round(k * 10)) < 1e-9 and round(k * 10) % 3 == 0:
            print(f"   k={k:.1f}  Brier {b:.5f}")
    K, Bmodel = best
    print(f"  best k = {K:.2f}   design Brier {Bmodel:.5f}")

    # market Brier on the same frames for reference
    se = ns = 0
    for w in design:
        y = 1 if w["d"]["_winner"] == "Up" else 0
        for fr in w["frames"]:
            if np.isfinite(fr["Up"]) and 0 < fr["Up"] < 1:
                se += (fr["Up"] - y) ** 2
                ns += 1
    Bmkt = se / ns
    print(f"  market (ask as probability) Brier {Bmkt:.5f} on {ns} frames")
    print(f"  -> model {'BEATS' if Bmodel < Bmkt else 'does NOT beat'} the market "
          f"({(Bmkt - Bmodel) * 1000:+.2f} milli-Brier)")

    print("\n  calibration after fit (design):")
    P, Y = [], []
    for w in design:
        y = 1 if w["d"]["_winner"] == "Up" else 0
        for fr in w["frames"]:
            if np.isfinite(fr["Up"]):
                P.append(p_up(fr, K)); Y.append(y)
    P, Y = np.array(P), np.array(Y, float)
    for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
        m = (P >= lo) & (P < hi)
        if m.sum() > 5:
            print(f"    [{lo:.2f},{hi:.2f})  n={m.sum():>6}  model {P[m].mean():.4f}  realised {Y[m].mean():.4f}")

    # ---- 2. edge surface, design only --------------------------------------
    print("\n=== 2) EDGE SURFACE ON DESIGN (calibrated p, first qualifying frame per window) ===")
    cells = []
    for tlo, thi in ((30, 240), (240, 262), (262, 298), (240, 298)):
        for thr in (0.005, 0.01, 0.02, 0.03, 0.05):
            r = collect(design, tlo, thi, thr, K)
            res = S.report(r, f"T {tlo}-{thi}s edge>={thr:.3f}")
            if res:
                cells.append(((tlo, thi, thr), res))
    print("\n=== 3) PLATEAU ===")
    for k_, v in sorted([c for c in cells if c[1]["n"] >= 30],
                        key=lambda c: -c[1]["net"])[:8]:
        print(f"    T {k_[0]}-{k_[1]}s edge>={k_[2]:.3f}  N={v['n']:>3} net=${v['net']:+7.2f} "
              f"roi={v['net']/(5*v['n'])*100:+6.1f}% t={v['t']:+5.2f} "
              f"CI=[{v['lo']:+.2f},{v['hi']:+.2f}]")
    return wins, design, hold, K, cells


if __name__ == "__main__":
    main()
