"""_pm_twap_strategy.py - probability model + trading rule, design -> holdout.

MECHANISM
  These BTC 5-minute binaries settle on the Chainlink 60 s TWAP: its value at the
  window end versus its value at the open.  Verified on this log: the sign of
  (cl_end - cl_open) reproduces the recorded winner in 249/250 windows.

  So at any moment T the settlement is an average whose window [240,300] is
  partly OBSERVED.  With S = price to beat and P(.) the coinbase path:

      settle(T) = [int(240..T) P dt + P(T) * (300 - T)] / 60
      p_up(T)   = Phi((settle(T) - S) / sd)          tau = unsettled seconds
      sd        = sqrt(sigma_1s^2 * tau / 3 + sigma_strike^2)

  sigma_1s is the measured per-second BTC innovation, sigma_strike the error of
  the reconstructed strike.  No fitted gates, no ML, no look-ahead.

PROTOCOL (fixed before the holdout is touched)
  * design  = earlier 70% of windows by start time
  * holdout = later 30%, evaluated once after freezing one rule
  * rule    = first frame in the time band where model edge over the quoted ask
              (after the 0.07*p*(1-p) taker fee) >= threshold; bet that side, $5,
              hold to settlement
  * bar     = holdout bootstrap CI must exclude zero and a 1 s fill delay must
              not flip the sign

Run: python _pm_twap_strategy.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402

FEE = 0.07
STAKE = 5.0
SIGMA_1S = 4.63
SIGMA_STRIKE = 7.62
STRIKE_BIAS = -7.38
STEP = 3.0
MAX_SPREAD = 0.06
MIN_DEPTH = 5.0
DESIGN_FRAC = 0.70


def ffill(a):
    a = np.array(a, float)
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]):
            last = a[i]
        elif np.isfinite(last):
            a[i] = last
    return a


def trapz_cum(t, y):
    out = np.zeros(len(t))
    for i in range(1, len(t)):
        if np.isfinite(y[i]) and np.isfinite(y[i - 1]):
            out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
        else:
            out[i] = out[i - 1]
    return out


def area(t, cum, a, b):
    def at(x):
        if x <= t[0]:
            return 0.0
        if x >= t[-1]:
            return cum[-1]
        return float(np.interp(x, t, cum))
    return at(b) - at(a)


def prep(W):
    """Per window: strike, cumulative coinbase integral, and priced model frames."""
    by_ws = {d["ws"]: d for d in W}
    out = []
    for d in W:
        t = np.asarray(d["t"], float)
        cb = ffill(d["cb_px"])
        if np.isfinite(cb).sum() < 50:
            continue
        prev = by_ws.get(d["ws"] - 300)
        S = np.nan
        if prev is not None:
            pt = np.asarray(prev["t"], float)
            pc = np.asarray(prev["cb_px"], float)
            m = (pt >= 240) & np.isfinite(pc)
            if m.sum() >= 3:
                S = float(np.trapezoid(pc[m], pt[m]) / (pt[m][-1] - pt[m][0])) + STRIKE_BIAS
        if not np.isfinite(S):
            m = (t <= 3) & np.isfinite(cb)
            if m.sum() < 2:
                continue
            S = float(cb[m].mean()) + STRIKE_BIAS
        cum = trapz_cum(t, cb)

        frames = []
        for T in np.arange(30.0, 298.1, STEP):
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                break
            P = float(np.interp(T, t, cb))
            if not np.isfinite(P):
                continue
            elapsed = min(max(T - 240.0, 0.0), 60.0)
            locked = area(t, cum, 240.0, min(max(T, 240.0), 300.0))
            settle = (locked + P * (60.0 - elapsed)) / 60.0
            tau = 60.0 - elapsed
            sd = np.sqrt(SIGMA_1S ** 2 * tau / 3.0 + SIGMA_STRIKE ** 2)
            p_up = float(norm.cdf((settle - S) / sd))
            fr = dict(T=float(T), p_up=p_up, i=i)
            for side, p in (("Up", p_up), ("Down", 1.0 - p_up)):
                a, b, s = d[side + "_ask"][i], d[side + "_bid"][i], d[side + "_asksz"][i]
                ok = (all(np.isfinite(v) for v in (a, b, s)) and 0 < b <= a < 1
                      and (a - b) <= MAX_SPREAD and s * a >= MIN_DEPTH)
                fr[side] = float(a) if ok else np.nan
                fr[side + "_edge"] = (p - a - FEE * a * (1 - a)) if ok else -9.9
                fr[side + "_p"] = p
            frames.append(fr)
        out.append(dict(d=d, ws=d["ws"], S=S, t=t, cb=cb, frames=frames))
    out.sort(key=lambda w: w["ws"])
    return out


def run(wins, t_lo, t_hi, thr, one_per_window=True):
    rows = []
    for w in wins:
        d = w["d"]
        for fr in w["frames"]:
            if fr["T"] < t_lo or fr["T"] > t_hi:
                continue
            side = "Up" if fr["Up_edge"] >= fr["Down_edge"] else "Down"
            e = fr[side + "_edge"]
            if e < thr:
                if one_per_window:
                    continue
                continue
            rows.append(dict(wi=w["ws"], T=fr["T"], side=side, ask=fr[side],
                             p=fr[side + "_p"], edge=e,
                             won=1 if d["_winner"] == side else 0))
            if one_per_window:
                break
    return rows


def pnl(rows, stake=STAKE):
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    sh = stake / ask
    return sh * won - stake - sh * FEE * ask * (1 - ask)


def report(rows, label, quiet=False):
    if not rows:
        if not quiet:
            print(f"  {label:<34} no trades")
        return None
    p = pnl(rows)
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    wi = [r["wi"] for r in rows]
    m, t, _ = B.cluster_t(wi, p)
    lo, hi = B.boot_ci(wi, p)
    bp = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean())) if len(won) else 1.0
    st = STAKE * len(rows)
    srt = np.sort(p)[::-1]
    top5 = srt[:5].sum() / p.sum() if p.sum() else np.nan
    if not quiet:
        print(f"  {label:<34} N={len(rows):>3} w={int(won.sum()):>3} pay={ask.mean():.4f} "
              f"real={won.mean():.4f} net=${p.sum():+7.2f} roi={p.sum()/st*100:+6.1f}% "
              f"t={t:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] bp={bp:.3f} "
              f"worst=${p.min():+.2f} top5={top5*100:4.0f}%")
    return dict(n=len(rows), net=p.sum(), t=t, lo=lo, hi=hi, p=p, rows=rows,
                top5=top5, bp=bp)


def main():
    W = B.load_all()
    wins = prep(W)
    cut = int(len(wins) * DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 112)
    print("TWAP-SETTLEMENT MODEL -> TRADING RULE")
    print("=" * 112)
    print(f"windows {len(wins)}   design {len(design)}  holdout {len(hold)}   "
          f"design {design[0]['ws']}..{design[-1]['ws']}   holdout {hold[0]['ws']}..{hold[-1]['ws']}")

    # ---- calibration on the design slice ----------------------------------
    print("\n=== 1) IS THE MODEL CALIBRATED? (design, every 3rd second) ===")
    P, Y, A = [], [], []
    for w in design:
        y = 1 if w["d"]["_winner"] == "Up" else 0
        for fr in w["frames"]:
            if np.isfinite(fr["Up"]):
                P.append(fr["p_up"]); Y.append(y); A.append(fr["Up"])
    P, Y, A = np.array(P), np.array(Y, float), np.array(A)
    print(f"  frames {len(P)}   model mean {P.mean():.4f}   market-ask mean {A.mean():.4f}   realised {Y.mean():.4f}")
    print(f"  Brier  model {np.mean((P-Y)**2):.5f}   market {np.mean((A-Y)**2):.5f}   "
          f"(lower is better)")
    print("  bucket         n    model   realised")
    for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
        m = (P >= lo) & (P < hi)
        if m.sum() > 5:
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>7}  {P[m].mean():.4f}  {Y[m].mean():.4f}")

    # ---- edge surface on design -------------------------------------------
    print("\n=== 2) EDGE SURFACE ON DESIGN (first qualifying frame per window) ===")
    cells = []
    for tlo, thi in ((30, 240), (240, 262), (262, 298), (240, 298), (30, 298)):
        for thr in (0.005, 0.01, 0.02, 0.03, 0.05):
            r = run(design, tlo, thi, thr)
            res = report(r, f"T {tlo}-{thi}s  edge>={thr:.3f}")
            if res:
                cells.append(((tlo, thi, thr), res))

    # ---- freeze -----------------------------------------------------------
    print("\n=== 3) PLATEAU (neighbours must agree) ===")
    ok = [c for c in cells if c[1]["n"] >= 30]
    ok.sort(key=lambda c: -c[1]["net"])
    for k, v in ok[:10]:
        print(f"    T {k[0]}-{k[1]}s  edge>={k[2]:.3f}  N={v['n']:>3} net=${v['net']:+7.2f} "
              f"roi={v['net']/(5*v['n'])*100:+6.1f}%  t={v['t']:+5.2f} "
              f"CI=[{v['lo']:+.2f},{v['hi']:+.2f}]")
    return wins, design, hold, cells


if __name__ == "__main__":
    main()
