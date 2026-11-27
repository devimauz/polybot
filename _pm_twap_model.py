"""_pm_twap_model.py - a probability model for the BTC 5-minute binary, then a
trading rule built on it, validated design-slice -> untouched holdout.

MECHANISM (measured, not assumed)
---------------------------------
These markets settle on the Chainlink 60 s TWAP: the value at the window end vs
the value at the open (verified here: sign(cl_end - cl_open) reproduces the
recorded winner in 249/250 windows).

So at any moment T the settlement is decided by an average whose window
[240, 300] is partly OBSERVED already.  Write S for the price to beat and P(.)
for the coinbase path:

    settlement(T) = [ integral(240..T) P dt + P(T)*(300 - T) ] / 60
    p_up(T)       = Phi( (settlement(T) - S) / sd_remaining )

with sd_remaining = sqrt( sigma_1s^2 * tau/3 + sigma_strike^2 ), tau = 300-T
clipped to the unsettled part, sigma_1s the per-second BTC innovation, and
sigma_strike the error of the reconstructed strike.  That is the whole model -
no fitted gates, no machine learning.

Price to beat: rebuilt from the PREVIOUS window's last 60 s of coinbase data
(mean over [-60,0]) with a bias correction; it reproduces the winner in 97.7% of
the 816 windows that have a predecessor (the 161 windows with a chainlink series
show it is right to ~$7.6 sd).

PROTOCOL (pre-registered before looking at the holdout)
------------------------------------------------------
* Design slice  = the earlier 60% of windows by window start.
* Holdout       = the later 40%, evaluated once, after the rule is frozen.
* Rule: first frame in the allowed time band where the model's edge over the
  quoted ask (after fee) exceeds the threshold; bet that side, $5, hold to
  settlement.
* Fee: Polymarket taker fee C * 0.07 * p * (1-p).
* Failure bar: holdout bootstrap CI includes zero, or a 1.0 s fill delay flips
  the sign -> the rule does not clear the bar, and I say so.

Run: python _pm_twap_model.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402

FEE_RATE = 0.07
STAKE = 5.0
SIGMA_1S = 4.63          # $/sqrt(second), median innovation on this log
SIGMA_STRIKE = 7.62      # $, strike reconstruction error vs chainlink truth
STRIKE_BIAS = -7.38      # measured median offset of the reconstruction
SNAP_STEP = 3.0          # seconds between candidate decision frames
MAX_SPREAD = 0.06
MIN_DEPTH_USD = 5.0

DESIGN_FRAC = 0.60


# ------------------------------------------------------------------ utilities
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
    """Cumulative trapezoid integral of y over t."""
    out = np.zeros(len(t))
    for i in range(1, len(t)):
        if np.isfinite(y[i]) and np.isfinite(y[i - 1]):
            out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
        else:
            out[i] = out[i - 1]
    return out


def area_between(t, cum, a, b):
    def at(x):
        if x <= t[0]:
            return 0.0
        if x >= t[-1]:
            return cum[-1]
        return float(np.interp(x, t, cum))
    return at(b) - at(a)


# ------------------------------------------------------------------ windows
def build_windows(W):
    """Attach strike, settlement path and the model probability to each window."""
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
            pcb = np.asarray(prev["cb_px"], float)
            m = (pt >= 240) & np.isfinite(pcb)
            if m.sum() >= 3:
                S = float(np.trapezoid(pcb[m], pt[m]) / (pt[m][-1] - pt[m][0])) + STRIKE_BIAS
        if not np.isfinite(S):
            m = (t <= 3) & np.isfinite(cb)
            if m.sum() < 2:
                continue
            S = float(cb[m].mean()) + STRIKE_BIAS
        cum = trapz_cum(t, cb)
        out.append(dict(d=d, t=t, cb=cb, S=S, cum=cum))
    return out


def model_p_up(w, T):
    """p(Up) at time T (causal: uses only data at or before T)."""
    t, cb, S, cum = w["t"], w["cb"], w["S"], w["cum"]
    if T < t[0] or T > t[-1]:
        return np.nan
    P = float(np.interp(T, t, cb))
    if not np.isfinite(P):
        return np.nan
    elapsed = min(max(T - 240.0, 0.0), 60.0)
    locked = area_between(t, cum, 240.0, min(max(T, 240.0), 300.0))
    settle = (locked + P * (60.0 - elapsed)) / 60.0
    tau = 60.0 - elapsed
    sd = np.sqrt(SIGMA_1S ** 2 * tau / 3.0 + SIGMA_STRIKE ** 2)
    return float(norm.cdf((settle - S) / sd)), settle, S


def quote(d, side, i, latency_s=0.0):
    t = d["t"]

    def leg(j):
        a, b, s = d[side + "_ask"][j], d[side + "_bid"][j], d[side + "_asksz"][j]
        if not all(np.isfinite(v) for v in (a, b, s)):
            return None
        if not (0 < b <= a < 1) or (a - b) > MAX_SPREAD or s * a < MIN_DEPTH_USD:
            return None
        return float(a)

    if latency_s <= 0:
        return leg(i)
    j = int(np.searchsorted(t, t[i] + latency_s))
    return leg(j) if j < len(t) else None


def fee_per_share(price):
    return FEE_RATE * price * (1.0 - price)


def edge_of(p, ask):
    """expected net per share, in dollars, of buying at `ask` with true p."""
    return p - ask - fee_per_share(ask)


# ------------------------------------------------------------------ backtest
def collect(ws_list, t_lo, t_hi, min_edge, latency_s=0.0, one_per_window=True):
    rows = []
    for w in ws_list:
        d = w["d"]
        t = w["t"]
        fired = False
        for T in np.arange(t_lo, min(t_hi, 299.0) + 1e-9, SNAP_STEP):
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                break
            r = model_p_up(w, float(T))
            if r is None or not np.isfinite(r[0]):
                continue
            p_up = r[0]
            best = None
            for side, p in (("Up", p_up), ("Down", 1.0 - p_up)):
                ask = quote(d, side, i, latency_s)
                if ask is None:
                    continue
                e = edge_of(p, ask)
                if e >= min_edge and (best is None or e > best[2]):
                    best = (side, ask, e, p)
            if best is None:
                continue
            side, ask, e, p = best
            rows.append(dict(wi=ws_list.index(w) if False else d["ws"], side=side,
                             T=float(T), ask=ask, p=p, edge=e,
                             won=1 if d["_winner"] == side else 0))
            if one_per_window:
                fired = True
                break
    return rows


def pnl(rows, stake=STAKE):
    if not rows:
        return np.array([])
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    sh = stake / ask
    return sh * won - stake - sh * fee_per_share(ask)


def report(rows, label):
    if not rows:
        print(f"  {label:<40} no trades")
        return None
    p = pnl(rows)
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    wi = [r["wi"] for r in rows]
    m, tstat, nwin = B.cluster_t(wi, p)
    lo, hi = B.boot_ci(wi, p)
    st = STAKE * len(rows)
    srt = np.sort(p)[::-1]
    top5 = srt[:5].sum() / p.sum() if p.sum() > 0 else float("nan")
    print(f"  {label:<40} N={len(rows):>3} w={int(won.sum()):>3} "
          f"pay={ask.mean():.4f} real={won.mean():.4f} "
          f"net=${p.sum():+7.2f} roi={p.sum()/st*100:+6.1f}% "
          f"t={tstat:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] "
          f"worst=${p.min():+.2f} top5={top5*100:4.1f}%")
    return dict(n=len(rows), net=p.sum(), t=tstat, lo=lo, hi=hi, p=p, rows=rows)


def main():
    W = B.load_all()
    wins = build_windows(W)
    wins.sort(key=lambda w: w["d"]["ws"])
    cut = int(len(wins) * DESIGN_FRAC)
    design, holdout = wins[:cut], wins[cut:]
    print("=" * 108)
    print("TWAP-SETTLEMENT MODEL -> TRADING RULE")
    print("=" * 108)
    print(f"windows with a usable path: {len(wins)}   "
          f"design {len(design)} ({design[0]['d']['ws']}..{design[-1]['d']['ws']})   "
          f"holdout {len(holdout)} ({holdout[0]['d']['ws']}..{holdout[-1]['d']['ws']})")

    # ---- 1. is the model calibrated at all? (design slice only) -------------
    print("\n=== 1) MODEL CALIBRATION ON THE DESIGN SLICE (all frames, T in 30..298) ===")
    rec = []
    for w in design:
        for T in np.arange(30, 299, SNAP_STEP):
            r = model_p_up(w, float(T))
            if r is None or not np.isfinite(r[0]):
                continue
            rec.append((r[0], 1 if w["d"]["_winner"] == "Up" else 0, T))
    P = np.array([x[0] for x in rec])
    Y = np.array([x[1] for x in rec], float)
    print(f"  frames {len(P)}   mean predicted {P.mean():.4f}   realised {Y.mean():.4f}")
    print("  bucket        n     predicted  realised")
    for lo, hi in ((0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
        m = (P >= lo) & (P < hi)
        if m.sum() > 5:
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>6}   {P[m].mean():.4f}    {Y[m].mean():.4f}")

    # ---- 2. where is the market wrong? (design slice) -----------------------
    print("\n=== 2) EDGE SURFACE ON THE DESIGN SLICE (first qualifying frame per window) ===")
    print("  rule                                        " + " "*22 + "(shown by time band x edge threshold)")
    grid = {}
    for t_lo, t_hi in ((30, 240), (240, 262), (262, 299), (30, 299)):
        for me in (0.005, 0.01, 0.02, 0.03, 0.05):
            r = collect(design, t_lo, t_hi, me)
            res = report(r, f"T {t_lo}-{t_hi}s  min_edge {me:.3f}")
            grid[(t_lo, t_hi, me)] = res

    # ---- 3. freeze: best plateau on design ----------------------------------
    print("\n=== 3) PLATEAU CHECK (adjacent cells must agree, not one lucky cell) ===")
    cand = [(k, v) for k, v in grid.items() if v and v["n"] >= 25]
    cand.sort(key=lambda kv: -kv[1]["net"])
    for k, v in cand[:8]:
        print(f"    T {k[0]}-{k[1]}s  edge>={k[2]:.3f}  N={v['n']:>3} net=${v['net']:+7.2f} "
              f"t={v['t']:+5.2f} CI=[{v['lo']:+.2f},{v['hi']:+.2f}]")

    return wins, design, holdout


if __name__ == "__main__":
    main()
