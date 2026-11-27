"""_pm_lastlook.py - the sharpest version of the mechanism: the final seconds.

At T seconds into the window the settlement is the mean of [240,300].  By T=295,
55 of those 60 seconds are already observed and fixed, so the outcome is
essentially decided by data in hand.  If the book has not caught up - the
leading side still offered below its locked-in value - that is a trade that
needs no forecast, only arithmetic.

The probability comes from the same calibrated model as _pm_twap_strategy2
(z / (k*sd), k fitted on the design slice), so this cannot manufacture
confidence the noise scale does not support.

Protocol: thresholds and the time window are chosen on the DESIGN slice; the
holdout is evaluated once.

Run: python _pm_lastlook.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402

FEE = S.FEE
STAKE = S.STAKE
K = 2.2                  # calibrated noise inflation (design fit)
MAX_SPREAD = 0.06
MIN_DEPTH = 5.0
DESIGN_FRAC = S.DESIGN_FRAC


def frames_tail(w, t_lo=270.0, t_hi=299.0, step=1.0):
    """Model probability + executable asks for every second in the tail."""
    d = w["d"]
    t = np.asarray(d["t"], float)
    cb = np.asarray(w["cb"], float)
    cum = S.trapz_cum(t, cb)
    out = []
    for T in np.arange(t_lo, t_hi + 1e-9, step):
        i = int(np.searchsorted(t, T))
        if i >= len(t):
            break
        P = float(np.interp(T, t, cb))
        if not np.isfinite(P):
            continue
        elapsed = min(max(T - 240.0, 0.0), 60.0)
        locked = S.area(t, cum, 240.0, min(max(T, 240.0), 300.0))
        settle = (locked + P * (60.0 - elapsed)) / 60.0
        tau = 60.0 - elapsed
        sd = K * np.sqrt(S.SIGMA_1S ** 2 * tau / 3.0 + S.SIGMA_STRIKE ** 2)
        p_up = float(norm.cdf((settle - w["S"]) / sd))
        rec = dict(T=float(T), i=i, p_up=p_up, z=settle - w["S"], sd=sd)
        for side, p in (("Up", p_up), ("Down", 1.0 - p_up)):
            a, b, sz = d[side + "_ask"][i], d[side + "_bid"][i], d[side + "_asksz"][i]
            ok = (all(np.isfinite(v) for v in (a, b, sz)) and 0 < b <= a < 1
                  and (a - b) <= MAX_SPREAD and sz * a >= MIN_DEPTH)
            rec[side] = float(a) if ok else np.nan
            rec[side + "_p"] = p
        out.append(rec)
    return out


def run(wins, t_lo, t_hi, p_min, max_ask=0.995, one_per_window=True):
    rows = []
    for w in wins:
        for fr in frames_tail(w, t_lo, t_hi):
            side = "Up" if fr["p_up"] >= 0.5 else "Down"
            p = fr[side + "_p"]
            ask = fr[side]
            if not np.isfinite(ask) or p < p_min or ask > max_ask:
                continue
            edge = p - ask - FEE * ask * (1 - ask)
            if edge <= 0:
                continue
            rows.append(dict(wi=w["ws"], T=fr["T"], side=side, ask=float(ask),
                             p=float(p), edge=float(edge),
                             won=1 if w["d"]["_winner"] == side else 0))
            if one_per_window:
                break
    return rows


def show(rows, label):
    if not rows:
        print(f"  {label:<38} no trades")
        return None
    p = S.pnl(rows)
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    wi = [r["wi"] for r in rows]
    m, t, _ = B.cluster_t(wi, p)
    lo, hi = B.boot_ci(wi, p)
    st = STAKE * len(rows)
    print(f"  {label:<38} N={len(rows):>3} w={int(won.sum()):>3} ask={ask.mean():.4f} "
          f"real={won.mean():.4f} net=${p.sum():+7.2f} roi={p.sum()/st*100:+6.1f}% "
          f"t={t:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] worst=${p.min():+.2f}")
    return dict(n=len(rows), net=p.sum(), t=t, lo=lo, hi=hi, p=p, rows=rows)


def main():
    W = B.load_all()
    wins = S.prep(W)
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 112)
    print("FINAL-SECONDS TRADE: buy the side whose settlement is already locked in")
    print("=" * 112)
    print(f"windows {len(wins)}   design {len(design)}   holdout {len(hold)}   (k={K})")

    print("\n=== DESIGN: scan the last 30 s ===")
    for t_lo, t_hi in ((270, 285), (285, 292), (292, 299), (285, 299), (270, 299)):
        for pmin in (0.95, 0.97, 0.985, 0.995):
            show(run(design, t_lo, t_hi, pmin), f"T {t_lo}-{t_hi}s  p>={pmin}")
        print()

    print("=== DESIGN: how often is the last frame even reachable? ===")
    for w in design[:5]:
        fr = frames_tail(w, 292, 299)
        print(f"   {w['d']['_slug']}: {len(fr)} frames in 292-299s, "
              f"max p {max([f['p_up'] for f in fr] + [0]):.4f}")
    return wins, design, hold


if __name__ == "__main__":
    main()
