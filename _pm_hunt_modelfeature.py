#!/usr/bin/env python3
"""_pm_hunt_modelfeature.py — the settlement model's DISAGREEMENT with the book
as a price-surface feature across ALL times (not just T>=240 where Late-Lock lives).

The Late-Lock Tail (T>=240, model_p - ask >= 6c) is a confirmed non-latency
engine. Open question: is the disagreement signal (model_p - ask) informative
at OTHER times too, or is it only a tail phenomenon? If yes at some T, that is
a NEW non-latency family (model-gated entry earlier in the window).

DESIGN slice only; freeze by plateau + halves; one holdout shot.

Model (from settlement-mechanics-btc5m.md):
  settle(T) = (integral_240..T cb + cb(T)*(300-T)) / 60   [only when T>240]
  For T<=240 the whole settlement window is unobserved; we use the FULL
  remaining-window expectation under the random-walk model instead.
  p_up = Phi((settle_exp - strike)/sd)
  sd = sigma1s * 2.2 * sqrt(tau/3) + 7.62   (fitted, from the skill)
  sigma1s ~ 4.63, strike = cb at window open (reconstruct), bias ~ -7.4
"""
import os, pickle, math
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0
SIGMA1S = 4.63
SIGMA_STRIKE = 7.62
NOISE_SCALE = 2.2
BIAS = -7.4

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def strike_of(d):
    """CB price at window open (t=0). Use first finite cb_px."""
    cb = d["cb_px"]
    f = np.where(np.isfinite(cb))[0]
    return cb[f[0]] if len(f) else None

def settle_partial_mean(d, T):
    """Mean cb_px over the OBSERVED part of [240, min(T,300)]."""
    t = d["t"]; cb = d["cb_px"]
    m = (t >= 240) & (t <= min(T, 300)) & np.isfinite(cb)
    return cb[m].mean() if m.any() else None

def model_p(d, T, side):
    """P(settle side wins) at clock T, using the random-walk model.
    Returns (p, ok). For T>=240 we use the observed partial; for T<240 the
    expectation of the settlement equals current cb (martingale), with the
    remaining-window noise."""
    strike = strike_of(d)
    if strike is None: return None, False
    # current cb
    t = d["t"]
    i = int(np.searchsorted(t, T, side="right")) - 1
    if i < 0: return None, False
    # walk back to last finite cb at/before i
    j = i
    while j >= 0 and not np.isfinite(d["cb_px"][j]): j -= 1
    if j < 0: return None, False
    cur_cb = d["cb_px"][j]
    if T >= 240:
        part = settle_partial_mean(d, T)
        if part is None: part = cur_cb
        tau = max(300 - T, 1.0)
        # expected settlement = (observed_part * (T-240) + cur_cb * tau) / 60
        obs_frac = min(max(T - 240, 0), 60)
        settle_exp = (part * obs_frac + cur_cb * tau) / 60.0
    else:
        # full remaining window unobserved: martingale => settle_exp = cur_cb
        tau = 60.0  # full settlement window ahead
        settle_exp = cur_cb
    sd = SIGMA1S * NOISE_SCALE * math.sqrt(tau / 3.0) + SIGMA_STRIKE
    dist = settle_exp - strike + BIAS  # bias correction
    p_up = norm.cdf(dist / sd)
    p = p_up if side == "Up" else (1 - p_up)
    return p, True

def first_trade(d, T, band=(0.10, 0.90), margin=0.0):
    """First frame at/after T where model_p - ask >= margin (either side),
    ask in band, depth clears a $5 ticket."""
    t = d["t"]
    i0 = int(np.searchsorted(t, T, side="right")) - 1
    if i0 < 0: return None
    for i in range(i0, len(t)):
        for side in ("Up", "Down"):
            ask = d[f"{side}_ask"][i]; asz = d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0] <= ask <= band[1]): continue
            sh = int(STAKE/ask)
            if sh < 5 or not np.isfinite(asz) or asz < sh: continue
            # use the model at this frame's clock
            Tt = d["t"][i]
            p, ok = model_p(d, Tt, side)
            if not ok or p is None: continue
            if p - ask < margin: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            return dict(ws=d["ws"], side=side, ask=ask, won=won, pnl=pnl,
                        p=p, margin=p-ask, T=Tt, sh=sh)
    return None

def cluster_t(rows):
    if not rows: return float("nan")
    pnl=np.array([r["pnl"] for r in rows]); ws=np.array([r["ws"] for r in rows])
    uq=np.unique(ws); s=np.array([pnl[ws==w].sum() for w in uq])
    if len(s)<2: return float("nan")
    m=s.mean(); se=s.std(ddof=1)/math.sqrt(len(s))
    return m/se if se>0 else float("nan")

def report(tag, rows):
    if not rows: print(f"  {tag}: N=0"); return
    n=len(rows); pnl=np.array([r["pnl"] for r in rows])
    wr=np.mean([r["won"] for r in rows]); ask=np.mean([r["ask"] for r in rows])
    pm=np.mean([r["p"] for r in rows]); mg=np.mean([r["margin"] for r in rows])
    print(f"  {tag}: N={n} wr={wr:.3f} ask={ask:.3f} model_p={pm:.3f} margin={mg:+.3f} "
          f"net=${pnl.sum():+.0f} mean=${pnl.mean():+.3f} t={cluster_t(rows):+.2f}")

def main():
    W = load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    for T in (120, 180, 210, 240, 260):
        print(f"\n=== DESIGN: model-gated entry, first qualifying frame at/after T={T} ===")
        for margin in (0.0, 0.03, 0.06):
            rows = [r for d in design if (r := first_trade(d, T, margin=margin))]
            report(f"margin={margin:.2f}", rows)

if __name__ == "__main__":
    main()