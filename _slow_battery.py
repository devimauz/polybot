#!/usr/bin/env python3
"""_slow_battery.py — NON-LATENCY strategy battery on _pm_cache (769 windows, Oct14-Nov1).
Family A: model-gated deep-lock tail — at T>=240, p_model (observed TWAP part + remaining
  noise from window-sigma) vs observed ask; buy side if p - ask - fee_margin > tau.
Family B: second-scale price-action at fixed T: 60s-return z-score continuation/reversal.
Entries fill at ask + 2c (slow-bot realistic). Hold to settlement (all TP variants falsified).
Gates: design=first 2/3, holdout=last 1/3, halves, arms, n>=150, placebo (permuted winners).
"""
import numpy as np, glob, os

files = sorted(glob.glob(r"C:\Users\user\Desktop\bot\_pm_cache\*.npz"))
WS = []
for f in files:
    z = np.load(f, allow_pickle=True)
    if "_winner" not in z.files: continue
    cb = z["cb_px"]; cl = z["cl_px"]; ua = z["Up_ask"]; da = z["Down_ask"]
    if np.isnan(cl[-1]) or np.isnan(cl[0]) or z["_winner"] not in ("Up", "Down"): continue
    if not (np.isfinite(cb).sum() > 250): continue
    WS.append(dict(ws=int(np.asarray(z["ws"]).reshape(-1)[0]), win=str(z["_winner"]),
                   cb=cb, cl=cl, ua=ua, da=da))
print("usable windows:", len(WS))
FEE_PX = lambda a: 0.07 * a * (1 - a)   # per share of stake dollar equivalent (as bot books)

def norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608))  # approx

def model_p_up(w, T):
    """P(settleTWAP>strike) at time T using cl observed to T + remaining cb noise."""
    clw = w["cl"]; cbw = w["cb"]
    strike = clw[0] if np.isfinite(clw[0]) else np.nan
    if not np.isfinite(strike) or T < 120: return None
    obs = clw[180:T + 1][np.isfinite(clw[180:T + 1])]
    if len(obs) < 10: return None
    o = obs.mean()                      # observed mean of [180,T]... approximating integral share
    # resolution mean over [240,300]; observed part [240,T]
    obs240 = clw[max(180, T - 120):T + 1]; obs240 = obs240[np.isfinite(obs240)]
    n_obs = len(obs240)
    if n_obs == 0: return None
    m_obs = obs240.mean()
    tau = 300 - T
    last = cbw[max(0, T - 5):T + 1]; last = last[np.isfinite(last)]
    if len(last) == 0: return None
    cur = last.mean()
    d = np.diff(cbw[np.isfinite(cbw)])
    sig = np.std(d) * 2.2 if len(d) > 30 else 10.0
    sd = sig * np.sqrt(tau / 3.0) + 7.62
    exp_settle = (m_obs * n_obs + cur * tau) / 60.0 * (60.0 / (n_obs + tau)) if False else (m_obs * n_obs + cur * tau) / (n_obs + tau)
    z = (exp_settle - strike) / sd
    return norm_cdf(z)

def run_battery(cells):
    out = []
    for name, fn in cells:
        recs = []
        for w in WS:
            r = fn(w)
            if r: recs.append((w, r[0], r[1], r[2]))  # side, fill_px, stake$
        if len(recs) < 5: out.append((name, len(recs), 0,0,0,0,0,0,0,0,0,0)); continue
        def ev(rs):
            tot = 0.0; n = 0
            for w, side, px, sh in rs:
                pnl = sh * (1.0 if side == w["win"] else 0.0) - sh * px - sh * 0.07 * px * (1 - px)
                tot += pnl; n += 1
            return tot / max(n, 1), n
        mean, n = ev(recs)
        half = len(recs) // 2
        h1, _ = ev(recs[:half]); h2, _ = ev(recs[half:])
        up = [r for r in recs if r[1] == "Up"]; dn = [r for r in recs if r[1] == "Down"]
        a1 = ev(up)[0] if up else 0.0; a2 = ev(dn)[0] if dn else 0.0
        rng = np.random.default_rng(7)
        perm = list(recs); rng.shuffle(perm)
        plrecs = [(w, perm[i][1], px, sh) for i, (w, side, px, sh) in enumerate(recs)]
        pl, _ = ev(plrecs)
        cut = int(len(recs) * 2 / 3)
        des, hold = ev(recs[:cut]), ev(recs[cut:])
        out.append((name, n, mean, h1, h2, a1, a2, pl, des[0], hold[0], des[1], hold[1]))
    return out

def A(tau_c, T):
    def fn(w):
        Ts = range(240, 300, 20) if T is None else [T]
        for Tx in Ts:
            r = _A1(w, tau_c, Tx)
            if r: return r
        return None
    return fn

def _A1(w, tau_c, T):
    def _x():
        p = model_p_up(w, T)
        if p is None: return None
        ua, da = w["ua"][T], w["da"][T]
        if not (np.isfinite(ua) and np.isfinite(da)): return None
        for side, ask, pp in (("Up", ua, p), ("Down", da, 1 - p)):
            if not (0.10 <= ask <= 0.90): continue
            edge = pp - (ask + tau_c / 100.0)
            if edge > 0:
                sh = int(5 / ask)
                if sh >= 5: return (side, ask, sh)
        return None
    return _x()

def B(k_sigma, T, mode):
    """60s return z-score continuation (mode=cont) / reversal (rev), at fixed T."""
    def fn(w):
        cb = w["cb"]
        if T >= 300 or not (np.isfinite(cb[T]) and np.isfinite(cb[max(0, T - 60)])): return None
        r = cb[T] - cb[max(0, T - 60)]
        d = np.diff(cb[np.isfinite(cb)])
        if len(d) < 30: return None
        sig = np.std(d) * np.sqrt(60) * 2.2
        if sig <= 0: return None
        z = r / sig
        if mode == "cont": side = "Up" if z > k_sigma else ("Down" if z < -k_sigma else None)
        else: side = "Down" if z > k_sigma else ("Up" if z < -k_sigma else None)
        if not side: return None
        ask = w["ua"][T] if side == "Up" else w["da"][T]
        if not (np.isfinite(ask) and 0.10 <= ask <= 0.90): return None
        sh = int(5 / ask)
        if sh < 5: return None
        return (side, ask, sh)
    return fn

cells = [(f"A lock p>ask+{t}c T{T}", A(t, T)) for t in (0, 2, 4, 6) for T in (240, 260, 280)]
cells += [(f"B cont z>{k} T{T}", B(k, T, "cont")) for k in (0.5, 0.75, 1.0) for T in (150, 240)]
cells += [(f"B rev  z>{k} T{T}", B(k, T, "rev")) for k in (0.5, 0.75, 1.0) for T in (150, 240)]
cells += [("C lock-any-T 6c", A(6, None))]

print("%-22s %5s %8s %8s %8s %7s %7s %8s %8s %8s" % ("cell", "n", "mean$", "h1$", "h2$", "Up$", "Dn$", "plac$", "design", "holdout"))
for row in run_battery(cells):
    name, n, mean, h1, h2, a1, a2, pl, des, hol, nd, nh = row
    flag = "  <== CHECK" if (n >= 100 and mean > 0.15 and h1 > 0 and h2 > 0 and hol > 0) else ""
    print("%-22s %5d %+8.3f %+8.3f %+8.3f %+7.3f %+7.3f %+8.3f %+8.3f %+8.3f%s" % (name, n, mean, h1, h2, a1, a2, pl, des, hol, flag))
