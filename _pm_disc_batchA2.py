#!/usr/bin/env python3
"""_pm_disc_batchA2.py — validate the cross-window-vol lead (A3) and path-geometry
lead (A2) to destruction, per the skill: both-halves + placebo + one holdout.

A3: prev-window |cb move| LOW -> current T=180 trade edge +$1.65 (design).
A2: current mid BELOW running mean (z low) -> mild +.

We freeze A3 (prev low-vol) and A2 (z-low) and test:
  - design both-halves
  - placebo (shuffle winners within ask band)
  - ONE holdout shot
"""
import os, pickle, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0]); d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def one_trade(d, T=180, band=(0.10, 0.90)):
    t = d["t"]; i0 = int(np.searchsorted(t, T, side="right")) - 1
    if i0 < 0: return None
    for i in range(i0, len(t)):
        for side in ("Up", "Down"):
            ask = d[f"{side}_ask"][i]; asz = d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0] <= ask <= band[1]): continue
            sh = int(STAKE/ask)
            if sh < 5 or not np.isfinite(asz) or asz < sh: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            return dict(ws=d["ws"], side=side, ask=ask, won=won, pnl=pnl, sh=sh)
    return None

def win_vol(d):
    cb = d["cb_px"]; f = np.isfinite(cb)
    if f.sum() < 10: return None
    return float(np.abs(cb[f][-1] - cb[f][0]))

def path_z(d, T=180):
    cb = d["cb_px"]; t = d["t"]
    m = (t <= T) & np.isfinite(cb)
    if m.sum() < 5: return None
    cbm = cb[m]; sd = float(np.std(cbm))
    if sd < 1e-6: return None
    return float((cbm[-1] - np.mean(cbm)) / sd)

def stats(rows):
    if not rows: return None
    pnl = np.array([r["pnl"] for r in rows])
    return dict(n=len(rows), wr=float(np.mean([r["won"] for r in rows])),
                ask=float(np.mean([r["ask"] for r in rows])),
                be=float(np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])),
                net=float(pnl.sum()), mean=float(pnl.mean()), med=float(np.median(pnl)))

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} be={st['be']:.3f} "
          f"net=${st['net']:+.0f} mean=${st['mean']:+.3f} med=${st['med']:+.3f}")

def cluster_t(rows):
    if not rows: return float('nan')
    pnl=np.array([r["pnl"] for r in rows]); ws=np.array([r["ws"] for r in rows])
    uq=np.unique(ws); s=np.array([pnl[ws==w].sum() for w in uq])
    if len(s)<2: return float('nan')
    m=s.mean(); se=s.std(ddof=1)/math.sqrt(len(s))
    return m/se if se>0 else float('nan')

def placebo(rows, band_edges=(0.10,0.30,0.50,0.70,0.90), draws=300, seed=7):
    """Shuffle winners within ask band; P(shuffled net >= real net)."""
    rng=np.random.default_rng(seed)
    real = np.array([r["pnl"] for r in rows]).sum()
    # band index per row
    bi = np.digitize([r["ask"] for r in rows], band_edges) - 1
    wins = np.array([r["won"] for r in rows], dtype=float)
    sh = np.array([r["sh"] for r in rows], dtype=float)
    ask = np.array([r["ask"] for r in rows], dtype=float)
    ge=0
    for _ in range(draws):
        w = wins.copy()
        for b in np.unique(bi):
            m = bi==b
            w[m] = rng.permutation(wins[m])
        pnl = sh*(w) - sh*ask - sh*FEE*ask*(1-ask)
        if pnl.sum() >= real: ge+=1
    return ge/draws

def main():
    W = load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    for d in design+holdout:
        d["_vol"] = win_vol(d); d["_z"] = path_z(d, 180); d["_trade"] = one_trade(d, 180)

    # build design rows with prev-vol
    vmap = {d["ws"]: d["_vol"] for d in W if d["_vol"] is not None}
    zmap = {d["ws"]: d["_z"] for d in W}
    drows=[]
    for d in design:
        r=d["_trade"]
        if r is None: continue
        r=dict(r); r["_prevvol"]=vmap.get(d["ws"]-300); r["_z"]=d["_z"]
        drows.append(r)

    # ---- A3: prev-window low-vol ----
    d3 = [r for r in drows if r["_prevvol"] is not None]
    vals=np.array([r["_prevvol"] for r in d3]); q1,q2=np.quantile(vals,[1/3,2/3])
    low = [r for r in d3 if r["_prevvol"]<=q1]
    print(f"\n=== A3 prev low-vol (<= {q1:.0f}$) — DESIGN ===")
    show("prev low-vol", stats(low))
    print(f"  cluster-t {cluster_t(low):+.2f}  placebo P(shuffled>=real)={placebo(low):.3f}")
    low.sort(key=lambda r:r["ws"]); m=len(low)//2
    show("  1st half", stats(low[:m])); show("  2nd half", stats(low[m:]))

    # HOLDOUT one shot
    hrowsv = {d["ws"]: d["_vol"] for d in holdout if d["_vol"] is not None}
    h3=[]
    for d in holdout:
        r=d["_trade"]
        if r is None: continue
        pv=hrowsv.get(d["ws"]-300)
        if pv is None or pv>q1: continue
        r=dict(r); h3.append(r)
    print(f"\n=== A3 prev low-vol — HOLDOUT (frozen cut {q1:.0f}$) ===")
    show("prev low-vol holdout", stats(h3))
    if h3:
        h3.sort(key=lambda r:r["ws"]); m=len(h3)//2
        show("  ho 1st half", stats(h3[:m])); show("  ho 2nd half", stats(h3[m:]))

    # ---- A2: path z low ----
    d2 = [r for r in drows if r["_z"] is not None]
    vals=np.array([r["_z"] for r in d2]); q1z,q2z=np.quantile(vals,[1/3,2/3])
    zlow = [r for r in d2 if r["_z"]<=q1z]
    print(f"\n=== A2 path z-low (<= {q1z:+.2f}) — DESIGN ===")
    show("z low", stats(zlow))
    print(f"  cluster-t {cluster_t(zlow):+.2f}  placebo={placebo(zlow):.3f}")
    zlow.sort(key=lambda r:r["ws"]); m=len(zlow)//2
    show("  1st half", stats(zlow[:m])); show("  2nd half", stats(zlow[m:]))
    h2=[]
    for d in holdout:
        r=d["_trade"]
        if r is None or d["_z"] is None or d["_z"]>q1z: continue
        h2.append(dict(r))
    print(f"\n=== A2 path z-low — HOLDOUT (frozen cut {q1z:+.2f}) ===")
    show("z low holdout", stats(h2))

if __name__ == "__main__":
    main()