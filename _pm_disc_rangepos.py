#!/usr/bin/env python3
"""_pm_disc_rangepos.py — rigorous test of the ONE thread that held on holdout:
range_pos (where current cb sits in its [min,max] range) high, in ask 0.50-0.70,
at T=120. Design +0.58 (halves +0.15/+1.01, placebo 0.14, median +$2.13) ->
holdout +1.36 (n=16, median +$2.27).

Per the skill: freeze from design, require both-halves + plateau, one holdout
shot, report N / wr / ask / be / net / median / t / cluster-t / placebo.
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

def trade_for(d, T, band):
    t=d["t"]; i0=int(np.searchsorted(t,T,side="right"))-1
    if i0<0: return None
    for i in range(i0,len(t)):
        for side in ("Up","Down"):
            ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0]<=ask<=band[1]): continue
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            won=(d["_winner"]==side)
            pnl=sh*(1.0 if won else 0.0)-sh*ask-fee(ask,sh)
            return dict(ws=d["ws"],side=side,ask=ask,won=won,pnl=pnl,sh=sh)
    return None

def range_pos(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; lo=np.min(v); hi=np.max(v)
    if hi-lo<1e-9: return np.nan
    return float((v[-1]-lo)/(hi-lo))

def ret30(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-30)&(t<=T)&np.isfinite(cb)
    if m.sum()<2: return None
    v=cb[m]; return float((v[-1]-v[0])/(v[0]+1e-9)*100)

def stats(rows):
    if not rows: return None
    pnl=np.array([r["pnl"] for r in rows])
    return dict(n=len(rows), wr=float(np.mean([r["won"] for r in rows])),
                ask=float(np.mean([r["ask"] for r in rows])),
                be=float(np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])),
                net=float(pnl.sum()), mean=float(pnl.mean()), med=float(np.median(pnl)),
                worst=float(pnl.min()))

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} be={st['be']:.3f} "
          f"net=${st['net']:+.0f} mean=${st['mean']:+.3f} med=${st['med']:+.3f} worst=${st['worst']:+.2f}")

def cluster_t(rows):
    if not rows: return float('nan')
    pnl=np.array([r["pnl"] for r in rows]); ws=np.array([r["ws"] for r in rows])
    uq=np.unique(ws); s=np.array([pnl[ws==w].sum() for w in uq])
    if len(s)<2: return float('nan')
    m=s.mean(); se=s.std(ddof=1)/math.sqrt(len(s))
    return m/se if se>0 else float('nan')

def placebo(rows, draws=300, seed=11):
    rng=np.random.default_rng(seed)
    real=np.array([r["pnl"] for r in rows]).sum()
    edges=[0.10,0.30,0.50,0.70,0.90]
    bi=np.digitize([r["ask"] for r in rows],edges)-1
    wins=np.array([r["won"] for r in rows],dtype=float)
    sh=np.array([r["sh"] for r in rows],dtype=float)
    ask=np.array([r["ask"] for r in rows],dtype=float)
    ge=0
    for _ in range(draws):
        w=wins.copy()
        for b in np.unique(bi):
            m=bi==b; w[m]=rng.permutation(wins[m])
        pnl=sh*w - sh*ask - sh*FEE*ask*(1-ask)
        if pnl.sum()>=real: ge+=1
    return ge/draws

def collect(W, T, band, feat, cut, arm):
    rows=[]
    for d in W:
        tr=trade_for(d,T,band)
        if tr is None: continue
        val=feat(d,T)
        if val is None or not np.isfinite(val): continue
        if arm=="hi" and val < cut: continue
        if arm=="lo" and val > cut: continue
        rows.append(tr)
    return rows

def main():
    W=load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    T=120; band=(0.50,0.70)

    # ---- design: where to put the cut ----
    drows=[]
    for d in design:
        tr=trade_for(d,T,band)
        if tr is None: continue
        rp=range_pos(d,T)
        if rp is None or not np.isfinite(rp): continue
        tr=dict(tr); tr["rp"]=rp; drows.append(tr)
    vals=np.array([r["rp"] for r in drows])
    print(f"\n=== DESIGN T={T} band={band}: range_pos distribution (n={len(drows)}) ===")
    for q in (0.5,0.6,0.7,0.75,0.8,0.85,0.9):
        print(f"  quantile {q:.2f} = {np.quantile(vals,q):.3f}")

    # sweep cut on design: find where hi arm is positive with n>=20 + both halves
    print("\n-- design cut sweep (range_pos >= cut, hi arm) --")
    best=None
    for cut in (0.7,0.75,0.8,0.85,0.9):
        rows=[r for r in drows if r["rp"]>=cut]
        st=stats(rows)
        if st is None or st['n']<20: 
            show(f"cut>={cut}", st); continue
        rows.sort(key=lambda r:r["ws"]); m=len(rows)//2
        h1=stats(rows[:m]); h2=stats(rows[m:])
        show(f"cut>={cut}", st)
        print(f"     halves {h1['mean']:+.2f}/{h2['mean']:+.2f}  t={cluster_t(rows):+.2f} plac={placebo(rows):.3f}")
        if h1['mean']>0 and h2['mean']>0:
            if best is None or st['mean']>best[1]['mean']: best=(cut,st,rows)

    if best is None:
        print("\nNo design cut had both halves positive -> no freeze.")
        return
    cut, dst, drows_frozen = best
    print(f"\n=== FROZEN: range_pos >= {cut} (design n={dst['n']}, mean {dst['mean']:+.2f}) ===")

    # ---- HOLDOUT one shot ----
    hrows=[]
    for d in holdout:
        tr=trade_for(d,T,band)
        if tr is None: continue
        rp=range_pos(d,T)
        if rp is None or not np.isfinite(rp) or rp<cut: continue
        hrows.append(tr)
    print(f"\n=== HOLDOUT (one shot): range_pos >= {cut} ===")
    show("holdout", stats(hrows))
    if hrows:
        hrows.sort(key=lambda r:r["ws"]); m=len(hrows)//2
        show("  ho 1st half", stats(hrows[:m]))
        show("  ho 2nd half", stats(hrows[m:]))
        print(f"  cluster-t {cluster_t(hrows):+.2f}  placebo={placebo(hrows):.3f}")

    # ---- also test the ret30 near-duplicate ----
    print(f"\n=== ret30 >= {np.quantile([ret30(d,T) for d in design if ret30(d,T) is not None],0.8):.3f} (hi arm, band {band}, T={T}) ===")
    dvals=[ret30(d,T) for d in design if ret30(d,T) is not None and trade_for(d,T,band)]
    rcut=float(np.quantile(dvals,0.8))
    dr=[r for r in drows]  # need ret30 per row; recompute
    dr2=[]
    for d in design:
        tr=trade_for(d,T,band)
        if tr is None: continue
        r30=ret30(d,T)
        if r30 is None or not np.isfinite(r30): continue
        tr=dict(tr); tr["r30"]=r30; dr2.append(tr)
    dr_hi=[r for r in dr2 if r["r30"]>=rcut]
    show("design ret30-hi", stats(dr_hi))
    if dr_hi:
        dr_hi.sort(key=lambda r:r["ws"]); m=len(dr_hi)//2
        show("  1st half", stats(dr_hi[:m])); show("  2nd half", stats(dr_hi[m:]))
    hr=[]
    for d in holdout:
        tr=trade_for(d,T,band)
        if tr is None: continue
        r30=ret30(d,T)
        if r30 is None or not np.isfinite(r30) or r30<rcut: continue
        hr.append(tr)
    show("holdout ret30-hi", stats(hr))

if __name__=="__main__":
    main()