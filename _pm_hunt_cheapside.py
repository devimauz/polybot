#!/usr/bin/env python3
"""_pm_hunt_cheapside.py — honest test of the ONE cell that wasn't pure base rate:
buy the CHEAP side (ask 0.10-0.30) when taker flow is AGAINST it (imb<0).

The flow placebo said the pooled fade edge is base rate. But the cheap band
(0.10-0.30) is where the skill says cheap-side cells need the price-implied
breakeven comparison (not a permutation placebo) because outcomes cluster.

Per the skill:
- cheap-side market orders are fill-constrained: $5/0.20 = 25 shares; check
  depth, not just dollar stake.
- fee as % of stake is 5.6% at 20c — the structural drag.
- honest placebo for cheap-side = price-implied breakeven row.

DESIGN slice only; freeze by both-halves; one holdout shot. No re-run of the
falsified generic fade — this is the cheap-band-under-flow-specific cell.
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
        d["ws"] = int(d["ws"][0])
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def load_trades():
    z = np.load(os.path.join(CACHE, "trades.npz"), allow_pickle=True)
    slugs=list(z["slugs"]); starts=z["starts"]; lens=z["lens"]; flat=z["flat"]
    TR={}
    for k,s0,ln in zip(slugs,starts,lens):
        a=flat[:,s0:s0+ln]; o=np.argsort(a[0],kind="stable"); TR[str(k)]=a[:,o]
    return TR

def fee(p,sh): return sh*FEE*p*(1.0-p)

def flow_imb(TR, d, T):
    a=TR.get(d["_slug"])
    if a is None: return None
    t=a[0]; side=a[3]; oc=a[4]; sz=a[2]
    m=(t>=d["ws"]) & (t<d["ws"]+T)
    if not m.any(): return None
    side=side[m]; oc=oc[m]; sz=sz[m]
    up_buy=sz[(oc==0)&(side==0)].sum(); up_sell=sz[(oc==0)&(side==1)].sum()
    dn_buy=sz[(oc==1)&(side==0)].sum(); dn_sell=sz[(oc==1)&(side==1)].sum()
    signed=up_buy-up_sell-dn_buy+dn_sell; total=up_buy+up_sell+dn_buy+dn_sell
    if total<=0: return None
    return signed/total

def collect(W, TR, T, band, imb_max):
    """Buy the side in `band` when flow against it (imbalance_for_side < imb_max)."""
    rows=[]
    for d in W:
        i=int(np.searchsorted(d["t"],T,side="right"))-1
        if i<0: continue
        imb=flow_imb(TR,d,T)
        if imb is None: continue
        for side in ("Up","Down"):
            ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0]<=ask<=band[1]): continue
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            # imbalance from the perspective of THIS side
            side_imb = imb if side=="Up" else -imb
            if side_imb >= imb_max: continue   # flow not against this side
            won=(d["_winner"]==side)
            pnl=sh*(1.0 if won else 0.0)-sh*ask-fee(ask,sh)
            rows.append(dict(ws=d["ws"],side=side,ask=ask,won=won,pnl=pnl,
                             side_imb=side_imb,sh=sh,asz=asz,T=T))
    return rows

def stats(rows):
    if not rows: return None
    pnl=np.array([r["pnl"] for r in rows])
    return dict(n=len(rows),wr=np.mean([r["won"] for r in rows]),
                ask=np.mean([r["ask"] for r in rows]),
                be=np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows]),
                net=pnl.sum(),mean=pnl.mean(),
                med=np.median(pnl),worst=pnl.min())

def show(tag,st):
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

def main():
    W=load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")
    TR=load_trades()

    T=120
    # sweep the cheap band + flow-against threshold on DESIGN
    print(f"\n=== DESIGN T={T}: buy cheap side when flow against it ===")
    for blo,bhi in [(0.10,0.25),(0.10,0.30),(0.15,0.30),(0.20,0.35)]:
        for imax in (0.0, -0.1, -0.2, -0.3):
            rows=collect(design,TR,T,(blo,bhi),imax)
            st=stats(rows)
            if st and st['n']>=15:
                show(f"ask[{blo:.2f},{bhi:.2f}) imb<{imax:+.1f}", st)

    # Freeze the best-looking cheap cell and check halves + depth coverage
    print(f"\n=== FREEZE candidate: ask[0.10,0.30) imb<0 (DESIGN) ===")
    rows=collect(design,TR,T,(0.10,0.30),0.0)
    st=stats(rows); show("cheap fade", st)
    if st and st['n']>=10:
        rows.sort(key=lambda r:r["ws"]); m=len(rows)//2
        show("1st half", stats(rows[:m]))
        show("2nd half", stats(rows[m:]))
        # depth coverage: how many have asz >= sh?
        cov=np.mean([1.0 if r["asz"]>=r["sh"] else 0.0 for r in rows])
        print(f"  depth-covered: {cov*100:.0f}% (asz>=ticket shares)")
        print(f"  cluster-t: {cluster_t(rows):+.2f}")

    # HOLDOUT one shot on the frozen cell
    print(f"\n=== HOLDOUT: ask[0.10,0.30) imb<0 ===")
    hrows=collect(holdout,TR,T,(0.10,0.30),0.0)
    show("cheap fade holdout", stats(hrows))
    if hrows:
        hrows.sort(key=lambda r:r["ws"]); m=len(hrows)//2
        show("holdout 1st half", stats(hrows[:m]))
        show("holdout 2nd half", stats(hrows[m:]))

if __name__=="__main__":
    main()