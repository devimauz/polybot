#!/usr/bin/env python3
"""_pm_fav.py — non-latency, >=50% win-rate candidate: buy the heavy favourite
(ask >= threshold) in the last N seconds, hold to settlement.

This is the ONLY non-latency shape that can produce a high win rate (the
leader-wins curve is 89-93% at T>=240). The risk: at 80c+ the price is
calibrated, so the edge is thin. We measure win rate, edge (wr - be), net, DD.

DESIGN slice only; then one holdout shot. If edge <= 0, it's the calibrated
favourite (no tradeable edge) — report that honestly.
"""
import os, pickle, math
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(ROOT,"_pm_cache")
FEE=0.07; STAKE=5.0

def load_all():
    man=pickle.load(open(os.path.join(CACHE,"manifest.pkl"),"rb"))
    out=[]
    for m in man:
        if not m["_winner"]: continue
        z=np.load(os.path.join(CACHE,m["_slug"]+".npz"),allow_pickle=True)
        d={k:z[k] for k in z.files if k!="_winner"}
        d["_winner"]=m["_winner"]; d["_slug"]=m["_slug"]; d["ws"]=int(d["ws"][0])
        d["t"]=d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d:d["ws"])
    return out

def fee(p,sh): return sh*FEE*p*(1.0-p)

def trade(d, Tmin, thr):
    """First frame at/after Tmin where either ask >= thr, band ok, depth clears."""
    t=d["t"]; i0=int(np.searchsorted(t,Tmin,side="right"))-1
    if i0<0: return None
    for i in range(i0,len(t)):
        for side in ("Up","Down"):
            ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or ask<thr or ask>=0.995: continue
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            won=1.0 if d["_winner"]==side else 0.0
            pnl=sh*won - sh*ask - fee(ask,sh)
            return dict(ws=d["ws"],side=side,ask=ask,sh=sh,won=bool(won),pnl=pnl)
    return None

def stats(rows):
    if not rows: return None
    p=np.array([r["pnl"] for r in rows])
    cum=np.cumsum(p); dd=cum-np.maximum.accumulate(cum)
    wr=np.mean([r["won"] for r in rows])
    ask=np.mean([r["ask"] for r in rows])
    be=np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])
    return dict(n=len(p),wr=float(wr),ask=float(ask),be=float(be),
                edge=float(wr-be),mean=float(p.mean()),net=float(p.sum()),
                dd=float(dd.min()),worst=float(p.min()))

def show(tag,s):
    if s is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={s['n']} win%={100*s['wr']:.1f} ask={s['ask']:.3f} be={s['be']:.3f} "
          f"edge={s['edge']:+.3f} net=${s['net']:+.0f} mean=${s['mean']:+.3f} DD=${s['dd']:+.0f}")

def main():
    W=load_all(); split=int(len(W)*0.70)
    design,holdout=W[:split],W[split:]
    print(f"Loaded {len(W)} | design={len(design)} holdout={len(holdout)}")

    print("\n=== DESIGN: favourite buy, by (Tmin, ask threshold) ===")
    for Tmin in (200, 240, 260):
        for thr in (0.80, 0.85, 0.90):
            rows=[t for d in design if (t:=trade(d,Tmin,thr))]
            show(f"T>={Tmin} ask>={thr}", stats(rows))

    # freeze the best-looking: highest win% with edge>0 and N>=30
    print("\n=== freeze candidate: T>=240, ask>=0.85 (design) ===")
    rows=[t for d in design if (t:=trade(d,240,0.85))]
    s=stats(rows); show("design", s)
    if s:
        rows.sort(key=lambda r:r["ws"]); m=len(rows)//2
        show("  1st half", stats(rows[:m])); show("  2nd half", stats(rows[m:]))

        print("\n=== HOLDOUT (one shot): T>=240, ask>=0.85 ===")
        h=[t for d in holdout if (t:=trade(d,240,0.85))]
        show("holdout", stats(h))

if __name__=="__main__":
    main()