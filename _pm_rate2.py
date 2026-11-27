#!/usr/bin/env python3
"""_pm_rate2.py — can we 5x Late-Lock's trade rate without lowering the edge?

Late-Lock fires on ~10% of 5m windows (model disagrees with book by >=6c at
T>=240). The other 90% it skips. To reach $10/hour we need ~5x more trades.

Approach: on the windows Late-Lock SKIPS, search for a SECOND condition that
still has edge. Candidates (all non-latency, all on the settlement model):
  - smaller margin (4c, 2c) at T>=240
  - earlier T (180, 120) with the model
  - the model at T>=240 but band 0.10-0.90 vs wider

We measure: does relaxing the gate add trades WITHOUT killing the per-trade edge?
Key: the added trades must keep mean > 0 (else we just add volume at a loss).

DESIGN slice only; report trades/day and mean/trade at each gate.
"""
import os, pickle, math
import numpy as np
from scipy.stats import norm

ROOT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(ROOT,"_pm_cache")
FEE=0.07; STAKE=5.0
SIGMA1S=4.63; SIGMA_STRIKE=7.62; NOISE=2.2; BIAS=-7.4

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

def strike_of(d):
    cb=d["cb_px"]; f=np.where(np.isfinite(cb))[0]
    return float(cb[f[0]]) if len(f) else None

def model_p(d,T,side):
    strike=strike_of(d)
    if strike is None: return None
    t=d["t"]; i=int(np.searchsorted(t,T,side="right"))-1
    if i<0: return None
    j=i
    while j>=0 and not np.isfinite(d["cb_px"][j]): j-=1
    if j<0: return None
    cur=d["cb_px"][j]
    if T>=240:
        m=(t>=240)&(t<=min(T,300))&np.isfinite(d["cb_px"])
        part=d["cb_px"][m].mean() if m.any() else cur
        tau=max(300-T,1.0); obs=min(max(T-240,0),60)
        exp=(part*obs+cur*tau)/60.0
    else:
        tau=60.0; exp=cur
    sd=SIGMA1S*NOISE*math.sqrt(tau/3.0)+SIGMA_STRIKE
    p_up=norm.cdf((exp-strike+BIAS)/sd)
    return p_up if side=="Up" else 1-p_up

def first_trade(d, T, margin, band):
    t=d["t"]; i0=int(np.searchsorted(t,T,side="right"))-1
    if i0<0: return None
    for i in range(i0,len(t)):
        for side in ("Up","Down"):
            ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0]<=ask<=band[1]): continue
            p=model_p(d,t[i],side)
            if p is None or p<ask+margin: continue
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            won=1.0 if d["_winner"]==side else 0.0
            pnl=sh*won - sh*(ask+0.02) - fee(ask+0.02,sh)  # 2c fill
            return dict(ws=d["ws"],side=side,ask=ask,pnl=pnl)
    return None

def stats(rows):
    if not rows: return None
    p=np.array([r["pnl"] for r in rows])
    return dict(n=len(p),mean=float(p.mean()),net=float(p.sum()),
                wr=float((p>0).mean()))

def main():
    W=load_all(); split=int(len(W)*0.70)
    design,holdout=W[:split],W[split:]
    span=(design[-1]["ws"]-design[0]["ws"])/86400.0
    print(f"design: {len(design)} windows over {span:.1f} days = {len(design)/span:.0f} windows/day")

    print("\n=== gate sweep: trades/day, mean/trade, net (design, 2c fill) ===")
    print("T    margin  band        N    perDay  mean$/tr  net$")
    for T in (180, 210, 240, 260):
        for margin in (0.02, 0.04, 0.06, 0.08):
            for band in ((0.10,0.90),(0.10,0.70),(0.30,0.90)):
                rows=[t for d in design if (t:=first_trade(d,T,margin,band))]
                s=stats(rows)
                if s is None or s['n']<10: continue
                print(f"{T:3d}  {margin:.2f}   {band}  {s['n']:4d}  {s['n']/span:5.1f}  "
                      f"{s['mean']:+7.3f}  {s['net']:+7.0f}")

    # the production gate (T240, 6c, 0.10-0.90) for reference
    print("\n=== PRODUCTION gate: T=240, margin=6c, band 0.10-0.90 ===")
    rows=[t for d in design if (t:=first_trade(d,240,0.06,(0.10,0.90)))]
    s=stats(rows)
    print(f"  N={s['n']} perDay={s['n']/span:.1f} mean=${s['mean']:.3f} net=${s['net']:.0f}")

    # can we ADD a 2nd gate (different T or band) on the windows the 1st skips?
    print("\n=== stacking: production gate + a 2nd gate on skipped windows ===")
    prod=[t for d in design if (t:=first_trade(d,240,0.06,(0.10,0.90)))]
    prod_ws=set(t["ws"] for t in prod)
    for T2,m2,b2 in ((240,0.04,(0.10,0.90)),(240,0.02,(0.10,0.90)),(260,0.04,(0.10,0.90))):
        add=[t for d in design if d["ws"] not in prod_ws and (t:=first_trade(d,T2,m2,b2))]
        allrows=prod+add
        s=stats(allrows); sa=stats(add)
        if s is None or sa is None: continue
        print(f"  +T{T2} m{m2} {b2}: adds {sa['n']} trades (mean ${sa['mean']:.3f}), "
              f"total {s['n']} ({s['n']/span:.1f}/day, mean ${s['mean']:.3f}, net ${s['net']:.0f})")

if __name__=="__main__":
    main()