#!/usr/bin/env python3
"""_pm_theta8.py — replay the θ8 Coinbase-move stale-buy engine on the local
763-window cache, with the skill's fill model (causal clock, ask+chase, depth).

This is the ONE engine that historically has BOTH a >=50% win rate AND real
profit. The skill's numbers (N=2023, +$1.48/trade, wr 58.4%, t=+4.07) come
from the AMS ms-probe corpus; here we reproduce it on the local Oct cache as an
independent check, then report win rate + profit + drawdown at $5.

Rule: when Coinbase moves >=8$ within 200ms, buy the ALIGNED side's ask if it is
still stale (<= pre-move ask) and in band 0.10-0.90, chase 1c, $5, hold to settle.
"""
import os, pickle, math
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(ROOT,"_pm_cache")
FEE=0.07; STAKE=5.0; THETA=8.0; CHASE=0.01

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

def replay(W):
    trades=[]
    for d in W:
        t=d["t"]; cb=d["cb_px"]
        # find cb moves >= THETA within 200ms, dedup 3s
        f=np.where(np.isfinite(cb))[0]
        if len(f)<5: continue
        cbv=cb[f]; tv=t[f]
        last=-1e9
        for k in range(1,len(cbv)):
            if tv[k]-last<3.0: continue
            # look back 200ms
            m=(tv>=tv[k]-0.2)&(tv<=tv[k])
            if not m.any(): continue
            ref=cbv[m][0]
            if abs(cbv[k]-ref)<THETA: continue
            move=cbv[k]-ref
            last=tv[k]
            side="Up" if move>0 else "Down"
            # find the book frame at/just after tv[k]
            i=int(np.searchsorted(t,tv[k],side="left"))
            if i>=len(t): continue
            # ask must still be stale: <= pre-move ask (approx: ask at trigger)
            ask=d[f"{side}_ask"][i]
            if not np.isfinite(ask): continue
            pre_ask=ask  # best available proxy for pre-move ask
            if not (0.10<=ask<=0.90): continue
            # chase 1c: fill at ask+CHASE
            fill=ask+CHASE
            if fill>=1.0: continue
            asz=d[f"{side}_asksz"][i]
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            won=1.0 if d["_winner"]==side else 0.0
            pnl=sh*won - sh*fill - fee(fill,sh)
            trades.append(dict(ws=d["ws"],side=side,ask=ask,fill=fill,sh=sh,
                               won=bool(won),pnl=pnl))
            # one trade per move (already deduped by 3s)
    return trades

def stats(pnls):
    p=np.array(pnls)
    cum=np.cumsum(p); dd=cum-np.maximum.accumulate(cum)
    return dict(n=len(p),wr=float((p>0).mean()),mean=float(p.mean()),
                net=float(p.sum()),dd=float(dd.min()),
                worst=float(p.min()),best=float(p.max()))

def main():
    W=load_all(); n=len(W); split=int(n*0.70)
    design,holdout=W[:split],W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    td=replay(design); th=replay(holdout)
    sd=stats([t["pnl"] for t in td]); sh=stats([t["pnl"] for t in th])
    print(f"\n=== θ8 stale-buy, $5, 1c chase, 4c... (chase only) ===")
    print(f"  DESIGN  : N={sd['n']} win%={100*sd['wr']:.1f} mean=${sd['mean']:+.3f} "
          f"net=${sd['net']:+.0f} maxDD=${sd['dd']:+.0f}")
    print(f"  HOLDOUT : N={sh['n']} win%={100*sh['wr']:.1f} mean=${sh['mean']:+.3f} "
          f"net=${sh['net']:+.0f} maxDD=${sh['dd']:+.0f}")
    # pooled
    allp=[t["pnl"] for t in td+th]
    sa=stats(allp)
    print(f"  POOLED  : N={sa['n']} win%={100*sa['wr']:.1f} mean=${sa['mean']:+.3f} "
          f"net=${sa['net']:+.0f} maxDD=${sa['dd']:+.0f} worst=${sa['worst']:+.2f}")

    # add a 4c slip variant (worst-case chase)
    print(f"\n=== θ8 with 4c slip (worst-case) ===")
    for label,ww in (("design",td),("holdout",th)):
        pnls=[]
        for t in ww:
            fill=t["ask"]+0.04
            if fill>=1.0: continue
            sh=t["sh"]
            won=1.0 if t["won"] else 0.0
            pnls.append(sh*won - sh*fill - fee(fill,sh))
        s=stats(pnls)
        print(f"  {label:8s}: N={s['n']} win%={100*s['wr']:.1f} mean=${s['mean']:+.3f} "
              f"net=${s['net']:+.0f} maxDD=${s['dd']:+.0f}")

if __name__=="__main__":
    main()