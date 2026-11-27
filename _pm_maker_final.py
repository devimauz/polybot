#!/usr/bin/env python3
"""_pm_maker_final.py — maker complete-set with QUEUE-EXACT fills from real prints.

The one maker construction that is structurally positive: buy BOTH legs at the
bid (collect the 1.1c overround), maker fee = 0. But a resting bid only fills
when an aggressor SELL trades THROUGH it (or cumulative sell volume at the price
>= displayed depth). We use the real 1.28M prints (trades.npz) to simulate
queue-exact fills — this is the test the skill says no snapshot proxy can pass.

Win rate: a complete set that fills both legs wins ~82% (both legs cost < $1,
exactly one pays $1). If only one leg fills, it's a directional bet (50/50-ish).
We report win rate, net, and the fill rate.
"""
import os, pickle, math
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(ROOT,"_pm_cache")
FEE_TAKER=0.07
MAKER_FEE=0.0

def load_all():
    man=pickle.load(open(os.path.join(CACHE,"manifest.pkl"),"rb"))
    out=[]
    for m in man:
        if not m["_winner"]: continue
        z=np.load(os.path.join(CACHE,m["_slug"]+".npz"),allow_pickle=True)
        d={k:z[k] for k in z.files if k!="_winner"}
        d["_winner"]=m["_winner"]; d["_slug"]=m["_slug"]; d["ws"]=int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d:d["ws"])
    return out

def load_trades():
    z=np.load(os.path.join(CACHE,"trades.npz"),allow_pickle=True)
    slugs=list(z["slugs"]); starts=z["starts"]; lens=z["lens"]; flat=z["flat"]
    TR={}
    for k,s0,ln in zip(slugs,starts,lens):
        a=flat[:,s0:s0+ln]; o=np.argsort(a[0],kind="stable"); TR[str(k)]=a[:,o]
    return TR

def sim_maker_fill(d, TR, oc, t0):
    """Queue-exact: buy 1 share at the displayed best bid for outcome oc at t0.
    Fills if an aggressor SELL print trades STRICTLY below our bid, OR cumulative
    sell volume printed AT our price >= displayed depth. Returns (filled, fill_px)
    or (False, None)."""
    i=int(np.searchsorted(d["t"],t0,side="left"))
    if i>=len(d["t"]): return False,None
    Bq=d[f"{'Up' if oc==0 else 'Down'}_bid"][i]
    D=d[f"{'Up' if oc==0 else 'Down'}_bidsz"][i]
    if not np.isfinite(Bq) or Bq<=0.01 or Bq>=0.99: return False,None
    a=TR.get(d["_slug"])
    if a is None: return False,None
    # a[0] is ABSOLUTE epoch; convert our window-relative t0 to epoch
    t0_epoch = d["ws"] + t0
    # aggressor SELL on this outcome: side==1 (sell), oc==oc
    m=(a[4]==oc)&(a[3]==1)&(a[0]>=t0_epoch)&(a[0]<=d["ws"]+300)
    if not m.any(): return False,None
    p=a[1][m]; sz=a[2][m]
    # strictly below our bid = through-fill
    through = p < Bq - 1e-9
    # at our price, cumulative volume >= our depth
    atb = np.isclose(p, Bq, atol=1e-4)
    cum_at_bid = sz[atb].sum()
    if through.any():
        return True, Bq
    if cum_at_bid >= (D if np.isfinite(D) else 1e9):
        return True, Bq
    return False,None

def main():
    W=load_all(); split=int(len(W)*0.70)
    design,holdout=W[:split],W[split:]
    print(f"Loaded {len(W)} | design={len(design)} holdout={len(holdout)}")
    TR=load_trades()
    print(f"trades: {sum(len(v) for v in TR.values())} prints, {len(TR)} slugs")

    for label,ww in (("DESIGN",design),("HOLDOUT",holdout)):
        both=0; both_net=0.0; both_pnl=[]
        one=0; one_net=0.0; one_win=0
        for d in ww:
            t0=d["t"][0]
            # buy 1 share each leg at first frame bid
            fU,pU=sim_maker_fill(d,TR,0,t0)
            fD,pD=sim_maker_fill(d,TR,1,t0)
            if fU and fD:
                cost=pU+pD
                if cost < 1.0:
                    pnl=1.0-cost  # one leg pays $1, we hold both
                    both+=1; both_net+=pnl; both_pnl.append(pnl)
            elif fU or fD:
                one+=1
                px = pU if fU else pD
                won = (d["_winner"]=="Up") if fU else (d["_winner"]=="Down")
                pnl = (1.0 if won else 0.0) - px
                one_net+=pnl; one_win += (1 if won else 0)
        bp=np.array(both_pnl)
        print(f"\n=== {label} ===")
        print(f"  both-leg fills: {both}, net ${both_net:+.2f}, mean ${bp.mean() if len(bp) else 0:.4f}, "
              f"win% {100*(bp>0).mean() if len(bp) else 0:.0f}")
        print(f"  one-leg fills: {one}, net ${one_net:+.2f}, win% {100*one_win/max(one,1):.0f}")
        print(f"  TOTAL net: ${both_net+one_net:+.2f} over {len(ww)} windows")

if __name__=="__main__":
    main()