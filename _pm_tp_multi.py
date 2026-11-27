#!/usr/bin/env python3
"""_pm_tp_multi.py — re-run Late-Lock from _pm_cache npz WITH the mark path, to
answer the two user levers honestly:

1) TAKE-PROFIT on the cheap-side entry (mark path from cb -> model p).
   For each trade, replay the mark at each second after entry; if mark >=
   entry*(1+tp) we 'sell' at that mark (minus 4c slip + fee), else hold to settle.
   Compare hold vs TP5/10/20/50%.

2) MORE TRADES PER WINDOW: instead of 1 entry/window (first qualifying T),
   take EVERY qualifying T in {240,260,280} (up to 3/window) and report whether
   net rises or just correlated risk accumulates.

Late-Lock rule (unchanged): at T, model p = Phi((exp_settle-strike)/sd); buy side
where p > ask + 6c, band 0.10-0.90, fill ask+4c, $5, hold.
"""
import os, pickle, math, datetime
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE=0.07; SLIP=0.04; STAKE=5.0
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

def model_p(d, T, side):
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

def mark_path(d, side, t0, t1):
    """Return (times, model_p) every ~5s from t0..t1 for `side`. This is the
    mark-to-market proxy: the contract's fair value = model P(win) at each T."""
    ts=[]; ps=[]
    for T in np.arange(t0, t1, 5.0):
        p=model_p(d,T,side)
        if p is not None:
            ts.append(T); ps.append(p)
    return np.array(ts), np.array(ps)

def run():
    W=load_all()
    n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    # ---- collect trades from HOLDOUT (design already frozen) ----
    trades=[]
    for d in holdout:
        for T in (240,260,280):
            p=model_p(d,T,"Up")
            if p is None: continue
            for side,pp in (("Up",p),("Down",1-p)):
                # need ask at T
                i=int(np.searchsorted(d["t"],T,side="right"))-1
                if i<0: continue
                ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
                if not np.isfinite(ask) or not (0.10<=ask<=0.90): continue
                if pp < ask+0.06: continue
                sh=int(STAKE/ask)
                if sh<5 or not np.isfinite(asz) or asz<sh: continue
                won=(d["_winner"]==side)
                trades.append(dict(ws=d["ws"],T=T,side=side,ask=ask,sh=sh,won=won,p=p,d=d))
                break  # one side per T (the qualifying side)
        # NOTE: we break out of side loop per T; multiple T can still qualify

    # dedupe: keep first qualifying T per window (the current rule) vs all
    print(f"\nraw qualifying trades (all T, all windows): {len(trades)}")
    from collections import Counter
    byws=Counter(t["ws"] for t in trades)
    print(f"windows with >=1: {len(byws)}, entries/window dist: {dict(Counter(byws.values()))}")

    # ---- 1) HOLD vs TP (first-entry-per-window = current rule) ----
    first={}
    for t in trades:
        if t["ws"] not in first: first[t["ws"]]=t
    first_trades=list(first.values())
    print(f"\n=== 1) HOLD vs TAKE-PROFIT (first entry/window, $5, 4c slip) ===")
    for tp in (0.0, 0.05, 0.10, 0.20, 0.50):
        net=0.0; n=0; wins=0
        for t in first_trades:
            d=t["d"]; ask=t["ask"]; sh=t["sh"]; side=t["side"]; T0=t["T"]
            fill=ask+SLIP
            if fill>=1.0: continue
            if tp==0.0:
                won=1.0 if t["won"] else 0.0
                pnl=sh*won - sh*fill - fee(fill,sh)
            else:
                # replay mark path; sell if mark >= fill*(1+tp)
                ts,ps=mark_path(d,side,T0,300)
                sold=False; pnl=None
                target=fill*(1+tp)
                for T,mk in zip(ts,ps):
                    if mk>=target:
                        # sell at mark - slip - fee
                        pnl=sh*mk - sh*(mk-SLIP) - fee(mk-SLIP,sh)  # ~ -slip-fee
                        # simpler: sell proceeds = sh*(mark) - fee; we paid sh*fill
                        pnl=sh*mk - sh*fill - fee(mk,sh)
                        sold=True; break
                if not sold:
                    won=1.0 if t["won"] else 0.0
                    pnl=sh*won - sh*fill - fee(fill,sh)
            net+=pnl; n+=1; wins+= (pnl>0)
        print(f"  TP {int(tp*100):2d}%: N={n} win%={100*wins/n:.0f} net=${net:+.0f} mean=${net/n:+.3f}")

    # ---- 2) MORE PER WINDOW: all qualifying T (up to 3) ----
    print(f"\n=== 2) MORE TRADES PER WINDOW (all qualifying T, hold, $5, 4c slip) ===")
    for mode,label in ((first_trades,"1st only (current)"),(trades,"all T (2-3/window)")):
        net=0.0; n=0; wins=0
        for t in mode:
            d=t["d"]; ask=t["ask"]; sh=t["sh"]
            fill=ask+SLIP
            if fill>=1.0: continue
            won=1.0 if t["won"] else 0.0
            pnl=sh*won - sh*fill - fee(fill,sh)
            net+=pnl; n+=1; wins+=(pnl>0)
        print(f"  {label}: N={n} win%={100*wins/n:.0f} net=${net:+.0f} mean=${net/n:+.3f}")

if __name__=="__main__":
    run()