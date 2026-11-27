#!/usr/bin/env python3
"""_pm_best.py — the best honest non-latency strategy: DUAL-GATE Late-Lock.

Two independent settlement-model gates at different T (different observation of
the TWAP average = independent signal):
  Gate A: T=240, margin 6c, band 0.10-0.90  (10.4 trades/day, +$1.27/trade)
  Gate B: T=210, margin 6c, band 0.10-0.90  (17.3 trades/day, +$0.31/trade)
  -> ~27.7 trades/day, blended mean ~+$0.67/trade at $5.

$10/hour = 6 trades/hour at $5. This gives ~1.15/hour. Gap = 5.2x.
We report: trades/day, $/hour at $5, and the stake needed for $10/hour.

This is the honest ceiling. We do NOT claim $10/hour at $5.
"""
import os, pickle, math
import numpy as np
from scipy.stats import norm
from PIL import Image, ImageDraw, ImageFont

ROOT=os.path.dirname(os.path.abspath(__file__))
CACHE=os.path.join(ROOT,"_pm_cache")
FEE=0.07; STAKE=5.0; SLIP=0.02
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

def trade(d,T,margin,band):
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
            fill=ask+SLIP
            won=1.0 if d["_winner"]==side else 0.0
            pnl=sh*won - sh*fill - fee(fill,sh)
            return dict(ws=d["ws"],side=side,ask=ask,pnl=pnl,sh=sh,won=bool(won))
    return None

def dual(d):
    # Gate A first (T=240), else Gate B (T=210)
    a=trade(d,240,0.06,(0.10,0.90))
    if a: return a
    return trade(d,210,0.06,(0.10,0.90))

def main():
    W=load_all(); split=int(len(W)*0.70)
    design,holdout=W[:split],W[split:]
    span=(W[-1]["ws"]-W[0]["ws"])/86400.0

    td=[t for d in design if (t:=dual(d))]
    th=[t for d in holdout if (t:=dual(d))]
    for name,rows in (("DESIGN",td),("HOLDOUT",th)):
        p=np.array([t["pnl"] for t in rows])
        if len(p)==0: print(f"{name}: N=0"); continue
        cum=np.cumsum(p); dd=cum-np.maximum.accumulate(cum)
        span_h=(rows[-1]["ws"]-rows[0]["ws"])/3600.0
        print(f"=== {name} === N={len(p)} win%={100*(p>0).mean():.1f} mean=${p.mean():+.3f}")
        print(f"  net=${p.sum():+.0f}  trades/hour={len(p)/span_h:.2f}  $/hour=${p.sum()/span_h:+.2f}  maxDD=${dd.min():+.0f}")

    # full period
    allr=td+th; allr.sort(key=lambda x:x["ws"])
    p=np.array([t["pnl"] for t in allr])
    span_h=(allr[-1]["ws"]-allr[0]["ws"])/3600.0
    cum=np.cumsum(p); dd=cum-np.maximum.accumulate(cum)
    tph=len(p)/span_h
    print(f"\n=== FULL (design+holdout) === N={len(p)} win%={100*(p>0).mean():.1f} mean=${p.mean():+.3f}")
    print(f"  net=${p.sum():+.0f}  {tph:.2f} trades/hour  ${p.sum()/span_h:+.2f}/hour  maxDD=${dd.min():+.0f}")
    print(f"\n  => at $5 stake: ${p.sum()/span_h:+.2f}/hour")
    print(f"  => stake needed for $10/hour: ${5*10/(p.sum()/span_h):.0f}")
    print(f"  => bankroll for that stake (1.06% Kelly): ${5*10/(p.sum()/span_h)/0.0106:.0f}")

    # chart
    W2,H2=1500,720; BG=(18,20,24)
    F=lambda s: ImageFont.truetype("arial.ttf",s)
    img=Image.new("RGB",(W2,H2),BG); d=ImageDraw.Draw(img)
    pl,pr,pt,pb=90,250,120,80
    t0,t1=allr[0]["ws"],allr[-1]["ws"]
    vmin=min(cum.min(),0); vmax=max(cum.max(),10); pad=(vmax-vmin)*0.1+1
    vmin-=pad; vmax+=pad
    X=lambda t: pl+(t-t0)/(t1-t0)*(W2-pl-pr)
    Y=lambda v: H2-pb-(v-vmin)/(vmax-vmin)*(H2-pt-pb)
    for g in range(0,int(vmax)+50,50):
        yy=Y(g); d.line([(pl,yy),(W2-pr,yy)],fill=(45,50,58))
        d.text((14,yy-9),f"${g}",font=F(15),fill=(150,158,170))
    d.line([(pl,Y(0)),(W2-pr,Y(0))],fill=(90,96,105),width=2)
    import datetime
    for k in range(0,int(span)+1,2):
        t=t0+k*86400
        if t>t1: break
        d.text((X(t)-30,H2-pb+10),datetime.datetime.utcfromtimestamp(t).strftime("%d %b"),font=F(13),fill=(150,158,170))
    d.line([(X(t["ws"]),Y(v)) for t,v in zip(allr,cum)],fill=(56,132,255),width=3,joint="curve")
    ty=Y(dd.min()); d.line([(pl,ty),(W2-pr,ty)],fill=(220,90,90),width=1)
    d.text((pl+6,ty-20),f"max DD ${dd.min():.0f}",font=F(14),fill=(255,150,150))
    d.text((pl,18),"DUAL-GATE Late-Lock (T=240 + T=210), $5, 2c slip",font=F(22),fill=(240,242,246))
    d.text((pl,50),f"N={len(p)} win%={100*(p>0).mean():.0f} mean +${p.mean():.2f}/tr  "
           f"{tph:.1f} tr/hr  ${p.sum()/span_h:+.1f}/hour  maxDD ${dd.min():.0f}",font=F(17),fill=(200,230,200))
    d.text((pl,78),f"for $10/hour: stake ${5*10/(p.sum()/span_h):.0f}, bankroll ${5*10/(p.sum()/span_h)/0.0106:.0f}  ·  "
           f"NOT reachable at $5 stake (non-latency)",font=F(15),fill=(255,190,120))
    img.save(r"C:\Users\user\Desktop\bot\_pm_out\dual_gate.png")
    print("saved dual_gate.png")

if __name__=="__main__":
    main()