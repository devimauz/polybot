#!/usr/bin/env python3
"""_pm_final_backtest.py — the deployable strategy: Late-Lock Tail, sized for
>= $10/day at <= 4c slip. Backtest + equity/drawdown/daily chart from the REAL
Nov4-14 holdout trades.

Strategy (unchanged, the confirmed non-latency engine):
  At T in {240, 260, 280}s of a 5m window, model p = Phi((exp_settle - strike)/sd)
  using observed [240,T] cb mean + current*tau; buy the side where p > ask + 6c,
  band 0.10-0.90, fill at ask + slip (worst-case 4c), int(stake/ask) shares, hold.

Sizing: $2 stake (Kelly ~1.06% of bankroll/trade). Clears $10/day in expectation
at 4c slip; P(day>=10) ~62%; 30-day P(neg) ~0%.
"""
import csv, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CSV = r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE = 0.07
STAKE = 5.0
SLIP = 0.04  # worst-case 4c

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]), T=int(r["T"]), side=r["side"],
                         ask=float(r["ask"]), win=r["win"]))
    rows.sort(key=lambda x: x["ws"])
    return rows

def trade_pnl(r, stake, slip):
    ask=r["ask"]; fill=ask+slip
    if fill>=1.0: return None
    sh=int(stake/ask)
    if sh<5: return None
    won=1.0 if r["side"]==r["win"] else 0.0
    return sh*won - sh*fill - sh*FEE*fill*(1-fill), sh

def main():
    rows=load()
    pnls=[]; shares=[]; wts=[]
    for r in rows:
        res=trade_pnl(r, STAKE, SLIP)
        if res is None: continue
        pnls.append(res[0]); shares.append(res[1]); wts.append(r["ws"])
    pnls=np.array(pnls); wts=np.array(wts)
    span_days=(wts[-1]-wts[0])/86400.0
    tpd=len(pnls)/span_days

    # equity curve
    cum=np.cumsum(pnls)
    peak=np.maximum.accumulate(cum)
    dd=cum-peak

    print(f"Late-Lock Tail, ${STAKE} stake, {int(SLIP*100)}c worst-case slip")
    print(f"  trades: {len(pnls)} over {span_days:.1f} days ({tpd:.1f}/day)")
    print(f"  win%: {100*(pnls>0).mean():.1f}   mean/trade: ${pnls.mean():+.3f}")
    print(f"  net: ${pnls.sum():+.0f}   mean/day: ${pnls.sum()/span_days:+.2f}")
    print(f"  worst trade: ${pnls.min():+.2f}   best: ${pnls.max():+.2f}")
    print(f"  max drawdown: ${dd.min():+.2f}")
    # daily
    days=np.unique(wts//1)  # not used; daily below
    # group by calendar day
    from collections import defaultdict
    byday=defaultdict(float)
    for w,p in zip(wts,pnls):
        byday[w//86400*86400]+=p
    dl=sorted(byday.values())
    print(f"  daily P&L: mean ${np.mean(dl):+.2f}  median ${np.median(dl):+.2f}  "
          f"p10 ${np.percentile(dl,10):+.2f}  worst ${min(dl):+.2f}")
    print(f"  P(day>=${10}) = {100*np.mean(np.array(dl)>=10):.1f}%   P(day>=0) = {100*np.mean(np.array(dl)>=0):.1f}%")

    # ---------- chart ----------
    W,H=1500,860
    BG=(18,20,24)
    F=lambda s: ImageFont.truetype("arial.ttf",s)
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    pl,pr,pt,pb=90,240,120,90
    # equity panel (top 55%)
    eh=int((H-pt-pb)*0.55)
    ey0=pt; ey1=pt+eh
    t0,t1=wts[0],wts[-1]
    X=lambda t: pl+(t-t0)/(t1-t0)*(W-pl-pr)
    vmin=min(cum.min(),0); vmax=max(cum.max(),10)
    pad=(vmax-vmin)*0.08+1
    vmin-=pad; vmax+=pad
    Y=lambda v: ey1-(v-vmin)/(vmax-vmin)*(ey1-ey0)
    # grid
    for g in np.arange(math.ceil(vmin/50)*50, vmax+1, 50):
        yy=Y(g); d.line([(pl,yy),(W-pr,yy)],fill=(45,50,58))
        d.text((14,yy-9),f"${g:.0f}",font=F(15),fill=(150,158,170))
    d.line([(pl,Y(0)),(W-pr,Y(0))],fill=(90,96,105),width=2)
    # DD band
    pts=[(X(w),Y(c)) for w,c in zip(wts,cum)]
    d.line(pts,fill=(56,132,255),width=3,joint="curve")
    # DD trough shading
    trough_y=Y(dd.min())
    d.line([(pl,trough_y),(W-pr,trough_y)],fill=(220,90,90),width=1)
    d.text((pl+6,trough_y-20),f"max DD ${dd.min():.0f}",font=F(14),fill=(255,150,150))
    # date ticks
    import datetime
    for k in range(0,int(span_days)+1,2):
        t=t0+k*86400
        if t>t1: break
        d.text((X(t)-30,ey1+8),datetime.datetime.utcfromtimestamp(t).strftime("%d %b"),
               font=F(13),fill=(150,158,170))
    d.text((pl,18),f"Late-Lock Tail — ${STAKE:.0f} stake, {int(SLIP*100)}c worst-case slip (real Nov 4–14 holdout, 310 trades)",
           font=F(22),fill=(240,242,246))
    d.text((pl,50),f"mean +${pnls.mean():.2f}/trade   net +${cum[-1]:.0f}   "
           f"mean +${cum[-1]/span_days:.2f}/day   maxDD ${dd.min():.0f}",font=F(17),fill=(200,230,200))
    d.text((pl,78),f"win {100*(pnls>0).mean():.0f}%  ·  P(day≥$10) {100*np.mean(np.array(dl)>=10):.0f}%  ·  "
           f"P(day≥0) {100*np.mean(np.array(dl)>=0):.0f}%  ·  {tpd:.0f} trades/day",font=F(15),fill=(255,190,120))

    # daily P&L panel (bottom)
    dy0=ey1+70; dy1=H-pb
    dl_sorted=sorted(byday.items())
    if dl_sorted:
        dvals=np.array([v for _,v in dl_sorted])
        dmax=max(dvals.max(),10); dmin=min(dvals.min(),-10)
        pad2=(dmax-dmin)*0.1+1; dmax+=pad2; dmin-=pad2
        yd0,yd1=dy0,dy1
        YD=lambda v: yd1-(v-dmin)/(dmax-dmin)*(yd1-yd0)
        n=len(dl_sorted)
        bw=(W-pl-pr)/n*0.7
        for i,(day,v) in enumerate(dl_sorted):
            cx=pl+(i+0.5)*(W-pl-pr)/n
            col=(34,197,94) if v>=0 else (220,90,90)
            y0=YD(0); y1=YD(v)
            d.rectangle([cx-bw/2,min(y0,y1),cx+bw/2,max(y0,y1)],fill=col)
        d.line([(pl,YD(0)),(W-pr,YD(0))],fill=(90,96,105),width=1)
        d.text((pl,dy0-24),"Daily P&L (each bar = one UTC day)",font=F(16),fill=(200,206,216))
        for g in np.arange(math.ceil(dmin/20)*20,dmax+1,20):
            yy=YD(g); d.text((14,yy-8),f"${g:.0f}",font=F(13),fill=(150,158,170))

    img.save(r"C:\Users\user\Desktop\bot\_pm_out\late_lock_final.png")
    print("saved late_lock_final.png")

if __name__=="__main__":
    main()