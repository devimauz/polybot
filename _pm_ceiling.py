#!/usr/bin/env python3
"""_pm_ceiling.py — the honest $/hour ceiling for a non-latency strategy, and
the stake/bankroll needed to reach $10/hour. Single production gate only
(T=240, 6c) — the dual gate was tested and fails out-of-sample.
"""
import csv, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CSV=r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE=0.07; SLIP=0.04

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]),side=r["side"],ask=float(r["ask"]),win=r["win"]))
    rows.sort(key=lambda x:x["ws"])
    return rows

def pnl_at(rows,stake,slip):
    out=[]
    for r in rows:
        fill=r["ask"]+slip
        if fill>=1.0: continue
        sh=int(stake/r["ask"])
        if sh<5: continue
        won=1.0 if r["side"]==r["win"] else 0.0
        out.append(sh*won - sh*fill - sh*FEE*fill*(1-fill))
    return np.array(out)

def main():
    rows=load()
    span_h=(rows[-1]["ws"]-rows[0]["ws"])/3600.0
    print(f"Late-Lock holdout: {len(rows)} trades over {span_h:.1f} hours = {len(rows)/span_h:.2f} trades/hour\n")

    print("=== $/hour by stake (4c worst-case slip) ===")
    print("stake   $/hour    $/day    maxDD")
    for stake in (5,10,15,20,25,30,40,50):
        p=pnl_at(rows,stake,SLIP)
        cum=np.cumsum(p); dd=cum.min()
        per_h=p.sum()/span_h
        print(f"  ${stake:2d}   {per_h:+7.2f}   {per_h*24:+7.0f}   ${dd:+6.0f}")

    print("\n=== stake needed for $10/hour ===")
    for stake in range(5,60,5):
        p=pnl_at(rows,stake,SLIP)
        per_h=p.sum()/span_h
        if per_h>=10:
            print(f"  -> ${stake} stake: ${per_h:.2f}/hour  (bankroll ~${stake/0.0106:.0f} at 1.06% Kelly)")
            break
    else:
        print("  not reached up to $60 stake")

    # chart: $/hour vs stake
    stakes=list(range(5,55,5))
    perhs=[pnl_at(rows,s,SLIP).sum()/span_h for s in stakes]
    W,H=1100,620; BG=(18,20,24)
    F=lambda s: ImageFont.truetype("arial.ttf",s)
    img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
    pl,pr,pt,pb=90,60,100,80
    vmax=max(perhs+[10])*1.1; vmin=0
    X=lambda i: pl+i/(len(stakes)-1)*(W-pl-pr)
    Y=lambda v: H-pb-(v-vmin)/(vmax-vmin)*(H-pt-pb)
    for g in range(0,int(vmax)+10,10):
        yy=Y(g); d.line([(pl,yy),(W-pr,yy)],fill=(45,50,58))
        d.text((14,yy-9),f"${g}",font=F(15),fill=(150,158,170))
    # $10/hr target line
    d.line([(pl,Y(10)),(W-pr,Y(10))],fill=(255,190,120),width=2)
    d.text((W-pr-160,Y(10)-22),"target $10/hour",font=F(15),fill=(255,190,120))
    d.line([(X(i),Y(v)) for i,v in enumerate(perhs)],fill=(56,132,255),width=3,joint="curve")
    for i,(s,v) in enumerate(zip(stakes,perhs)):
        d.ellipse([X(i)-4,Y(v)-4,X(i)+4,Y(v)+4],fill=(56,132,255))
        d.text((X(i)-12,H-pb+10),f"${s}",font=F(13),fill=(150,158,170))
    d.text((pl,18),"Late-Lock Tail: $/hour vs stake (4c slip, real holdout)",font=F(22),fill=(240,242,246))
    d.text((pl,50),f"{len(rows)/span_h:.1f} trades/hour  ·  reaches $10/hour at the stake where the line crosses the orange line",
           font=F(15),fill=(200,206,216))
    img.save(r"C:\Users\user\Desktop\bot\_pm_out\ceiling.png")
    print("\nsaved ceiling.png")

if __name__=="__main__":
    main()