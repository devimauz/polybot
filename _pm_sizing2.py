#!/usr/bin/env python3
"""_pm_sizing2.py — daily-level risk for the Late-Lock 4c-slip design.
Answers: what is P(a random day clears $10)? worst day? bankroll for $2 stake?
"""
import csv
import numpy as np

CSV = r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE = 0.07

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]), side=r["side"], ask=float(r["ask"]),
                         sh0=int(r["sh"]), win=r["win"]))
    rows.sort(key=lambda x: x["ws"])
    return rows

def repriced(rows, slip_c):
    s=slip_c/100.0; out=[]
    for r in rows:
        fill=r["ask"]+s
        if fill>=1.0: continue
        sh=r["sh0"]; won=1.0 if r["side"]==r["win"] else 0.0
        out.append(sh*won - sh*fill - sh*FEE*fill*(1-fill))
    return np.array(out)

def mc_days(pnl, stake, days=30, paths=20000, seed=5):
    rng=np.random.default_rng(seed)
    scale=stake/5.0; p=pnl*scale
    tpd=len(pnl)/10.8
    per_day_trades=int(round(tpd))
    # simulate `days` days, each = per_day_trades draws
    idx=rng.integers(0,len(p),size=(paths,days,per_day_trades))
    daily=p[idx].sum(axis=2)  # (paths, days)
    return daily, per_day_trades

def main():
    rows=load()
    p4=repriced(rows,4)
    print(f"Late-Lock 4c-slip: {len(p4)} trades, mean ${p4.mean():.3f}/trade, sd ${p4.std(ddof=1):.3f}")
    tpd=len(p4)/10.8
    print(f"trade rate: {tpd:.1f}/day\n")

    for stake in (2,3,5):
        daily, per_day = mc_days(p4, stake)   # daily = (paths, days)
        all_days = daily.flatten()
        print(f"=== ${stake} stake ({per_day} trades/day) ===")
        print(f"  per-day P&L: mean ${all_days.mean():+.2f}  median ${np.median(all_days):+.2f}  "
              f"p10 ${np.percentile(all_days,10):+.2f}  p90 ${np.percentile(all_days,90):+.2f}")
        print(f"  P(day >= $10) = {100*(all_days>=10).mean():.1f}%")
        print(f"  P(day >= $0)  = {100*(all_days>=0).mean():.1f}%")
        print(f"  worst simulated day = ${all_days.min():+.2f}")
        # 30-day total
        tot = daily.sum(axis=1)
        print(f"  30-day total: median ${np.median(tot):+.0f}  p10 ${np.percentile(tot,10):+.0f}  "
              f"P(neg)={100*(tot<0).mean():.1f}%  P(>=$300)={100*(tot>=300).mean():.1f}%")
        # bankroll: worst 30-day drawdown within a path (approx: min cumsum)
        cum = np.cumsum(daily, axis=1)
        dd = cum - np.maximum.accumulate(cum, axis=1)
        maxdd = -dd.min(axis=1)
        print(f"  30d max drawdown: median ${np.median(maxdd):.0f}  p90 ${np.percentile(maxdd,90):.0f}  "
              f"p99 ${np.percentile(maxdd,99):.0f}  worst ${maxdd.max():.0f}")
        print()

if __name__=="__main__":
    main()