#!/usr/bin/env python3
"""_pm_improve.py — two LEGITIMATE ways to raise profit without touching win%.

The 35% win rate is structural: Late-Lock buys the ~20c just-reversed side
(pays ~22c, breakeven ~24%). Raising win% = buying expensive favourites at fair
price = no edge. So we don't touch win%.

Two real levers (both rule the skill allows — neither is a TP cap, which is
falsified: TP cuts the +100-300% winners that pay for the losers):

  A) CONVICTION SIZING — stake proportional to model margin (p - ask - 6c).
     Bigger bets where the model disagrees most; smaller where marginal.
  B) CONTINUOUS GATE — enter at ANY T in [240,280] (first qualifying frame),
     not just 3 fixed Ts. More trades, same edge, same win%.

We measure: profit at $5 EQUITY stake (scale conviction to a $5 budget),
trades/day, win%, maxDD, and whether the improvement survives at 4c slip.
"""
import csv, math
import numpy as np

CSV = r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE = 0.07
SLIP = 0.04
BUDGET = 5.0  # max stake per trade = $5 (the venue min / the cap)

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]), T=int(r["T"]), side=r["side"],
                         ask=float(r["ask"]), win=r["win"]))
    rows.sort(key=lambda x: x["ws"])
    return rows

def pnl_fixed(r, stake):
    fill=r["ask"]+SLIP
    if fill>=1.0: return None
    sh=int(stake/r["ask"])
    if sh<5: return None
    won=1.0 if r["side"]==r["win"] else 0.0
    return sh*won - sh*fill - sh*FEE*fill*(1-fill)

def pnl_scaled(r, margin):
    """Stake proportional to conviction, capped at BUDGET. margin = p - ask - 6c.
    We don't have p in the CSV, so use a proxy: the ask discount vs 0.5 (cheaper
    = more conviction that it's a real reversal). Actually the CSV only has ask;
    we approximate conviction by (0.5 - ask) clipped, scaled so max stake = BUDGET."""
    # conviction in [0,1]: cheaper side = stronger model disagreement
    conv = max(0.0, min(1.0, (0.5 - r["ask"]) / 0.4))  # ask 0.10 -> 1.0, 0.50 -> 0
    stake = max(BUDGET*conv, 5.0)  # never below venue min $5... but that defeats it
    # To keep >= $5 venue min AND scale, we scale ABOVE $5 only for high conv
    stake = 5.0 + (BUDGET-5.0)*conv if BUDGET > 5.0 else BUDGET*conv
    # Actually: allow stake in [5, BUDGET] scaled by conv; below-5 conv -> 5 (min)
    stake = max(5.0, min(BUDGET, 5.0 + (BUDGET-5.0)*conv)) if BUDGET>5.0 else 5.0
    fill=r["ask"]+SLIP
    if fill>=1.0: return None
    sh=int(stake/r["ask"])
    if sh<5: return None
    won=1.0 if r["side"]==r["win"] else 0.0
    return sh*won - sh*fill - sh*FEE*fill*(1-fill)

def summarize(name, pnls, span):
    if len(pnls)==0: print(f"{name}: N=0"); return
    p=np.array(pnls)
    cum=np.cumsum(p); dd=cum-np.maximum.accumulate(cum)
    print(f"\n=== {name} ===")
    print(f"  N={len(p)}  win%={100*(p>0).mean():.1f}  mean/trade=${p.mean():+.3f}")
    print(f"  net=${p.sum():+.0f}  mean/day=${p.sum()/span:+.2f}  maxDD=${dd.min():+.0f}  worst=${p.min():+.2f}")

def main():
    rows=load()
    span=(rows[-1]["ws"]-rows[0]["ws"])/86400.0
    print(f"Late-Lock holdout: {len(rows)} trades over {span:.1f} days ({len(rows)/span:.1f}/day)")

    # Baseline: $5 fixed
    base=[p for r in rows if (p:=pnl_fixed(r,5.0)) is not None]
    summarize("A) BASELINE $5 fixed stake", base, span)

    # Conviction sizing: $5..$20 (cap 4x the min, scaled by how cheap = conviction)
    for cap in (10.0, 15.0, 20.0):
        BUDGET=cap
        scaled=[p for r in rows if (p:=pnl_scaled(r, pnl_conv_proxy(r) if False else 0)) is not None]
        # recompute with conv proxy inline
        pnls=[]
        for r in rows:
            conv=max(0.0,min(1.0,(0.5-r["ask"])/0.4))
            stake=max(5.0, min(cap, 5.0+(cap-5.0)*conv))
            fill=r["ask"]+SLIP
            if fill>=1.0: continue
            sh=int(stake/r["ask"])
            if sh<5: continue
            won=1.0 if r["side"]==r["win"] else 0.0
            pnls.append(sh*won - sh*fill - sh*FEE*fill*(1-fill))
        summarize(f"B) CONVICTION sizing, stake $5..${cap:.0f}", pnls, span)

    # Also: what if we only take the CHEAPEST asks (strongest conviction)?
    print("\n=== C) CHEAP-SIDE filter: only ask <= threshold (strongest model disagreement) ===")
    for thr in (0.15, 0.20, 0.25):
        pnls=[]
        for r in rows:
            if r["ask"]>thr: continue
            fill=r["ask"]+SLIP
            if fill>=1.0: continue
            sh=int(5.0/r["ask"])
            if sh<5: continue
            won=1.0 if r["side"]==r["win"] else 0.0
            pnls.append(sh*won - sh*fill - sh*FEE*fill*(1-fill))
        summarize(f"ask<={thr:.2f}, $5", pnls, span)

if __name__=="__main__":
    main()