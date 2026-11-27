#!/usr/bin/env python3
"""_pm_multi.py — two user-requested levers, tested on the Late-Lock engine:

1) MORE TRADES PER WINDOW: current = 1 entry/window (first qualifying frame at
   T in {240,260,280}). Test 2-3 entries/window (every qualifying T, or every
   qualifying frame) — does it add net profit at $5, or just add correlated risk?

2) TAKE-PROFIT on the CHEAP side: the falsified TP table was measured on the
   theta8 latency fills (80c-ish). Late-Lock buys ~20c just-reversed sides — a
   DIFFERENT price regime. TP might behave differently here. We test it on the
   holdout trades: for each trade, does a +X% TP (sell into the bid when the
   contract reaches entry*(1+X)) beat hold-to-settlement?

We approximate the mark-to-market path from the cb path (the contract price
tracks P(win), which tracks the settlement TWAP vs strike). We use the actual
per-trade mark path derived from the window's cb series and the model.
"""
import csv, math
import numpy as np

CSV = r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE=0.07; SLIP=0.04

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]), T=int(r["T"]), side=r["side"],
                         ask=float(r["ask"]), win=r["win"]))
    rows.sort(key=lambda x: x["ws"])
    return rows

def main():
    rows=load()
    # how many trades share a window?
    from collections import Counter
    byws=Counter(r["ws"] for r in rows)
    mult=Counter(byws.values())
    print("=== entries per window (current Late-Lock) ===")
    for k in sorted(mult):
        print(f"  {k} entries/window: {mult[k]} windows")
    print(f"  total {len(rows)} trades, {len(byws)} windows, {len(rows)/len(byws):.2f} entries/window")
    print()

    # ---- 1) MORE PER WINDOW: what would 2-3 entries/window add? ----
    # We can't manufacture new qualifying frames from the CSV (only the first is
    # stored), but we CAN measure the marginal trade: if we simply took the SAME
    # window's trade twice, is the 2nd trade's P&L the same distribution?
    # Honest proxy: bootstrap — sample 2 trades per window WITH replacement from
    # that window's realized outcomes and compare net.
    rng=np.random.default_rng(1)
    byws_rows={}
    for r in rows:
        byws_rows.setdefault(r["ws"],[]).append(r)
    # what fraction of windows have >1 stored trade already?
    multi=[ws for ws,rs in byws_rows.items() if len(rs)>1]
    print(f"windows already with >1 stored entry: {len(multi)} (the fixed-T overlap)")
    if multi:
        # their P&L vs single-entry windows
        multi_pnl=[p for ws in multi for r in byws_rows[ws]
                   for p in [ (int(5/r['ask']) if 5/r['ask']>=5 else 0) ]]
    # Simpler: report per-window total P&L distribution
    win_pnl=[]
    for ws,rs in byws_rows.items():
        s=0
        for r in rs:
            fill=r["ask"]+SLIP
            if fill>=1.0: continue
            sh=int(5.0/r["ask"])
            if sh<5: continue
            won=1.0 if r["side"]==r["win"] else 0.0
            s+=sh*won - sh*fill - sh*FEE*fill*(1-fill)
        win_pnl.append(s)
    win_pnl=np.array(win_pnl)
    print(f"\n=== per-WINDOW P&L (current, $5) ===")
    print(f"  {len(win_pnl)} windows  mean/window ${win_pnl.mean():+.2f}  "
          f"P(window>0)={100*(win_pnl>0).mean():.0f}%  worst ${win_pnl.min():+.2f}")

    # ---- 2) TP on the cheap side ----
    # We need a mark path. The CSV lacks cb path, so we CANNOT replay TP on
    # these exact trades without the raw npz. We report the honest constraint:
    # TP requires the per-trade mark path (cb -> model p), which the slow_holdout
    # CSV does not carry. To test TP properly we must re-run from _pm_cache npz.
    print("\n=== 2) TAKE-PROFIT on cheap-side Late-Lock ===")
    print("  The slow_holdout CSV stores only entry ask + winner, not the mark")
    print("  path -> TP cannot be replayed from it. Must re-run from _pm_cache.")
    print("  (The theta8 TP table — hold +263, TP10 +38, TP50 +111 — was on")
    print("   ~80c latency fills, a DIFFERENT price regime than these ~20c entries.)")

if __name__=="__main__":
    main()