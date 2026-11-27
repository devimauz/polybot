"""Calibration: where is the market mispriced? Edge surface by (time band x ask band).
Edge per share for buying side S at ask a: win_rate - a - entry_fee, where fee = 0.05*min(a,1-a) per share.
"""
from loader import load
import collections

ws = load()

# frames are ~1s apart; sample every frame
TB = [(0,60),(60,120),(120,180),(180,240),(240,270),(270,285),(285,292),(292,296),(296,300)]
AB = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.93),(0.93,0.95),(0.95,0.99)]

agg = collections.defaultdict(lambda: [0,0,0.0])  # (tb,ab) -> [n, wins, ask_sum]
for w in ws.values():
    win_side = w["winner"]
    for fr in w["frames"]:
        t = fr["t"]
        if not (0 <= t < 300): continue
        for side in ("up","dn"):
            s = fr[side]
            a = s["ask"]
            for (t0,t1) in TB:
                if t0 <= t < t1: break
            for (a0,a1) in AB:
                if a0 <= a < a1: break
            agg[(t0,a0)][0]+=1
            agg[(t0,a0)][2]+=a
            if (side=="up") == (win_side=="Up"): agg[(t0,a0)][1]+=1

print(f"{'time':>10} {'ask':>7} {'n':>7} {'win%':>6} {'ask':>6} {'edge/sh':>8}")
rows=[]
for (t0,a0),(n,wins,asum) in sorted(agg.items()):
    if n < 200: continue
    wr = wins/n; a=asum/n
    fee = 0.05*min(a,1-a)
    edge = wr - a - fee
    rows.append((edge,t0,a0,n,wr,a))
    print(f"{t0:>4}-{t0 if False else '':<5} {a0:.2f}-{a0+0.1 if a0<0.9 else 0.99:<4} {n:>7} {wr*100:>5.1f}% {a:>6.3f} {edge*100:>+7.2f}%")
# top cells
rows.sort(reverse=True)
print("\nTOP cells:")
for edge,t0,a0,n,wr,a in rows[:12]:
    print(f"t={t0:>=3}-{[t1 for tt,t1 in TB if tt==t0][0]:<3} ask={a0:.2f}+ n={n:>6} wr={wr:.3f} ask_avg={a:.3f} edge={edge:+.4f}/sh")