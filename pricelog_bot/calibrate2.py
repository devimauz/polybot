"""Calibration v2: edge surface by (time band x ask band), ask buckets over full [0,1)."""
from loader import load
import collections

ws = load()
TB = [(0,30),(30,60),(60,90),(90,120),(120,150),(150,180),(180,210),(210,240),(240,260),(260,275),(275,285),(285,292),(292,297),(297,300)]
EDGES = [0,.02,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.85,.90,.93,.95,.97,.99,1.0]
def aband(a):
    for i in range(len(EDGES)-1):
        if EDGES[i] <= a < EDGES[i+1]:
            return i
    return len(EDGES)-2

agg = collections.defaultdict(lambda: [0,0,0.0])
for w in ws.values():
    win_up = w["winner"] == "Up"
    for fr in w["frames"]:
        t = fr["t"]
        if not (0 <= t < 300): continue
        ti = next((i for i,(x,y) in enumerate(TB) if x<=t<y), None)
        if ti is None: continue
        for side in ("up","dn"):
            a = fr[side]["ask"]
            k = (ti, aband(a))
            agg[k][0]+=1; agg[k][2]+=a
            if (side=="up")==win_up: agg[k][1]+=1

def lbl(k):
    ti, ai = k
    t0,t1 = TB[ti]
    return f"t={t0:>3}-{t1:<3} ask={EDGES[ai]:.2f}-{EDGES[ai+1]:.2f}"

rows=[]
for k,(n,wins,asum) in agg.items():
    if n < 300: continue
    wr=wins/n; a=asum/n
    fee = 0.05*min(a,1-a)
    rows.append((wr-a-fee, k, n, wr, a))
rows.sort(reverse=True)
print("cells sorted by edge/sh (n>=300):")
for e,k,n,wr,a in rows:
    print(f"{lbl(k):<28} n={n:>7} wr={wr:.3f} ask={a:.3f} edge={e*100:+.2f}%/sh")