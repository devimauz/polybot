"""Window-level calibration: for each cell, count windows whose dog/leader ask is in band at first qualifying frame, and win rate."""
from loader import load
import collections
ws = load(); items = sorted(ws.items())
TB = [(30,90),(90,150),(150,210),(210,250),(250,270),(270,285)]
AB = [(0.02,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),
      (0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.85),(0.85,0.90),(0.90,0.93),(0.93,0.95),(0.95,0.99)]
def dog_side(fr): return "dn" if fr["up"]["ask"] <= fr["dn"]["ask"] else "up"

# two views: DOG buys and LEADER buys
for view in ("dog","leader"):
    agg = collections.defaultdict(lambda: [0,0,0.0])
    for slug,w in items:
        win_up = w["winner"]=="Up"
        hit = set()
        for fr in w["frames"]:
            t = fr["t"]
            ti = next((i for i,(x,y) in enumerate(TB) if x<=t<y), None)
            if ti is None: continue
            s = dog_side(fr) if view=="dog" else ("up" if dog_side(fr)=="dn" else "dn")
            a = fr[s]["ask"]
            ai = next((i for i,(x,y) in enumerate(AB) if x<=a<y), None)
            if ai is None: continue
            hit.add((ti,ai))
            # win?
            won = (s=="up")==win_up
            k=(ti,ai); agg[k][0]+=1; agg[k][2]+=a
            if won: agg[k][1]+=1
        # per window only count first hit cell once is complex; frame counts fine for wr, but window count:
    print(f"== {view.upper()} ==")
    rows=[]
    for k,(n,wins,asum) in agg.items():
        if n < 100: continue
        wr=wins/n; a=asum/n; fee=0.05*min(a,1-a)
        rows.append((wr-a-fee,k,n,wr,a))
    rows.sort(reverse=True)
    for e,k,n,wr,a in rows:
        t0,t1=TB[k[0]]; a0,a1=AB[k[1]]
        print(f"t={t0:>3}-{t1:<3} ask={a0:.2f}-{a1:<4} n_frames={n:>6} wr={wr:.3f} ask={a:.3f} edge={e*100:+6.2f}%/sh")