#!/usr/bin/env python3
"""_pm15_hour.py — hour-of-day decomposition of the 15m theta8 stale-buy EV (6-day corpus).
If a pocket hours exist where 15m books underreact, that's the ONLY remaining un-falsified
15m candidate. Pre-registered: report ALL 24 buckets, no cherry-picking; gates: n>=400,
mean>0 in BOTH time halves of the bucket, win% beats price-implied breakeven."""
import json, glob, gzip, os, math, datetime
from collections import deque, defaultdict

LOG = "/root/poly/laglog"
cb, t15 = [], []
for fp in sorted(glob.glob(os.path.join(LOG, "lag-*.jsonl")) + glob.glob(os.path.join(LOG, "lag-*.jsonl.gz"))):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt") as f:
        for L in f:
            if '"src":"cb"' in L and '"tp":"m"' in L:
                r = json.loads(L); cb.append((r["t_ns"], r["p"]))
            elif '"src":"pm15"' in L:
                r = json.loads(L); t15.append((r["t_ns"], r["baU"], r["baD"]))
cb.sort(); t15.sort()
T0 = t15[0][0]
cb = [x for x in cb if x[0] >= T0]
tt = [x[0] for x in cb]
cum = [0.0]*(len(cb)+1)
for i,(_,p) in enumerate(cb): cum[i+1]=cum[i]+p
def cbm(a,b):
    i=bisect_left(tt,a*1e9); j=bisect_left(tt,b*1e9)
    return None if j<=i else (cum[j]-cum[i])/(j-i)
from bisect import bisect_left
def winner(w,d=900):
    s=cbm(w-60,w); f=cbm(w+d-60,w+d)
    return None if s is None or f is None else ("Up" if f>s else "Down")
t15tt=[x[0] for x in t15]
def before(t):
    i=bisect_left(t15tt,t); return t15[i-1] if i else None
def after(t):
    i=bisect_left(t15tt,t); return t15[i] if i<len(t15) else None
jumps=[]; q=deque(); last=0
for t_ns,p in cb:
    while q and t_ns-q[0][0]>3e8: q.popleft()
    if q and abs(p-q[0][1])>=8 and t_ns-last>3e9: jumps.append((t_ns,p-q[0][1])); last=t_ns
    q.append((t_ns,p))
fee=lambda sh,px: sh*0.07*px*(1-px)
rows=defaultdict(list)  # hour -> [pnl]
rowsd=defaultdict(list)
for te,d in jumps:
    pre=before(te)
    if not pre: continue
    si=1 if d>0 else 2
    pa=pre[si]
    post=after(te+100*10**6)
    if not (post and post[si]<=pa and 0.10<=pa<=0.90): continue
    w=int(te//10**9//900*900)
    wn=winner(w)
    if not wn: continue
    sh=int(5/pa)
    if sh<5: continue
    pnl=sh*(1.0 if ("Up" if d>0 else "Down")==wn else 0.0)-sh*pa-fee(sh,pa)
    h=datetime.datetime.fromtimestamp(te/1e9,datetime.UTC).hour
    rows[h].append(pnl); rowsd[h].append((te,pnl,pa))
print("hour  n     mean$   net$    win%  breakev%  first-half  second-half")
for h in range(24):
    v=rows.get(h) or []
    n=len(v)
    if n<50: print("%2d  %4d  ---- thin" % (h,n)); continue
    mean=sum(v)/n
    wr=100*sum(1 for x in v if x>0)/n
    be=100*sum(0.07*p*(1-p)+p for _,_,p in rowsd[h] if False)  # placeholder
    # breakeven: avg ask + avg fee/share vs 1.0 => win% needed = mean(ask*(1+0.07*(1-ask)))
    need=100*sum((p+0.02)*(1+0.07*(1-p)) for _,_,p in rowsd[h])/n
    half=n//2
    v2=sorted(rowsd[h])
    h1=sum(x[1] for x in v2[:half])/max(half,1)
    h2=sum(x[1] for x in v2[half:])/max(n-half,1)
    mark=""
    if n>=400 and mean>0.3 and h1>0 and h2>0: mark="  <== POCKET?"
    print("%2d  %4d  %+0.3f  %+0.0f  %5.1f  %5.1f     %+0.2f/%+0.2f%s" % (h,n,mean,sum(v),wr,need,h1,h2,mark))
