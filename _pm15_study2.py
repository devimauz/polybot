#!/usr/bin/env python3
"""_pm15_study2.py — fast: cb cumsum + bisect timelines."""
import json, glob, gzip, os, bisect
from collections import deque

LOG = "/root/poly/laglog"
cb, t5, t15 = [], [], []
for fp in sorted(glob.glob(os.path.join(LOG, "lag-*.jsonl")) + glob.glob(os.path.join(LOG, "lag-*.jsonl.gz"))):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt") as f:
        for L in f:
            if '"src":"cb"' in L and '"tp":"m"' in L:
                r = json.loads(L); cb.append((r["t_ns"], r["p"]))
            elif '"src":"pm15"' in L:
                r = json.loads(L); t15.append((r["t_ns"], r["baU"], r["baD"], r["bbU"], r["bbD"]))
            elif '"src":"pm"' in L and '"ev"' in L:
                r = json.loads(L); t5.append((r["t_ns"], r["baU"], r["baD"], r["bbU"], r["bbD"]))
cb.sort(); t5.sort(); t15.sort()
T0 = t15[0][0]
cb = [x for x in cb if x[0] >= T0]
t5 = [x for x in t5 if x[0] >= T0]
print("coverage h %.1f | cb %d | pm5 %d | pm15 %d" % ((t15[-1][0]-T0)/3.6e12, len(cb), len(t5), len(t15)))

tt = [x[0] for x in cb]
cum = [0.0]*(len(cb)+1)
for i, (_, p) in enumerate(cb): cum[i+1] = cum[i]+p
def cb_mean(a_s, b_s):
    i = bisect.bisect_left(tt, a_s*1e9); j = bisect.bisect_left(tt, b_s*1e9)
    return None if j <= i else (cum[j]-cum[i])/(j-i)

def winner(w, dur):
    st = cb_mean(w-60, w); fi = cb_mean(w+dur-60, w+dur)
    if st is None or fi is None: return None
    return "Up" if fi > st else "Down"

def before(tl, times, t_ns):
    i = bisect.bisect_left(times, t_ns)
    return tl[i-1] if i else None
def after(tl, times, t_ns):
    i = bisect.bisect_left(times, t_ns)
    return tl[i] if i < len(tl) else None

jumps=[]; q=deque(); last=0
for t_ns,p in cb:
    while q and t_ns-q[0][0] > 3e8: q.popleft()
    if q and abs(p-q[0][1])>=8 and t_ns-last>3e9:
        jumps.append((t_ns, p-q[0][1])); last=t_ns
    q.append((t_ns,p))
print("jumps >=8$/300ms:", len(jumps))

fee=lambda sh,px: sh*0.07*px*(1-px)
DUR={"5m":300,"15m":900}
TL={"5m":t5,"15m":t15}
print("\n== THETA8 STALE-BUY: survival curve + EV (buy pre-jump ask at +100ms if stale, band .10-.90) ==")
for mkt,dur in DUR.items():
    tl=TL[mkt]; tms=[x[0] for x in tl]
    tot={d:[0,0] for d in (100,250,500,1000,2000)}
    pnls=[]
    for te,d in jumps:
        pre=before(tl,tms,te)
        if not pre: continue
        si=1 if d>0 else 2
        pa=pre[si]
        w=int(te//10**9//dur*dur)
        for dt in tot:
            post=after(tl,tms,te+dt*10**6)
            if post is None: continue
            tot[dt][1]+=1
            if post[si]<=pa: tot[dt][0]+=1
        post=after(tl,tms,te+100*10**6)
        if post and post[si]<=pa and 0.10<=pa<=0.90:
            wn=winner(w,dur)
            if wn:
                sh=int(5/pa)
                if sh>=5:
                    pnls.append(sh*(1.0 if ("Up" if d>0 else "Down")==wn else 0.0)-sh*pa-fee(sh,pa))
    surv=" ".join("%dms:%.0f%%" % (dt,100*v[0]/max(v[1],1)) for dt,v in tot.items())
    if pnls:
        n=len(pnls)
        print("%-4s stale-survive %s | EV n %d net $%.0f mean %+.2f win %.0f%%" % (
            mkt,surv,n,sum(pnls),sum(pnls)/n,100*sum(1 for x in pnls if x>0)/n))
    else:
        print("%-4s stale-survive %s | EV n 0" % (mkt,surv))

print("\n== LAST-2MIN TAKES (leader by bids, ask .35-.80, $5 sizing) ==")
for mkt,dur in [("15m",900),("5m",300)]:
    tl=TL[mkt]; tms=[x[0] for x in tl]
    ws=[]
    for w in range(int(T0//10**9//dur*dur), int(tl[-1][0]//10**9//dur*dur), dur):
        win=winner(w,dur)
        if not win: continue
        i=bisect.bisect_left(tms,(w+dur-120)*10**9)
        j=bisect.bisect_left(tms,(w+dur-30)*10**9)
        if i>=j: continue
        x=tl[i]
        leader="Up" if x[3]>x[4] else "Down"
        ask=x[1] if leader=="Up" else x[2]
        if 0.35<=ask<=0.80:
            sh=int(5/ask)
            ws.append(sh*(1.0 if leader==win else 0.0)-sh*ask-fee(sh,ask))
    n=len(ws)
    if n:
        print("%-4s n %d net $%.0f mean %+.2f win %.0f%%" % (mkt,n,sum(ws),sum(ws)/n,100*sum(1 for x in ws if x>0)/n))
    else:
        print("%-4s no entries" % mkt)
