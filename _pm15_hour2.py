#!/usr/bin/env python3
"""_pm15_hour2.py — hour-of-day 15m stale-buy EV using OFFICIAL gamma winners.
One gamma call per covered 15m window (~600), cached to 15m_winners.json."""
import json, glob, gzip, os, math, datetime, urllib.request, time
from collections import deque, defaultdict
from bisect import bisect_left

LOG = "/root/poly/laglog"
CACHE = "/root/poly/15m_winners.json"
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) lag-probe"}

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
def proxy_winner(w):
    s=cbm(w-60,w); f=cbm(w+840,w+900)
    return None if s is None or f is None else ("Up" if f>s else "Down")

# official winners: windows with >=98.5% resolved outcome
winners = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def official(w):
    k=str(w)
    if k in winners: return winners[k]
    try:
        req=urllib.request.Request(f"https://gamma-api.polymarket.com/events?slug=btc-updown-15m-{w}",headers=HDR)
        e=json.load(urllib.request.urlopen(req,timeout=8))
        mk=e[0]["markets"][0]
        op_=json.loads(mk.get("outcomePrices") or "[]")
        r=None
        if len(op_)==2:
            a,b2=float(op_[0]),float(op_[1])
            if mk.get("closed") and a>=0.985: r="Up"
            elif mk.get("closed") and b2>=0.985: r="Down"
        winners[k]=r or ""
    except Exception:
        winners[k]=""
    return winners[k]

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

ws_cov = sorted({int(t//10**9//900*900) for t in t15tt})
print("windows to query:", len(ws_cov))
for i,w in enumerate(ws_cov):
    official(w)
    if i%80==79: json.dump(winners,open(CACHE,"w")); print("gammas",i+1); time.sleep(0.5)
json.dump(winners,open(CACHE,"w"))
off_ok=sum(1 for v in winners.values() if v)
print("official winners:", off_ok)

fee=lambda sh,px: sh*0.07*px*(1-px)
# PROXY calibration vs official
agree=tot=0
for w in ws_cov[:400]:
    pw=proxy_winner(w); ow=official(w)
    if pw and ow: tot+=1; agree+= (pw==ow)
print("proxy agreement: %.1f%% (n %d)" % (100*agree/max(tot,1), tot))

def analyze(getw, label):
    rows=defaultdict(list); rowsd=defaultdict(list); drop=0
    for te,d in jumps:
        pre=before(te)
        if not pre: continue
        si=1 if d>0 else 2
        pa=pre[si]
        post=after(te+100*10**6)
        if not (post and post[si]<=pa and 0.10<=pa<=0.90): continue
        w=int(te//10**9//900*900)
        wn=getw(w)
        if not wn:
            drop+=1; continue
        sh=int(5/pa)
        if sh<5: continue
        pnl=sh*(1.0 if ("Up" if d>0 else "Down")==wn else 0.0)-sh*pa-fee(sh,pa)
        h=datetime.datetime.fromtimestamp(te/1e9,datetime.timezone.utc).hour
        rows[h].append(pnl); rowsd[h].append((te,pnl,pa))
    print("\n=== %s ===  (dropped no-winner: %d)" % (label, drop))
    print("hour  n     mean$    net$   win%   needed%  halves")
    for h in range(24):
        v=rows.get(h) or []
        n=len(v)
        if n<50: print("%2d  %4d  thin" % (h,n)); continue
        mean=sum(v)/n
        wr=100*sum(1 for x in v if x>0)/n
        need=100*sum(p*(1+0.07*(1-p))+0.02*p for _,_,p in rowsd[h])/n
        v2=sorted(rowsd[h]); half=n//2
        h1=sum(x[1] for x in v2[:half])/max(half,1); h2=sum(x[1] for x in v2[half:])/max(n-half,1)
        mark="  <== POCKET?" if (n>=400 and mean>0.3 and h1>0 and h2>0) else ""
        print("%2d  %4d  %+0.3f  %+0.0f  %5.1f  %5.1f  %+0.2f/%+0.2f%s" % (h,n,mean,sum(v),wr,need,h1,h2,mark))

analyze(proxy_winner, "PROXY winners (cb-TWAP)")
analyze(lambda w: official(w) or None, "OFFICIAL gamma winners")
