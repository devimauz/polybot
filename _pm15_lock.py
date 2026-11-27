#!/usr/bin/env python3
"""_pm15_lock.py — Late-Lock Tail (frozen rule set) replayed on the 15-min corpus.
Model identical: p from observed settle-part + window-sigma remaining noise;
buy side when p - (ask+6c) > 0 at T in last 60s; fill ask+2c; hold; $5 stake."""
import json, glob, gzip, os, math
from collections import defaultdict

LOG = "/root/poly/laglog"
DUR = 900
cb_sec = {}
pm15 = {}
for fp in sorted(glob.glob(os.path.join(LOG, "lag-*.jsonl")) + glob.glob(os.path.join(LOG, "lag-*.jsonl.gz"))):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt") as f:
        for L in f:
            if '"src":"cb"' in L and '"tp":"m"' in L:
                r = json.loads(L); cb_sec[int(r["t_ns"]//10**9)] = r["p"]
            elif '"src":"pm15"' in L:
                r = json.loads(L); pm15[int(r["t_ns"]//10**9)] = (r["baU"], r["baD"])
tmin = max(min(pm15), min(cb_sec)); tmax = min(max(pm15), max(cb_sec))
print("15m coverage h: %.1f | pm15 secs %d" % ((tmax-tmin)/1e9/3600, len(pm15)))

def cbs(a, b):
    return [cb_sec.get(s, None) for s in range(a, b)]

def norm_cdf(x): return 0.5*(1+math.tanh(x*0.7978845608))
def np_ok(x): return isinstance(x,(int,float)) and x==x

def replay(trades):
    out=[]
    for w, T, side, ask, sh, win in trades:
        pay = min(ask+0.02, 0.99)
        pnl = sh*(1.0 if side==win else 0.0) - sh*pay - sh*0.07*pay*(1-pay)
        out.append((w, pnl, side))
    return out

def report(name, rows):
    n=len(rows)
    if n<3: print("%-18s n %d (too small)"%(name,n)); return
    rows.sort()
    mean=sum(r[1] for r in rows)/n
    wr=100*sum(1 for r in rows if r[1]>0)/n
    half=n//2
    h1=sum(r[1] for r in rows[:half])/half; h2=sum(r[1] for r in rows[half:])/(n-half)
    up=[r[1] for r in rows if r[2]=="Up"]; dn=[r[1] for r in rows if r[2]=="Down"]
    print("%-18s n %3d mean %+0.3f win %.0f%% halves %+0.2f/%+0.2222f".replace("0.2222","0.2") % (
        name, n, mean, wr, h1, h2) + " arms U%+.2f(%d)/D%+.2f(%d)" % (
        sum(up)/len(up) if up else 0, len(up), sum(dn)/len(dn) if dn else 0, len(dn)))

for margin in (0.04, 0.06, 0.10):
    trades=[]; early={}
    for w in range(int(tmin//DUR*DUR), int(tmax//DUR*DUR), DUR):
        win=None
        pre=[x for x in cbs(w-60,w) if x]
        tail=[x for x in cbs(w+DUR-60,w+DUR) if x]
        if len(pre)<40 or len(tail)<40: continue
        strike=sum(pre)/len(pre); settle=sum(tail)/len(tail)
        win = "Up" if settle>strike else "Down"
        allp=[cb_sec.get(s) for s in range(w, w+DUR)]
        vals=[x for x in allp if x]
        if len(vals)<60: continue
        diffs=[vals[i+1]-vals[i] for i in range(len(vals)-1)]
        sig=math.sqrt(sum(d*d for d in diffs)/len(diffs))*2.2 if len(diffs)>30 else 10.0
        for T in (DUR-60, DUR-40, DUR-20):
            s=w+T
            t=pm15.get(s)
            if not t: continue
            idx=T
            obs=[x for x in allp[DUR-60:T] if x]
            if len(obs)<3: continue
            cur=allp[T] if allp[T] else obs[-1]
            tau_s=DUR-T
            sd=sig*math.sqrt(tau_s/3.0)+7.62
            exps=(sum(obs)/len(obs)*len(obs)+cur*tau_s)/(len(obs)+tau_s)
            p=norm_cdf((exps-strike)/sd)
            for side,ask,pp in (("Up",t[0],p),("Down",t[1],1-p)):
                if np_ok(ask) and 0.10<=ask<=0.90 and pp-(ask+margin)>0:
                    sh=int(5/ask)
                    if sh>=5: trades.append((w,T,side,ask,sh,win))
                    break
    rows=replay(trades)
    report("15m lock+%.0fc (T240s)" % (margin*100), rows)
