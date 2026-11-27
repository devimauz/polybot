#!/usr/bin/env python3
"""_pm15_study.py — first-look strategy study on 15-min books vs 5-min control,
using AMS probe logs (cb prints + pm/pm15 touch timelines). All in one pass.
Honest small-n: this is a first measurement, not a verdict."""
import json, glob, gzip, os
from collections import deque

LOG = "/root/poly/laglog"
cb = []   # (t_ns, price)
t5 = []   # (t_ns, baU, baD, bbU, bbD)
t15 = []

for fp in sorted(glob.glob(os.path.join(LOG, "lag-*.jsonl")) + glob.glob(os.path.join(LOG, "lag-*.jsonl.gz"))):
    op = gzip.open if fp.endswith(".gz") else open
    try:
        with op(fp, "rt") as f:
            for L in f:
                if '"src":"cb"' in L and '"tp":"m"' in L:
                    r = json.loads(L)
                    cb.append((r["t_ns"], r["p"]))
                elif '"src":"pm15"' in L:
                    r = json.loads(L)
                    t15.append((r["t_ns"], r["baU"], r["baD"], r["bbU"], r["bbD"]))
                elif '"src":"pm"' in L:
                    r = json.loads(L)
                    t5.append((r["t_ns"], r["baU"], r["baD"], r["bbU"], r["bbD"]))
    except Exception as e:
        print("file err", fp, e)

cb.sort(); t5.sort(); t15.sort()
if not t15: raise SystemExit("no 15m data")
T0 = t15[0][0]  # coverage floor for FAIR comparison (both markets only after 15m logging began)
print("coverage: cb %d | pm5 %d | pm15 %d | shared window %.1f h" % (
    len(cb), len(t5), len(t15), (t15[-1][0]-T0)/3.6e12))

cb_s = [(t/1e9, p) for t, p in cb]

def cb_mean(a, b):
    lo, hi = a*1e9, b*1e9
    s = [p for t, p in cb_s if a <= t < b]
    if not s: return None
    return sum(s)/len(s)

# winner reconstruction per window
def winner(w, dur):
    strike = cb_mean(w-60, w)
    fin = cb_mean(w+dur-60, w+dur)
    if strike is None or fin is None: return None
    return "Up" if fin > strike else "Down"

def touch_after(tl, t_ns, dt_ms):
    import bisect
    times = [x[0] for x in tl]
    i = bisect.bisect_right(times, t_ns+dt_ms*10**6)
    if i >= len(tl): return None
    return tl[i]

def touch_before(tl, t_ns):
    import bisect
    times = [x[0] for x in tl]
    i = bisect.bisect_left(times, t_ns)
    if i == 0: return None
    return tl[i-1]

# jump events on cb (>=8$ within 300ms), dedup 3s
jumps = []
q = deque()
last_fire = 0
for t_ns, p in cb:
    while q and t_ns - q[0][0] > 3e8: q.popleft()
    if q:
        d = p - q[0][1]
        if abs(d) >= 8 and t_ns - last_fire > 3e9 and t_ns >= T0:
            jumps.append((t_ns, d))
            last_fire = t_ns
    q.append((t_ns, p))
print("jumps >=8$/300ms since 15m-logging start:", len(jumps))

DUR = {"5m": 300, "15m": 900}
TL = {"5m": t5, "15m": t15}
fee = lambda sh, px: sh*0.07*px*(1-px)

print("\n== STALE-ASK SURVIVAL after jump (ask not yet repriced) and BUY EV ==")
for mkt, dur in DUR.items():
    tl = TL[mkt]
    tot = {d: [0, 0] for d in (100, 250, 500, 1000, 2000)}
    pnls = []
    band = {"0.1-0.3": [0, 0.0], "0.3-0.5": [0, 0.0], "0.5-0.8": [0, 0.0], "0.8-0.9": [0, 0.0]}
    for te, d in jumps:
        pre = touch_before(tl, te)
        if not pre: continue
        side_ask = 1 if d > 0 else 2  # baU or baD index
        pre_ask = pre[side_ask]
        w = int(te//10**9 // dur * dur)
        for dt in tot:
            post = touch_after(tl, te, dt)
            if post is None: continue
            if post[side_ask] <= pre_ask + 1e-9:
                tot[dt][0] += 1
            tot[dt][1] += 1
        # buy at pre-ask at +100ms if still stale, band, winner known
        post = touch_after(tl, te, 100)
        if post and post[side_ask] <= pre_ask and 0.10 <= pre_ask <= 0.90:
            ws = winner(w, dur)
            if ws:
                sh = int(5/pre_ask)
                if sh >= 5:
                    pnl = sh*(1.0 if ("Up" if d > 0 else "Down") == ws else 0.0) - sh*pre_ask - fee(sh, pre_ask)
                    pnls.append(pnl)
                    k = ("0.1-0.3" if pre_ask < .3 else "0.3-0.5" if pre_ask < .5 else "0.5-0.8" if pre_ask < .8 else "0.8-0.9")
                    band[k][0] += 1; band[k][1] += pnl
    surv = " | ".join("%dms %d/%d (%.0f%%)" % (dt, v[0], v[1], 100*v[0]/max(v[1],1)) for dt, v in tot.items())
    if pnls:
        n = len(pnls); net = sum(pnls)
        wins = sum(1 for x in pnls if x > 0)
        print("%-4s stale: %s\n     EV: n %d net $%.0f mean %+.2f win %.0f%%\n     bands: %s" % (
            mkt, surv, n, net, net/n, 100*wins/n,
            {k: v[0] for k, v in band.items()}))
    else:
        print("%-4s stale: %s\n     EV: no qualifying buys yet" % (mkt, surv))

def time_guard(tl): return tl[-1][0]

print("\n== LAST-2-MIN TAKES (leader bid side, ask .35-.80) ==")
for mkt, dur in [("15m", 900), ("5m", 300)]:
    tl = TL[mkt]
    ws = sorted({int(t//10**9//dur*dur) for t, *_ in tl if t >= T0})
    res = []
    for w in ws:
        if w + dur > (tl[-1][0] if mkt == "15m" else tl[-1][0]): pass
        win = winner(w, dur)
        if not win: continue
        lo = (w + dur - 120)*10**9; hi = (w + dur - 30)*10**9
        cands = [x for x in tl if lo <= x[0] <= hi]
        if not cands: continue
        x = cands[0]
        leader = "Up" if x[3] > x[4] else "Down"
        ask = x[1] if leader == "Up" else x[2]
        if 0.35 <= ask <= 0.80:
            sh = int(5/ask)
            pnl = sh*(1.0 if leader == win else 0.0) - sh*ask - fee(sh, ask)
            res.append(pnl)
    if res:
        n = len(res)
        print("%-4s cell: n %d net $%.0f mean %+.2f win %.0f%%" % (mkt, n, sum(res), sum(res)/n, 100*sum(1 for x in res if x>0)/n))
    else:
        print("%-4s cell: no entries yet" % mkt)
