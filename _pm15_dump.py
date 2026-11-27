#!/usr/bin/env python3
"""_pm15_dump.py — dump per-trade rows of the 15m theta8 stale-buy (official winners)
to /root/poly/pm15_pocket_trades.csv: ws,hour,side,ask,sh,pnl"""
import json, glob, gzip, os, math, datetime, csv
from collections import deque
from bisect import bisect_left

LOG = "/root/poly/laglog"
WIN = json.load(open("/root/poly/15m_winners.json"))
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
t15tt = [x[0] for x in t15]
def before(t):
    i = bisect_left(t15tt, t); return t15[i-1] if i else None
def after(t):
    i = bisect_left(t15tt, t); return t15[i] if i < len(t15) else None
jumps = []; q = deque(); last = 0
for t_ns, p in cb:
    while q and t_ns-q[0][0] > 3e8: q.popleft()
    if q and abs(p-q[0][1]) >= 8 and t_ns-last > 3e9:
        jumps.append((t_ns, p-q[0][1])); last = t_ns
    q.append((t_ns, p))
fee = lambda sh, px: sh*0.07*px*(1-px)
rows = []
for te, d in jumps:
    pre = before(te)
    if not pre: continue
    si = 1 if d > 0 else 2
    pa = pre[si]
    post = after(te+100*10**6)
    if not (post and post[si] <= pa and 0.10 <= pa <= 0.90): continue
    w = str(int(te//10**9//900*900))
    wn = WIN.get(w) or ""
    if wn not in ("Up", "Down"): continue
    sh = int(5/pa)
    if sh < 5: continue
    side = "Up" if d > 0 else "Down"
    pnl = sh*(1.0 if side == wn else 0.0) - sh*pa - fee(sh, pa)
    h = datetime.datetime.fromtimestamp(te/1e9, datetime.timezone.utc).hour
    rows.append((int(w), h, side, round(pa, 3), sh, round(pnl, 3)))
with open("/root/poly/pm15_pocket_trades.csv", "w", newline="") as fh:
    wc = csv.writer(fh); wc.writerow(["ws", "hour", "side", "ask", "sh", "pnl"])
    wc.writerows(rows)
print("dumped", len(rows))
