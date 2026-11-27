#!/usr/bin/env python3
"""_slow_holdout.py — replay the deep-lock tail cells (A/C families) on the AMS ms corpus
(Nov 4-14), the design-independent holdout. cb 1s resample + touch series + reconstructed
winners (Coinbase-TWAP proxy). Depth gate: fill only if visible ask depth at T >= shares."""
import json, glob, gzip, os, bisect, numpy as np
from collections import defaultdict

LOG = "/root/poly/laglog"
# gather cb prints and touches per second
cb_sec = {}     # ws_s -> last price in that second
pm = defaultdict(dict)   # ws_s -> (baU,baD,bbU,bbD, asz?) touches keep last
meta = {}
for fp in sorted(glob.glob(os.path.join(LOG, "lag-*.jsonl")) + glob.glob(os.path.join(LOG, "lag-*.jsonl.gz"))):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt") as f:
        for L in f:
            if '"src":"cb"' in L and '"tp":"m"' in L:
                r = json.loads(L); s = int(r["t_ns"] // 10**9)
                cb_sec[s] = r["p"]
            elif '"src":"pm"' in L:
                r = json.loads(L); s = int(r["t_ns"] // 10**9)
                pm[s] = (r["baU"], r["baD"], r["bbU"], r["bbD"])
print("seconds covered: cb %d pm %d" % (len(cb_sec), len(pm)))
tmin = max(min(cb_sec), min(pm)); tmax = min(max(cb_sec), max(pm))

def cb_series(a, b):
    return np.array([cb_sec.get(s, np.nan) for s in range(a, b)])

def norm_cdf(x): return 0.5 * (1 + np.tanh(x * 0.7978845608))

trades = []   # (ws, T, side, ask, sh, win)
for w in range(int(tmin // 300 * 300), int(tmax // 300 * 300), 300):
    cbw = cb_series(w, w + 300)
    fin = np.isfinite(cbw)
    if fin.sum() < 240: continue
    # strike proxy: mean cb over [w-60,w) needed; use first-seconds fallback to window open
    pre = cb_series(w - 60, w)
    if np.isfinite(pre).sum() < 40: continue
    strike = pre[np.isfinite(pre)].mean()
    tail = cbw[240:300][np.isfinite(cbw[240:300])]
    if len(tail) < 30: continue
    settle = tail.mean()
    win = "Up" if settle > strike else "Down"
    d = np.diff(cbw[fin]) * 2.2
    sig = float(np.std(d)) if len(d) > 30 else 10.0
    for T in (240, 260, 280):
        for margin in (0.04, 0.06):
            s = w + T
            t = pm.get(s)
            if not t: continue
            baU, baD, bbU, bbD = t
            obs = cbw[240:T]
            obs = obs[np.isfinite(obs)]
            if len(obs) < 3: continue
            m_obs = obs.mean(); cur = obs[-1]; tau_s = 300 - T
            sd = sig * np.sqrt(tau_s / 3.0) + 7.62
            exp_settle = (m_obs * len(obs) + cur * tau_s) / (len(obs) + tau_s)
            p = norm_cdf((exp_settle - strike) / sd)
            for side, ask, pp in (("Up", baU, p), ("Down", baD, 1 - p)):
                if not (0.10 <= ask <= 0.90): continue
                if pp - (ask + margin) <= 0: continue
                sh = int(5 / ask)
                if sh < 5: continue
                trades.append((w, T, side, ask, sh, win, margin))
res = defaultdict(list)
for w, T, side, ask, sh, win, mg in trades:
    for gate in ((240,), (240, 260, 280)):
        pass
    pnl = sh * (1.0 if side == win else 0.0) - sh * (ask + 0.02) - sh * 0.07 * (ask + 0.02) * (1 - ask)
    res[(mg, T)].append((pnl, side, w))
print("\n== HOLDOUT (Nov4-14 ms corpus, independent of Oct design) ==")
print("%-16s %5s %8s %7s %-22s %-22s" % ("cell", "n", "mean$", "win%", "Up$/Dn$", "first-half/2nd"))
for key in sorted(res):
    rs = res[key]
    if len(rs) < 5: continue
    n = len(rs); mean = sum(x[0] for x in rs) / n
    wr = 100 * sum(1 for x in rs if x[0] > 0) / n
    up = [x[0] for x in rs if x[1] == "Up"]; dn = [x[0] for x in rs if x[1] == "Down"]
    half = n // 2
    h1 = sum(x[0] for x in rs[:half]) / max(half, 1); h2 = sum(x[0] for x in rs[half:]) / max(n - half, 1)
    print("lock+%.0fc T%d %6d %8.3f %6.1f  U%+.2f/D%+.2f %s %+.2f/%+.2f" % (
        key[0] * 100, key[1], n, mean, wr,
        (sum(up) / len(up)) if up else 0, (sum(dn) / len(dn)) if dn else 0,
        f"ws {rs[0][2]}-{rs[-1][2]}", h1, h2))
# C-style: any-T earliest
early = {}
for w, T, side, ask, sh, win, mg in trades:
    if mg < 0.06: continue
    k = w
    if k not in early or T < early[k][0]: early[k] = (T, side, ask, sh, win)
rs = []
for k, (T, side, ask, sh, win) in sorted(early.items()):
    pnl = sh * (1.0 if side == win else 0.0) - sh * (ask + 0.02) - sh * 0.07 * (ask + 0.02) * (1 - ask)
    rs.append((pnl, side))
if rs:
    import csv
    with open("/root/poly/slow_holdout_trades.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh); wcsv.writerow(["ws","T","side","ask","sh","pnl","win"])
        for k in sorted(early):
            T, side, ask, sh, win = early[k]
            pay = min(ask+0.02, 0.99)
            wcsv.writerow([k, T, side, round(ask,2), sh,
                           round(sh*(1.0 if side==win else 0.0)-sh*pay-sh*0.07*pay*(1-pay), 3), win])
    n = len(rs); mean = sum(x[0] for x in rs) / n
    half = n // 2
    up = [x[0] for x in rs if x[1] == "Up"]; dn = [x[0] for x in rs if x[1] == "Down"]
    print("C any-T 6c    %5d %8.3f %6.1f  U%+.2f/D%+.2f half %+0.2f/%+0.2f" % (
        n, mean, 100 * sum(1 for x in rs if x[0] > 0) / n,
        sum(up) / max(len(up), 1), sum(dn) / max(len(dn), 1),
        sum(x[0] for x in rs[:half]) / half, sum(x[0] for x in rs[half:]) / (n - half)))
