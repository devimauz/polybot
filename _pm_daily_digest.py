#!/usr/bin/env python3
"""Aggregate last-24h settle rows from pm-latency-bot logs on the VPS."""
import json, glob, sys, os, collections

LOGDIR = "/root/poly/latencylog"
now = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if not now:
    now = 1794267623  # fallback; primary set by caller
start_cut = now - 86400
# clean paper era floor
clean_start = 1793801658

settles = []      # ws -> trade dict
skip_repriced = 0
socket_ends = 0
hb_counts = 0
cb_age_sum = 0.0
halt_flag = False
n_halts = 0
band = collections.Counter()

for fp in glob.glob(os.path.join(LOGDIR, "bot-*.jsonl")):
    ws = int(os.path.basename(fp)[4:-6])
    if ws < start_cut or ws < clean_start:
        continue
    with open(fp, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            src = r.get("src")
            if src == "settle":
                w = r.get("ws", ws)
                if w < start_cut or w < clean_start:
                    continue
                settles.append(r)
            elif src == "skip_repriced":
                skip_repriced += 1
            elif src == "socket_end":
                socket_ends += 1
            elif src == "hb":
                hb_counts += 1
                a = r.get("cb_age_s")
                if isinstance(a, (int, float)):
                    cb_age_sum += a
                if r.get("halt"):
                    halt_flag = True
                    n_halts += 1

# dedupe settle by ws (keep last)
byws = {}
for r in settles:
    byws[r.get("ws")] = r
settles = list(byws.values())
settles.sort(key=lambda r: r.get("ws", 0))

trades = len(settles)
pnls = []
for r in settles:
    p = r.get("pnl")
    if p is None:
        # fall back to other pnl keys
        p = r.get("net") or r.get("delta") or r.get("usd")
    pnls.append(float(p) if p is not None else 0.0)
    # entry band
    ep = r.get("entry") or r.get("price") or r.get("px")
    if isinstance(ep, (int, float)):
        if 0.1 <= ep <= 0.3: band["0.10-0.30"] += 1
        elif 0.3 < ep <= 0.5: band["0.30-0.50"] += 1
        elif 0.5 < ep <= 0.7: band["0.50-0.70"] += 1
        elif 0.7 < ep <= 0.9: band["0.70-0.90"] += 1
        else: band["other"] += 1

wins = sum(1 for p in pnls if p > 0)
net = sum(pnls)
mean = net / trades if trades else 0.0
# max drawdown on cumulative by ws order (already sorted)
cum = 0.0
peak = 0.0
maxdd = 0.0
for p in pnls:
    cum += p
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > maxdd:
        maxdd = dd
# longest losing streak
losestreak = 0
cur = 0
for p in pnls:
    if p <= 0:
        cur += 1
        if cur > losestreak:
            losestreak = cur
    else:
        cur = 0

res = {
    "cuts": {"now": now, "start_cut": start_cut},
    "trades": trades,
    "wins": wins,
    "win_pct": round(100.0 * wins / trades, 1) if trades else 0.0,
    "net": round(net, 2),
    "mean": round(mean, 4),
    "max_drawdown": round(maxdd, 2),
    "longest_losing_streak": losestreak,
    "entry_bands": dict(band),
    "skip_repriced": skip_repriced,
    "socket_ends": socket_ends,
    "hb_count": hb_counts,
    "hb_mean_cb_age_s": round(cb_age_sum / hb_counts, 2) if hb_counts else None,
    "halt_flag": halt_flag,
    "n_halts": n_halts,
    "ws_range": [settles[0]["ws"] if settles else None, settles[-1]["ws"] if settles else None],
}
print(json.dumps(res, indent=2))