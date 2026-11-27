import json, glob, datetime

now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
win_start = now - 86400
files = sorted(glob.glob('/root/poly/latencylog/bot-*.jsonl'))
files = [f for f in files if '/archive/' not in f]

# halt events: first hb with halt True per contiguous run
halts = []
cum = 0
pts = []
for f in files:
    base = f.split('/')[-1]
    try:
        wsf = int(base.split('-')[1].split('.')[0])
    except Exception:
        wsf = None
    for line in open(f):
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get('t_ns', 0) / 1e9
        if not (win_start <= t <= now):
            continue
        s = o.get('src')
        if s == 'settle':
            pts.append((o['ws'], o['pnl']))
        elif s == 'hb' and o.get('halt'):
            halts.append((wsf, o.get('pnl'), datetime.datetime.utcfromtimestamp(t).strftime('%m-%d %H:%M')))

# cumulative by ws
ws_sorted = sorted(dict(pts).keys())
pns = [dict(pts)[w] for w in ws_sorted]
cum = 0
series = []
peak = 0
mdd = 0
mdd_at = None
for i, w in enumerate(ws_sorted):
    cum += pns[i]
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > mdd:
        mdd = dd
        mdd_at = w
    series.append((w, cum))
# sample cumulative at block boundaries for arc
blocks = {}
for w, c in series:
    d = datetime.datetime.utcfromtimestamp(w).strftime('%m-%d %H')
    blocks[d] = c
print("HALT EVENTS (last 24h) count:", len(halts))
for h in halts[:20]:
    print("  ", h)
print("cumulative arc by hour (ws hh, cum$):")
for k in sorted(blocks):
    print(f"  {k}  {blocks[k]:.3f}")
print(f"total cum: {cum:.3f}  mdd {mdd:.3f} reached at ws {datetime.datetime.utcfromtimestamp(mdd_at).strftime('%m-%d %H:%M')}")