import json, glob, time, collections

now = time.time()
cut = now - 86400
files = glob.glob('/root/poly/latencylog/bot-*.jsonl')

settles = {}
entry = {}
ev = collections.Counter()   # skip_repriced, socket_end, reconnect counts
hb_last = None
per_win = collections.Counter()

for f in files:
    try:
        ws = int(f.split('-')[-1].split('.')[0])
    except Exception:
        continue
    in24 = (cut <= ws <= now)
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        s = d.get('src')
        if s == 'settle':
            if in24 and cut <= d['ws'] <= now and d['ws'] not in settles:
                settles[d['ws']] = d
        elif s == 'paper_order':
            entry.setdefault(ws, d.get('ask'))
        elif s in ('skip_repriced', 'socket_end', 'reconnect', 'socket_reconnect'):
            if in24:
                ev[s] += 1
        elif s == 'hb':
            if in24:
                hb_last = d

ws_sorted = sorted(settles.keys())
trades = len(ws_sorted)
wins = sum(1 for w in ws_sorted if settles[w]['pnl'] > 0)
net = sum(settles[w]['pnl'] for w in ws_sorted)
mean = net / trades if trades else 0.0

# cumulative / max drawdown (peak->trough on cumulative by ws)
cum = 0.0
peak = 0.0
maxdd = 0.0
for w in ws_sorted:
    cum += settles[w]['pnl']
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > maxdd:
        maxdd = dd

# longest losing streak (pnl<0 consecutive)
streak = best = 0
for w in ws_sorted:
    if settles[w]['pnl'] < 0:
        streak += 1
        best = max(best, streak)
    else:
        streak = 0

# entry-band distribution (0.1-0.3 / 0.3-0.5 / 0.5-0.7 / 0.7-0.9)
band = collections.Counter()
for w in ws_sorted:
    p = entry.get(w)
    if p is None:
        continue
    if p < 0.3: band['0.1-0.3'] += 1
    elif p < 0.5: band['0.3-0.5'] += 1
    elif p < 0.7: band['0.5-0.7'] += 1
    elif p <= 0.9: band['0.7-0.9'] += 1
nband = sum(band.values())

print('WINDOW last24h settle-ws range: %s .. %s (now=%.0f, cut=%.0f)' % (ws_sorted[0] if ws_sorted else None, ws_sorted[-1] if ws_sorted else None, now, cut))
print('trades=%d wins=%d net=%.2f mean=%.2f' % (trades, wins, net, mean))
print('winpct=%.1f maxdd=%.2f longest_losing_streak=%d' % (100.0*wins/trades if trades else 0, maxdd, best))
print('entry_band=%s nband=%d' % (dict(band), nband))
print('skip_repriced=%d socket_end=%d reconnect=%d' % (ev['skip_repriced'], ev['socket_end'], ev['reconnect']))
if hb_last:
    print('hb_last pnl=%.2f halt=%s cb_age_s=%s mid_age_s=%s' % (hb_last.get('pnl'), hb_last.get('halt'), hb_last.get('cb_age_s'), hb_last.get('mid_age_s')))
else:
    print('hb_last=None')
# running total & recent settle list (for sanity)
tot = hb_last.get('pnl') if hb_last else None
print('current_total_pnl_from_hb=%s' % tot)