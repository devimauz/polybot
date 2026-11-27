import json, glob, datetime, collections, sys

now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
win_start = now - 86400

files = sorted(glob.glob('/root/poly/latencylog/bot-*.jsonl'))
files = [f for f in files if '/archive/' not in f]

settle = {}   # ws -> pnl (dedup by ws)
paper_order_asks = []  # (ws, ask)
skip_repriced = 0
socket_end = 0
hb = {'cb_age_s': [], 'halt': False, 'n': 0}
skip_nobook = 0
reconnect = 0

for f in files:
    # filename ws = bot-<ws>.jsonl
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
        s = o.get('src')
        t = o.get('t_ns', 0) / 1e9
        # gate on the row's own timestamp in window
        if not (win_start <= t <= now):
            continue
        if s == 'settle':
            settle[o['ws']] = o['pnl']  # dedup by ws
        elif s == 'paper_order':
            paper_order_asks.append((wsf, o.get('ask')))
        elif s == 'skip_repriced':
            skip_repriced += 1
        elif s == 'skip_nobook':
            skip_nobook += 1
        elif s == 'socket_end':
            socket_end += 1
        elif s == 'hb':
            hb['cb_age_s'].append(o.get('cb_age_s'))
            if o.get('halt'):
                hb['halt'] = True
            hb['n'] += 1
        elif s == 'reconnect':
            reconnect += 1

# sort settlements by ws
ws_sorted = sorted(settle.keys())
pnls = [settle[w] for w in ws_sorted]
n = len(pnls)
wins = sum(1 for p in pnls if p > 0)
net = sum(pnls)
mean = net / n if n else 0
sd = (sum((p-mean)**2 for p in pnls)/(n-1))**0.5 if n > 1 else 0
se = sd / n**0.5 if n else 0
cum = 0
peak = 0
mdd = 0
for p in pnls:
    cum += p
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > mdd:
        mdd = dd

# longest losing streak
streak = 0
max_streak = 0
for p in pnls:
    if p < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0

# entry-band distribution from paper_order asks within the settle windows
bands = {'0.1-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, 'other': 0}
settle_ws = set(ws_sorted)
for (wf, ask) in paper_order_asks:
    if ask is None:
        continue
    if wf in settle_ws:
        if 0.1 <= ask < 0.3: bands['0.1-0.3'] += 1
        elif 0.3 <= ask < 0.5: bands['0.3-0.5'] += 1
        elif 0.5 <= ask < 0.7: bands['0.5-0.7'] += 1
        elif 0.7 <= ask < 0.9: bands['0.7-0.9'] += 1
        else: bands['other'] += 1

def sig(x):
    return f"{100*x:.1f}%"

print(f"now_utc={datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')} window_start={datetime.datetime.utcfromtimestamp(win_start).strftime('%m-%d %H:%M')} UTC")
print(f"{'metric':<22}{'value'}")
print(f"{'trades (settled)':<22}{n}")
print(f"{'wins':<22}{wins} ({sig(wins/n) if n else 0})")
print(f"{'net$':<22}{net:.3f}")
print(f"{'mean$/trade':<22}{mean:.3f}")
print(f"{'per-trade sd':<22}{sd:.3f}")
print(f"{'se of mean':<22}{se:.4f}")
print(f"{'max drawdown$':<22}{mdd:.3f}")
print(f"{'longest losing streak':<22}{max_streak}")
print(f"{'entry bands 0.1-0.3':<22}{bands['0.1-0.3']}")
print(f"{'         0.3-0.5':<22}{bands['0.3-0.5']}")
print(f"{'         0.5-0.7':<22}{bands['0.5-0.7']}")
print(f"{'         0.7-0.9':<22}{bands['0.7-0.9']}")
print(f"{'         other':<22}{bands['other']}")
print(f"{'skip_repriced':<22}{skip_repriced}")
print(f"{'skip_nobook':<22}{skip_nobook}")
print(f"{'socket_end':<22}{socket_end}")
print(f"{'reconnect':<22}{reconnect}")
print(f"{'hb rows':<22}{hb['n']}")
if hb['cb_age_s']:
    print(f"{'hb cb_age_s min/max':<22}{min(hb['cb_age_s'])} / {max(hb['cb_age_s'])}")
print(f"{'halt flag seen':<22}{hb['halt']}")