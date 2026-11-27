import json, os, glob, time, math

now = int(time.time())
CUT = now - 86400
FLOOR = 1793801658  # clean paper era start
LO = max(CUT, FLOOR)

settles = {}   # ws -> pnl
win_of = {}    # ws -> winner
orders = {}    # ws -> first entry ask
skip_r_lines = 0
skip_r_ev = set()
skip_nb = 0
sock_end = 0
last_hb = None

for p in sorted(glob.glob('/root/poly/latencylog/bot-*.jsonl')):
    if '/archive' in p:
        continue
    try:
        with open(p) as f:
            for line in f:
                if '"src"' not in line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                s = o.get('src')
                t = o.get('t_ns', 0) / 1e9
                if s == 'settle':
                    ws = o.get('ws')
                    if ws and ws >= LO and ws not in settles:
                        settles[ws] = o.get('pnl', 0.0)
                        win_of[ws] = o.get('winner')
                elif s == 'paper_order' and t >= LO:
                    w = int(t // 300 * 300)
                    if w not in orders and 'ask' in o:
                        orders[w] = o['ask']
                elif s == 'skip_repriced' and t >= LO:
                    skip_r_lines += 1
                    skip_r_ev.add((int(t), o.get('a_prev'), o.get('a_now'), o.get('aligned')))
                elif s == 'skip_nobook' and t >= LO:
                    skip_nb += 1
                elif s == 'socket_end' and t >= LO:
                    sock_end += 1
                elif s == 'hb' and t >= LO:
                    if last_hb is None or t > last_hb[0]:
                        last_hb = (t, o)
    except Exception as e:
        print('ERR', p, e)

wss = sorted(settles)
pnls = [settles[w] for w in wss]
n = len(pnls)
net = sum(pnls)
wins = sum(1 for x in pnls if x > 0)
mean = net / n if n else 0.0
sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (n - 1)) if n > 1 else 0.0

# max drawdown on cumulative by ws
cum = 0.0; peak = 0.0; mdd = 0.0
for x in pnls:
    cum += x
    peak = max(peak, cum)
    mdd = min(mdd, cum - peak)

# longest losing streak (pnl<0)
lls = cur = 0
for x in pnls:
    cur = cur + 1 if x < 0 else 0
    lls = max(lls, cur)

# entry bands from joined paper_order ask
bands = {'0.1-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, 'other': 0}
no_ask = 0
for w in wss:
    a = orders.get(w)
    if a is None:
        no_ask += 1
        continue
    if 0.1 <= a < 0.3: bands['0.1-0.3'] += 1
    elif 0.3 <= a < 0.5: bands['0.3-0.5'] += 1
    elif 0.5 <= a < 0.7: bands['0.5-0.7'] += 1
    elif 0.7 <= a <= 0.9: bands['0.7-0.9'] += 1
    else: bands['other'] += 1

hb = last_hb[1] if last_hb else {}
out = {
    'now': now, 'cut': LO, 'n': n, 'wins': wins, 'winrate': wins / n if n else 0,
    'net': round(net, 2), 'mean': round(mean, 3), 'sd': round(sd, 3),
    'se': round(sd / math.sqrt(n), 3) if n else 0,
    'mdd': round(mdd, 2), 'lls': lls,
    'bands': bands, 'no_ask': no_ask,
    'skip_repriced_lines': skip_r_lines, 'skip_repriced_events': len(skip_r_ev),
    'skip_nobook': skip_nb, 'socket_end': sock_end,
    'hb_halt': hb.get('halt'), 'hb_cb_age_s': hb.get('cb_age_s'),
    'hb_cum_pnl_since_boot': hb.get('pnl'),
    'best_trade': round(max(pnls), 2) if pnls else 0,
    'worst_trade': round(min(pnls), 2) if pnls else 0,
}
print(json.dumps(out))
