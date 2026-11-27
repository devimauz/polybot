#!/usr/bin/env python3
"""Full last-24h paper-bot stats, computed on the box."""
import json, glob, os, sys, collections

LOGDIR = "/root/poly/latencylog"
now = int(sys.argv[1])
start_cut = now - 86400
clean_start = 1793801658

orders = {}   # ws -> entry ask
settles = {}  # ws -> pnl
skip_repriced = 0
socket_ends = 0
hb_count = 0
cb_age_sum = 0.0
halt = False
n_halts = 0

for fp in glob.glob(os.path.join(LOGDIR, "bot-*.jsonl")):
    ws = int(os.path.basename(fp)[4:-6])
    if ws < start_cut or ws < clean_start:
        continue
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            src = r.get("src")
            w = r.get("ws", ws)
            if w < start_cut: continue
            if src == "paper_order":
                a = r.get("ask")
                if isinstance(a, (int, float)): orders[w] = a
            elif src == "settle":
                p = r.get("pnl")
                if isinstance(p, (int, float)): settles[w] = p
            elif src == "skip_repriced": skip_repriced += 1
            elif src == "socket_end": socket_ends += 1
            elif src == "hb":
                hb_count += 1
                a = r.get("cb_age_s")
                if isinstance(a,(int,float)): cb_age_sum += a
                if r.get("halt"): halt=True; n_halts+=1

allws = sorted(set(orders)|set(settles))
pnls = [settles[w] for w in allws if w in settles]
n=len(pnls)
wins=sum(1 for p in pnls if p>0)
net=sum(pnls)
mean=net/n if n else 0

cum=0.0; peak=0.0; maxdd=0.0
losestreak=cur=0
for p in pnls:
    cum+=p
    if cum>peak: peak=cum
    dd=peak-cum
    if dd>maxdd: maxdd=dd
    cur = cur+1 if p<=0 else 0
    if cur>losestreak: losestreak=cur

band=collections.Counter()
for w in allws:
    if w in settles:
        ep=orders.get(w)
        if ep is not None:
            if 0.1<=ep<0.3: band["0.10-0.30"]+=1
            elif 0.3<=ep<0.5: band["0.30-0.50"]+=1
            elif 0.5<=ep<0.7: band["0.50-0.70"]+=1
            elif 0.7<=ep<0.9: band["0.70-0.90"]+=1
            else: band["other"]+=1

# split half
h=n//2
h1=pnls[:h]; h2=pnls[h:]
res=dict(
  now=now, start_cut=start_cut,
  trades=n, wins=wins, win_pct=round(100*wins/n,1),
  net=round(net,2), mean=round(mean,3),
  max_drawdown=round(maxdd,2), longest_losing_streak=losestreak,
  entry_bands=dict(band), skip_repriced=skip_repriced,
  socket_ends=socket_ends, hb_count=hb_count,
  hb_mean_cb_age_s=round(cb_age_sum/hb_count,2) if hb_count else None,
  halt=halt, n_halts=n_halts,
  ws_range=[allws[0],allws[-1]],
  half1=[round(sum(h1),1),round(sum(h1)/len(h1),3) if h1 else 0, len(h1)],
  half2=[round(sum(h2),1),round(sum(h2)/len(h2),3) if h2 else 0, len(h2)],
  cumfinal=round(sum(pnls),2),
  win_mean=round(sum(p for p in pnls if p>0)/max(1,wins),3),
  loss_mean=round(sum(p for p in pnls if p<=0)/max(1,n-wins),3),
)
print(json.dumps(res, indent=2))