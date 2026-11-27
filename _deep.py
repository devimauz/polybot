#!/usr/bin/env python3
"""Deep survey v2: rich market fields from gamma, open markets only."""
import json, subprocess

def get(url):
    out = subprocess.check_output(["curl","-s","--compressed","--max-time","20",url], text=True)
    return json.loads(out)

ev = []
for off in range(0, 300, 50):
    page = get(f"https://gamma-api.polymarket.com/events?limit=50&offset={off}&closed=false&order=volume24hr&ascending=false")
    if not isinstance(page, list) or not page: break
    ev += page
    if len(page) < 50: break

rows=[]
for e in ev:
    for m in (e.get("markets") or []):
        if not isinstance(m, dict): continue
        if m.get("closed"): continue
        if not m.get("acceptingOrders"): continue
        vol = float(m.get("volume24hr") or 0)
        if vol < 30000: continue
        rows.append(dict(vol=vol, title=(e.get("title") or m.get("question",""))[:48],
                         ticker=e.get("ticker",""), spread=m.get("spread"),
                         bestask=m.get("bestAsk"), taker=m.get("takerBaseFee"),
                         tick=m.get("orderPriceMinTickSize")))
print(f"open tradeable markets >$30k 24h vol: {len(rows)}\n")

def cat(t):
    for p in ('nba','nfl','nhl','mlb','nca','ufc','tennis','atp','wta','f1','golf','pga','epl','soccer','boxing','mma','dota','cs','lol','valorant','counter'):
        if t.startswith(p): return 'sports/esports'
    for p in ('bitcoin','btc','ethereum','eth','crypto','price','solana','sol','xrp'):
        if t.startswith(p): return 'crypto'
    for p in ('fed','election','president','senate','congress','trump','biden','nominee','war','ceasefire','israel','iran','ukraine','russia'):
        if t.startswith(p): return 'politics/geo'
    return 'other'
from collections import defaultdict
g=defaultdict(list)
for r in rows: g[cat(r['ticker'])].append(r)

def med(a):
    a=[x for x in a if isinstance(x,(int,float))]
    return sorted(a)[len(a)//2] if a else float('nan')

print(f"{'category':16s} {'n':>4s} {'tot24hVol':>14s} {'medSpread':>10s} {'medTaker':>9s} {'medTick':>8s}")
print("-"*70)
for c,rs in sorted(g.items(), key=lambda x:-sum(r['vol'] for r in x[1])):
    vol=sum(r['vol'] for r in rs)
    print(f"{c:16s} {len(rs):4d} {vol:14,.0f} {med([r['spread'] for r in rs]):10.4f} {med([r['taker'] for r in rs]):9.4f} {med([r['tick'] for r in rs]):8.3f}")

print("\n=== top 20 markets (open) ===")
for r in sorted(rows, key=lambda x:-x['vol'])[:20]:
    print(f"  ${r['vol']:10,.0f}  sp={r['spread']}  ask={r['bestask']}  taker={r['taker']}  tick={r['tick']}  {r['title'][:40]}")