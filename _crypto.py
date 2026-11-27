#!/usr/bin/env python3
"""The decisive test: crypto 'What price will BTC hit' markets vs Coinbase spot.
If the market's implied probability disagrees with the objective BTC price, that's
a real arb. Sports have no objective reference (can't arb), politics is
subjective. Crypto is the only category with an external truth.

We take each open BTC 'hit $X' market, get its ask, and compute the objective
probability from spot (BTC currently above/below $X -> resolves YES/NO).
"""
import json, subprocess, urllib.request

def get(url):
    out = subprocess.check_output(["curl","-s","--compressed","--max-time","20",url], text=True)
    return json.loads(out)

# BTC spot from Coinbase (works on AMS)
try:
    cb = get("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
    btc = float(cb["price"])
    print(f"Coinbase BTC spot: ${btc:,.2f}\n")
except Exception as ex:
    print("spot fetch failed:", ex); btc=None

# pull crypto markets
ev=[]
for off in range(0,300,50):
    p=get(f"https://gamma-api.polymarket.com/events?limit=50&offset={off}&closed=false&order=volume24hr&ascending=false")
    if not isinstance(p,list) or not p: break
    ev+=p
    if len(p)<50: break

rows=[]
for e in ev:
    t=e.get("ticker","")
    if not any(t.startswith(x) for x in ('bitcoin','btc','ethereum','eth','solana','sol','xrp','crypto')):
        continue
    for m in (e.get("markets") or []):
        if not isinstance(m,dict) or m.get("closed") or not m.get("acceptingOrders"): continue
        q=m.get("question","")
        rows.append(dict(q=q, ticker=t, outcomes=m.get("outcomes"),
                         prices=m.get("outcomePrices"), spread=m.get("spread"),
                         bestask=m.get("bestAsk"), vol=float(m.get("volume24hr") or 0),
                         end=m.get("endDate","")))
print(f"open crypto markets: {len(rows)}\n")
print(f"{'24hVol':>9s}  {'ask':>5s}  {'spread':>6s}  question")
print("-"*90)
for r in sorted(rows,key=lambda x:-x['vol'])[:20]:
    print(f"{r['vol']:9,.0f}  {str(r['bestask']):>5s}  {str(r['spread']):>6s}  {r['q'][:60]}")

# For BTC 'hit $X by <date>' markets: objective prob = P(BTC >= X by date).
# At minimum, if BTC is already >= X, the 'hit' should resolve YES (prob ~1).
# If market ask < 0.9 while BTC already above X, that's a mispricing.
print("\n=== objective check: markets where BTC already above/below threshold ===")
if btc:
    import re
    for r in rows:
        q=r['q']
        # parse 'What price will Bitcoin hit in November?' -> threshold in outcomes
        # these markets have multiple outcomes (price buckets); skip complex ones
        m=re.search(r'\$(\d[\d,]*)\s*(k|K)?', q)
        if m:
            thr=float(m.group(1).replace(',',''))
            if m.group(2): thr*=1000
            above = btc >= thr
            print(f"  BTC ${btc:,.0f} vs thr ${thr:,.0f} -> {'ABOVE' if above else 'below'}; ask={r['bestask']}  {q[:40]}")