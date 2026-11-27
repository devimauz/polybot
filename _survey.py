#!/usr/bin/env python3
"""Survey Polymarket categories by 24h volume and liquidity — run ON the AMS VPS."""
import json, subprocess

def get(url):
    out = subprocess.check_output(["curl","-s","--compressed","--max-time","20",url], text=True)
    return json.loads(out)

# top events by 24h volume, open only (paginate, limit<=50)
ev = []
for off in range(0, 300, 50):
    page = get(f"https://gamma-api.polymarket.com/events?limit=50&offset={off}&closed=false&order=volume24hr&ascending=false")
    if not isinstance(page, list): break
    ev += page
    if len(page) < 50: break
print("events:", len(ev))
cats={}
for e in ev:
    if not isinstance(e, dict): continue
    c = e.get("category") or "other"
    cats[c] = cats.get(c,0)+1
print("by count:", cats)

# aggregate volume by category
vol={}
for e in ev:
    if not isinstance(e, dict): continue
    c=e.get("category") or "other"
    vol[c]=vol.get(c,0)+float(e.get("volume24hr") or 0)
print("\n=== 24h volume by category (top) ===")
for c,v in sorted(vol.items(), key=lambda x:-x[1])[:15]:
    print(f"  {c:16s} ${v:12,.0f}")

print("\n=== top 25 events by 24h volume ===")
for e in ev[:25]:
    if not isinstance(e, dict): continue
    mk = e.get("markets") or [{}]
    m = mk[0] if mk else {}
    print(f"  ${float(e.get('volume24hr') or 0):10,.0f}  {(e.get('category') or '?'):12s}  {(e.get('title') or '')[:55]}")

# also check sports specifically (the user mentioned sport)
print("\n=== sports-like events (by ticker prefix) ===")
sports=[e for e in ev if isinstance(e,dict) and any(e.get('ticker','').startswith(p) for p in ('nba','nfl','nhl','mlb','nca','ufc','tennis','atp','wta','f1','golf','pga','epl','soccer','ufc','boxing'))]
print("sports events in top 300:", len(sports))
for e in sports[:15]:
    print(f"  ${float(e.get('volume24hr') or 0):10,.0f}  {(e.get('title') or '')[:55]}")