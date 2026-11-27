from loader import load
import collections
ws = load(); items = sorted(ws.items())
c = collections.Counter(); cw = collections.Counter()
for slug, w in items:
    seen = set()
    for fr in w["frames"]:
        t = fr["t"]
        if not (90 <= t < 240): continue
        lead = "up" if fr["up"]["ask"] >= fr["dn"]["ask"] else "dn"
        a = fr[lead]["ask"]
        b = int(a*20)/20
        c[b]+=1
        seen.add(b)
    for b in seen: cw[b]+=1
print("leader ask bucket (20 buckets): frame counts:", dict(sorted(c.items())))
print("window counts (leader ever in bucket):", dict(sorted(cw.items())))
# spread of leader books at 0.8-0.95
import statistics
sp=[]
for slug,w in items[:40]:
    for fr in w["frames"]:
        lead = "up" if fr["up"]["ask"] >= fr["dn"]["ask"] else "dn"
        d=fr[lead]
        if 0.8<=d["ask"]<0.95 and 90<=fr["t"]<240: sp.append(d["ask"]-d["bid"])
print("leader spreads 0.8-0.95:", len(sp), statistics.median(sp) if sp else None)