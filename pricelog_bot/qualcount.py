"""How many windows (not frames) qualify per cell?"""
from loader import load
import itertools, collections
ws = load(); items = sorted(ws.items())
def dog_side(fr):
    return "dn" if fr["up"]["ask"] <= fr["dn"]["ask"] else "up"
cnt = collections.Counter()
for slug, w in items:
    seen = set()
    for fr in w["frames"]:
        t = fr["t"]
        if not (150 <= t < 292): continue
        d = dog_side(fr); a = fr[d]["ask"]
        for T0,T1 in [(150,240),(180,270),(210,275),(240,285),(240,292)]:
            if T0<=t<T1 and 0.05<=a<0.50:
                seen.add((T0, int(a*10)/10))
    for k in seen: cnt[k]+=1
for k in sorted(cnt): print(k, cnt[k])