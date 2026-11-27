"""_cal_worklist.py - crawl gamma for CLOSED markets per category, build AWS fetch list.

Fields kept per market: cond, tok (outcomes[0] clob token), win (1/0 = did
outcomes[0] win), q, end, closed (unix), vol, liq, spread, negrisk, tag.
Keeps binary markets, volumeNum >= 1000, closed within the crawlable window.
Writes _pm_out/cal_worklist.json (dedup by cond).
"""
import json, time, urllib.request, urllib.error

def g(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            code = e.code
            if code == 422:  # gamma rate-limit quirk
                time.sleep(3 * (k + 1)); continue
            if code == 429:
                time.sleep(10 * (k + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (k + 1))
    return []

import datetime
def unix(s):
    try:
        return int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None

TAGS = ["nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc", "weather", "finance", "politics"]
PAGES = 12  # 12 x 100 per tag, newest-closed first
seen = set()
out = []
for tag in TAGS:
    kept = 0
    for p in range(PAGES):
        off = p * 100
        r = g(f"https://gamma-api.polymarket.com/markets?closed=true&tag_slug={tag}&limit=100&offset={off}&order=closedTime&ascending=false")
        if not r:
            break
        for m in r:
            cond = m.get("conditionId")
            if not cond or cond in seen:
                continue
            try:
                outs = json.loads(m["outcomes"]); prs = json.loads(m["outcomePrices"])
                toks = json.loads(m["clobTokenIds"])
            except Exception:
                continue
            if len(outs) != 2 or len(toks) != 2:
                continue
            if sorted([float(x) for x in prs]) != [0.0, 1.0]:
                continue
            vol = float(m.get("volumeNum") or 0)
            if vol < 1000:
                continue
            cl = unix(m.get("closedTime") or "") or unix(m.get("endDate") or "")
            if cl is None:
                continue
            seen.add(cond)
            kept += 1
            out.append({
                "cond": cond, "tag": tag, "tok": toks[0],
                "win": 1 if float(prs[0]) == 1.0 else 0,
                "q": (m.get("question") or "")[:80], "closed": cl,
                "end": unix(m.get("endDate") or "") or cl,
                "vol": vol, "liq": float(m.get("liquidityNum") or 0),
                "spread": m.get("spread"), "negrisk": bool(m.get("negRisk")),
            })
        time.sleep(0.6)
    print(f"{tag:9s} kept={kept}", flush=True)
out.sort(key=lambda x: x["closed"])
json.dump(out, open("_pm_out/cal_worklist.json", "w"))
print("TOTAL markets:", len(out))
by = {}
for o in out:
    by[o["tag"]] = by.get(o["tag"], 0) + 1
print(by)
print("closed range:", datetime.datetime.utcfromtimestamp(out[0]['closed']), "->", datetime.datetime.utcfromtimestamp(out[-1]['closed']))
