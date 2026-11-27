"""_cal_worklist2.py - crawl gamma CLOSED markets per category per weekly window.

Windowed by end_date_min/max so we sample across time (newest-first alone only
reaches the last hour of mass-resolutions). Keeps binary, resolved [0,1]
markets with volumeNum >= 1000. Writes _pm_out/cal_worklist.json sorted by
closed time; used by the AWS prices-history fetcher (_cal_fetch.py).
"""
import json, time, urllib.request, urllib.error, datetime

def g(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            time.sleep(2.5 * (k + 1) * (2 if e.code == 429 else 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return []

def to_u(x):
    if not x:
        return None
    try:
        return int(datetime.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None

TAGS = ["nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc", "weather", "finance", "politics"]
now = datetime.datetime(2026, 10, 30, 2, tzinfo=datetime.timezone.utc)
WINDOWS = [(now - datetime.timedelta(days=7 * (w + 1)), now - datetime.timedelta(days=7 * w)) for w in range(10)][::-1]
PAGES = 3

seen, out = set(), []
for tag in TAGS:
    kept = 0
    for w0, w1 in WINDOWS:
        for p in range(PAGES):
            r = g("https://gamma-api.polymarket.com/markets?closed=true&tag_slug={t}"
                  "&limit=100&offset={o}&order=closedTime&ascending=false"
                  "&end_date_min={a}&end_date_max={b}".format(
                      t=tag, o=p * 100, a=w0.strftime("%Y-%m-%dT%H:%M:%SZ"), b=w1.strftime("%Y-%m-%dT%H:%M:%SZ")))
            if not r:
                break
            for m in r:
                cond = m.get("conditionId")
                if not cond or cond in seen:
                    continue
                try:
                    outs = json.loads(m["outcomes"]); prs = json.loads(m["outcomePrices"]); toks = json.loads(m["clobTokenIds"])
                except Exception:
                    continue
                if len(outs) != 2 or len(toks) != 2:
                    continue
                if sorted(float(x) for x in prs) != [0.0, 1.0]:
                    continue
                vol = float(m.get("volumeNum") or 0)
                if vol < 1000:
                    continue
                cl = to_u(m.get("closedTime")) or to_u(m.get("endDate"))
                en = to_u(m.get("endDate")) or cl
                if cl is None:
                    continue
                seen.add(cond)
                kept += 1
                out.append(dict(cond=cond, tag=tag, tok=toks[0],
                                win=1 if float(prs[0]) == 1.0 else 0,
                                q=(m.get("question") or "")[:80], closed=cl, end=en,
                                vol=vol, liq=float(m.get("liquidityNum") or 0),
                                spread=m.get("spread"), negrisk=bool(m.get("negRisk"))))
            if len(r) < 100:
                break
            time.sleep(0.45)
        time.sleep(0.3)
    print(f"{tag:9s} kept={kept}", flush=True)

out.sort(key=lambda x: x["closed"])
json.dump(out, open("_pm_out/cal_worklist.json", "w"))
print("TOTAL:", len(out))
by = {}
for o in out:
    by[o["tag"]] = by.get(o["tag"], 0) + 1
print(by)
if out:
    print("range:", datetime.datetime.fromtimestamp(out[0]["closed"], datetime.timezone.utc),
          "->", datetime.datetime.fromtimestamp(out[-1]["closed"], datetime.timezone.utc))
