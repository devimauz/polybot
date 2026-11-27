"""_15m_worklist.py - enumerate closed btc-updown-15m/4h windows via deterministic slugs."""
import json, urllib.request, urllib.error, time, datetime, os

def g(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=25))
        except urllib.error.HTTPError as e:
            time.sleep(2.0 * (k + 1) * (2 if e.code == 429 else 1))
        except Exception:
            time.sleep(1.5 * (k + 1))
    return []

now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
out = []
for per, tag, nback in ((900, "15m", 640), (14400, "4h", 160)):
    got = 0
    for i in range(3, nback):
        ws = (now // per) * per - i * per
        slug = f"btc-updown-{tag}-{ws}"
        e = g(f"https://gamma-api.polymarket.com/events?slug={slug}")
        if not e:
            continue
        m = e[0]["markets"][0]
        try:
            prs = json.loads(m["outcomePrices"]); toks = json.loads(m["clobTokenIds"])
        except Exception:
            continue
        if sorted(float(x) for x in prs) != [0.0, 1.0]:
            continue
        out.append(dict(slug=slug, tag=tag, ws=ws, cond=m["conditionId"], tok=toks[0],
                        win=1 if float(prs[0]) == 1 else 0,
                        vol=float(m.get("volumeNum") or 0)))
        got += 1
        if i % 5 == 0:
            time.sleep(0.15)
    print(tag, "closed markets:", got, flush=True)

json.dump(out, open("_pm_out/updown_worklist.json", "w"))
import statistics
for t in ("15m", "4h"):
    l = [o for o in out if o["tag"] == t]
    if l:
        print(t, "n=", len(l), "median vol $%.0f" % statistics.median(o["vol"] for o in l),
              "up-rate %.3f" % (sum(o["win"] for o in l) / len(l)))
