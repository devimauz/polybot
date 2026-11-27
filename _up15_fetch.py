"""_up15_fetch.py - fetch all taker prints for 15m/4h updown windows (AWS side)."""
import json, os, time, urllib.request, gzip

HOME = os.path.expanduser("~/poly/up15")
WL = json.load(open(os.path.join(HOME, "worklist.json")))
OUT = os.path.join(HOME, "prints.jsonl.gz")
DONE = os.path.join(HOME, "done.txt")
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()

def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return []

fh = gzip.open(OUT, "at", encoding="utf-8")
for i, m in enumerate(WL):
    if m["cond"] in done:
        continue
    rows = []
    for off in (0, 1000, 2000, 3000):
        d = get(f"https://data-api.polymarket.com/trades?market={m['cond']}&limit=1000&offset={off}&takerOnly=true")
        if not d:
            break
        rows += d
        if len(d) < 1000:
            break
    for t in rows:
        fh.write(json.dumps({"c": m["cond"], "ts": t["timestamp"], "p": t["price"],
                             "sz": t["size"], "b": 1 if t["side"] == "BUY" else 0,
                             "u": 1 if t["outcome"] == "Up" else 0}) + "\n")
    open(DONE, "a").write(m["cond"] + "\n")
    if i % 100 == 0:
        fh.flush()
        print(f"{i}/{len(WL)} prints={len(rows)}", flush=True)
    time.sleep(0.1)
print("ALLDONE", flush=True)
