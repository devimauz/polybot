"""_cal_trades_tail.py - fetch final trade prints per closed market (AWS side).

For a subset list ~/cal/subset.json ([{cond,tag,closed,vol}]): pulls up to 2
pages of data-api trades (newest-first => the LAST trades before close) and
writes ~/cal/tail.jsonl.gz rows {cond, t:[ts], p:[px], sz:[size], side:[1=BUY]}
Used only if the histogram analysis shows the settlement-lag family is worth
precision pricing.
"""
import json, os, time, urllib.request, gzip

HOME = os.path.expanduser("~/poly/cal")
SUB = json.load(open(os.path.join(HOME, "subset.json")))
OUT = os.path.join(HOME, "tail.jsonl.gz")
DONE = os.path.join(HOME, "taildone.txt")
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()

def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None

fh = gzip.open(OUT, "at", encoding="utf-8")
for i, m in enumerate(SUB):
    if m["cond"] in done:
        continue
    rows = []
    for off in (0, 500):
        d = get(f"https://data-api.polymarket.com/trades?market={m['cond']}&limit=500&offset={off}&takerOnly=true")
        if not d:
            break
        rows += d
        if len(d) < 500:
            break
    ts = [t["timestamp"] for t in rows]; px = [t["price"] for t in rows]
    sz = [t["size"] for t in rows]; sd = [1 if t["side"] == "BUY" else 0 for t in rows]
    oc = [1 if t["outcome"] == "Down" or t["outcomeIndex"] == 1 else 0 for t in rows]
    fh.write(json.dumps({"cond": m["cond"], "t": ts, "p": px, "sz": sz, "sd": sd, "oc": oc}) + "\n")
    open(DONE, "a").write(m["cond"] + "\n")
    if i % 200 == 0:
        fh.flush()
        print(f"{i}/{len(SUB)}", flush=True)
    time.sleep(0.1)
fh.close()
print("ALLDONE", flush=True)
