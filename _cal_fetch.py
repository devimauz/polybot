"""_cal_fetch.py - fetch 36h-pre-close price history per closed market (runs on AWS).

Reads ~/poly/cal/worklist.json, calls
  clob.polymarket.com/prices-history?market=<tok>&startTs=closed-36h&endTs=closed+2h&fidelity=5
appends one JSONL row per market to ~/poly/cal/hist.jsonl.gz (idempotent resume
via processed-conds file), 0.12s pause between calls.
"""
import json, os, time, urllib.request, gzip, sys

HOME = os.path.expanduser("~/poly/cal")
os.makedirs(HOME, exist_ok=True)
WL = json.load(open(os.path.join(HOME, "worklist.json")))
OUT = os.path.join(HOME, "hist.jsonl.gz")
DONE = os.path.join(HOME, "done.txt")

done = set()
if os.path.exists(DONE):
    done = set(open(DONE).read().split())

def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None

fh = gzip.open(OUT, "at", encoding="utf-8")
t0 = time.time()
for i, m in enumerate(WL):
    if m["cond"] in done:
        continue
    url = (f"https://clob.polymarket.com/prices-history?market={m['tok']}"
           f"&startTs={m['closed']-129600}&endTs={m['closed']+7200}&fidelity=5")
    d = get(url)
    row = dict(cond=m["cond"], h=(d or {}).get("history", []) if d else None)
    fh.write(json.dumps(row) + "\n")
    with open(DONE, "a") as db:
        db.write(m["cond"] + "\n")
    if i % 100 == 0:
        fh.flush()
    if i % 200 == 0:
        rate = (i + 1) / max(1e-9, time.time() - t0)
        print(f"{i}/{len(WL)} {rate:.1f}/s eta_min={(len(WL)-i)/max(rate,0.1)/60:.0f}", flush=True)
    time.sleep(0.12)
fh.close()
print("ALLDONE", len(WL), flush=True)
