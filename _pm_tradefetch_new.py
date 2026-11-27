"""_pm_tradefetch.py - fetch full taker trade prints for logged 5m windows (runs on AWS).

For every resolved window in manifest.json [(slug, cond)], pages
https://data-api.polymarket.com/trades?market=<cond>&takerOnly=true
and writes ~/poly/trades/<slug>.jsonl.gz with slim rows:
  {"ts":int,"side":"BUY|SELL","p":float,"sz":float,"oc":"Up|Down","tx":hash}
Idempotent: existing files are skipped so a crashed run resumes.
"""
import json, os, sys, time, gzip, urllib.request

MAN = json.load(open(os.path.expanduser("~/poly/tradefetch/manifest_new.json")))
OUT = os.path.expanduser("~/poly/trades")
os.makedirs(OUT, exist_ok=True)

BASE = "https://data-api.polymarket.com/trades"
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-fetch"}

def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:
            sys.stderr.write(f"retry {k} {url[:80]} {e}\n")
            time.sleep(2.0 * (k + 1))
    return None

done = fail = 0
for i, m in enumerate(MAN):
    slug, cond = m["slug"], m["cond"]
    fp = os.path.join(OUT, slug + ".jsonl.gz")
    if os.path.exists(fp):
        done += 1
        continue
    rows, off, cap, page = [], 0, 30000, []
    while off < cap:
        page = get(f"{BASE}?market={cond}&limit=1000&offset={off}&takerOnly=true")
        if page is None:
            break
        for t in page:
            rows.append({"ts": t["timestamp"], "side": t["side"], "p": t["price"],
                         "sz": t["size"], "oc": t["outcome"], "tx": t.get("transactionHash", "")})
        if len(page) < 1000:
            break
        off += 1000
    if page is None and off == 0:
        fail += 1
        print(f"{i} FAIL {slug}", flush=True)
        continue
    with gzip.open(fp, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    done += 1
    if i % 25 == 0:
        print(f"{i}/{len(MAN)} {slug} rows={len(rows)} done={done} fail={fail}", flush=True)
    time.sleep(0.2)
print(f"ALLDONE done={done} fail={fail}", flush=True)
