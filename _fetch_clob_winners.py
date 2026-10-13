"""Fetch authoritative CLOB winners for ALL btc windows in windows.jsonl.
Writes slug -> {winner, status} to clob_winners.json."""
import json, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

ROOT = "C:/Users/user/Desktop/bot"
PATH = ROOT + "/price_logs/twap/windows.jsonl"
OUT = ROOT + "/price_logs/twap/clob_winners.json"

cids = {}
with open(PATH, encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("asset") == "btc" and d.get("cid"):
            cids[d["window"]] = d["cid"]

print("btc cids to fetch:", len(cids), flush=True)
client = httpx.Client(timeout=20)

def fetch(slug):
    cid = cids[slug]
    for attempt in range(3):
        try:
            r = client.get(f"https://clob.polymarket.com/markets/{cid}", timeout=20)
            m = r.json()
            tokens = m.get("tokens") or []
            if not tokens:
                return slug, None, "no-tokens"
            w = [t.get("outcome") for t in tokens if t.get("winner")]
            return slug, (w[0] if w else None), "ok"
        except Exception as e:
            if attempt == 2:
                return slug, None, f"{type(e).__name__}:{str(e)[:40]}"
            time.sleep(1 + attempt)

results = {}
done = 0
with ThreadPoolExecutor(max_workers=15) as ex:
    futs = {ex.submit(fetch, s): s for s in cids}
    for f in as_completed(futs):
        slug, winner, status = f.result()
        results[slug] = {"winner": winner, "status": status}
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(cids)}", flush=True)

json.dump(results, open(OUT, "w"))
ok = sum(1 for v in results.values() if v["winner"])
print(f"DONE: resolved={ok}/{len(results)} -> {OUT}", flush=True)
