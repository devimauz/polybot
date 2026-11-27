"""_cal_list.py - list CLOSED resolved binary markets per category via gamma-api.

Writes _pm_out/cal_markets.jsonl: one row per market with what the history
fetcher + calibration analysis need:
  cond, tok_yes, vol, liq, spread, created, end, closed, outcome winner, question, negRisk, tag
Keeps only markets that resolved cleanly (price exactly 1/0) with volume >= MINVOL.
"""
import json, time, urllib.request

def g(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=25))

TAGS = {"sports": 1, "politics": 2, "finance": 120, "economics": 225, "weather": 84, "tech": 1401}
MINVOL = 2000.0
out = {}
for tag, tid in TAGS.items():
    n_ok = n_sk = 0
    for page in range(50):
        try:
            r = g(f"https://gamma-api.polymarket.com/markets?closed=true&tag_id={tid}"
                  f"&limit=500&offset={page*500}&order=closedTime&ascending=false")
        except Exception as e:
            print(tag, "page", page, "ERR", repr(e)[:80]); break
        if not r:
            break
        stop = False
        for m in r:
            ct = m.get("closedTime") or ""
            if ct and ct < "2026-06-01":
                stop = True
            try:
                vol = float(m.get("volumeNum") or 0)
                op = json.loads(m.get("outcomePrices") or "[]")
                oc = json.loads(m.get("outcomes") or "[]")
                toks = m.get("clobTokenIds")
                tok = json.loads(toks)[0] if toks else None
            except Exception:
                continue
            if vol < MINVOL or not tok or len(op) != 2:
                n_sk += 1; continue
            p0, p1 = float(op[0]), float(op[1])
            if not ((p0 == 1 and p1 == 0) or (p0 == 0 and p1 == 1)):
                n_sk += 1; continue
            out[m["conditionId"]] = {
                "cond": m["conditionId"], "tok": tok, "tag": tag, "q": m.get("question", "")[:90],
                "win": oc[0] if p0 == 1 else oc[1], "vol": vol,
                "liq": float(m.get("liquidityNum") or 0),
                "spread": m.get("spread"), "negRisk": m.get("negRisk"),
                "created": m.get("startDate") or "", "end": m.get("endDate") or "",
                "closed": ct,
            }
            n_ok += 1
        if stop or len(r) < 500:
            break
        time.sleep(0.2)
    print(f"{tag:10s} kept={n_ok} skipped={n_sk} unique total={len(out)}", flush=True)
    time.sleep(0.3)

with open("_pm_out/cal_markets.jsonl", "w", encoding="utf-8") as fh:
    for v in out.values():
        fh.write(json.dumps(v) + "\n")
print("TOTAL unique markets:", len(out))
