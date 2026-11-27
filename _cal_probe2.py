import json, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 research"}
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25))
tok = "16040015440196279900485035793550429453516625694844857319147506590755961451627"
for q in ["interval=max&fidelity=60", "interval=1m&fidelity=10", "interval=1w&fidelity=10"]:
    u = "https://clob.polymarket.com/prices-history?market=%s&%s" % (tok, q)
    try:
        h = get(u).get("history", [])
        print(q, "points:", len(h), "first:", h[0] if h else None, "last:", h[-1] if h else None)
    except Exception as e:
        print(q, "ERR", repr(e)[:100])
