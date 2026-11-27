import json, urllib.request
tok = "16040015440196279900485035793550429453516625694844857319147506590755961451627"
u = "https://clob.polymarket.com/prices-history?market=%s&interval=max&fidelity=60" % tok
r = json.load(urllib.request.urlopen(u, timeout=20))
h = r.get("history", [])
print("points:", len(h))
if h:
    print("first:", h[0])
    print("last:", h[-1])
