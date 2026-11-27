"""_cal_report2.py - diagnostics for the sports tilt found in _cal_report.

1. Event clustering: group markets by (tag, close-hour bucket) ~ same game;
   cluster-t and window... event-cluster bootstrap CI on key cells.
2. Chronological halves per cell (design = earlier 5 weeks, holdout = later 5).
3. Freshness gate: bars within 1h of lead point instead of 6h.
4. Per-tag monotonicity of edge vs price decile (is the tilt everywhere?).
5. What ARE the 15c buckets: top question patterns + negRisk share.
6. Dollars: $5-stake net per market with shares>=5 (p>=0.10 ok; p=0.15 -> 33sh).
"""
import json, os, sys, gzip, collections
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT, "_pm_out", "cal_worklist.json")))
HF = os.path.join(ROOT, "price_logs", "cal", "hist.jsonl.gz")
raw = open(HF, "rb").read()
import zlib
try:
    txt = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
except zlib.error:
    txt = gzip.decompress(raw)
HIST = {}
for line in txt.decode("utf-8", "replace").splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("h"):
        HIST[r["cond"]] = np.array([(x["t"], x["p"]) for x in r["h"]], float)

THETA = {"nba": .05, "nfl": .05, "mlb": .05, "nhl": .05, "soccer": .05, "tennis": .05, "ufc": .05,
         "weather": .07, "finance": .04, "politics": .04}
SPORT = {"nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc"}

def price_at(m, L, max_age=6 * 3600):
    h = HIST.get(m["cond"])
    if h is None or len(h) < 5:
        return np.nan
    t = m["closed"] - L
    sel = h[h[:, 0] <= t]
    if len(sel) == 0 or (t - sel[-1, 0]) > max_age:
        return np.nan
    return sel[-1, 1]

rows = []
for m in W:
    for L, tagl in ((86400, "24h"), (21600, "6h"), (3600, "1h")):
        p = price_at(m, L)
        if not np.isfinite(p):
            continue
        evk = (m["tag"], m["closed"] // (2 * 3600))  # 2h bucket ~ one game session
        rows.append(dict(cond=m["cond"], tag=m["tag"], grp="sports" if m["tag"] in SPORT else m["tag"],
                         L=L, lab=tagl, p=p, win=m["win"], closed=m["closed"], ev=evk,
                         q=m["q"], nr=m["negrisk"], vol=m["vol"]))
R = rows
print("rows:", len(R), flush=True)

def cluster_stats(sub):
    p = np.array([x["p"] for x in sub]); wr = np.array([x["win"] for x in sub], float)
    th = np.array([THETA[x["tag"]] for x in sub])
    fee = th * p * (1 - p)
    pnl = wr - p - fee                       # per-share net $ at 1 share
    ev = [x["ev"] for x in sub]
    uq = list(dict.fromkeys(ev))
    s = np.array([pnl[[i for i, e in enumerate(ev) if e == u]].sum() for u in uq])
    c = np.array([sum(1 for e in ev if e == u) for u in uq])
    mean = pnl.mean()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return len(sub), len(uq), mean * 100, (mean / se) if se > 0 else np.nan

def halves(sub):
    h = sub[0]["closed"] + (sub[-1]["closed"] - sub[0]["closed"]) / 2 if sub else 0
    h = sorted(x["closed"] for x in sub)
    mid = h[len(h) // 2]
    a = [x for x in sub if x["closed"] < mid]; b = [x for x in sub if x["closed"] >= mid]
    def m(c):
        if not c: return float("nan")
        return np.mean([x["win"] - x["p"] - THETA[x["tag"]] * x["p"] * (1 - x["p"]) for x in c]) * 100
    return m(a), m(b)

print("\n=== KEY CELLS: net edge c/share, event-clustered t (2h buckets) ===")
print("cell                        n  evts  mean_net_c  t    | design/holdout c")
for grp, lo, hi in [("sports", 0.10, 0.20), ("sports", 0.15, 0.25), ("sports", 0.05, 0.15),
                    ("sports", 0.90, 1.001), ("sports", 0.80, 0.90), ("sports", 0.20, 0.30),
                    ("sports", 0.30, 0.50), ("weather", 0.0, 0.10), ("weather", 0.20, 0.45),
                    ("finance", 0.20, 0.35), ("politics", 0.05, 0.15)]:
    for lab in ("24h", "6h", "1h"):
        sub = [x for x in R if x["grp"] == grp and x["lab"] == lab and lo <= x["p"] < hi]
        if len(sub) < 25:
            continue
        n, e, mn, t = cluster_stats(sub)
        d1, d2 = halves(sub)
        print(f"{grp[:5]}[{lo:.2f},{hi:.2f})@{lab:3s} {n:5d} {e:5d}   {mn:+6.2f}  {t:+5.2f} | {d1:+6.2f}/{d2:+6.2f}")

print("\n=== freshness gate: bars within 1h (sports 0.10-0.20) ===")
sub1 = []
for m in W:
    if m["tag"] not in SPORT: continue
    for L, lab in ((86400, "24h"), (21600, "6h")):
        p = price_at(m, L, max_age=3600)
        if np.isfinite(p) and 0.10 <= p < 0.20:
            sub1.append(dict(cond=m["cond"], tag=m["tag"], grp="sports", L=L, lab=lab, p=p,
                             win=m["win"], closed=m["closed"], ev=(m["tag"], m["closed"] // 7200),
                             q=m["q"], nr=m["negrisk"], vol=m["vol"]))
if sub1:
    n, e, mn, t = cluster_stats(sub1)
    print(f"sports[0.10,0.20) fresh1h  n={n} evts={e} net={mn:+.2f}c t={t:+.2f}")

print("\n=== per-tag monotonicity: net edge by price decile @6h ===")
for tag in sorted(SPORT):
    line = f"{tag:7s}"
    for lo in [i / 10 for i in range(10)]:
        sub = [x for x in R if x["tag"] == tag and x["lab"] == "6h" and lo <= x["p"] < lo + .1]
        if len(sub) >= 25:
            p = np.mean([x["p"] for x in sub]); wr = np.mean([x["win"] for x in sub])
            line += f" {100 * (wr - p - .05 * p * (1 - p)):+5.1f}"
        else:
            line += "      -"
    print(line)

print("\n=== what is the sports 10-20c bucket @6h? ===")
sub = [x for x in R if x["grp"] == "sports" and x["lab"] == "6h" and 0.10 <= x["p"] < 0.20]
pat = collections.Counter()
for x in sub:
    q = x["q"].lower()
    for k, v in {"spread:": "spread", "o/u": "ou", "vs.": "moneyline", "will ": "will?"}.items():
        if k in q:
            pat[v] += 1; break
    else:
        if ":" in q:
            pat["colon-prop"] += 1
        else:
            pat["other"] += 1
print("patterns:", dict(pat), "| negRisk share:", np.mean([x["nr"] for x in sub]).round(3))
for x in sub[:12]:
    print("   p=%.3f win=%d  %s" % (x["p"], x["win"], x["q"][:60]))

print("\n=== dollar view: $5-ish stake, shares=max(5,floor(5/p)), sports 10-20c @6h ===")
tot = 0.0; ntt = 0
for x in sub:
    sh = max(5.0, np.floor(5 / x["p"]))
    pnl = sh * x["win"] - sh * x["p"] - THETA["soccer"] * x["p"] * (1 - x["p"]) * sh
    tot += pnl; ntt += 1
print(f"n={ntt} total=${tot:+.2f}  per-market=${tot/max(ntt,1):+.3f}")
