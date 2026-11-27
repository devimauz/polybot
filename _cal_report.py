"""_cal_report.py - calibration + settlement-lag analysis on closed markets.

Inputs: _pm_out/cal_worklist.json (markets, tag, win, closed ts)
        price_logs/cal/hist.jsonl.gz  rows {cond, h:[{t,p},...]}  (fetched on AWS)

1) CALIBRATION: for lead times L in {24h, 6h, 1h, 15m, 5m}: winner-side token
   price p at the last point <= closed-L; bucket by p decile; realised wr of
   outcomes[0]; edge_yes = wr - p, edge_no = p - (1-wr) - ... computed per side.
   Per category, chronological half-slices (design = earlier 50% of closedTime,
   holdout = later 50%). Fee model: taker fee theta*p*(1-p) with theta per
   category (sports .05, finance .04, politics .04, weather .07->also shown at 0).
2) SETTLEMENT LAG: distribution of the FINAL pre-close price p_last for the
   winning side: if you buy the winner at p_last you earn (1-p_last) in <1 day.
   Histogram of 1-p_last by tag, and share of markets still <=0.985 / <=0.995
   one hour before close.

Prints tables to _pm_out/cal_report.txt
"""
import json, os, sys, gzip, collections
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT, "_pm_out", "cal_worklist.json")))
HF = os.path.join(ROOT, "price_logs", "cal", "hist.jsonl.gz")
if not os.path.exists(HF):
    sys.exit("no hist file: " + HF)

HIST = {}
import zlib
raw = open(HF, "rb").read()
try:
    txt = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
except zlib.error:
    txt = gzip.decompress(raw)  # complete file
for line in txt.decode("utf-8", "replace").splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("h"):
        HIST[r["cond"]] = np.array([(x["t"], x["p"]) for x in r["h"]], float)

THETA = {"nba": .05, "nfl": .05, "mlb": .05, "nhl": .05, "soccer": .05, "tennis": .05, "ufc": .05,
         "weather": .07, "finance": .04, "politics": .04}
LEADS = [86400, 21600, 3600, 900, 300]
GROUPS = {"nba": "sports", "nfl": "sports", "mlb": "sports", "nhl": "sports", "soccer": "sports",
          "tennis": "sports", "ufc": "sports", "weather": "weather", "finance": "finance", "politics": "politics"}

mk = []
for m in W:
    h = HIST.get(m["cond"])
    if h is None or len(h) < 5:
        continue
    row = dict(m)
    ok = True
    for L in LEADS:
        t = m["closed"] - L
        sel = h[h[:, 0] <= t]
        if len(sel) == 0 or (t - sel[-1, 0]) > 6 * 3600:
            row["p_" + str(L)] = np.nan
        else:
            row["p_" + str(L)] = sel[-1, 1]
    sel = h[h[:, 0] <= m["closed"]]
    row["p_last"] = sel[-1, 1] if len(sel) else np.nan
    mk.append(row)
print("markets with history:", len(mk), "of", len(W), flush=True)

closes = sorted(x["closed"] for x in mk)
mid = closes[len(closes) // 2]
for x in mk:
    x["half"] = "design" if x["closed"] < mid else "holdout"

def table(L, only_group=None):
    key = "p_" + str(L)
    rows = [x for x in mk if np.isfinite(x[key])]
    if only_group:
        rows = [x for x in rows if GROUPS[x["tag"]] == only_group]
    if not rows:
        return
    print(f"\n=== lead {L//3600 if L >= 3600 else L//60}h{'m' if L<3600 else ''} :: n={len(rows)} ===")
    print(" bucket      n   mean_p  real_wr  edge_yes_c  net_yes_c  |  mirror edge_no_c")
    for lo, hi in [(i/10, (i+1)/10) for i in range(10)]:
        b = [x for x in rows if lo <= x[key] < hi + (1e-9 if hi == 1.0 else 0) and x[key] >= lo]
        if hi == 1.0:
            b = [x for x in rows if lo <= x[key] <= 1.0]
        if len(b) < 20:
            continue
        p = np.mean([x[key] for x in b]); wr = np.mean([x["win"] for x in b])
        th = np.mean([THETA[x["tag"]] for x in b])
        fee = th * p * (1 - p)
        edge = (wr - p) * 100
        net = edge - fee * 100
        edge_no = ((1 - wr) - (1 - p)) * 100
        print(f" [{lo:.1f},{hi:.1f}) {len(b):5d}  {p:.4f}   {wr:.4f}   {edge:+6.2f}     {net:+6.2f}    | {edge_no:+6.2f}")

for grp in ("sports", "weather", "finance", "politics"):
    print("\n############ GROUP:", grp, "############")
    for L in LEADS:
        table(L, grp)

print("\n\n############ design vs holdout (pooled sports, lead 1h and 15m) ############")
for L in (3600, 900):
    for half in ("design", "holdout"):
        key = "p_" + str(L)
        rows = [x for x in mk if GROUPS[x["tag"]] == "sports" and np.isfinite(x[key]) and x["half"] == half]
        if not rows:
            continue
        lo_b = [x for x in rows if x[key] <= 0.25]
        hi_b = [x for x in rows if x[key] >= 0.75]
        def q(b):
            if len(b) < 15: return "n<15"
            p = np.mean([x[key] for x in b]); wr = np.mean([x["win"] for x in b])
            return f"n={len(b)} p={p:.3f} wr={wr:.3f} edge={(wr-p)*100:+.2f}c"
        print(f" lead {L}s {half:6s} cheap(p<=.25): {q(lo_b)} | fav(p>=.75): {q(hi_b)}")

print("\n############ SETTLEMENT LAG (final pre-close price of the WINNING side) ############")
by_tag = collections.defaultdict(list)
for x in mk:
    if np.isfinite(x.get("p_last", np.nan)):
        pwin = x["p_last"] if x["win"] == 1 else 1 - x["p_last"]
        by_tag[x["tag"]].append(pwin)
for tag, v in sorted(by_tag.items()):
    v = np.array(v)
    th = THETA[tag]
    med = np.median(v)
    # buy winner at p_last: profit 1-p per share; taker fee th*p(1-p); net yield per $ staked:
    p = max(min(np.median(v), .999), .001)
    net_yield = (1 - p - th * p * (1 - p)) / p * 100
    print(f"{tag:9s} n={len(v):5d} median p_win={med:.4f} p90={np.percentile(v,90):.4f} "
          f"share<=0.985: {np.mean(v<=0.985)*100:4.1f}%  share<=0.95: {np.mean(v<=0.95)*100:4.1f}%  net-yield@median: {net_yield:+.3f}%")
