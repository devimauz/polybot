"""_cal_exec.py - executability test for the sports 10-20c tilt.

Two hard questions the calibration table cannot answer:
 A) Staleness: are the 6h/24h bars live quotes or hours-old prints?
 B) Buyable? A calibration edge is only real if you could buy at (or below)
    that price. So: find actual TAKER BUY prints for the eventual winner at
    price in the target band, <= lead window before close, and measure
    realised edge from the PRINT (the price that really traded).

Also: a proper event-clustered bootstrap (not the slow O(n^2) version) and
the "one buy per event" de-correlation that a live bot would face.
"""
import json, os, sys, gzip, collections
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT, "_pm_out", "cal_worklist.json")))
by_cond = {m["cond"]: m for m in W}

raw = open(os.path.join(ROOT, "price_logs/cal/hist.jsonl.gz"), "rb").read()
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
print("markets with hist:", len(HIST), flush=True)

TAIL = os.path.join(ROOT, "price_logs/cal/tail.jsonl.gz")
TAILS = {}
if os.path.exists(TAIL):
    try:
        ttxt = gzip.decompress(open(TAIL, "rb").read()).decode("utf-8", "replace")
    except Exception:
        ttxt = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(open(TAIL, "rb").read()).decode("utf-8", "replace")
    for line in ttxt.splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        TAILS[r["cond"]] = r
print("markets with tail prints:", len(TAILS), flush=True)

THETA = {"nba": .05, "nfl": .05, "mlb": .05, "nhl": .05, "soccer": .05, "tennis": .05, "ufc": .05,
         "weather": .07, "finance": .04, "politics": .04}
SPORT = {"nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc"}

# ---------- A) staleness distribution of the 6h/24h bars ----------
print("\n=== A) how fresh is the bar we priced from? (sports) ===")
for L, lab in ((86400, "24h"), (21600, "6h")):
    ages = []
    for m in W:
        if m["tag"] not in SPORT:
            continue
        h = HIST.get(m["cond"])
        if h is None or len(h) < 5:
            continue
        t = m["closed"] - L
        sel = h[h[:, 0] <= t]
        if len(sel):
            ages.append(t - sel[-1, 0])
    a = np.array(ages)
    print(f" lead {lab:4s} n={len(a):5d} median_age={np.median(a)/60:7.1f}min "
          f"p25={np.percentile(a,25)/60:7.1f} p75={np.percentile(a,75)/60:8.1f} p90={np.percentile(a,90)/60:9.1f} "
          f"share<=30min={np.mean(a<=1800):.2f}")

# ---------- B) TAKER BUY prints as the executable price ----------
print("\n=== B) EXECUTABLE: taker BUY prints on the winning side, price band, <=6h before close ===")
CELLS = [("sports", SPORT, 0.10, 0.20, .05), ("sports", SPORT, 0.20, 0.35, .05),
         ("sports", SPORT, 0.40, 0.60, .05), ("sports", SPORT, 0.60, 0.80, .05),
         ("sports", SPORT, 0.80, 0.95, .05), ("weather", {"weather"}, 0.10, 0.35, .07)]
rng = np.random.default_rng(7)
for name, tags, lo, hi, th in CELLS:
    buys = []      # one per market: first taker BUY in band within 6h..30m of close
    for cond, tr in TAILS.items():
        m = by_cond.get(cond)
        if m is None or m["tag"] not in tags:
            continue
        t = np.array(tr["t"]); p = np.array(tr["p"]); sd = np.array(tr["sd"]); oc = np.array(tr["oc"])
        if len(t) == 0:
            continue
        # outcome-0 price; if the print was on the Down token, its outcomes[0]-price is 1-p
        po = np.where(oc == 0, p, 1 - p)
        # buy of outcomes[0]: taker BUY on token0 OR taker SELL on token1
        is_buy0 = np.where(oc == 0, sd == 1, sd == 0)
        dclo = m["closed"] - t
        msk = is_buy0 & (lo <= po) & (po < hi) & (dclo <= 21600) & (dclo >= 1800) & (t <= m["closed"])
        idx = np.where(msk)[0]
        if len(idx) == 0:
            continue
        j = idx[0]                       # earliest such print (closest to 6h before close)
        buys.append(dict(cond=cond, tag=m["tag"], px=po[j], win=float(m["win"]),
                         ev=(m["tag"], m["closed"] // 7200)))
    if len(buys) < 15:
        print(f"{name} [{lo:.2f},{hi:.2f})  n={len(buys)} (too few)")
        continue
    px = np.array([b["px"] for b in buys]); wr = np.array([b["win"] for b in buys])
    fee = th * px * (1 - px)
    net = wr - px - fee
    evs = list(dict.fromkeys(b["ev"] for b in buys))
    # event-clustered t on per-trade mean: SD of per-event means / sqrt(n_events)
    em = np.array([net[[i for i, b in enumerate(buys) if b["ev"] == u]].mean() for u in evs])
    se_mean = em.std(ddof=1) / np.sqrt(len(evs)) if len(evs) > 1 else np.nan
    tstat = (net.mean() / se_mean) if se_mean and se_mean > 0 else np.nan
    # de-correlated: keep ONE random buy per event
    per_ev = {}
    for i, b in enumerate(buys):
        per_ev.setdefault(b["ev"], []).append(i)
    solo = [rng.choice(v) for v in per_ev.values()]
    net_solo = net[solo]
    # bootstrap CI on per-share net (event-clustered) via precomputed group sums
    gsum, gcnt = [], []
    for u in evs:
        ii = [i for i, b in enumerate(buys) if b["ev"] == u]
        gsum.append(net[ii].sum()); gcnt.append(len(ii))
    gsum = np.array(gsum); gcnt = np.array(gcnt)
    boot = []
    for _ in range(4000):
        pick = rng.integers(0, len(evs), len(evs))
        boot.append(gsum[pick].sum() / gcnt[pick].sum())
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"{name}[{lo:.2f},{hi:.2f})  buys={len(buys):4d} evts={len(evs):4d} pay={px.mean():.3f} wr={wr.mean():.3f} "
          f"net={net.mean()*100:+5.2f}c t={tstat:+5.2f} "
          f"CI/trade=[{ci[0]*100:+.1f},{ci[1]*100:+.1f}]c | one-per-event n={len(solo)} net={net_solo.mean()*100:+5.2f}c "
          f"$@5stake=${np.sum([max(5,np.floor(5/x))*y for x,y in zip(px[solo],net_solo)]):+.2f}")

# ---------- C) is the tail of the tilt in cheap-side buys or fading favourites? ----------
print("\n=== C) mirror check: buy the COMPLEMENT of the 0.80-0.95 favourite band (i.e. fade fav) ===")
fade = []
for cond, tr in TAILS.items():
    m = by_cond.get(cond)
    if m is None or m["tag"] not in SPORT:
        continue
    t = np.array(tr["t"]); p = np.array(tr["p"]); sd = np.array(tr["sd"]); oc = np.array(tr["oc"])
    if len(t) == 0:
        continue
    po = np.where(oc == 0, p, 1 - p)
    is_buy0 = np.where(oc == 0, sd == 1, sd == 0)
    dclo = m["closed"] - t
    msk = is_buy0 & (po >= 0.80) & (po < 0.95) & (dclo <= 21600) & (dclo >= 1800)
    idx = np.where(msk)[0]
    if len(idx) == 0:
        continue
    j = idx[0]
    fade.append((po[j], 1 - float(m["win"])))
if len(fade) >= 15:
    f = np.array(fade)
    net = f[:, 1] - f[:, 0] - .05 * f[:, 0] * (1 - f[:, 0])
    print(f"fade-fav via complement buys: n={len(f)} pay={f[:,0].mean():.3f} realized={f[:,1].mean():.3f} net={net.mean()*100:+.2f}c")
else:
    print(f"fade-fav: n={len(fade)} too few to say")
