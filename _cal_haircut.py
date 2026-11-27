"""_cal_haircut.py - is the fade-the-favourite edge real for a TAKER working
the bid (print price minus spread haircut), and its dog-mirror?

Uses only the 5m-BTC-style discipline: entry price = actual print adjusted by a
haircut sweep; fee 0.05; event-clustered t + bootstrap CI per share.
"""
import json, os, sys, zlib, gzip
import numpy as np

ROOT = r"C:\Users\user\Desktop\bot"
sys.path.insert(0, ROOT)

def load(fp):
    raw = open(fp, "rb").read()
    try:
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode("utf-8", "replace")
    except zlib.error:
        return gzip.decompress(raw).decode("utf-8", "replace")

TAILS = {}
for line in load(os.path.join(ROOT, "price_logs/cal/tail.jsonl.gz")).splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("t"):
        TAILS[r["cond"]] = r
W = json.load(open(os.path.join(ROOT, "_pm_out/cal_worklist.json")))
by = {m["cond"]: m for m in W}
SPORT = ("nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc")

def mk(cond):
    tr = TAILS[cond]
    m = by[cond]
    t = np.array(tr["t"], float)
    o = np.argsort(t)
    t = t[o]
    p = np.array(tr["p"], float)[o]
    sd = np.array(tr["sd"])[o]
    oc = np.array(tr["oc"])[o]
    po = np.where(oc == 0, p, 1 - p)
    buy0 = np.where(oc == 0, sd == 1, sd == 0)
    dl = m["closed"] - t
    return po, buy0, dl, m

def build(lo, hi, lead):
    out = []
    for cond in TAILS:
        m = by.get(cond)
        if not m or m["tag"] not in SPORT or m["negrisk"]:
            continue
        po, buy0, dl, m = mk(cond)
        idx = np.where((po >= lo) & (po < hi) & buy0 & (dl <= lead * 3600) & (dl > 0))[0]
        for j in idx:
            out.append(dict(cond=cond, tag=m["tag"], closed=m["closed"],
                            p=float(po[j]), win=float(m["win"])))
    return out

def cell(rws, label, fee=0.05, side="sell"):
    if len(rws) < 25:
        print(f"{label:46s} n={len(rws):5d} (few)")
        return
    p = np.array([r["p"] for r in rws]); w = np.array([r["win"] for r in rws])
    sh = np.clip(np.floor(5 / p), 5, 40)
    sign = 1.0 if side == "buy" else -1.0
    pnl = sign * (w - p) * sh - fee * p * (1 - p) * sh
    evs = list(dict.fromkeys((r["tag"], r["closed"] // 7200) for r in rws))
    ii = [[k for k, r in enumerate(rws) if (r["tag"], r["closed"] // 7200) == u] for u in evs]
    gsum = np.array([pnl[x].sum() for x in ii]); gsh = np.array([sh[x].sum() for x in ii])
    gm = np.array([pnl[x].mean() for x in ii])
    tstat = gm.mean() / (gm.std(ddof=1) / np.sqrt(len(evs))) if len(evs) > 1 else np.nan
    rng = np.random.default_rng(9)
    boot = [(lambda k: gsum[k].sum() / gsh[k].sum())(rng.integers(0, len(evs), len(evs))) for _ in range(3000)]
    ci = np.percentile(boot, [2.5, 97.5]) * 100
    halves = sorted(set(r["closed"] for r in rws))
    midt = halves[len(halves) // 2]
    a = [i for i, r in enumerate(rws) if r["closed"] < midt]
    b = [i for i, r in enumerate(rws) if r["closed"] >= midt]
    ha = pnl[a].sum() / max(sh[a].sum(), 1) * 100 if a else float("nan")
    hb = pnl[b].sum() / max(sh[b].sum(), 1) * 100 if b else float("nan")
    print(f"{label:46s} n={len(rws):5d} evts={len(evs):4d} pay={p.mean():.3f} wr={w.mean():.3f} "
          f"net/sh={pnl.sum()/sh.sum()*100:+5.2f}c t={tstat:+5.2f} CI=[{ci[0]:+.1f},{ci[1]:+.1f}]c "
          f"halves={ha:+.1f}/{hb:+.1f}c tot=${pnl.sum():+8.0f}")

print("=== SELL the favourite after a BUY print at [lo,hi) within lead of close ===")
print("=== you would hit the BID: print minus haircut sweep. fee 0.05          ===")
for lo, hi, lead in ((0.80, 0.93, 6), (0.80, 0.93, 1), (0.65, 0.80, 1), (0.93, 0.99, 1)):
    rws = build(lo, hi, lead)
    if not rws:
        continue
    cell(rws, f"fav[{lo:.2f},{hi:.2f}) {lead}h haircut 0.0c")
    for cut in (0.01, 0.02, 0.03):
        cell([dict(r, p=r["p"] - cut) for r in rws], f"   haircut {cut*100:.0f}c")

print("\n=== BUY the dog when favourite SELLERS print (mirror flow) ===")
for lo, hi, lead in ((0.80, 0.93, 6), (0.80, 0.93, 1), (0.65, 0.80, 1), (0.93, 0.99, 1)):
    rws = []
    for cond in TAILS:
        m = by.get(cond)
        if not m or m["tag"] not in SPORT or m["negrisk"]:
            continue
        po, buy0, dl, m = mk(cond)
        sell0 = ~buy0
        idx = np.where((po >= lo) & (po < hi) & sell0 & (dl <= lead * 3600) & (dl > 0))[0]
        if len(idx) == 0:
            continue
        j = idx[-1]
        rws.append(dict(cond=cond, tag=m["tag"], closed=m["closed"],
                        p=float(1 - po[j]), win=1.0 - float(m["win"])))
    cell(rws, f"buy dog mirror fav[{lo:.2f},{hi:.2f}) {lead}h", side="buy")
    for cut in (0.01, 0.02, 0.03):
        cell([dict(r, p=r["p"] + cut) for r in rws], f"   +{cut*100:.0f}c ask haircut", side="buy")
