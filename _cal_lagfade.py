"""_cal_lagfade.py - three decisive print-level studies.

A) SETTLEMENT LAG: for each market with tail prints, min price printed on the
   EVENTUAL WINNER within last 30m / 2h before close, and the same within
   [1h,24h] window. If people sell the winner at 0.97 late, buying it is a
   near-free return. Print-based => no stale-bar artifact.

B) THE FADE, conditioned properly:
   B1) per-print passive-seller edge on favourite buys (mirror of _cal_exec B):
       every BUY print of outcomes[0] at p in band & dl<=6h: seller earned
       p - win. Clustered CI. This is what an ask-resting bot earns PER FILL.
   B2) same but restricted to prints where price later RISES above the print
       (the maker's adverse sub-population) vs prints that are the LAST touch
       of that price (nobody crossed after) => bounds for queue share.
   C) direction drift: after an in-band favourite print at dl in [60m, 6h],
       where is the NEXT print (up/down)? Fade is real only if favourites
       that print in-band tend to NOT keep rising (i.e. prints are not the
       front edge of a run).

All at market level, event-clustered (tag x 2h close bucket).
"""
import json, os, sys, zlib, gzip, collections
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT, "_pm_out", "cal_worklist.json")))
by_cond = {m["cond"]: m for m in W}
SPORT = ("nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc")

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

def arr(cond):
    tr = TAILS[cond]
    m = by_cond[cond]
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

# ---------- A) settlement lag ----------
print("=== A) settlement-lag, honest metric: LAST print before close on the winner ===")
for win_lag, lab in ((600, "last touch <=10m"), (1800, "last touch <=30m")):
    for grp, tags in (("sports", SPORT), ("weather", ("weather",))):
        vals = []
        for cond in TAILS:
            po, buy0, dl, m = arr(cond)
            if m["tag"] not in tags or (grp == "sports" and m["negrisk"]):
                continue
            sel = (dl <= win_lag) & (dl > 0)
            pw = np.where(m["win"] == 1, po, 1 - po)[sel]
            if len(pw):
                vals.append(pw[-1])           # the price the winner last traded at
        if not vals:
            continue
        v = np.array(vals)
        th = 0.05 if grp == "sports" else 0.07
        disc = 1 - v
        fee = th * v * (1 - v)
        print(f"{lab:20s} {grp:7s} n={len(v):5d} median_last_price={np.median(v):.4f} "
              f"mean_disc={disc.mean()*100:.2f}c mean_net_disc={(disc-fee).mean()*100:+.2f}c "
              f"share<=0.99={np.mean(v<=0.99)*100:.1f}% share<=0.98={np.mean(v<=0.98)*100:.1f}%")

# ---------- B1) per-print passive favourite seller ----------
print("\n=== B) per-print PASSIVE favourite-seller edge (you quote the ask they lift) ===")
rng = np.random.default_rng(5)
for lo, hi in ((0.65, 0.80), (0.80, 0.93), (0.93, 0.99)):
    for lead, lend in ((3600, 86400), (1800, 21600)):
        buys = []
        for cond in TAILS:
            po, buy0, dl, m = arr(cond)
            if m["tag"] not in SPORT or m["negrisk"]:
                continue
            msk = buy0 & (po >= lo) & (po < hi) & (dl >= lead) & (dl <= lend) & (dl > 0)
            for j in np.where(msk)[0]:
                buys.append((dl[j], po[j], float(m["win"]), (m["tag"], m["closed"] // 7200), cond, j, po.copy(), buy0.copy(), dl.copy()))
        if len(buys) < 30:
            print(f"fav[{lo:.2f},{hi:.2f}) lead {lead//60:4d}-{lend//60:4d}m prints={len(buys)} (few)")
            continue
        p = np.array([b[1] for b in buys]); win = np.array([b[2] for b in buys])
        gross = p - win
        # adverse bound: fills where price later trades STRICTLY above this print (certain ours)
        adv = []
        for (dlj, pj, wj, ev, cond, j, poA, buyA, dlA) in buys:
            later = (dlA < dlj) & (dlA > 0)
            above = later & (poA > pj) & buyA
            adv.append(above.any())
        adv = np.array(adv)
        sh = np.clip(np.floor(5 / p), 5, 40)
        pnl = gross * sh
        evs = list(dict.fromkeys(b[3] for b in buys))
        idx = {u: [i for i, b in enumerate(buys) if b[3] == u] for u in evs}
        em = np.array([pnl[idx[u]].mean() for u in evs])
        se = em.std(ddof=1) / np.sqrt(len(evs))
        gsum = np.array([pnl[idx[u]].sum() for u in evs])
        gsh = np.array([sh[idx[u]].sum() for u in evs])
        boot = [(lambda k: gsum[k].sum() / gsh[k].sum())(rng.integers(0, len(evs), len(evs))) for _ in range(2000)]
        ci = np.percentile(boot, [2.5, 97.5]) * 100
        advp = adv.mean()
        adv_edge = (p - win)[adv].mean() * 100 if adv.any() else float('nan')
        nonadv_edge = (p - win)[~adv].mean() * 100 if (~adv).any() else float('nan')
        tot = 2 * len(p) * 5
        print(f"fav[{lo:.2f},{hi:.2f}) {lead//60}~{lend//60}m prints={len(p):5d} evts={len(evs):4d} "
              f"pay={p.mean():.3f} wr={win.mean():.3f} seller_edge={ (p-win).mean()*100:+6.2f}c/sh "
              f"CI=[{ci[0]:+.1f},{ci[1]:+.1f}]c tot_if_all_mine=${(p-win).sum()*5:+8.0f} "
              f"| certain_fill_share={advp:.2f} edge_when_swept={adv_edge:+6.2f}c edge_untouched={nonadv_edge:+6.2f}c")

# ---------- C) drift after in-band favourite prints ----------
print("\n=== C) after a favourite print in [0.65,0.93) at 1-6h before close: where does it go? ===")
nxt = {"up": 0, "down": 0, "flat": 0, "none": 0}
fin_wr = []
for cond in TAILS:
    po, buy0, dl, m = arr(cond)
    if m["tag"] not in SPORT or m["negrisk"]:
        continue
    idx = np.where((po >= 0.65) & (po < 0.93) & (dl >= 3600) & (dl <= 21600) & (dl > 0))[0]
    if len(idx) == 0:
        continue
    j = idx[-1]
    fin_wr.append(float(m["win"]))
    after = np.where((dl < dl[j]) & (dl > 0))[0]
    if len(after) == 0:
        nxt["none"] += 1
        continue
    d = po[after[0]] - po[j]
    nxt["up" if d > 1e-9 else "down" if d < -1e-9 else "flat"] += 1
f = np.array(fin_wr)
n = sum(nxt.values())
print(f"last in-band print: n={n} fav final wr={f.mean():.3f} | next print: up={nxt['up']/n:.2f} "
      f"flat={nxt['flat']/n:.2f} down={nxt['down']/n:.2f} noback={nxt['none']/n:.2f}")
print(f"if you FADE that print taker-style at the print price: edge={ (1-f.mean()) - f.mean()*0 if False else 0}")
