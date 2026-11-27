"""_cal_freeze.py - PRECOMMITTED design/holdout test of the fade-the-favourite family.

Discovered on the design slice (closed 2026-01-01 .. mid-Sept window, ~70% of the
chronology) from real taker prints, NOT from stale bars.

RULE (frozen before looking at any holdout):
  As a MAKER, rest a limit SELL of the favourite at 1 - p with p in [0.65, 0.93]
  (i.e. bid the dog at 0.07-0.35) at least 1 h before the market closes.
  Filled iff there is an aggressor BUY print at-or-through our price before close.
  Hold to settlement. Maker fee 0 (rebates ignored).
  One entry per market. Size: 5 shares minimum, floor($5/price) capped at 40 shares
  (thin sports books).

Measured in three blocks: design (first 70% chronology), holdout (last 30% untouched
by the cell choice), and an outcome-placebo p-value.

Also reports the taker version (just crossing) so the cost of the passive route is
visible, since no bid-size data exists in the print stream (fill is an upper bound).
"""
import json, os, sys, zlib, gzip, collections
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(ROOT, "_pm_out", "cal_worklist.json")))
by_cond = {m["cond"]: m for m in W}

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

PLO, PHI, LEAD = 0.65, 0.93, 3600
closes = sorted(m["closed"] for m in W if m["cond"] in TAILS)
cut = closes[int(len(closes) * 0.70)]
print(f"markets with prints: {len(TAILS)}  design cut {__import__('datetime').datetime.fromtimestamp(cut, __import__('datetime').timezone.utc)}")

def trades_for(cond):
    tr = TAILS[cond]
    m = by_cond[cond]
    t = np.array(tr["t"], float); p = np.array(tr["p"], float)
    sd = np.array(tr["sd"]); oc = np.array(tr["oc"])
    po = np.where(oc == 0, p, 1 - p)                      # outcomes[0] price
    buy0 = np.where(oc == 0, sd == 1, sd == 0)            # aggressor bought outcomes[0]
    sell0 = np.where(oc == 0, sd == 0, sd == 1)
    dl = m["closed"] - t
    o = np.argsort(t)
    return po[o], buy0[o], sell0[o], dl[o]

rows = []
for cond in TAILS:
    m = by_cond[cond]
    if m["tag"] not in ("nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc"):
        continue
    po, buy0, sell0, dl = trades_for(cond)
    if len(po) == 0:
        continue
    # candidate entry: last print with outcomes[0] in the favourite band, >=1h before close
    cand = np.where((po >= PLO) & (po < PHI) & (dl >= LEAD) & (dl <= 24 * 3600))[0]
    if len(cand) == 0:
        continue
    i = cand[-1]
    p_enter = po[i]
    win0 = float(m["win"])
    # --- MAKER SELL outcomes[0] at p_enter (equivalently buy outcomes[1] at 1-p):
    later = np.where((dl < dl[i]) & (dl > 0) & (np.arange(len(po)) > i))[0]
    fills_strict = [j for j in later if buy0[j] and po[j] > p_enter]   # definitely ours
    fills_incl = [j for j in later if buy0[j] and po[j] >= p_enter]    # upper bound
    # --- TAKER SELL: cross at p_enter now (pay taker fee, no queue risk)
    if not m["negrisk"]:
        rows.append(dict(cond=cond, tag=m["tag"], closed=m["closed"], p=p_enter, win=win0,
                         maker_strict=len(fills_strict) > 0,
                         maker_incl=len(fills_incl) > 0,
                         ms_t=dl[fills_strict[0]] if fills_strict else np.nan,
                         vol=m["vol"], ev=(m["tag"], m["closed"] // 7200)))

R = rows
if not R:
    sys.exit("no candidates")
R.sort(key=lambda r: r["closed"])
print("candidate favourite markets (print-priced, band, >=1h lead):", len(R))

def net_seller(p, win, shares, fee):
    """We SOLD outcomes[0] at p: receive p per share, pay 1 if it wins."""
    return shares * (p - win) - fee * p * (1 - p) * shares

def report(subset, label, mode):
    """mode: 'taker' | 'maker_strict' | 'maker_incl'"""
    if mode == "taker":
        rows2 = subset
        fee = 0.05
    else:
        rows2 = [r for r in subset if r[mode]]
        fee = 0.0
    if len(rows2) < 10:
        print(f"{label:24s} n={len(rows2):4d} (too few)")
        return None
    p = np.array([r["p"] for r in rows2]); win = np.array([r["win"] for r in rows2])
    sh = np.clip(np.floor(5 / p), 5, 40)
    pnl = net_seller(p, win, sh, fee)
    per_sh = pnl.sum() / sh.sum() * 100
    evs = list(dict.fromkeys(r["ev"] for r in rows2))
    idx_by_ev = [[i for i, r in enumerate(rows2) if r["ev"] == u] for u in evs]
    gsum = np.array([pnl[ix].sum() for ix in idx_by_ev])
    em = np.array([pnl[ix].mean() for ix in idx_by_ev])
    se = em.std(ddof=1) / np.sqrt(len(evs)) if len(evs) > 1 else np.nan
    tstat = per_sh / (se / np.clip(np.mean(sh), 1e-9, None)) if se == se and se > 0 else np.nan
    rng = np.random.default_rng(3)
    gsh = np.array([sh[ix].sum() for ix in idx_by_ev])
    boot = [(lambda k: gsum[k].sum() / gsh[k].sum())(rng.integers(0, len(evs), len(evs))) for _ in range(3000)]
    ci = np.percentile(boot, [2.5, 97.5]) * 100
    print(f"{label:24s} n={len(rows2):4d} evts={len(evs):4d} sold@{p.mean():.3f} fav_wr={win.mean():.3f} "
          f"edge={(p.mean()-win.mean())*100:+5.2f}c net/sh={per_sh:+5.2f}c tot=${pnl.sum():+8.2f} "
          f"t={tstat:+5.2f} CI/sh=[{ci[0]:+.1f},{ci[1]:+.1f}]c worst=${pnl.min():+.2f}")
    return pnl, rows2

design = [r for r in R if r["closed"] < cut]
holdout = [r for r in R if r["closed"] >= cut]
print("\n=== TAKER version (cross now, sell the favourite; pays taker fee) ===")
for lab, sub in (("design", design), ("holdout", holdout), ("ALL", R)):
    report(sub, "taker " + lab, "taker")
print("\n=== MAKER version, STRICT fills only (print strictly above our offer = certain ours) ===")
report(design, "maker design strict", "maker_strict")
report(holdout, "maker holdout strict", "maker_strict")
report(R, "maker ALL strict", "maker_strict")
print("\n=== MAKER version, INCLUSIVE upper bound (>= our price; queue-ahead ignored) ===")
report(design, "maker design incl", "maker_incl")
report(holdout, "maker holdout incl", "maker_incl")
print("\nfill rate strict design: %.2f  holdout: %.2f" % (
    np.mean([r["maker_strict"] for r in design]), np.mean([r["maker_strict"] for r in holdout])))

# outcome placebo on the strict-filled maker set
allr = [r for r in R if r["maker_strict"]]
p = np.array([r["p"] for r in allr]); win = np.array([r["win"] for r in allr])
sh = np.clip(np.floor(5 / p), 5, 40)
obs = net_seller(p, win, sh, 0.0).sum()
rng = np.random.default_rng(11)
wr_obs = win.mean()
pl = []
for _ in range(4000):
    w2 = (rng.random(len(win)) < wr_obs).astype(float)
    pl.append(net_seller(p, w2, sh, 0.0).sum())
pl = np.array(pl)
print(f"outcome-placebo: obs=${obs:+.2f} shuffled median=${np.median(pl):+.2f} p(>=obs)={np.mean(pl>=obs):.4f}")

# tag split + mean fill lead
print("\ntag split (maker strict):")
for tag in ("nba", "nfl", "mlb", "nhl", "soccer", "tennis", "ufc"):
    sub = [r for r in allr if r["tag"] == tag]
    if len(sub) >= 8:
        pp = np.array([r["p"] for r in sub]); ww = np.array([r["win"] for r in sub])
        ss = np.clip(np.floor(5 / pp), 5, 40)
        pnl = net_seller(pp, ww, ss, 0.0)
        mt = np.array([r["ms_t"] for r in sub if np.isfinite(r["ms_t"])])
        print(f"  {tag:7s} n={len(sub):4d} net=${pnl.sum():+7.2f} per_trade=${pnl.mean():+.3f} "
              f"median_fill_lead={np.median(mt)/3600 if len(mt) else float('nan'):.1f}h")
