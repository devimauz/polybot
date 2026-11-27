#!/usr/bin/env python3
"""_pm_hunt_hour.py — validate the UTC-hour-of-day lead on the 5m design slice.

Per the skill: a 24-bucket scan is a hypothesis, not a belief. We report
cluster-t (window-clustered), both-halves, and the price-implied breakeven,
and we split the design slice into early/late to see which hours hold.
"""
import os, pickle, datetime, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def one_trade(d, T=180, band=(0.10, 0.90)):
    t = d["t"]
    i0 = int(np.searchsorted(t, T, side="right")) - 1
    if i0 < 0: return None
    for i in range(i0, len(t)):
        for side in ("Up", "Down"):
            ask = d[f"{side}_ask"][i]; asz = d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0] <= ask <= band[1]): continue
            sh = int(STAKE/ask)
            if sh < 5 or not np.isfinite(asz) or asz < sh: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            return dict(ws=d["ws"], side=side, ask=ask, won=won, pnl=pnl, sh=sh)
    return None

def cluster_t(rows):
    # window-clustered: several rows per window possible; here 1/window
    pnl = np.array([r["pnl"] for r in rows])
    ws = np.array([r["ws"] for r in rows])
    uq = np.unique(ws)
    s = np.array([pnl[ws==w].sum() for w in uq])
    if len(s) < 2: return float("nan")
    m = s.mean(); se = s.std(ddof=1)/math.sqrt(len(s))
    return m/se if se>0 else float("nan")

def report(tag, rows):
    if not rows: print(f"  {tag}: N=0"); return
    n=len(rows); pnl=np.array([r["pnl"] for r in rows])
    wr=np.mean([r["won"] for r in rows]); ask=np.mean([r["ask"] for r in rows])
    net=pnl.sum(); mean=pnl.mean()
    t=cluster_t(rows)
    # price-implied breakeven: mean ask*(1+fee) — the win rate needed to break even
    be=np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])
    half=n//2
    h1=pnl[:half].mean() if half else float('nan')
    h2=pnl[half:].mean() if (n-half) else float('nan')
    print(f"  {tag}: N={n} wr={wr:.3f} ask={ask:.3f} be={be:.3f} net=${net:+.0f} "
          f"mean=${mean:+.3f} t={t:+.2f} halves={h1:+.2f}/{h2:+.2f}")

def main():
    W = load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    drows = [r for d in design if (r := one_trade(d))]
    print(f"\n=== DESIGN T=180 baseline ===")
    report("all", drows)

    byh = {}
    for r in drows:
        h = datetime.datetime.utcfromtimestamp(r["ws"]).hour
        byh.setdefault(h, []).append(r)
    print("\n-- UTC hour (DESIGN), sorted by mean --")
    order = sorted(byh, key=lambda h: -np.mean([r["pnl"] for r in byh[h]]))
    for h in order:
        report(f"h{h:02d}", byh[h])

    # Freeze the clearly-positive hours and clearly-negative hours (design)
    pos = [h for h in byh if len(byh[h])>=10 and np.mean([r["pnl"] for r in byh[h]])>0.3]
    neg = [h for h in byh if len(byh[h])>=10 and np.mean([r["pnl"] for r in byh[h]])<-0.3]
    print(f"\n  POSITIVE hours (design, N>=10, mean>+$0.3): {sorted(pos)}")
    print(f"  NEGATIVE hours (design, N>=10, mean<-$0.3): {sorted(neg)}")

    # design halves check for the positive-hour set
    pos_rows = [r for r in drows if datetime.datetime.utcfromtimestamp(r["ws"]).hour in pos]
    print(f"\n-- POSITIVE-hour POOLED (design) --")
    report("pos-pool", pos_rows)
    # split pos-pool by its own median ws
    pos_rows.sort(key=lambda r: r["ws"])
    m = len(pos_rows)//2
    report("pos 1st half", pos_rows[:m])
    report("pos 2nd half", pos_rows[m:])

    # The honest test: on holdout, does the FROZEN positive-hour set hold?
    hrows = [r for d in holdout if (r := one_trade(d))]
    print(f"\n=== HOLDOUT T=180 ===")
    report("all", hrows)
    hpos = [r for r in hrows if datetime.datetime.utcfromtimestamp(r["ws"]).hour in pos]
    hneg = [r for r in hrows if datetime.datetime.utcfromtimestamp(r["ws"]).hour in neg]
    print("-- frozen sets on HOLDOUT --")
    report("pos-hours", hpos)
    report("neg-hours (avoid)", hneg)

if __name__ == "__main__":
    main()