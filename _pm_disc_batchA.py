#!/usr/bin/env python3
"""_pm_disc_batchA.py — genuinely-new non-latency axes, batch A (DESIGN slice only).

Axes never mined on this log (exhausted: price level, time, book shape,
maker fills, model-late, flow imbalance, streaks, model T-variants):

  A1 VOLATILITY REGIME — does the edge of a generic T=180 entry depend on the
     window's realised move / cb innovation vol? A new axis, not a price re-slice.
  A2 PATH GEOMETRY — is the current mid's distance from the running window MEAN
     (not the open) predictive? A basis/mean-reversion axis distinct from
     favourite-band and median-cross.
  A3 CROSS-WINDOW VOL — does the PREVIOUS window's realised vol predict whether
     the CURRENT window trends or reverts at T=180?

Neutral baseline trade: first frame at/after T=180, either side, ask in
[0.10,0.90], depth clears a $5 ticket, hold to settlement. We slice THIS by
each new axis. DESIGN = first 70%. One holdout shot only on a frozen candidate.
"""
import os, pickle, math, datetime
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

def frame_at(d, T):
    i = int(np.searchsorted(d["t"], T, side="right")) - 1
    return i if i >= 0 else None

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
            return dict(ws=d["ws"], side=side, ask=ask, won=won, pnl=pnl,
                        sh=sh, mid=(d[f"{side}_ask"][i]+d[f"{side}_bid"][i])/2,
                        i=i, T=t[i])
    return None

def win_vol(d):
    """Realised |move| and per-second innovation vol of cb over the window."""
    cb = d["cb_px"]; t = d["t"]
    f = np.isfinite(cb)
    if f.sum() < 10: return None
    cb = cb[f]; t = t[f]
    total_move = abs(cb[-1] - cb[0])
    # per ~1s innovation: diff over the median dt
    dif = np.diff(cb)
    return dict(total_move=total_move, sd_innov=float(np.std(dif)) if len(dif)>1 else np.nan,
                n=len(cb), cb0=cb[0], cb_now=cb[-1])

def path_features(d, T):
    """Distance of current mid from the running window MEAN of cb."""
    cb = d["cb_px"]; t = d["t"]
    m = (t <= T) & np.isfinite(cb)
    if m.sum() < 5: return None
    cbm = cb[m]; tm = t[m]
    mean_cb = float(np.mean(cbm))
    cur_cb = float(cbm[-1])
    # basis: current cb minus running mean, normalised by running sd
    sd = float(np.std(cbm))
    basis = (cur_cb - mean_cb)
    z = basis / sd if sd > 1e-6 else np.nan
    # signed recent drift over last 30s
    m30 = (t <= T) & (t >= T-30) & np.isfinite(cb)
    drift = float(cb[m30][-1] - cb[m30][0]) if m30.sum() >= 2 else np.nan
    return dict(basis=basis, z=z, mean_cb=mean_cb, cur_cb=cur_cb, drift=drift, sd=sd)

def stats(rows):
    if not rows: return None
    pnl = np.array([r["pnl"] for r in rows])
    return dict(n=len(rows), wr=float(np.mean([r["won"] for r in rows])),
                ask=float(np.mean([r["ask"] for r in rows])),
                be=float(np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])),
                net=float(pnl.sum()), mean=float(pnl.mean()))

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} be={st['be']:.3f} "
          f"net=${st['net']:+.0f} mean=${st['mean']:+.3f}")

def terciles(rows, key):
    vals = np.array([r[key] for r in rows])
    q1, q2 = np.quantile(vals, [1/3, 2/3])
    return (q1, q2,
            [r for r in rows if r[key] <= q1],
            [r for r in rows if q1 < r[key] <= q2],
            [r for r in rows if r[key] > q2])

def main():
    W = load_all(); n = len(W); split = int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    # attach features
    for d in design + holdout:
        wv = win_vol(d); d["_vol"] = wv
        pf = path_features(d, 180); d["_path"] = pf
        d["_trade"] = one_trade(d, 180)

    drows = [dict(r, _vol=d["_vol"], _path=d["_path"], ws=d["ws"])
             for d in design if (r := d["_trade"]) is not None]
    print(f"\n=== DESIGN T=180: {len(drows)} trades ===")
    show("baseline all", stats(drows))

    # A1: volatility regime (total |move| terciles)
    print("\n-- A1 VOLATILITY REGIME: total |cb move| terciles (DESIGN) --")
    d2 = [r for r in drows if r["_vol"]]
    q1,q2,lo,mid,hi = terciles(d2, "_vol") if False else (None,)*5
    vals = np.array([r["_vol"]["total_move"] for r in d2])
    q1,q2 = np.quantile(vals,[1/3,2/3])
    for lab,sub in [("low-vol",[r for r in d2 if r["_vol"]["total_move"]<=q1]),
                    ("mid-vol",[r for r in d2 if q1<r["_vol"]["total_move"]<=q2]),
                    ("high-vol",[r for r in d2 if r["_vol"]["total_move"]>q2])]:
        show(f"{lab}", stats(sub))
    print(f"  (cuts {q1:.0f}, {q2:.0f} $)")

    # A2: path geometry — basis z terciles
    print("\n-- A2 PATH GEOMETRY: (cur_cb - running_mean)/sd terciles (DESIGN) --")
    d3 = [r for r in drows if r["_path"] and np.isfinite(r["_path"]["z"])]
    vals = np.array([r["_path"]["z"] for r in d3])
    q1,q2 = np.quantile(vals,[1/3,2/3])
    for lab,sub in [("below-mean (z low)",[r for r in d3 if r["_path"]["z"]<=q1]),
                    ("mid",[r for r in d3 if q1<r["_path"]["z"]<=q2]),
                    ("above-mean (z high)",[r for r in d3 if r["_path"]["z"]>q2])]:
        show(f"{lab}", stats(sub))
    print(f"  (cuts {q1:+.2f}, {q2:+.2f})")

    # A3: cross-window vol — prev window total_move vs current edge
    print("\n-- A3 CROSS-WINDOW VOL: prev window |move| terciles vs current edge (DESIGN) --")
    vmap = {d["ws"]: d["_vol"]["total_move"] for d in design if d["_vol"]}
    d4 = [r for r in drows if (r["ws"]-300) in vmap]
    for r in d4: r["_prevvol"] = vmap[r["ws"]-300]
    vals = np.array([r["_prevvol"] for r in d4])
    q1,q2 = np.quantile(vals,[1/3,2/3])
    for lab,sub in [("prev low-vol",[r for r in d4 if r["_prevvol"]<=q1]),
                    ("prev mid-vol",[r for r in d4 if q1<r["_prevvol"]<=q2]),
                    ("prev high-vol",[r for r in d4 if r["_prevvol"]>q2])]:
        show(f"{lab}", stats(sub))
    print(f"  (cuts {q1:.0f}, {q2:.0f} $)")

if __name__ == "__main__":
    main()