#!/usr/bin/env python3
"""_pm_hunt_timeofday.py — non-latency hunt: time-of-day / day-of-week structure
on the 5m window, plus richer serial structure than the falsified 2-streak.

DESIGN slice only. Freeze candidates with plateau + both-halves check, then
ONE holdout evaluation. No falsified family re-run.

Baseline rule we slice: at a fixed T, buy the side whose ask is in a band and
holds to settlement (the neutral price-surface trade). We ask: does the EDGE of
that trade depend on (a) UTC hour, (b) day of week, (c) serial history beyond
2 same-win streaks?
"""
import os, pickle, datetime
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

def one_trade_per_window(d, T, band=(0.10, 0.90)):
    """Neutral rule: first frame at/after T where either ask is in band and
    depth clears a $5 ticket. Returns (side, ask, won, pnl, ws) or None."""
    t = d["t"]
    i0 = int(np.searchsorted(t, T, side="right")) - 1
    if i0 < 0: return None
    for i in range(i0, len(t)):
        for side in ("Up", "Down"):
            ask = d[f"{side}_ask"][i]
            asz = d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0] <= ask <= band[1]): continue
            sh = int(STAKE/ask)
            if sh < 5 or not np.isfinite(asz) or asz < sh: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            return (side, ask, won, pnl, d["ws"])
    return None

def stats(rows):
    if not rows: return None
    pnl = np.array([r[3] for r in rows])
    wr = np.mean([r[2] for r in rows])
    ask = np.mean([r[1] for r in rows])
    return dict(n=len(rows), wr=wr, ask=ask, net=pnl.sum(), mean=pnl.mean(),
                wins=int(pnl.__gt__(0).sum()))

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} "
          f"net=${st['net']:+.0f} mean=${st['mean']:+.3f} win={st['wins']}")

def main():
    W = load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")
    print(f"design span {datetime.datetime.utcfromtimestamp(design[0]['ws'])} -> "
          f"{datetime.datetime.utcfromtimestamp(design[-1]['ws'])}")

    T = 180
    drows = [r for d in design if (r := one_trade_per_window(d, T))]
    print(f"\n=== DESIGN T={T}: {len(drows)} trades (1/window) ===")
    show("baseline all", stats(drows))

    # (a) UTC hour-of-day
    print("\n-- by UTC hour (DESIGN) --")
    byh = {}
    for r in drows:
        h = datetime.datetime.utcfromtimestamp(r[4]).hour
        byh.setdefault(h, []).append(r)
    for h in sorted(byh):
        st = stats(byh[h])
        mark = ""
        if st and st["n"]>=15 and st["mean"]>0.3: mark=" <== positive"
        show(f"h{h:02d}", st)

    # (b) day of week
    print("\n-- by weekday (DESIGN) --")
    byw = {}
    for r in drows:
        w = datetime.datetime.utcfromtimestamp(r[4]).weekday()
        byw.setdefault(w, []).append(r)
    names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for w in sorted(byw):
        show(names[w], stats(byw[w]))

    # (c) serial structure: winner of prev K windows vs current
    # (falsified was 2-streak buy<0.62 early; we test richer: does the current
    # window's T=180 edge depend on the PREVIOUS window's winner / run length?)
    print("\n-- serial: current edge by previous-window winner (DESIGN) --")
    # map ws -> winner
    wmap = {d["ws"]: d["_winner"] for d in W}
    # need the 5m grid: each window is 300s; prev = ws-300
    byprev = {}
    for r in drows:
        prev = wmap.get(r[4]-300)
        if prev is None: continue
        byprev.setdefault(prev, []).append(r)
    for k in sorted(byprev):
        show(f"prev={k}", stats(byprev[k]))

    # run-length: how many consecutive same-winner windows preceded this one
    print("\n-- serial: current edge by run-length of same winner (DESIGN) --")
    # build winner sequence over ALL windows (design+holdout) by ws
    seq = [(d["ws"], d["_winner"]) for d in W]
    seq.sort()
    run = {}
    cnt = 0; cur = None
    for ws, w in seq:
        if w == cur: cnt += 1
        else: cur = w; cnt = 1
        run[ws] = cnt
    byrun = {}
    for r in drows:
        rl = run.get(r[4], 1)
        byrun.setdefault(rl, []).append(r)
    for k in sorted(byrun):
        st = stats(byrun[k])
        if st and st["n"]>=10: show(f"run={k}", st)

if __name__ == "__main__":
    main()