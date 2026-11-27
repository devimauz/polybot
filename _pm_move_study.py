"""_pm_move_study.py - repricing-speed distribution after Coinbase moves, measured
on the 15-day 0.28s-cadence snapshot corpus (answers what the 22-min probe can't:
how fast does the book reprice after LARGE moves, across regimes).

For each window with frame cadence <0.5s:
  - per-window sigma_1s from cb_px diffs; detection: |cb[i]-cb[i-m]| with m=2 frames.
  - event buckets by z = |d2| / (sigma_1s*sqrt(m*dt)); cooldown 2s; skip t<10, t>260.
  - ALREADY-REPRICED fraction: aligned ask at the trigger frame itself already moved.
  - stale@+1 frame (0.28s) / +2 frames (0.57s) / +1.0s: would an order at the last
    pre-move ask still be marketable.
  - reprice lag in frames for kask +1c / +2c.
  - EV of buying the aligned ask at +1 frame (5 shares, taker fee, settle to winner):
    mean/median per trade, win-rate, cluster-t, chronological halves, Up/Down arms.
The EV part is the same move-follow family falsified at band level - quoted here
only as the size of the prize, NOT as a shippable rule; the TIMING distribution is
the new information.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

SPLIT = 1793316600
W = [d for d in B.load_all() if d["_winner"] in ("Up", "Down") and d["ws"] <= SPLIT]
W = [d for d in W if len(d["t"]) > 300 and float(np.median(np.diff(d["t"]))) < 0.5]
print(f"fast-cadence OLD windows: {len(W)}")

ZB = [(2.5, 4), (4, 6), (6, 10), (10, 99)]
M = 2


def study(W):
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]; cb = d["cb_px"]; n = len(t)
        med_dt = float(np.median(np.diff(t)))
        dif = np.diff(cb[np.isfinite(cb)])
        sig = float(np.std(dif)) if len(dif) > 10 else 0.0
        if sig <= 0:
            continue
        winner = d["_winner"]
        last_fire = -10.0
        for i in range(M + 2, n - 4):
            ti = float(t[i])
            if ti < 10 or ti > 260 or not np.isfinite(cb[i]) or not np.isfinite(cb[i - M]):
                continue
            d2 = float(cb[i]) - float(cb[i - M])
            z = abs(d2) / (sig * np.sqrt(M))
            if z < ZB[0][0] or ti - last_fire < 2.0:
                continue
            last_fire = ti
            side = "Up" if d2 > 0 else "Down"
            ask = d[side + "_ask"]; asz = d[side + "_asksz"]
            a0 = ask[i - M]  # pre-move touch (what we saw at detection input)
            if not (np.isfinite(a0) and 0.02 <= a0 <= 0.98):
                continue
            alr = np.isfinite(ask[i]) and ask[i] >= a0 + 0.01
            j1 = i + 1 if i + 1 < n and np.isfinite(ask[i + 1]) else None
            j2 = i + 2 if i + 2 < n and np.isfinite(ask[i + 2]) else None
            stale1 = j1 is not None and ask[j1] <= a0 + 0.005
            stale2 = j2 is not None and ask[j2] <= a0 + 0.005
            k = i + M
            j100 = int(np.searchsorted(t, ti + 1.0))
            stale1s = j100 < n and np.isfinite(ask[j100]) and ask[j100] <= a0 + 0.005
            for kap in (0.01, 0.02):
                msk = np.where(ask[i:] >= a0 + kap)[0]
                rows.append(dict(wi=wi, z=z, kap=kap, side=side,
                                 lag_fr=msk[0] if len(msk) else 9999,
                                 alr=alr, stale1=stale1, stale2=stale2, stale1s=stale1s,
                                 a_act=ask[j1] if j1 else np.nan,
                                 won=1.0 if winner == side else 0.0))
    return rows


R = study(W)
print(f"events (dual-kappa rows): {len(R)}")
for (zlo, zhi) in ZB:
    for kap in (0.01, 0.02):
        s = [r for r in R if zlo <= r["z"] < zhi and r["kap"] == kap]
        if len(s) < 8:
            print(f"z[{zlo},{zhi}) k=+{kap:.2f}: N={len(s)} thin"); continue
        lag = np.array([r["lag_fr"] for r in s])
        print(f"z[{zlo},{zhi}) k=+{kap:.2f}: N={len(s):5d} "
              f"already-repriced@trigger={np.mean([r['alr'] for r in s]):.2f} "
              f"stale@+0.3s={np.mean([r['stale1'] for r in s]):.2f} "
              f"stale@+0.6s={np.mean([r['stale2'] for r in s]):.2f} "
              f"stale@+1.0s={np.mean([r['stale1s'] for r in s]):.2f} "
              f"lag frames p50={int(np.median(lag))} p90={int(np.percentile(lag,90))}")

print("\nEV if we buy the aligned ask at +1 frame (0.28s) whenever still stale (5 shares):")
for (zlo, zhi) in ZB:
    s = [r for r in R if zlo <= r["z"] < zhi and r["kap"] == 0.01 and r["stale1"]
         and np.isfinite(r["a_act"]) and r["a_act"] <= 0.98]
    if len(s) < 8:
        print(f"z[{zlo},{zhi}): N={len(s)} thin"); continue
    pnl, wi, won = [], [], []
    for r in s:
        sh = 5.0 / r["a_act"]
        pnl.append(sh * r["won"] - 5.0 - 0.07 * r["a_act"] * (1 - r["a_act"]) * sh)
        wi.append(r["wi"]); won.append(r["won"])
    pnl = np.array(pnl); wi = np.array(wi); won = np.array(won)
    m, tstat, nw = B.cluster_t(wi, pnl)
    lo, hi = B.boot_ci(wi, pnl)
    h = max(wi) // 2
    h1 = pnl[wi <= h].sum(); h2 = pnl[wi > h].sum()
    up = pnl[[r["side"] == "Up" for r in s]]; dn = pnl[[r["side"] == "Down" for r in s]]
    print(f"z[{zlo},{zhi}): N={len(pnl)} (clusters={nw}) wr={won.mean():.3f} "
          f"net=${pnl.sum():+.2f} mean=${pnl.mean():+.3f} med=${np.median(pnl):+.3f} "
          f"t={tstat:+.2f} CI=[{lo:+.2f},{hi:+.2f}] halves=${h1:+.2f}/${h2:+.2f} "
          f"Up=${up.sum():+.1f}({len(up)}) Down=${dn.sum():+.1f}({len(dn)})")
