"""_pm_v4_trend.py - trend battery: tape-follow (validate legacy momentum CSV),
book-trend continuation, cross-window outcome streaks.

Precommitted before results (see chat): A/B/C families, one trade per window,
fill at +1s with band+depth gates, hold to settlement, OLD 519 design slice with
halves/arms checks, plateau across grid, single NEW holdout look for survivors.
Stake $5; fee 0.07*p*(1-p) per share; min 5 shares implicit.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

SPLIT = 1793316600
L = 1.0
STAKE = 5.0
FEE = lambda p, sh: sh * 0.07 * p * (1 - p)

W_all = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
OLD = [d for d in W_all if d["ws"] <= SPLIT]
NEW = [d for d in W_all if d["ws"] > SPLIT]
by_ws = {d["ws"]: d["_winner"] for d in W_all}
print(f"design OLD={len(OLD)}  holdout NEW={len(NEW)} (not touched until a rule freezes)")


def mk_trade(d, i_entry, side, plo, phi):
    """Fill side's ask at first frame >= t[i_entry]+L; band+depth gate at fill.
    Returns (pnl, price, won) or None."""
    t = d["t"]; n = len(t)
    j = int(np.searchsorted(t, float(t[i_entry]) + L))
    if j >= n - 1:
        return None
    a = d[side + "_ask"][j]; sz = d[side + "_asksz"][j]
    if not (np.isfinite(a) and plo <= a <= phi) or not np.isfinite(sz) or sz * a < STAKE or sz < 5:
        return None
    sh = STAKE / a
    won = 1.0 if d["_winner"] == side else 0.0
    return sh * won - STAKE - FEE(a, sh), a, won, j


def report(rows, label):
    if not rows:
        print(f"{label:34s} N=0    no trades")
        return None
    pnl = np.array([r[0] for r in rows]); price = np.array([r[1] for r in rows])
    won = np.array([r[2] for r in rows]); wi = np.array([r[3] for r in rows])
    tot = pnl.sum(); roi = 100 * tot / (STAKE * len(pnl))
    wr = won.mean(); paid = price.mean()
    m, tstat, nw = B.cluster_t(wi, pnl)
    lo, hi = B.boot_ci(wi, pnl)
    h1 = sum(r[0] for r in rows if r[3] < nw / 2); h2 = sum(r[0] for r in rows if r[3] >= nw / 2)
    ups = [r for r in rows if r[4] == "Up"]; dns = [r for r in rows if r[4] == "Down"]
    su = sum(r[0] for r in ups); sd = sum(r[0] for r in dns)
    # streaks + dd
    seq = 0; worst = 0; run = 0
    for r in sorted(rows, key=lambda x: x[3]):
        if r[0] < 0: run += 1; worst = max(worst, run)
        else: run = 0
    cum = np.cumsum(pnl); dd = (np.maximum.accumulate(cum) - cum).max()
    print(f"{label:34s} N={len(pnl):4d} wr={wr:.3f} paid={paid:.3f} "
          f"net=${tot:+7.2f} roi={roi:+5.1f}% t={tstat:+.2f} CI=[{lo:+.2f},{hi:+.2f}] "
          f"halves={h1:+7.2f}/{h2:+7.2f} arms U:{su:+6.1f}({len(ups)}) D:{sd:+6.1f}({len(dns)}) "
          f"worstL={worst} dd=${dd:.0f}")
    return tot, h1, h2, su, sd


def gate(res):
    return res and res[0] > 0 and res[1] > 0 and res[2] > 0 and res[3] * res[4] > 0


# ---------------- A: tape-follow (legacy momentum CSV rule) ----------------
print("\n=== A) tape-follow at t=150 (move over prior LOOK s, band .55-.93) ===")
Ares = {}
for LOOK in (100, 150):
    for TH in (0.0002, 0.0004, 0.0006):   # bps as fraction
        for wi, d in enumerate(OLD):
            t = d["t"]; cb = d["cb_px"]
            i = int(np.searchsorted(t, 150.0))
            ib = int(np.searchsorted(t, 150.0 - LOOK))
            if i >= len(t) - 2 or ib < 0 or not (np.isfinite(cb[i]) and np.isfinite(cb[ib])):
                continue
            mv = (cb[i] - cb[ib]) / cb[ib]
            if abs(mv) < TH:
                continue
            side = "Up" if mv > 0 else "Down"
            a0 = d[side + "_ask"][i]
            if not (np.isfinite(a0) and 0.55 <= a0 <= 0.93):
                continue
            r = mk_trade(d, i, side, 0.55, 0.93)
            if r:
                Ares.setdefault((LOOK, TH), []).append(r + (side, wi, d["ws"]))
for key, rows in sorted(Ares.items()):
    rows2 = [(r[0], r[1], r[2], r[5], r[4]) for r in rows]
    gate(report(rows2, f"A look={key[0]} th={1e4*key[1]:.0f}bps"))

# ---------------- B: book-trend continuation ----------------
print("\n=== B) book-trend: side's ask rose >=k over past W, t in [30,240] ===")
Bres = {}
for Wsec, k in ((20, 0.02), (20, 0.04), (60, 0.04), (60, 0.06)):
    rows = []
    for wi, d in enumerate(OLD):
        t = d["t"]; n = len(t)
        fired = False
        i = int(np.searchsorted(t, 30.0))
        while i < n - 2 and t[i] < 240 and not fired:
            ib = int(np.searchsorted(t, float(t[i]) - Wsec))
            if ib >= 0:
                du = d["Up_ask"][i] - d["Up_ask"][ib]
                dd = d["Down_ask"][i] - d["Down_ask"][ib]
                side = "Up" if du >= k and du >= dd else ("Down" if dd >= k else None)
                if side:
                    a0 = d[side + "_ask"][i]
                    if np.isfinite(a0) and 0.50 <= a0 <= 0.93:
                        r = mk_trade(d, i, side, 0.50, 0.93)
                        if r:
                            rows.append((r[0], r[1], r[2], wi, side))
                            fired = True
            i += 1
    Bres[(Wsec, k)] = gate(report([(x[0], x[1], x[2], x[3], x[4]) for x in rows],
                                  f"B W={Wsec}s k={k*100:.0f}c"))

# ---------------- C: cross-window streaks ----------------
print("\n=== C) streak: prev 2 windows same winner, buy that side t<60, ask<=0.62 ===")
rows = []
for wi, d in enumerate(OLD):
    p1 = by_ws.get(d["ws"] - 300); p2 = by_ws.get(d["ws"] - 600)
    if not p1 or p1 != p2:
        continue
    t = d["t"]
    i = int(np.searchsorted(t, 20.0))
    while i < len(t) - 2 and t[i] < 60:
        a0 = d[p1 + "_ask"][i]
        if np.isfinite(a0) and 0.45 <= a0 <= 0.62:
            r = mk_trade(d, i, p1, 0.45, 0.62)
            if r:
                rows.append((r[0], r[1], r[2], wi, p1))
            break
        i += 1
Cgate = gate(report(rows, "C streak>=2"))

# ---------------- survivors -> NEW holdout, once ----------------
surv = [("A", k) for k, g in Ares.items() if g] + [("B", k) for k, g in Bres.items() if g]
if Cgate:
    surv.append(("C", None))
print("\ndesign survivors:", surv if surv else "NONE")
if surv:
    print("\n=== HOLDOUT NEW (single look) ===")
    # (re-run the same rule functions over NEW with identical parameters)
    for fam, key in surv:
        if fam == "A":
            LOOK, TH = key; rows = []
            for wi, d in enumerate(NEW):
                t = d["t"]; cb = d["cb_px"]
                i = int(np.searchsorted(t, 150.0)); ib = int(np.searchsorted(t, 150.0 - LOOK))
                if i >= len(t) - 2 or ib < 0 or not (np.isfinite(cb[i]) and np.isfinite(cb[ib])):
                    continue
                mv = (cb[i] - cb[ib]) / cb[ib]
                if abs(mv) < TH:
                    continue
                side = "Up" if mv > 0 else "Down"
                a0 = d[side + "_ask"][i]
                if not (np.isfinite(a0) and 0.55 <= a0 <= 0.93):
                    continue
                r = mk_trade(d, i, side, 0.55, 0.93)
                if r:
                    rows.append((r[0], r[1], r[2], wi, side))
            report(rows, f"NEW A look={LOOK} th={1e4*TH:.0f}bps")
        elif fam == "B":
            Wsec, k = key; rows = []
            for wi, d in enumerate(NEW):
                t = d["t"]; n = len(t); fired = False
                i = int(np.searchsorted(t, 30.0))
                while i < n - 2 and t[i] < 240 and not fired:
                    ib = int(np.searchsorted(t, float(t[i]) - Wsec))
                    if ib >= 0:
                        du = d["Up_ask"][i] - d["Up_ask"][ib]
                        dd = d["Down_ask"][i] - d["Down_ask"][ib]
                        side = "Up" if du >= k and du >= dd else ("Down" if dd >= k else None)
                        if side:
                            a0 = d[side + "_ask"][i]
                            if np.isfinite(a0) and 0.50 <= a0 <= 0.93:
                                r = mk_trade(d, i, side, 0.50, 0.93)
                                if r:
                                    rows.append((r[0], r[1], r[2], wi, side)); fired = True
                        i += 1
                    else:
                        i += 1
            report(rows, f"NEW B W={Wsec} k={k*100:.0f}c")
        else:
            rows = []
            for wi, d in enumerate(NEW):
                p1 = by_ws.get(d["ws"] - 300); p2 = by_ws.get(d["ws"] - 600)
                if not p1 or p1 != p2:
                    continue
                t = d["t"]; i = int(np.searchsorted(t, 20.0))
                while i < len(t) - 2 and t[i] < 60:
                    a0 = d[p1 + "_ask"][i]
                    if np.isfinite(a0) and 0.45 <= a0 <= 0.62:
                        r = mk_trade(d, i, p1, 0.45, 0.62)
                        if r:
                            rows.append((r[0], r[1], r[2], wi, p1))
                        break
                    i += 1
            report(rows, "NEW C streak>=2")
