"""_pm_v2_battery.py - maker complete-set with EXIT-AND-CANCEL response.

Oct-30 verdict falsified S1/S2 maker constructions: the both-fill subset earns
the 1.2c overround (+1.1c/share, ~82% of windows), but the ~18% one-leg tail
held to settlement averages -$1.98 and eats the whole income. Hedge-at-fill
(buy the other ask) was tried and failed.

NEW mechanism tested here: treat the one-leg fill as an inventory emergency,
not a position to settle:
  post 5-share bids at BOTH touches at first sane frame >= Ts (queue-exact
  fills from real prints, as in _pm_maker_s2b.py),
  when one leg fills: at the first frame >= fill + L ->
     - if the OTHER leg also filled before that frame: complete set, hold ($1).
     - else: CANCEL the resting leg and SELL the filled side into the displayed
       bid (taker fee, depth-gated); if the sell cannot execute, hold to settle.
This caps the tail at (adverse move + spread + fee) instead of -$2.

Splits: OLD = windows <= 1793316600 (the Oct-30 design set, already mined by
me), NEW = the 254 windows logged AFTER that battery finished = untouched
replication. Gates on OLD: both halves positive, Ts plateau, L in {0.5,1.0}
both positive. Only then ONE look at NEW with the full diagnostic battery.
"""
import os, sys, pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

SIZE = 5.0
SPLIT = 1793316600
FEE = lambda p, sh: sh * 0.07 * p * (1 - p)

z = np.load(os.path.join(ROOT, "_pm_cache", "trades.npz"), allow_pickle=True)
slugs, starts, lens, flat = list(z["slugs"]), z["starts"], z["lens"], z["flat"]
TR = {}
for k, s0, ln in zip(slugs, starts, lens):
    a = flat[:, s0:s0 + ln]
    TR[str(k)] = a[:, np.argsort(a[0], kind="stable")]


def sim_prints(a, ws, oc, t0, Br, Dq):
    """Queue-exact fill time of a SIZE-share bid at price Br (4dp) behind
    displayed depth Dq, from taker-SELL prints on that outcome token."""
    Br = round(float(Br), 4)
    if a is None or not np.isfinite(Dq) or Dq < 0:
        return None, Br
    m = (a[4] == oc) & (a[3] == 1) & (a[0] > t0) & (a[0] <= ws + 300)
    p = a[1][m]; sz = a[2][m]; ts = a[0][m]
    if len(ts) == 0:
        return None, Br
    pr = np.round(p, 4)
    through = pr < Br
    t_through = ts[through].min() if through.any() else np.inf
    atb = pr == Br
    cum = np.cumsum(np.where(atb, sz, 0.0))
    idx = np.where(cum >= Dq + SIZE)[0]
    t_atb = ts[idx[0]] if len(idx) else np.inf
    t = min(t_through, t_atb)
    return (None, Br) if np.isinf(t) else (t, Br)


def maker_window(d, i0, L, policy, tr_u=None, tr_d=None, ws_here=None):
    """One complete-set construction. Returns (kind, net, extra dict).
    kind in: both (complete set earned), none (no fill), exit (one-leg sold),
    hold (one-leg naked to settlement or both-filled late)."""
    t = d["t"]
    ws = ws_here if ws_here is not None else d["ws"]
    a = None if tr_u is None else tr_u
    bu, bd = d["Up_bid"][i0], d["Down_bid"][i0]
    t0 = float(t[i0]) + d["ws"]
    src_ws = d["ws"] if ws_here is None else ws_here
    fu, Bu = sim_prints(a, src_ws, 0, t0, bu, d["Up_bidsz"][i0])
    fd, Bd = sim_prints(a, src_ws, 1, t0, bd, d["Down_bidsz"][i0])
    win_up = d["_winner"] == "Up"
    # placebo uses d["_winner"] only in the hold path; under permutation we
    # pass a shuffled winner through the dict copy.
    if fu is not None and fd is not None:
        return ("both", SIZE * (1 - Bu - Bd), {})
    if fu is None and fd is None:
        return ("none", 0.0, {})
    up_leg = fu is not None
    tf, Bf = (fu, Bu) if up_leg else (fd, Bd)
    t_other = fd if up_leg else fu
    if policy == "hold":
        w = 1.0 if ((win_up if up_leg else (not win_up))) else 0.0
        return ("hold", SIZE * (w - Bf), {"up_leg": up_leg})
    # exit policy
    j = int(np.searchsorted(t, tf - d["ws"] + L))
    if j >= len(t):
        w = 1.0 if ((win_up if up_leg else (not win_up))) else 0.0
        return ("hold", SIZE * (w - Bf), {"up_leg": up_leg, "late": True})
    if t_other is not None and t_other <= float(t[j]) + d["ws"]:
        return ("both", SIZE * (1 - Bu - Bd), {})   # both before we could act
    Sx = d[("Up_bid" if up_leg else "Down_bid")][j]
    Dx = d[("Up_bidsz" if up_leg else "Down_bidsz")][j]
    if np.isfinite(Sx) and Sx >= 0.02 and Dx >= SIZE:
        f = FEE(Sx, SIZE)
        return ("exit", SIZE * (Sx - Bf) - f,
                {"up_leg": up_leg, "slip": Bf - Sx, "fee": f,
                 "react": float(t[j]) + d["ws"] - tf})
    w = 1.0 if ((win_up if up_leg else (not win_up))) else 0.0
    return ("hold", SIZE * (w - Bf), {"up_leg": up_leg, "nobid": True})


def first_sane(d, Ts):
    t = d["t"]
    i = int(np.searchsorted(t, Ts))
    while i < len(t) - 2:
        bu, bd = d["Up_bid"][i], d["Down_bid"][i]
        if np.isfinite(bu) and np.isfinite(bd) and bu > 0.02 and bd > 0.02 and bu + bd <= 1.00:
            return i
        i += 1
    return None


def run_set(W, Ts, L, policy, trmap=None):
    out = []
    for wi, d in enumerate(W):
        if d["_slug"] not in TR if trmap is None else trmap.get(d["_slug"]) is None:
            continue
        i0 = first_sane(d, Ts)
        if i0 is None:
            continue
        a = TR[d["_slug"]] if trmap is None else trmap[d["_slug"]]
        ws_here = d["ws"] if trmap is None else trmap["ws"][d["_slug"]]
        kind, net, ex = maker_window(d, i0, L, policy, a, ws_here=ws_here)
        out.append((wi, kind, net, ex))
    return out


def summarize(out, label):
    tot = sum(r[2] for r in out)
    by = {}
    for r in out:
        by.setdefault(r[1], [0, 0.0])
        by[r[1]][0] += 1; by[r[1]][1] += r[2]
    pnl = np.array([r[2] for r in out]); wi = np.array([r[0] for r in out])
    m, tstat, nw = B.cluster_t(wi, pnl)
    print(f"{label:44s} N={len(out):4d} " +
          " ".join(f"{k}:{v[0]}=${v[1]:+.2f}" for k, v in sorted(by.items())) +
          f"  TOTAL=${tot:+8.2f} t={tstat:+.2f}")
    return tot, tstat


W_all = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
OLD = [d for d in W_all if d["ws"] <= SPLIT]
NEW = [d for d in W_all if d["ws"] > SPLIT]
print(f"windows: resolved={len(W_all)}  OLD(design, mined)={len(OLD)}  NEW(untouched)={len(NEW)}")
print()

print("=== 0) reproduction of S2 (hold tail, no hedge) on OLD ===")
for Ts in (60, 120, 180, 240):
    run_set(OLD, Ts, 1.0, "hold")
    # summarize halves inline
# (full lines printed below via summarize)
print("(summarize)")
for Ts in (120, 180):
    o = run_set(OLD, Ts, 1.0, "hold")
    summarize(o, f"OLD hold Ts={Ts} L=1.0")
    h = len(OLD) // 2
    print(f"    halves: {sum(r[2] for r in o if r[0]<h):+.2f} / {sum(r[2] for r in o if r[0]>=h):+.2f}")
print()

# ---------- 1. EXIT policy on OLD (design gate) ----------
print("=== 1) EXIT-and-cancel policy on OLD design slice ===")
old_res = {}
for Ts in (60, 120, 180, 240):
    for L in (0.5, 1.0):
        o = run_set(OLD, Ts, L, "exit")
        old_res[(Ts, L)] = o
        tot, tstat = summarize(o, f"OLD exit Ts={Ts} L={L}")
        h = len(OLD) // 2
        h1 = sum(r[2] for r in o if r[0] < h); h2 = sum(r[2] for r in o if r[0] >= h)
        print(f"    halves: {h1:+.2f} / {h2:+.2f}   plateau-eligible={tot>0 and h1>0 and h2>0}")
print()

# GATE: freeze nothing unless some (Ts,L) is positive pooled AND both halves.
elig = [(Ts, L) for (Ts, L), o in old_res.items()
        if sum(r[2] for r in o) > 0
        and sum(r[2] for r in o if r[0] < len(OLD)//2) > 0
        and sum(r[2] for r in o if r[0] >= len(OLD)//2) > 0]
print(f"design-gate eligible (Ts,L) configs to carry to NEW: {elig if elig else 'NONE -> maker-exit rule is dead, NEW stays untouched'}")
print()

# ---------- 2. TAKER replication on the untouched NEW tail ----------
# Re-test the previously frozen/discredited rules on data logged AFTER the
# Oct-30 battery.  This cannot "save" a rule (the sign flip would be just the
# same noise reflected), but it completes the honest verdict on all current data.
print("=== 2) taker rules on NEW (untouched tail, 1s fill latency) ===")

def taker_battery(W, bands, label):
    for (tlo, thi, plo, phi) in bands:
        for L in (0.5, 1.0):
            pnl, wi = [], []
            for k, d in enumerate(W):
                t = d["t"]; n = len(t)
                i = int(np.searchsorted(t, tlo))
                fired = False
                while i < n and t[i] < thi:
                    a = d["Up_ask"][i]; sz = d["Up_asksz"][i]
                    dn = d["Down_ask"][i]; dsz = d["Down_asksz"][i]
                    for side, ask, depth in (("Up", a, sz), ("Down", dn, dsz)):
                        if not (np.isfinite(ask) and plo <= ask < phi) or fired:
                            continue
                        j = int(np.searchsorted(t, t[i] + L))
                        if j >= n:
                            continue
                        aj = d[side + "_ask"][j]; szj = d[side + "_asksz"][j]
                        if not (np.isfinite(aj) and plo <= aj < phi):
                            continue
                        if szj * aj < B.STAKE or depth * ask < B.STAKE:
                            continue
                        sh = B.STAKE / aj
                        won = 1.0 if d["_winner"] == side else 0.0
                        pnl.append(sh * won - B.STAKE - B.fee(aj, sh)); wi.append(k)
                        fired = True
                    i += 1
            if not pnl:
                print(f"{label} T[{tlo},{thi}) ask[{plo},{phi}) L={L}: no trades")
                continue
            p = np.array(pnl); w = np.array(wi)
            m, tstat, nw = B.cluster_t(w, p)
            lo, hi = B.boot_ci(w, p)
            tot = p.sum()
            print(f"{label} T[{tlo},{thi}) ask[{plo},{phi}) L={L:.1f}: N={len(p)} "
                  f"net=${tot:+.2f} roi={100*tot/(5.0*len(p)):+.1f}% t={tstat:+.2f} "
                  f"CI=[{lo:+.2f},{hi:+.2f}]/trade")

taker_battery(NEW, [(262, 300, 0.80, 0.96), (240, 262, 0.88, 0.96)], "NEW")
print()
taker_battery(OLD, [(262, 300, 0.80, 0.96), (240, 262, 0.88, 0.96)], "OLD")

