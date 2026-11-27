"""_pm_v3_bounce.py - PASSIVE ROUND-TRIP (maker-in, maker-out) scalp with queue-exact prints.

The untested version of "more trades per window": every falsified scalp was
taker-in/taker-out (pays spread+fee twice). Here BOTH legs are passive:
  join the displayed bid of side X as maker (5 shares behind displayed depth);
  when a taker-SELL print through-fills us, join the displayed ask of X as maker
  (5 shares behind displayed ask depth) at the first frame >= fill + L;
  a taker-BUY print through the ask completes the trip: net = SIZE*(ask-bid), fee 0.
  Unwound inventory at t>=T_END gets a fallback response:
     hold  -> settlement
     taker -> sell at bid (taker fee)
     set   -> post the OTHER leg's bid as maker near T_END; if through-filled,
              complete the set (payoff $1/pair = the user's "hedge trade", passive)
Modes: X = favourite leg (higher bid) or dog leg (lower bid).
Gates (precommitted, OLD design slice): pooled net > 0, both halves > 0, and the
mechanism must not be one big cell. NEW (254 untouched) only if something survives.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

SIZE = 5.0
SPLIT = 1793316600
KMAX = 3            # max round trips per window (the "more trades" ask)
T_ENTRY_END = 250   # last frame a fresh cycle may start
T_END = 255         # inventory must be resolved by here
FEE = lambda p, sh: sh * 0.07 * p * (1 - p)

z = np.load(os.path.join(ROOT, "_pm_cache", "trades.npz"), allow_pickle=True)
slugs, starts, lens, flat = list(z["slugs"]), z["starts"], z["lens"], z["flat"]
TR = {}
for k, s0, ln in zip(slugs, starts, lens):
    a = flat[:, s0:s0 + ln]
    TR[str(k)] = a[:, np.argsort(a[0], kind="stable")]


def fills_bid(a, ws, oc, t0, Br, Dq):
    """Our 5-share BID at Br (4dp) behind displayed depth Dq: filled by SELL prints."""
    Br = round(float(Br), 4)
    if a is None or not np.isfinite(Dq) or Dq < 0:
        return None
    m = (a[4] == oc) & (a[3] == 1) & (a[0] > t0) & (a[0] <= ws + 300)
    p = a[1][m]; sz = a[2][m]; ts = a[0][m]
    if len(ts) == 0:
        return None
    pr = np.round(p, 4)
    t_thr = ts[pr < Br].min() if (pr < Br).any() else np.inf
    cum = np.cumsum(np.where(pr == Br, sz, 0.0))
    idx = np.where(cum >= Dq + SIZE)[0]
    t_at = ts[idx[0]] if len(idx) else np.inf
    t = min(t_thr, t_at)
    return None if np.isinf(t) else t


def fills_ask(a, ws, oc, t0, Ar, Dq):
    """Our 5-share ASK at Ar behind displayed ask depth Dq: filled by BUY prints."""
    Ar = round(float(Ar), 4)
    if a is None or not np.isfinite(Dq) or Dq < 0:
        return None
    m = (a[4] == oc) & (a[3] == 0) & (a[0] > t0) & (a[0] <= ws + 300)
    p = a[1][m]; sz = a[2][m]; ts = a[0][m]
    if len(ts) == 0:
        return None
    pr = np.round(p, 4)
    t_thr = ts[pr > Ar].min() if (pr > Ar).any() else np.inf
    cum = np.cumsum(np.where(pr == Ar, sz, 0.0))
    idx = np.where(cum >= Dq + SIZE)[0]
    t_at = ts[idx[0]] if len(idx) else np.inf
    t = min(t_thr, t_at)
    return None if np.isinf(t) else t


def get(d, side, key, i):
    v = d[side + "_" + key][i]
    return float(v) if np.isfinite(v) else None


def run(W, mode, resp, Ts, L):
    rows = []
    for wi, d in enumerate(W):
        a = TR.get(d["_slug"])
        if a is None:
            continue
        t = d["t"]; n = len(t); ws = d["ws"]
        win_up = d["_winner"] == "Up"
        i = int(np.searchsorted(t, Ts))
        trips = 0
        while i < n and t[i] < T_ENTRY_END and trips < KMAX:
            ub, db = get(d, "Up", "bid", i), get(d, "Down", "bid", i)
            if ub is None or db is None or ub <= 0.02 or db <= 0.02 or ub + db > 1.00:
                i += 1; continue
            side = "Up" if (ub >= db) == (mode == "fav") else "Down"
            oc = 0 if side == "Up" else 1
            Br = get(d, side, "bid", i); Dq = get(d, side, "bidsz", i)
            t0 = ws + float(t[i])
            tf = fills_bid(a, ws, oc, t0, Br, Dq)
            if tf is None:
                break  # GTC bid never filled before the window died
            t_end_trip = tf  # advanced below once we know how the trip resolved
            # passive exit: join the ask at first frame >= fill + L
            j = int(np.searchsorted(t, tf - ws + L))
            net, kind = None, None
            if j < n and t[j] < T_END:
                Ar = get(d, side, "ask", j); Dx = get(d, side, "asksz", j)
                if Ar is not None and 0.02 < Ar < 0.98:
                    ta = fills_ask(a, ws, oc, ws + float(t[j]), Ar, Dx or 0.0)
                    if ta is not None:
                        net = SIZE * (round(Ar, 4) - round(Br, 4)); kind = "trip"
                        t_end_trip = ta
            if net is None:  # not unwound passively -> fallback response
                if resp == "hold":
                    w = 1.0 if (win_up if oc == 0 else not win_up) else 0.0
                    net, kind = SIZE * (w - Br), "hold"
                else:
                    k = int(np.searchsorted(t, max(T_END, tf - ws + L)))
                    if resp == "taker":
                        Sk = get(d, side, "bid", k) if k < n else None
                        if Sk is not None and Sk >= 0.02:
                            net = SIZE * (Sk - Br) - FEE(Sk, SIZE); kind = "taker"
                        else:
                            w = 1.0 if (win_up if oc == 0 else not win_up) else 0.0
                            net, kind = SIZE * (w - Br), "hold"
                    else:  # "set": post other leg's bid as maker; complete if through-filled
                        oth = "Down" if side == "Up" else "Up"; oc2 = 1 - oc
                        B2 = get(d, oth, "bid", k) if k < n else None
                        D2 = get(d, oth, "bidsz", k) if k < n else None
                        tf2 = fills_bid(a, ws, oc2, ws + float(t[k]), B2, D2) if (B2 is not None and k < n) else None
                        if tf2 is not None:
                            net = SIZE * (1 - round(Br, 4) - round(B2, 4)); kind = "set"
                        else:
                            w = 1.0 if (win_up if oc == 0 else not win_up) else 0.0
                            net, kind = SIZE * (w - Br), "hold"
            rows.append((wi, kind, net))
            trips += 1
            i = max(i + 1, int(np.searchsorted(t, t_end_trip - ws + 0.01)))
    return rows


def summarize(rows, label, W):
    if not rows:
        print(f"{label:46s} no trades"); return
    pnl = np.array([r[2] for r in rows]); wi = np.array([r[0] for r in rows])
    tot = pnl.sum()
    kinds = {}
    for r in rows:
        kinds.setdefault(r[1], [0, 0.0]); kinds[r[1]][0] += 1; kinds[r[1]][1] += r[2]
    m, tstat, nw = B.cluster_t(wi, pnl)
    h = len(W) // 2
    h1 = sum(r[2] for r in rows if r[0] < h); h2 = sum(r[2] for r in rows if r[0] >= h)
    print(f"{label:46s} N={len(rows):5d} " + " ".join(f"{k}:{v[0]}=${v[1]:+.1f}" for k, v in sorted(kinds.items()))
          + f" TOT=${tot:+8.2f} t={tstat:+.2f} halves={h1:+.1f}/{h2:+.1f}")
    return tot, h1, h2


W_all = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
OLD = [d for d in W_all if d["ws"] <= SPLIT]
NEW = [d for d in W_all if d["ws"] > SPLIT]
fast = [d for d in OLD if len(d["t"]) > 3 and float(np.median(np.diff(d["t"]))) < 0.5]
print(f"OLD={len(OLD)} fast-cadence OLD={len(fast)} NEW(untouched)={len(NEW)}  KMAX={KMAX} size=5sh")

survivors = []
for mode in ("fav", "dog"):
    for resp in ("hold", "taker", "set"):
        for Ts in (60, 120):
            for L in (0.3, 1.0):
                W = fast if L == 0.3 else OLD
                rows = run(W, mode, resp, Ts, L)
                r = summarize(rows, f"{mode}/{resp} Ts={Ts} L={L} [{'fast' if W is fast else 'OLD'}]", W)
                if r and r[0] > 0 and r[1] > 0 and r[2] > 0:
                    survivors.append((mode, resp, Ts, L))
print("design survivors:", survivors if survivors else "NONE")
np.save(os.path.join(ROOT, "_pm_out", "v3_done.npy"), np.array([1]))
if survivors:
    for s in survivors:
        rows = run(NEW, s[0], s[1], s[2], s[3] if s[3] <= 1.0 else 1.0)
        summarize(rows, f"HOLDOUT NEW {s}", NEW)
