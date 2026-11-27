"""_pm_maker_s2b.py - complete-set maker with hedge-at-fill response, design slice.

Post 5 shares bid at the displayed best bid of BOTH legs at first frame T>=Ts
(conservative queue fill from real trade prints). Response to a single-leg fill:
buy the other leg's ask at the first frame >=1s after the fill, FOK gate
(depth>=5 shares), and only if the locked total cost Bu+Ad < LOCKCAP (the loss
you accept to become riskless). If the hedge cannot execute, hold naked.
Also variant: skip the whole window unless bid-sum <= SUMMAX (wide overround).

Maker fee 0 (Polymarket crypto: makers pay no fee), taker fee 0.07*p*(1-p) on
hedge buys. Design = first 571 resolved windows; holdout untouched.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

SIZE = 5.0
z = np.load(os.path.join(ROOT, "_pm_cache", "trades.npz"), allow_pickle=True)
slugs = list(z["slugs"]); starts = z["starts"]; lens = z["lens"]; flat = z["flat"]
TR = {}
for k, s0, ln in zip(slugs, starts, lens):
    a = flat[:, s0:s0 + ln]
    TR[str(k)] = a[:, np.argsort(a[0], kind="stable")]

def sim_fill_at(d, oc, t0, Bq, Dq, price_round=4):
    Br = round(float(Bq), price_round)
    a = TR.get(d["_slug"])
    if a is None or not np.isfinite(Dq) or Dq < 0:
        return None, Br
    m = (a[4] == oc) & (a[3] == 1) & (a[0] > t0) & (a[0] <= d["ws"] + 300)
    p = a[1][m]; sz = a[2][m]; ts = a[0][m]
    if len(ts) == 0:
        return None, Br
    pr = np.round(p, price_round)
    through = pr < Br
    t_through = ts[through].min() if through.any() else np.inf
    atb = pr == Br
    cum = np.cumsum(np.where(atb, sz, 0.0))
    idx = np.where(cum >= Dq + SIZE)[0]
    t_atb = ts[idx[0]] if len(idx) else np.inf
    t = min(t_through, t_atb)
    return (None, Br) if np.isinf(t) else (t, Br)

W = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
design = W[:571]
have = [d for d in design if d["_slug"] in TR]

FEE = lambda p, sh: sh * 0.07 * p * (1 - p)

def run(Ts, summax=1.00, lockcap=1.05, hedge=True):
    res = []
    for wi, d in enumerate(have):
        t = d["t"]; i = int(np.searchsorted(t, Ts))
        if i >= len(t) - 3:
            continue
        bu, bd = d["Up_bid"][i], d["Down_bid"][i]
        if not (np.isfinite(bu) and np.isfinite(bd)) or bu <= 0.02 or bd <= 0.02:
            continue
        if bu + bd > summax:
            continue
        t0 = float(t[i]) + d["ws"]
        fu, Bu = sim_fill_at(d, 0, t0, bu, d["Up_bidsz"][i])
        fd, Bd = sim_fill_at(d, 1, t0, bd, d["Down_bidsz"][i])
        win_up = d["_winner"] == "Up"
        if fu is not None and fd is not None:
            res.append((wi, "both", SIZE * (1 - Bu - Bd)))
            continue
        if (fu is None) and (fd is None):
            res.append((wi, "none", 0.0)); continue
        up_leg = fu is not None
        tf, Bf = (fu, Bu) if up_leg else (fd, Bd)
        net = None
        if hedge:
            j = int(np.searchsorted(t, tf - d["ws"] + 1.0))
            if j < len(t):
                ao = d[("Down_ask" if up_leg else "Up_ask")][j]
                so = d[("Down_asksz" if up_leg else "Up_asksz")][j]
                if np.isfinite(ao) and so >= SIZE and Bf + ao < lockcap:
                    # locked settlement $1 per pair on min shares
                    sh = SIZE
                    res.append((wi, "hedged", sh * (1 - Bf - ao) - FEE(ao, sh)))
                    continue
        w = 1.0 if (win_up if up_leg else not win_up) else 0.0
        res.append((wi, "naked", SIZE * (w - Bf)))
    tot = sum(r[2] for r in res)
    by = {}
    for r in res:
        by.setdefault(r[1], [0, 0.0])
        by[r[1]][0] += 1; by[r[1]][1] += r[2]
    h = len(have) // 2
    h1 = sum(r[2] for r in res if r[0] < h); h2 = sum(r[2] for r in res if r[0] >= h)
    print(f"Ts={Ts:3d} sum<={summax:.3f} lockcap={lockcap:.3f} hedge={hedge}: "
          + " ".join(f"{k}:{v[0]}=${v[1]:+.2f}" for k, v in sorted(by.items()))
          + f"  TOTAL=${tot:+8.2f} halves=${h1:+.2f}/${h2:+.2f}")
    return res

print("windows:", len(have))
for Ts in (60, 120, 180, 240):
    run(Ts, hedge=True)
print("--- variants ---")
run(120, hedge=False)                       # naked tail, no hedge
run(120, summax=0.99, hedge=True)           # only wide overround entries
run(180, summax=0.99, hedge=True)
run(120, summax=0.99, lockcap=1.02, hedge=True)
run(180, summax=0.995, hedge=True)
