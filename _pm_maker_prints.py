"""_pm_maker_prints.py - PRINT-VERIFIED maker fill studies on the BTC 5m log.

Fill model (exact queue, conservative):
  - At frame i we post a GTC buy limit for 5 shares at the displayed best bid B
    of one outcome token, joining the back of the displayed queue (depth D).
  - The order is considered filled at the first AGGRESSOR SELL print for that
    token later in the window with price strictly below B (the whole level was
    swept through us), or cumulative sell volume printed AT B >= D + 5 shares.
  - No queue credit for cancels ahead of us (conservative). Maker fee = 0.
  - Orders die at window close; unfilled = no P&L.

Studies (design slice = first 571 resolved windows; last 246 untouched):
  S1 single-leg touch join, edge surface by T-band x bid-band (halves + arms)
  S2 complete-set maker: bid both legs; sub-case P&L (both/one/none), with the
     realistic response to one-leg fills: hedge-take the other leg (+latency,
     FOK depth gate, taker fee) vs naked hold.
  S3 latency stress 0.5s / 1.0s (re-read the touch after the lag).

Run with the hermes venv python (numpy/scipy), cwd Desktop\bot.
"""
import os, sys, pickle
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

FEE = 0.07
SIZE = 5.0

W = B.load_all()
RES = [d for d in W if d["_winner"] in ("Up", "Down")]
N_DESIGN = 571
design = RES[:N_DESIGN]

# ---- trades cache
tp = os.path.join(ROOT, "_pm_cache", "trades.npz")
if not os.path.exists(tp):
    sys.exit("no trades cache - run _pm_trades_build.py first")
z = np.load(tp, allow_pickle=True)
slugs = list(z["slugs"]); starts = z["starts"]; lens = z["lens"]; flat = z["flat"]
TR = {}
for k, s0, ln in zip(slugs, starts, lens):
    a = flat[:, s0:s0 + ln]
    o = np.argsort(a[0], kind="stable")
    TR[str(k)] = a[:, o]

def prints_for(slug, oc_down, t_after=None):
    a = TR.get(slug)
    if a is None:
        return None
    m = (a[4] == oc_down)
    if t_after is not None:
        m &= (a[0] > t_after)
    return a[:, m]

def sim_fill(d, oc, t0, price_round=4):
    """Return (fill_ts, fill_px_level) or (None, None) for a buy at displayed
    best bid B posted at epoch t0 for outcome oc (0=Up,1=Down)."""
    i = int(np.searchsorted(d["t"], t0 - d["ws"]))
    if i >= len(d["t"]):
        return None, None
    Bq = d[oc and "Down_bid" or "Up_bid"][i]
    D = d[oc and "Down_bidsz" or "Up_bidsz"][i]
    if not np.isfinite(Bq) or Bq <= 0.01 or Bq >= 0.99:
        return None, None
    Br = round(float(Bq), price_round)
    a = TR.get(d["_slug"])
    if a is None:
        return None, None
    m = (a[4] == oc) & (a[3] == 1) & (a[0] > t0) & (a[0] <= d["ws"] + 300)
    p = a[1][m]; sz = a[2][m]; ts = a[0][m]
    if len(ts) == 0:
        return None, None
    pr = np.round(p, price_round)
    through = pr < Br
    atb = pr == Br
    t_through = ts[through].min() if through.any() else np.inf
    cum = np.cumsum(np.where(atb, sz, 0.0))
    idx = np.where(cum >= D + SIZE)[0]
    t_atb = ts[idx[0]] if len(idx) else np.inf
    t = min(t_through, t_atb)
    if np.isinf(t):
        return None, None
    return t, Br

TB = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 270), (270, 300)]
BB = [(0.05, 0.35), (0.35, 0.55), (0.55, 0.70), (0.70, 0.80), (0.80, 0.88), (0.88, 0.94), (0.94, 0.99)]

def run_S1(slice_windows, latency=0.0, label=""):
    ev = []
    for wi, d in enumerate(slice_windows):
        t = d["t"]
        won_up = d["_winner"] == "Up"
        for oc, side in ((0, "Up"), (1, "Down")):
            won = 1.0 if (won_up if oc == 0 else not won_up) else 0.0
            seen = set()
            for j in range(len(t)):
                Tg = float(t[j])
                if Tg < 60:
                    continue
                if Tg >= 300 - latency:
                    break
                i = int(np.searchsorted(t, Tg + latency))
                if i >= len(t):
                    continue
                arr_bid = d[side + "_bid"]; arr_sz = d[side + "_bidsz"]
                Bi = arr_bid[i]
                if not np.isfinite(Bi):
                    continue
                tb = next((x for x in TB if x[0] <= Tg < x[1]), None)
                bb = next((x for x in BB if x[0] <= Bi < x[1]), None)
                if tb is None or bb is None or (tb, bb) in seen:
                    continue
                seen.add((tb, bb))
                tf, Bq = sim_fill_at(d, oc, float(t[i]) + d["ws"], Bi, arr_sz[i])
                if tf is not None:
                    ev.append((wi, side, Tg, float(Bq), won, tf - float(t[i]) - d["ws"]))
    return ev

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

# ---- S1 surface on design slice
print(f"windows total {len(RES)}, design {len(design)} (print-verified maker)", flush=True)
have_tr = [d for d in design if d["_slug"] in TR]
print(f"design windows WITH prints: {len(have_tr)}", flush=True)

ev = [] if os.environ.get("S2ONLY") else run_S1(have_tr, 0.0)
if not ev and not os.environ.get("S2ONLY"):
    sys.exit("no fills?")
E = np.array([(e[0], e[2], e[3], e[4], e[5]) for e in ev], float) if ev else np.zeros((0, 5))
SIDE = np.array([e[1] for e in ev])
wi = E[:, 0]; fillp = E[:, 2]; won = E[:, 3]; wait = E[:, 4]

def cell_line(m, prefix):
    if m.sum() == 0:
        return None
    net = SIZE * won[m] - SIZE * fillp[m]
    print(f"{prefix} fills={int(m.sum()):4d} pay={fillp[m].mean():.4f} wr={won[m].mean():.3f} "
          f"net/fill=${net.sum():+8.2f} net/sh={net.sum()/ (SIZE*m.sum())*100:+5.2f}c")
    return net

print("\n=== S1 touch-join maker, PRINT FILLS, design first 571 (posts: first per side per TxB band, T>=60) ===")
# posts denominator approximated by counting windows entering each cell - skip; fills only.
Tg = E[:, 1]
for (tlo, thi) in TB:
    for (blo, bhi) in BB:
        m = (Tg >= tlo) & (Tg < thi) & (fillp >= blo) & (fillp < bhi)
        if m.sum() == 0:
            continue
        cell_line(m, f"T[{tlo:3d},{thi:3d}) B[{blo:.2f},{bhi:.2f}) pooled|")
        h = wi[m] <= (len(have_tr) // 2)
        for sd in ("Up", "Down"):
            ms = m & (SIDE == sd)
            if ms.sum() >= 3:
                n = cell_line(ms, f"                       {sd:4s}    |")
        hm1 = m & (wi <= len(have_tr) // 2); hm2 = m & (wi > len(have_tr) // 2)
        n1 = SIZE * (won[hm1] - fillp[hm1]).sum(); n2 = SIZE * (won[hm2] - fillp[hm2]).sum()
        print(f"                       halves  | h1=${n1:+7.2f} ({hm1.sum()} fills)  h2=${n2:+7.2f} ({hm2.sum()} fills)")

# ---- S2 complete-set maker
print("\n=== S2 complete-set: bid BOTH legs at touch, first frame T>=Ts ===")
for TS in (60, 120, 180, 240):
    rec = []
    for wi_, d in enumerate(have_tr):
        t = d["t"]
        i = int(np.searchsorted(t, TS))
        if i >= len(t) - 3:
            continue
        bu, bd = d["Up_bid"][i], d["Down_bid"][i]
        au, ad = d["Up_ask"][i], d["Down_ask"][i]
        if not (np.isfinite(bu) and np.isfinite(bd) and np.isfinite(au) and np.isfinite(ad)):
            continue
        if not (0.02 < bu < 0.98 and 0.02 < bd < 0.98):
            continue
        fu, Bu = sim_fill_at(d, 0, float(t[i]) + d["ws"], bu, d["Up_bidsz"][i])
        fd, Bd = sim_fill_at(d, 1, float(t[i]) + d["ws"], bd, d["Down_bidsz"][i])
        win_up = d["_winner"] == "Up"
        wu = 1.0 if win_up else 0.0; wd = 1.0 if not win_up else 0.0
        both = fu is not None and fd is not None
        if both:
            net = SIZE * (1.0 - Bu - Bd)
            rec.append((wi_, "both", net, 0.0))
        elif fu is not None:
            # naked long Up at Bu
            rec.append((wi_, "up_only", SIZE * (wu - Bu), float(au)))
        elif fd is not None:
            rec.append((wi_, "dn_only", SIZE * (wd - Bd), float(ad)))
        else:
            rec.append((wi_, "none", 0.0, 0.0))
    import collections
    cnt = collections.Counter(r[1] for r in rec)
    nets = {k: sum(r[2] for r in rec if r[1] == k) for k in cnt}
    tot = sum(nets.values())
    halfn = len(have_tr) // 2
    h1 = sum(r[2] for r in rec if r[0] < halfn); h2 = sum(r[2] for r in rec if r[0] >= halfn)
    print(f"Ts={TS:3d}s  both={cnt.get('both',0):3d} up={cnt.get('up_only',0):3d} dn={cnt.get('dn_only',0):3d} none={cnt.get('none',0):3d} "
          f"| net$ both={nets.get('both',0):+7.2f} up_only={nets.get('up_only',0):+7.2f} dn_only={nets.get('dn_only',0):+7.2f} TOTAL={tot:+7.2f} halves=${h1:+.2f}/${h2:+.2f}")
