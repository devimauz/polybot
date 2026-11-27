"""Test strategy families the endgame grid CANNOT see.

build_rows() only samples T in [240,300), so everything so far lives in the
last minute.  These families need the whole window:
  A. EARLY-FAVOURITE  - buy the favourite early, when it is cheap
  B. LATE DIP        - the favourite's ask DROPS late (market moving toward
                        the TWAP resolution)
  C. LATE MOMENTUM   - the favourite's ask RISES late
  D. PRICE DISLOCATION - the favourite's ask is far from the 60s-TWAP-implied
                        probability

Each: same validity filters (both legs alive, tight spread, depth), same
causal clock, train/OOS split, and a PLACEBO that reassigns real outcomes
across windows and RE-SELECTS the rule.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B

W = B.load_all()
NW = len(W)
CUT = int(NW * 0.60)
TAKE = 10.0
FEE = 0.07
MAX_SPREAD = 0.04
MIN_DEPTH = 10.0
rng = np.random.default_rng(777)


def wide_rows(t_lo=0, t_hi=300, step=3):
    """Every frame in [t_lo, t_hi) with the same validity filters as the
    endgame builder, but with the FULL window available."""
    out = []
    for wi, d in enumerate(W):
        t = d["t"]
        n = len(t)
        for T in range(t_lo, t_hi, step):
            i = int(np.searchsorted(t, T))
            if i >= n:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up = 0.5 * (mu + (1.0 - md))
            if not (0.002 < p_up < 0.998):
                continue
            for side in ("Up", "Down"):
                a, b, sz = d[side + "_ask"][i], d[side + "_bid"][i], d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (a, b, sz)):
                    continue
                if not (0 < b <= a < 1) or (a - b) > MAX_SPREAD or sz * a < MIN_DEPTH:
                    continue
                o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                if not (np.isfinite(o) and 0.001 < o < 0.999):
                    continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                out.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side,
                                ask=float(a), bid=float(b), mkt=float(mkt),
                                p_up=float(p_up), won=1 if d["_winner"] == side else 0,
                                i=i))
    return out


def first_per_window(rs):
    best = {}
    for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    return list(best.values())


def stat(sel, stake=TAKE):
    if not sel:
        return None
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    wi = [r["wi"] for r in sel]
    sh = stake / a
    p = sh * w - stake - sh * FEE * a * (1 - a)
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    return dict(n=len(sel), wins=int(w.sum()), net=float(p.sum()), t=float(t),
                lo=float(lo), hi=float(hi), roi=float(p.sum()) / (stake * len(sel)),
                c5=c5, worst=float(p.min()), meanask=float(a.mean()))


def thirds(sel):
    o = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        if s:
            a = np.array([r["ask"] for r in s]); w = np.array([r["won"] for r in s], dtype=float)
            sh = TAKE / a
            o.append(float((sh * w - TAKE - sh * FEE * a * (1 - a)).sum()))
        else:
            o.append(0.0)
    return o


WR = wide_rows(0, 300, 3)
print(f"wide rows: {len(WR)}   windows: {NW}   TRAIN 0..{CUT-1}  OOS {CUT}..{NW-1}\n")

# ---------------------------------------------------------------- A. early favourite
print("=" * 108)
print("A. EARLY-FAVOURITE: buy the favourite early, when its ask is in a band")
print("=" * 108)
print(f"  {'window':<22} {'band':<14} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'net$':>9} {'t':>6} {'CI95':>17}  {'thirds'}")
best_a = []
for t0, t1 in ((0, 60), (0, 120), (30, 90), (60, 150), (60, 240), (120, 240)):
    for lo, hi in ((0.60, 0.98), (0.70, 0.95), (0.80, 0.96), (0.50, 0.90)):
        sel = first_per_window([r for r in WR if t0 <= r["T"] < t1 and r["mkt"] >= 0.5
                                and lo <= r["ask"] < hi])
        if len(sel) < 20:
            continue
        v = stat(sel)
        if v is None:
            continue
        tr = [r for r in sel if r["wi"] < CUT]
        te = [r for r in sel if r["wi"] >= CUT]
        vtr, vte = stat(tr), stat(te)
        if not vtr or not vte:
            continue
        h = thirds(sel)
        print(f"  T[{t0},{t1}){'':<10} [{lo:.2f},{hi:.2f})   {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} "
              f"{v['wins']/v['n']:>7.4f} {v['net']:>+9.2f} {v['t']:>+6.2f} [{v['lo']:>+5.2f},{v['hi']:>+5.2f}]  "
              f"[{h[0]:>+7.1f},{h[1]:>+7.1f},{h[2]:>+7.1f}]   train {vtr['net']:>+7.2f} / oos {vte['net']:>+7.2f}")
        best_a.append((t0, t1, lo, hi, vtr["net"], vte["net"]))

# ---------------------------------------------------------------- B/C. dip & momentum
print("\n" + "=" * 108)
print("B/C. LATE DIP and LATE MOMENTUM: the favourite's ask moves in the last minute")
print("=" * 108)
by_win = {}
for r in WR:
    by_win.setdefault(r["wi"], []).append(r)
for wi in by_win:
    by_win[wi].sort(key=lambda r: r["T"])


def late_moves(min_move, kind, t_lo=240):
    """Select the favourite quote late in the window whose ask has moved by
    `min_move` cents from the window's earlier reference."""
    out = []
    for wi, rs in by_win.items():
        favs = [r for r in rs if r["mkt"] >= 0.5]
        if not favs:
            continue
        for r in favs:
            if r["T"] < t_lo:
                continue
            # reference: the favourite ask 30-90s earlier in the same window
            ref = [q["ask"] for q in favs if r["T"] - 90 <= q["T"] < r["T"] - 30]
            if not ref:
                continue
            base = float(np.mean(ref))
            mv = r["ask"] - base
            if kind == "dip" and mv <= -min_move:
                out.append(r)
            if kind == "mom" and mv >= min_move:
                out.append(r)
    return first_per_window(out)


print(f"  {'kind':<8} {'move':>6} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'net$':>9} {'t':>6} {'CI95':>17}")
for kind in ("dip", "mom"):
    for mm in (0.02, 0.04, 0.06, 0.08, 0.10):
        sel = late_moves(mm, kind)
        if len(sel) < 15:
            continue
        v = stat(sel)
        tr = stat([r for r in sel if r["wi"] < CUT])
        te = stat([r for r in sel if r["wi"] >= CUT])
        if not v or not tr or not te:
            continue
        print(f"  {kind:<8} {mm:>6.2f} {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} "
              f"{v['wins']/v['n']:>7.4f} {v['net']:>+9.2f} {v['t']:>+6.2f} [{v['lo']:>+5.2f},{v['hi']:>+5.2f}]"
              f"   train {tr['net']:>+7.2f} / oos {te['net']:>+7.2f}")

# ---------------------------------------------------------------- D. dislocation
print("\n" + "=" * 108)
print("D. PRICE DISLOCATION: the favourite's ask far from the 60s-TWAP-implied probability")
print("=" * 108)
print("   (uses the Coinbase 60s TWAP as a proxy for the resolution probability)")
print(f"  {'window':<16} {'gap band':<16} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'net$':>9} {'t':>6} {'CI95':>17}")
for t0, t1 in ((0, 300), (150, 300), (240, 300)):
    for g0, g1 in ((0.05, 1.0), (0.10, 1.0), (0.05, 0.20), (-1.0, -0.05)):
        sel = []
        for wi, d in enumerate(W):
            t = d["t"]
            i = int(np.searchsorted(t, t0))
            if i >= len(t):
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up = 0.5 * (mu + (1.0 - md))
            for side in ("Up", "Down"):
                a = d[side + "_ask"][i]
                if not np.isfinite(a):
                    continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                if mkt < 0.5:
                    continue
                # implied prob of the favourite from the coinbase TWAP proxy
                cb = d["cb_px"][i]
                if not np.isfinite(cb) or cb <= 0:
                    continue
                # the TWAP-implied probability needs a strike; use the window
                # open price as the strike proxy
                strike = d["cb_px"][0]
                if not np.isfinite(strike) or strike <= 0:
                    continue
                # d1 proxy: 60s TWAP vs strike, in units of the trailing vol
                rel = (cb - strike) / strike
                p_fav = 0.5 * (1 + rel * 12.0)   # crude, monotone map
                p_fav = min(0.995, max(0.005, p_fav))
                gap = mkt - p_fav
                if g0 <= gap < g1 and a < 0.99:
                    sel.append(dict(wi=wi, T=float(t0), side=side, ask=float(a),
                                    mkt=float(mkt), won=1 if d["_winner"] == side else 0))
        sel = first_per_window(sel)
        if len(sel) < 15:
            continue
        v = stat(sel)
        tr = stat([r for r in sel if r["wi"] < CUT])
        te = stat([r for r in sel if r["wi"] >= CUT])
        if not v or not tr or not te:
            continue
        print(f"  T[{t0},{t1}){'':<3} [{g0:+.2f},{g1:+.2f})   {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} "
              f"{v['wins']/v['n']:>7.4f} {v['net']:>+9.2f} {v['t']:>+6.2f} [{v['lo']:>+5.2f},{v['hi']:>+5.2f}]"
              f"   train {tr['net']:>+7.2f} / oos {te['net']:>+7.2f}")
print("\nDONE")
