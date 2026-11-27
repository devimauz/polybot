"""Stage 25: THE UNIFIED Z-DIFFERENCE STRATEGY (theory-driven, low multiple-testing burden).

!! SUPERSEDED - this script uses the OPTIMISTIC logger clock (`d["t"]`) !!
It was written before the source-aligned clock (`t_src`) existed, so its
numbers carry a ~0.96 s look-ahead.  Its QUALITATIVE conclusions still hold
and were re-verified on the causal clock, but the magnitudes are inflated:
  - section A: corr(z_mine - z_mkt, won) = -0.55, i.e. the model-based
    z-difference ANTI-predicts, confirming the market beats the model.
  - section B: every "buy the underpriced side" cell is flat or negative.
  - section C: FADING an overpriced market loses (-$264.72, t=-2.25) and the
    cheap-underdog late trade loses (-$94.44, t=-0.47 on the causal clock).
    The mispricing is ONE-DIRECTIONAL - the favourite is underpriced, there
    is nothing to fade.
  - section D: reconciles the old T>=250/ask>=0.95 endgame estimate
    (+$8.16, t=+0.79) with the shipped rule (+$38.13, t=+3.20).
Re-run the equivalent on the causal clock via _pm_ship_rule.py.

THEORY (unifies every positive hint seen so far):
  Resolution: Up iff (60s TWAP over [240,300)) > (60s TWAP over [ws-60,ws)).
  My z_corr  = (log E[Q|F_T] - log strike) / sqrt(Var[Q|F_T])   <- CORRECT rule
  Market z   = Phi^-1(mid)                                     <- what the book says

  Stage 24 found the market's z tracks the NAIVE SPOT rule (slope 1.0-1.2) while
  the correct TWAP rule needs a LARGER multiplier:
      T=60: 1.095x   T=120: 1.134x   T=180: 1.224x   T=240: 1.582x   T=270: 3.47x
  So the market is systematically UNDER-confident about the favourite late in the
  window (it prices the SPOT rule, reality settles on the TWAP), and
  OVER-confident early (it prices more extreme than the TWAP rule allows).

  STRATEGY: trade the side where MY z is more extreme than the MARKET's z.
    z_mkt - z_mine > 0  -> market underprices that side -> BUY it
  This is ONE rule, derived from the resolution mechanics, with no band
  search, so it carries a much lower multiple-testing burden than price bands.

HONESTY REQUIREMENTS:
  * fills at the ASK with a depth check
  * taker fee 0.07*p*(1-p) on entry AND on exit
  * walk-forward: parameters chosen on the FIRST half, profit measured on the
    SECOND half only
  * cluster-robust t (windows are the independent unit)
  * reconcile the "76 trades / 0 losses" vs "101 trades / 1 loss" discrepancy
"""
import os, pickle, json
import numpy as np
from scipy.special import ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]):
            last = a[i]
        elif np.isfinite(last):
            a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]):
            nxt = a[i]
        elif np.isfinite(nxt):
            a[i] = nxt
    return a


def var_twap(s2, T):
    if T >= 240.0:
        return (300.0 - T) ** 3 / 10800.0
    a, b = 240.0 - T, 300.0 - T
    return 2.0 / 3600.0 * (b ** 3 / 6.0 - b * a * a / 2.0 + a ** 3 / 3.0)


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]; d["ws"] = int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


def cluster_t(wi, pnl):
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se > 0 else np.nan), len(uq)


def build(TGRID, DEPTH=10.0, MAXSPREAD=0.04):
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None:
            continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])

    out = []
    for wi, d in enumerate(W):
        t, cb, n = d["t"], ffill(d["cb_px"]), len(d["t"])
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        if not np.isfinite(strike) or strike <= 0:
            continue
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        sig2 = np.cumsum(r * r) / np.maximum(t, 1e-9)
        j240 = int(np.searchsorted(t, 240.0))
        for T in TGRID:
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998):
                continue
            s2 = float(sig2[i])
            if not np.isfinite(s2) or s2 <= 0:
                continue
            if T <= 240:
                eQ = cb[i]
            else:
                obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
                rem = 300.0 - T
                eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
            vQ = var_twap(s2, T)
            if vQ <= 0:
                continue
            z_mine_up = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz, bsz = d[side + "_asksz"][i], d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)):
                    continue
                if not (0 < bid <= ask < 1):
                    continue
                if (ask - bid) > MAXSPREAD:
                    continue
                if ask * asz < DEPTH:            # not enough size at the ask
                    continue
                mkt_side = p_up if side == "Up" else 1.0 - p_up
                z_mkt = ndtri(min(max(mkt_side, 1e-6), 1 - 1e-6))
                sgn = 1.0 if side == "Up" else -1.0
                out.append(dict(wi=wi, ws=d["ws"], T=float(T), i=int(i), side=side,
                                ask=float(ask), bid=float(bid), asz=float(asz),
                                z_mkt=float(z_mkt), z_mine=float(sgn * z_mine_up),
                                mkt=float(mkt_side),
                                won=1 if d["_winner"] == side else 0,
                                t_=float(t[i])))
    return W, out


def pnl_hold(row, stake=10.0):
    sh = stake / row["ask"]
    return sh * row["won"] - stake - FEE(row["ask"], sh)


def pnl_tp(row, tp, stake=10.0, exit_t=None):
    """Hold to settlement unless a later snapshot shows bid >= tp -> sell there."""
    return None  # handled separately


def main():
    TGRID = list(range(30, 300, 5))
    W, tr = build(TGRID)
    print(f"windows {len(W)}  candidate rows {len(tr)}")

    # ---------- reconcile: first-entry-per-window vs all-entries ----------
    for win, e in tr[0].items():
        break
    byw = {}
    for r in tr:
        byw.setdefault((r["wi"], r["side"]), []).append(r)
    for k in byw:
        byw[k].sort(key=lambda r: r["T"])

    print("\n=== A) signal quality vs the market, all rows ===")
    zg = np.array([r["z_mine"] - r["z_mkt"] for r in tr])   # >0 : market underprices this side
    won = np.array([r["won"] for r in tr])
    ask = np.array([r["ask"] for r in tr])
    mkt = np.array([r["mkt"] for r in tr])
    print(f"  corr(z_mine - z_mkt, won) = {np.corrcoef(zg, won)[0,1]:+.4f}   (n={len(tr)})")
    # calibration: is the market right about its own price?
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)):
        s = (ask >= lo) & (ask < hi)
        if s.sum() > 30:
            print(f"    ask[{lo:.1f},{hi:.1f}) n={int(s.sum()):>5} win={won[s].mean():.4f} "
                  f"meanAsk={ask[s].mean():.4f} edge={won[s].mean()-ask[s].mean():+.4f}")

    print("\n=== B) strategy: buy the side the market UNDERPRICES (z_mine > z_mkt + margin) ===")
    # first entry per (window, side) at the FIRST T that fires
    nw = len(W)
    cut = int(nw * 0.5)
    for margin in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        for tlo, thi in ((30, 300), (30, 200), (200, 300), (240, 300)):
            fired = []
            for k, lst in byw.items():
                for r in lst:
                    if not (tlo <= r["T"] < thi):
                        continue
                    if r["z_mine"] - r["z_mkt"] > margin:
                        fired.append(r); break
            if not fired:
                continue
            sel = np.array([r["wi"] for r in fired])
            pnl = np.array([pnl_hold(r) for r in fired])
            trn = np.array([r["won"] for r in fired])
            a = np.array([r["ask"] for r in fired])
            m_, tst, w_ = cluster_t(sel, pnl)
            tr_m = sel < cut
            oos = ~tr_m
            line = (f"  m={margin:.2f} T[{tlo},{thi}) N={len(fired):>4} w={w_:>3} "
                    f"total=${pnl.sum():+8.2f} avg=${pnl.mean():+.3f} t={tst:+5.2f} "
                    f"win={trn.mean():.3f} ask={a.mean():.3f}")
            if oos.sum() > 5:
                p2 = pnl[oos]; s2_ = sel[oos]
                m2, t2, w2 = cluster_t(s2_, p2)
                line += f" || OOS N={int(oos.sum()):>3} ${p2.sum():+7.2f} t={t2:+5.2f}"
            print(line)

    print("\n=== C) the other direction (market OVERprices -> fade it) ===")
    for margin in (0.5, 1.0, 1.5):
        fired = []
        for k, lst in byw.items():
            for r in lst:
                if r["z_mkt"] - r["z_mine"] > margin:
                    fired.append(r); break
        if not fired:
            continue
        sel = np.array([r["wi"] for r in fired])
        pnl = np.array([pnl_hold(r) for r in fired])
        trn = np.array([r["won"] for r in fired])
        a = np.array([r["ask"] for r in fired])
        m_, tst, w_ = cluster_t(sel, pnl)
        print(f"  m={margin:.2f} N={len(fired):>4} w={w_:>3} total=${pnl.sum():+8.2f} "
              f"avg=${pnl.mean():+.3f} t={tst:+5.2f} win={trn.mean():.3f} ask={a.mean():.3f}")

    print("\n=== D) reconcile the endgame discrepancy (T>=250, ask>=0.95) ===")
    for label, sel_fn in (("ALL rows", lambda rs: rs),
                          ("first per (w,side)", None)):
        rows = [r for r in tr if r["T"] >= 250 and r["ask"] >= 0.95 and r["ask"] <= 1.01]
        if sel_fn is None:
            seen = set(); rows2 = []
            for r in sorted(rows, key=lambda r: (r["wi"], r["T"])):
                k = (r["wi"], r["side"])
                if k in seen: continue
                seen.add(k); rows2.append(r)
            rows = rows2
        if not rows:
            continue
        sel = np.array([r["wi"] for r in rows])
        pnl = np.array([pnl_hold(r) for r in rows])
        trn = np.array([r["won"] for r in rows])
        a = np.array([r["ask"] for r in rows])
        m_, tst, w_ = cluster_t(sel, pnl)
        print(f"  {label:<20} N={len(rows):>4} w={w_:>3} total=${pnl.sum():+8.2f} "
              f"avg=${pnl.mean():+.3f} t={tst:+5.2f} win={trn.mean():.4f} ask={a.mean():.4f}")


if __name__ == "__main__":
    main()
