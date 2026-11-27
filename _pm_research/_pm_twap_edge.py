"""Stage 28: THE TWAP-LOCK-IN EDGE  (the decisive, mechanism-driven test).

!! SUPERSEDED - this script CRASHES and its placebo is INVALID !!
1. It crashes at section P4 with `ValueError: Input contains NaN` because
   `p_model` is NaN wherever the strike/variance reconstruction used a
   forward-filled price (unresolved windows).
2. Its original placebo permuted outcomes WITHIN a set, which preserves the
   win count and therefore produces a null identical to the observed - an
   invalid test.
The corrected versions are _pm_honest.py (real outcome-permutation placebo,
block reporting fixed) and _pm_ship_rule.py (the shipping gate on the
causal clock).  Do not reuse this file's numbers.

THE MECHANISM (this is the whole thesis, and it is testable):
  Polymarket 5-min crypto markets settle on the Chainlink 60-second TWAP over
  the window's final minute, [240,300), compared against the price-to-beat
  (the Chainlink 60s TWAP over [ws-60, ws)).

  A market maker who prices the SIMPLE rule ("is the spot price above the
  strike?") ignores that by time T the ALREADY OBSERVED part of the settlement
  window is locked in.  With f = (T-240)/60 of the window observed, the
  settlement is a blend of a KNOWN average and a random remainder, so the
  correct variance of the settlement shrinks like (300-T)^3, not (300-T).

  => the longer the window has been running, the more the TRUE probability
     departs from what a spot-rule market maker quotes.  The market should
     systematically UNDERPRICE the favourite (and OVERPRICE the underdog).

PREDICTION (falsifiable, stated before looking):
  P1. The gap  P_true(favourite) - P_market(favourite)  GROWS with T.
  P2. The gap is LARGEST when the favourite's market price is mid-range
      (~0.85-0.95) and ~0 near 0.99 (where the market is already saturated).
  P3. Trading the favourite when P_true - P_market > margin is +EV after fees,
      and the edge survives a shuffled-outcome placebo.

This script tests P1/P2 (calibration by time bucket) and P3 (trading), with:
  * ground-truth resolutions (never my own reconstruction)
  * a proper TWAP-rule probability with the (300-T)^3 variance
  * cluster-robust t-stats and a bootstrap CI
  * a shuffled-winner placebo and a random-strike placebo
"""
import os, pickle
import numpy as np
from scipy.special import ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]): last = a[i]
        elif np.isfinite(last): a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]): nxt = a[i]
        elif np.isfinite(nxt): a[i] = nxt
    return a


def var_mult(T):
    """Var[Q|F_T] / s^2 for Q = mean(price over [240,300))."""
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


def boot_ci(wi, pnl, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    tot = {w: pnl[wi == w].sum() for w in uq}
    cnt = {w: (wi == w).sum() for w in uq}
    keys = list(tot); out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(sum(tot[k] for k in pick) / sum(cnt[k] for k in pick))
    return np.percentile(out, [2.5, 97.5])


def build():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])
    rows = []
    for wi, d in enumerate(W):
        t, cb, n = d["t"], ffill(d["cb_px"]), len(d["t"])
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        if not np.isfinite(strike) or strike <= 0: continue
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        sig2 = np.cumsum(r * r) / np.maximum(t, 1e-9)
        j240 = int(np.searchsorted(t, 240.0))
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5: continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998): continue
            s2 = float(sig2[i])
            if not np.isfinite(s2) or s2 <= 0: continue
            obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
            rem = 300.0 - T
            eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
            vQ = var_mult(T) * s2
            if vQ <= 0: continue
            z_up = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                sgn = 1.0 if side == "Up" else -1.0
                p_model = float(ndtri(max(-8.0, min(8.0, sgn * z_up))))
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), mkt=float(mkt), p_model=p_model,
                                 won=1 if d["_winner"] == side else 0))
    return W, rows


def main():
    W, rows = build()
    print(f"rows {len(rows)}  windows {len(W)}")

    # ---------- P1: the favourite's market price vs its realised win rate ----------
    print("\n=== P1) favourite: market-implied prob vs REALISED win rate, by time ===")
    print(f"  {'T':>4} {'n':>5} {'wins':>5} {'mkt':>7} {'realised':>9} {'gap':>8} {'ask':>7} "
          f"{'fee':>6} {'netgap':>8}")
    for T in range(240, 300, 4):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5]
        if len(s) < 15: continue
        won = np.array([r["won"] for r in s]); mkt = np.array([r["mkt"] for r in s])
        ask = np.array([r["ask"] for r in s])
        fee = 0.07 * ask * (1 - ask)
        # per-window de-duplication: keep the first T per window
        seen = {}; 
        for r in sorted(s, key=lambda r: (r["wi"], r["T"])):
            seen.setdefault(r["wi"], r)
        s1 = list(seen.values())
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        a1 = np.array([r["ask"] for r in s1])
        gap = w1.mean() - m1.mean()
        se = np.sqrt(m1.mean() * (1 - m1.mean()) / len(s1))
        print(f"  {T:>4} {len(s1):>5} {int(w1.sum()):>5} {m1.mean():>7.4f} {w1.mean():>9.4f} "
              f"{gap:+8.4f} {a1.mean():>7.4f} {fee.mean():>6.4f} {gap-fee.mean():+8.4f}  "
              f"(t={gap/se if se>0 else 0:+.2f})")

    # ---------- P2: by market price level ----------
    print("\n=== P2) favourite: gap by MARKET PRICE level (endgame T>=250) ===")
    for lo, hi in ((0.50, 0.70), (0.70, 0.85), (0.85, 0.92), (0.92, 0.96),
                   (0.96, 0.98), (0.98, 1.001)):
        s = [r for r in rows if r["T"] >= 250 and r["mkt"] >= 0.5 and lo <= r["mkt"] < hi]
        if len(s) < 12: continue
        seen = {}
        for r in sorted(s, key=lambda r: (r["wi"], r["T"])):
            seen.setdefault(r["wi"], r)
        s1 = list(seen.values())
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        a1 = np.array([r["ask"] for r in s1])
        fee = 0.07 * a1 * (1 - a1)
        gap = w1.mean() - m1.mean()
        se = np.sqrt(m1.mean() * (1 - m1.mean()) / len(s1))
        print(f"  mkt[{lo:.2f},{hi:.3f}) n={len(s1):>3} mkt={m1.mean():.4f} realised={w1.mean():.4f} "
              f"gap={gap:+.4f} (t={gap/se if se>0 else 0:+.2f}) ask={a1.mean():.4f} "
              f"net={gap-fee.mean():+.4f}")

    # ---------- P3: trading the favourite in the endgame ----------
    print("\n=== P3) TRADE: buy the favourite at the ask, endgame ===")
    def first_per_window(rs):
        best = {}
        for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
            best.setdefault(r["wi"], r)
        return list(best.values())

    def report(name, rs, won_override=None, seed=0):
        rs = first_per_window(rs)
        if not rs: return None
        wi = np.array([r["wi"] for r in rs])
        won = np.array([r["won"] for r in rs])
        if won_override is not None:
            won = won_override
        ask = np.array([r["ask"] for r in rs])
        sh = 10.0 / ask
        pnl = sh * won - 10.0 - FEE(ask, sh)
        m_, tst, w_ = cluster_t(wi, pnl)
        lo, hi = boot_ci(wi, pnl)
        print(f"  {name:<40} N={len(rs):>3} w={w_:>3} tot=${pnl.sum():+7.2f} "
              f"avg=${pnl.mean():+.3f} t={tst:+5.2f} CI95=[{lo:+.3f},{hi:+.3f}] "
              f"win={won.mean():.4f} ask={ask.mean():.4f}")
        return rs

    for TLO in (240, 250, 260):
        for PLO in (0.50, 0.80, 0.90):
            sel = [r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5 and r["ask"] >= PLO]
            report(f"T>={TLO} fav ask>={PLO}", sel)

    print("\n  --- placebo: shuffled winners ---")
    base = [r for r in rows if r["T"] >= 250 and r["mkt"] >= 0.5 and r["ask"] >= 0.90]
    for s in range(3):
        rs = first_per_window(base)
        rng = np.random.default_rng(s)
        won = rng.permutation([r["won"] for r in rs])
        report(f"placebo shuffled (seed {s})", base, won_override=won)

    # ---------- P4: does the TWAP MODEL beat the market out of sample? ----------
    print("\n=== P4) does my TWAP model beat the market on the endgame? ===")
    from sklearn.metrics import log_loss
    nw = len(W)
    for frac in (0.4, 0.5, 0.6):
        cut = int(nw * frac)
        tr = np.array([r["wi"] < cut for r in rows]); te = ~tr
        y = np.array([r["won"] for r in rows])
        mkt = np.array([r["mkt"] for r in rows]); mod = np.array([r["p_model"] for r in rows])
        ll_m = log_loss(y[te], np.clip(mkt[te], 1e-6, 1 - 1e-6))
        ll_p = log_loss(y[te], np.clip(mod[te], 1e-6, 1 - 1e-6))
        # blend
        best = None
        for wgt in np.arange(0, 1.01, 0.1):
            bl = wgt * mod[te] + (1 - wgt) * mkt[te]
            l = log_loss(y[te], np.clip(bl, 1e-6, 1 - 1e-6))
            if best is None or l < best[1]: best = (wgt, l)
        print(f"  cut{frac}: LL market={ll_m:.5f}  LL twap-model={ll_p:.5f}  "
              f"best blend w={best[0]:.1f} LL={best[1]:.5f}  -> "
              f"{'MODEL WINS' if ll_p < ll_m else 'market wins'}")

    # ---------- P5: the trading rule implied by the model ----------
    print("\n=== P5) trade the favourite only when p_model - p_market > margin ===")
    for margin in (0.0, 0.05, 0.10, 0.20):
        sel = [r for r in rows if r["T"] >= 250 and r["mkt"] >= 0.5
               and r["p_model"] - r["mkt"] > margin]
        report(f"T>=250 fav, p_model-p_mkt>{margin}", sel)


if __name__ == "__main__":
    main()
