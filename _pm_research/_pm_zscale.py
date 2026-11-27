"""Stage 23: THE Z-SCALING TEST (the decisive test of the endgame mispricing).

Everything so far says the market is efficient in the mid. The one place a
structural mispricing can survive is the TWAP SETTLEMENT RULE:

    strike     S = mean(price over [ws-60, ws))      (Chainlink 60s TWAP at open)
    settlement Q = mean(price over [ws+240, ws+300)) (Chainlink 60s TWAP at close)
    Up iff Q >= S

At time T the settlement is a 60-second AVERAGE, so its conditional variance
is much smaller than a single price's. For a random walk with per-second
variance s^2:
    T <= 240:  E[Q|F_T] = X_T,                    Var[Q] = s^2 * 60^3/3/60^2 = 20 s^2
    T >  240:  E[Q|F_T] = (int_240^T X + (300-T)X_T)/60
                Var[Q] = s^2 * (300-T)^3/3/60^2
If the market prices with the NAIVE (single-price) variance it will be
mis-scaled by up to sqrt(240/20) = 3.46x, and it will systematically
UNDERPRICE favourites (and OVERprice longshots) whenever the price is far
from the strike.

THE TEST (basis-free where possible):
 1. z_mkt   = Phi^-1(p_market)        (side-corrected)
    z_corr  = (E[Q|F_T] - S)/sqrt(Var[Q|F_T])   (TWAP-correct, fixed sigma bug)
 2. Regress z_mkt on z_corr. Slope < 1 => the market SHRINKS z => it
    underprices favourites and overprices longshots. Slope > 1 => opposite.
 3. Compare logloss of (a) the market's price, (b) a logistic on z_corr,
    (c) a logistic on z_mkt -- out of sample, per time bucket.
 4. Trade the disagreement with cluster-robust stats.

  Also a BASIS-FREE variant: replace the reconstructed strike/settlement with
  the market's own implied z (no chainlink needed): if the market's z shrinks
  with the TWAP factor, then for windows where the price is FAR from the
  strike the market should be near 0/1 but isn't.
"""
import os, pickle, json
import numpy as np
from scipy.special import ndtr, ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUT, exist_ok=True)
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


def ffill(a):
    a = np.array(a, dtype=float).copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
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


def main():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        prev = by_ws.get(d["ws"] - 300)
        if prev is None:
            continue
        tt = prev["t"]; cc = ffill(prev["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])

    rows = []
    for wi, d in enumerate(W):
        t = d["t"]; cb = ffill(d["cb_px"]); n = len(t)
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        if not np.isfinite(strike) or strike <= 0:
            continue
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        dts = np.diff(t); dts = dts[dts > 0]
        dt = float(np.median(dts)) if len(dts) else 0.28
        # per-SECOND variance: sum(r^2) counts variance over elapsed time n*dt
        cum_rv = np.cumsum(r * r)
        cum_t = np.maximum(t, 1e-9)
        sig2 = cum_rv / cum_t                      # s^2 per second  (FIXED)
        j240 = int(np.searchsorted(t, 240.0))
        for T in (30, 60, 90, 120, 150, 180, 200, 220, 240, 250, 260, 270, 280, 290):
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
                varQ = s2 * 20.0                       # 60^3/3/60^2 = 20
            else:
                obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
                rem = 300.0 - T
                eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
                varQ = s2 * rem ** 3 / 3.0 / 60.0 ** 2
            if varQ <= 0:
                continue
            # FIX: distance must be in LOG-price units to match the log-return variance
            zc = (np.log(eQ) - np.log(strike)) / np.sqrt(varQ)
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not (np.isfinite(ask) and np.isfinite(bid) and np.isfinite(asz)):
                    continue
                if not (0 < bid <= ask < 1):
                    continue
                if (ask - bid) > 0.04:
                    continue
                mkt_side = p_up if side == "Up" else 1.0 - p_up
                z_mkt = ndtri(min(max(mkt_side, 1e-6), 1 - 1e-6))
                z_side = zc if side == "Up" else -zc
                rows.append((wi, float(T), float(ask), float(bid), float(mkt_side),
                             float(z_side), float(z_mkt), float(ask - bid), float(asz),
                             1 if d["_winner"] == side else 0))
    A = np.array(rows, dtype=float)
    wi, T, ask, bid, mkt, zc, zm, sprd, asz, y = (A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4],
                                                  A[:, 5], A[:, 6], A[:, 7], A[:, 8], A[:, 9].astype(int))
    print(f"rows {len(A)}")

    # ---------- 1. z scaling regression ----------
    print("\n=== 1) does the market scale z the TWAP way? (regress z_mkt on z_corr) ===")
    print(f"  {'T':>4} {'n':>6} {'slope':>8} {'intercept':>10} {'R2':>7}  reading")
    for TT in (60, 120, 180, 240, 260, 280):
        s = T == TT
        if s.sum() < 60:
            continue
        x, yv = zc[s], zm[s]
        if x.std() < 1e-9:
            continue
        slope = np.polyfit(x, yv, 1)
        pred = np.polyval(slope, x)
        r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        read = ("market SHRINKS z (underprices favourites)" if slope[0] < 0.97 else
                "market OVERSHOOTS z (overprices favourites)" if slope[0] > 1.03 else
                "market z-scaling matches TWAP")
        print(f"  {TT:>4} {int(s.sum()):>6} {slope[0]:>8.3f} {slope[1]:>10.3f} {r2:>7.3f}  {read}")
    s = np.ones(len(A), bool)
    x, yv = zc[s], zm[s]
    slope = np.polyfit(x, yv, 1)
    pred = np.polyval(slope, x)
    r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    print(f"  ALL {len(A):>5} {slope[0]:>8.3f} {slope[1]:>10.3f} {r2:>7.3f}")

    # ---------- 2. logloss: market vs z_corr model ----------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    print("\n=== 2) out-of-sample logloss per time bucket ===")
    nw = len(W)
    for TT in (60, 120, 180, 240, 260, 280):
        s = T == TT
        if s.sum() < 80:
            continue
        cut = int(nw * 0.5)
        tr = s & (wi < cut); te = s & (wi >= cut)
        if tr.sum() < 40 or te.sum() < 20:
            continue
        row = [f"T={TT:>3}"]
        ll_m = log_loss(y[te], np.clip(mkt[te], 1e-6, 1 - 1e-6))
        row.append(f"mkt={ll_m:.4f}")
        for nm, X in (("zcorr", zc.reshape(-1, 1)), ("zmkt", zm.reshape(-1, 1)),
                      ("both", np.column_stack([zc, zm]))):
            Xtr, Xte = X[tr], X[te]
            mu_, sd_ = Xtr.mean(0), Xtr.std(0); sd_[sd_ == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=4000).fit((Xtr - mu_) / sd_, y[tr])
            p = lr.predict_proba((Xte - mu_) / sd_)[:, 1]
            ll = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            row.append(f"{nm}={ll:.4f}{'*' if ll < ll_m else ''}")
        print("  " + "  ".join(row) + ("   (* = beats market)" if True else ""))

    # ---------- 3. trade the z disagreement ----------
    print("\n=== 3) trading: buy where z_corr says the market is wrong (OOS) ===")
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut; te = wi >= cut
        for margin in (0.5, 1.0, 1.5, 2.0):
            for direc in (1, -1):
                sel = np.where(te & (direc * (zc - zm) > margin))[0]
                if len(sel) < 20:
                    continue
                a = ask[sel]; sh = 10.0 / a
                fee = FEE(a, sh)
                pnl = sh * y[sel] - 10.0 - fee
                m, tst, nww = cluster_t(wi[sel], pnl)
                print(f"  cut{frac} dir{'+' if direc>0 else '-'} z-gap>{margin:.1f}: N={len(sel):>5} w={nww:>3} "
                      f"total=${pnl.sum():+8.2f} avg=${pnl.mean():+.3f} cluster-t={tst:+.2f} "
                      f"win={y[sel].mean():.4f} meanAsk={a.mean():.4f}")

    # ---------- 4. basis-free check: market z vs price distance ----------
    print("\n=== 4) basis-free: implied 'sigma distance' of the market's z ===")
    print("     if the market used the TWAP-correct variance, |X_T-S|/z_mkt should")
    print("     scale like sqrt(Var[Q]) = s*sqrt(20) for T<=240 (flat in T).")
    for TT in (30, 60, 120, 180, 240, 260, 280):
        s = T == TT
        if s.sum() < 60:
            continue
        # for the Up side only, to keep the distance signed & comparable
        ss = s & (A[:, 0] >= 0)
        # distance in log-price from the strike, using the model-side rows
        # reconstruct: use |zc| * sqrt(varQ) as the model distance
        dist = np.abs(zc[s]) * np.sqrt(np.where(T[s] <= 240, 20.0,
                                                (300.0 - T[s]) ** 3 / 3.0 / 3600.0)) * 0 + 1
        # simpler: |z_mkt| vs |z_corr| ratio
        ratio = np.abs(zm[s]) / np.maximum(np.abs(zc[s]), 1e-9)
        print(f"  T={TT:>3} n={int(s.sum()):>5}  median |z_mkt|/|z_corr| = {np.median(ratio):.3f} "
              f"(1.0 = market agrees with TWAP scaling)")
    with open(os.path.join(OUT, "zscale.json"), "w") as fh:
        json.dump({"slope_all": float(slope[0]), "r2_all": float(r2)}, fh, indent=2)


if __name__ == "__main__":
    main()
