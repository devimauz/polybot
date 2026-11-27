"""Stage 20: THE ENDGAME TEST (final decisive test).

Everything taker-side has lost because the market is efficient at MID-PRICE
calibration. But two things are still untested properly:

 (1) MY z_correct was computed with a WRONG vol estimator (per-snapshot
     variance divided by elapsed seconds, inflating sigma^2 by ~1/dt).
     Fixed here: sigma^2_sec = sum(r^2)/(n*dt).
 (2) The ENDGAME: with the 60s-TWAP settlement rule, the outcome becomes
     nearly determined before the market prices it to 1.00. The taker fee at
     p=0.97 is 0.0002/share -- essentially free. So the endgame is the ONLY
     place where a mispricing can survive fees.

SETUP
  strike     S = mean(price over [ws-60, ws))        (reconstructed)
  settlement Q = mean(price over [ws+240, ws+300))  (reconstructed)
  Up iff Q >= S.
  At time t, with X = price path (coinbase proxy):
    t <= 240: E[Q] = X_t,           Var[Q] = sigma^2 * 60^3/3/60^2 = 20 sigma^2
    t >  240: E[Q] = (int_240^t X + (300-t) X_t)/60
               Var[Q] = sigma^2 * (300-t)^3/3/60^2
  p_model = Phi((E[Q] - S)/sqrt(Var[Q]))

TESTS
  A. Is p_model better calibrated than the market's price? (logloss, walk-fwd)
  B. Endgame grid: time x price band, cluster-robust t, holdout, drop-best-k.
  C. Both fee schedules (official 0.07 linear, and 0.25*exponent-2 variant).
"""
import os, pickle, json
import numpy as np
from scipy.special import ndtr

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUT, exist_ok=True)

FEE_A = lambda p, sh: 0.07 * p * (1 - p) * sh          # official crypto taker
FEE_B = lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh    # 5m-binary variant
STRIKE = 0.0


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
        t = prev["t"]; cb = ffill(prev["cb_px"])
        m = (t >= 240.0) & np.isfinite(cb)
        if m.sum() >= 10:
            pre[d["ws"]] = (t[m] - 300.0, cb[m])

    # ---- build per-(window,T,side) records ----
    recs = []
    for wi, d in enumerate(W):
        t = d["t"]; cb = ffill(d["cb_px"]); n = len(t)
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]
            m = tt >= -60.0
            strike = float(np.mean(pp[m])) if m.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        if not np.isfinite(strike) or strike <= 0:
            continue
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        dts = np.diff(t); dts = dts[dts > 0]
        dt = float(np.median(dts)) if len(dts) else 0.28
        # per-second variance: sum(r^2)/(n*dt)
        cum_rv = np.cumsum(r * r)
        cum_t = np.maximum(t, 1e-9)
        sig2 = cum_rv / np.maximum(cum_t / dt, 1e-9)     # sigma^2 per second
        j240 = int(np.searchsorted(t, 240.0))
        for T in (60, 120, 180, 210, 230, 245, 255, 265, 275, 285, 292):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up_mkt = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up_mkt < 0.998):
                continue
            s2 = float(sig2[i])
            if not np.isfinite(s2) or s2 <= 0:
                continue
            if T <= 240:
                eQ = cb[i]
                varQ = s2 * 60.0 ** 3 / 3.0 / 60.0 ** 2      # = 20 s2
            else:
                obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
                rem = 300.0 - T
                eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
                varQ = s2 * rem ** 3 / 3.0 / 60.0 ** 2
            if varQ <= 0:
                continue
            z = (eQ - strike) / np.sqrt(varQ)
            p_model_up = float(ndtr(z))
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                if not (np.isfinite(ask) and np.isfinite(bid)) or not (0 < bid <= ask < 1):
                    continue
                mkt_side = p_up_mkt if side == "Up" else 1.0 - p_up_mkt
                mod_side = p_model_up if side == "Up" else 1.0 - p_model_up
                recs.append((wi, float(T), float(ask), float(bid), float(mkt_side), float(mod_side),
                             1 if d["_winner"] == side else 0))
    A = np.array(recs, dtype=float)
    wi, T, ask, bid, mkt, mod, y = (A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5],
                                    A[:, 6].astype(int))
    print(f"rows {len(A)}  windows {len(W)}")

    # ---------- A. calibration & logloss ----------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    nw = len(W)
    print("\n=== A) does the TWAP settlement model beat the market's price? ===")
    for TT in (60, 120, 180, 230, 255, 275, 285):
        s = T == TT
        if s.sum() < 40:
            continue
        ll_m = log_loss(y[s], np.clip(mkt[s], 1e-6, 1 - 1e-6))
        ll_c = log_loss(y[s], np.clip(mod[s], 1e-6, 1 - 1e-6))
        print(f"  T={TT:>3} n={int(s.sum()):>4}  LL market={ll_m:.4f}  LL twap-model={ll_c:.4f}  "
              f"{'MODEL BETTER' if ll_c<ll_m else 'market better'}")
    print("  walk-forward (all features):")
    X = np.column_stack([mkt, mod, ask, ask - bid])
    for frac in (0.4, 0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut; te = wi >= cut
        mu_, sd_ = X[tr].mean(0), X[tr].std(0); sd_[sd_ == 0] = 1
        lr = LogisticRegression(C=1.0, max_iter=4000).fit((X[tr] - mu_) / sd_, y[tr])
        p = lr.predict_proba((X[te] - mu_) / sd_)[:, 1]
        print(f"    cut{frac}: LL market={log_loss(y[te], np.clip(mkt[te],1e-6,1-1e-6)):.4f} "
              f"LL model={log_loss(y[te], np.clip(p,1e-6,1-1e-6)):.4f}")

    # ---------- B. endgame grid ----------
    print("\n=== B) ENDGAME grid: buy the FAVOURITE at the ask, hold to settlement ===")
    print("    (fee_A = 0.07 p(1-p) official;  fee_B = 0.25 (p(1-p))^2 variant)")
    TB = [(230, 250), (245, 265), (255, 275), (265, 285), (275, 300)]
    PB = [(0.90, 0.97), (0.93, 0.99), (0.95, 1.01), (0.97, 1.01)]
    grid = []
    for tlo, thi in TB:
        cells = []
        for plo, phi in PB:
            s = (T >= tlo) & (T < thi) & (ask >= plo) & (ask < phi)
            if s.sum() < 8:
                cells.append("n<8"); continue
            # favourite only: the row whose ask is the higher one -> both rows
            # present; keep rows where this side is the favourite
            fav = s & (ask >= 0.5)
            # dedupe: for each (window,T) take the row with the larger ask
            key = {}
            for k in np.where(fav)[0]:
                kk = (wi[k], T[k])
                if kk not in key or ask[k] > ask[key[kk]]:
                    key[kk] = k
            idx = np.array(list(key.values()))
            if len(idx) < 8:
                cells.append("n<8"); continue
            a = ask[idx]; yy = y[idx]
            sh = 10.0 / a
            pnlA = sh * yy - 10.0 - FEE_A(a, sh)
            pnlB = sh * yy - 10.0 - FEE_B(a, sh)
            m, tst, nww = cluster_t(wi[idx], pnlA)
            cells.append(f"n{len(idx):>3} w{yy.mean():.3f} ${pnlA.sum():+7.1f} t{tst:+5.2f}")
            grid.append((tlo, thi, plo, phi, len(idx), pnlA, pnlB, wi[idx], yy, a))
        print(f"  t[{tlo},{thi}): " + " | ".join(f"{c:>26}" for c in cells))
    print("    (bands: 0.90-0.97 | 0.93-0.99 | 0.95-1.01 | 0.97-1.01)")

    # ---------- C. best cells: holdout + drop-best-k + both fees ----------
    print("\n=== C) detail on the promising cells ===")
    for (tlo, thi, plo, phi, n, pnlA, pnlB, w2, yy, a) in grid:
        m, tst, nww = cluster_t(w2, pnlA)
        if tst < 1.0 or n < 20:
            continue
        order = np.argsort(-pnlA)
        keep = np.ones(len(pnlA), bool); keep[order[:3]] = False
        m2, t2, _ = cluster_t(w2[keep], pnlA[keep])
        wss = np.unique(w2); cut = wss[len(wss) // 2]
        A1 = pnlA[w2 <= cut]; B1 = pnlA[w2 > cut]
        print(f"  t[{tlo},{thi}) ask[{plo},{phi}) n={n} win={yy.mean():.4f} meanAsk={a.mean():.4f}")
        print(f"     feeA total=${pnlA.sum():+.2f} avg=${pnlA.mean():+.3f} t={tst:+.2f} | "
              f"feeB total=${pnlB.sum():+.2f} avg=${pnlB.mean():+.3f}")
        print(f"     holdout: first=${A1.sum():+.2f} (n{len(A1)}) second=${B1.sum():+.2f} (n{len(B1)}) | "
              f"drop-top3: ${pnlA[keep].sum():+.2f} t={t2:+.2f}")


if __name__ == "__main__":
    main()
