"""Stage 19: THE VOL-SCALING TEST (decisive; fixes the z sign bug).

!! SUPERSEDED - DO NOT TRUST THE NUMBERS THIS SCRIPT PRINTS !!
This script has a known UNITS BUG: it compares an ABSOLUTE price distance
against a LOG-RETURN variance and omits the transport term, so the
"ratio(mkt/TWAPscale)" column in section 3 is meaningless (it prints values
like 257490).  Its z also explodes near expiry, which is what invalidated
the "48/48 model-confirmed" rule.
Use _pm_zscale.py (transport term added) and _pm_rule.py (spot-rule vs
TWAP-rule) instead - they are the corrected versions.
The one conclusion that SURVIVES here is section 1: corr(mkt, y) exceeds
corr(z_correct, y) at EVERY horizon, and the model's z collapses at T=270
while the market's does not.  The market beats the model, so a model-based
entry is not the source of the edge - the PRICE is.

CONFIRMED RESOLUTION RULE (Polymarket 5-min crypto, since 2026-08-07):
    strike     = Chainlink 60s TWAP value AT THE OPEN  = mean(price over [ws-60, ws))
    settlement = Chainlink 60s TWAP value AT THE CLOSE= mean(price over [ws+240, ws+300))
    Up iff settlement >= strike.
  (Reconstructed from the logs: 263/269 = 97.8% agreement, vs 91.1% for final spot.)

THE HYPOTHESIS (H-scale):
  The settlement is an AVERAGE of 60 prices, so its conditional variance is NOT
  the variance of a single price. For a diffusive price with per-second
  variance sigma^2, at time T < 240:
        E[M | F_T] = X_T                        (martingale)
        Var[M | F_T] = sigma^2 * 60/3 = 20 sigma^2
  and for 240 < T < 300 (partial observation of the settlement window):
        E[M | F_T] = (int_240^T X + (300-T)*X_T)/60
        Var[M | F_T] = sigma^2 * (300-T)^2 / 3
  A participant who normalises the distance-to-strike by SPOT vol
  (sigma*sqrt(300-T)) rather than by the TWAP scale is mis-scaled by up to
  ~3.5x early in the window -> the market's price is too close to 0.5 when the
  true probability is far from 0.5 (or vice versa). That is tradeable.

TEST:
  1. For every window and a grid of T, compute p_correct = Phi(z_correct) with
     the TWAP-correct conditional mean/variance, using the SIDE-CORRECT sign.
  2. Measure corr(z_correct, y) and compare with corr(z_naive, y) and the
     market's own price. (This is where the previous run had a sign bug.)
  3. Back out the market's IMPLIED vol per T and compare with sigma*sqrt(20)
     (TWAP-correct) and sigma*sqrt(300-T) (spot-naive). If the market tracks
     the former, H-scale is dead; if it tracks the latter, the market is
     mis-scaled and we have a real edge.
  4. Trade p_correct vs the ask where they disagree, walk-forward, with depth
     and spread filters, cluster-robust stats.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUT, exist_ok=True)
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
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


def norm_cdf(x):
    from scipy.special import ndtr
    return ndtr(x)


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


def main():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    print(f"windows {len(W)}")

    # ---- pre-window 60s (for the strike) ----
    pre = {}
    for d in W:
        prev = by_ws.get(d["ws"] - 300)
        if prev is None:
            continue
        t = prev["t"]; cb = ffill(prev["cb_px"])
        m = (t >= 240.0) & np.isfinite(cb)
        if m.sum() < 10:
            continue
        pre[d["ws"]] = (t[m] - 300.0, cb[m])

    recs = []          # per (window, T, side)
    implied = []       # market implied sigma per (window,T)
    corr_rows = []
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
        # per-second vol: realised variance of log returns / elapsed seconds
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        dt = np.median(np.diff(t)) if n > 1 else 0.28
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.28
        cum_rv = np.cumsum(r * r)
        cum_t = np.maximum(t, 1e-9)
        sig2 = cum_rv / cum_t                      # per-second variance
        j240 = int(np.searchsorted(t, 240.0))
        for T in (30, 60, 90, 120, 150, 180, 210, 240, 260, 270, 280):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up_mkt = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up_mkt < 0.998):
                continue
            s2 = sig2[i]                            # per-second variance
            if not np.isfinite(s2) or s2 <= 0:
                continue
            sig = np.sqrt(s2)
            # --- TWAP-correct conditional mean/variance of the settlement ---
            if T <= 240:
                eM = cb[i]
                varM = s2 * 60.0 / 3.0
            else:
                obs = cb[j240:i + 1]
                obs = obs[np.isfinite(obs)]
                rem = 300.0 - T
                eM = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
                varM = s2 * rem * rem / 3.0
            if varM <= 0:
                continue
            z_corr = (eM - strike) / np.sqrt(varM)          # Up-side z
            # naive: distance measured in spot vol over the remaining horizon
            z_naive = (cb[i] - strike) / (sig * np.sqrt(max(300.0 - T, 1.0)))
            p_corr_up = norm_cdf(np.array([z_corr]))[0]
            p_naive_up = norm_cdf(np.array([z_naive]))[0]
            # market implied vol (from p_up_mkt, invert with the TWAP-correct scale)
            from statistics import NormalDist
            ND = NormalDist()
            try:
                z_mkt = ND.inv_cdf(min(max(p_up_mkt, 1e-6), 1 - 1e-6))
            except Exception:
                continue
            implied.append((T, abs(cb[i] - strike) / max(abs(z_mkt), 1e-9) / max(np.sqrt(varM), 1e-12) * np.sqrt(varM),
                            sig, z_mkt, cb[i] - strike))
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                if not (np.isfinite(ask) and np.isfinite(bid)) or not (0 < bid <= ask < 1):
                    continue
                mkt_side = p_up_mkt if side == "Up" else 1.0 - p_up_mkt
                corr_side = z_corr if side == "Up" else -z_corr
                naive_side = z_naive if side == "Up" else -z_naive
                recs.append((wi, T, float(ask), float(bid), float(mkt_side),
                             float(corr_side), float(naive_side),
                             1 if d["_winner"] == side else 0))

    A = np.array(recs, dtype=float)
    wi, T, ask, bid, mkt, zc, zn, y = (A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4],
                                       A[:, 5], A[:, 6], A[:, 7].astype(int))
    print(f"rows {len(A)}")
    print("\n=== 1) predictive power (side-correct z) ===")
    for TT in (60, 120, 180, 240, 270):
        s = T == TT
        if s.sum() < 40:
            continue
        print(f"  T={TT:>3} n={int(s.sum()):>5}  corr(z_correct,y)={np.corrcoef(zc[s], y[s])[0,1]:+.4f} "
              f"corr(z_naive,y)={np.corrcoef(zn[s], y[s])[0,1]:+.4f} corr(mkt,y)={np.corrcoef(mkt[s], y[s])[0,1]:+.4f}")

    # ---- 2. calibration of p_correct vs the market ----
    print("\n=== 2) calibration: p_correct (TWAP-correct) vs p_market ===")
    print(f"  {'T':>4} {'n':>6} {'bin':>16} {'p_correct':>10} {'p_market':>10} {'diff':>8}")
    for TT in (60, 120, 180, 240):
        s = T == TT
        if s.sum() < 40:
            continue
        pc = norm_cdf(zc[s])
        pm = mkt[s]
        edges = np.quantile(pc, [0, .25, .5, .75, 1.0])
        for k in range(4):
            b = (pc >= edges[k]) & (pc <= edges[k + 1] if k == 3 else pc < edges[k + 1])
            if b.sum() < 8:
                continue
            print(f"  {TT:>4} {int(b.sum()):>6} [{edges[k]:.3f},{edges[k+1]:.3f}] "
                  f"{pc[b].mean():>10.4f} {pm[b].mean():>10.4f} {pm[b].mean()-pc[b].mean():>+8.4f}")

    # ---- 3. implied vol scaling ----
    print("\n=== 3) market implied vol vs the two candidate scalings ===")
    IM = np.array(implied, dtype=float)
    for TT in (60, 120, 180, 240, 270):
        s = IM[:, 0] == TT
        if s.sum() < 20:
            continue
        sig = IM[s, 2]
        # distance in sigmas the market implies
        z_mkt = IM[s, 3]
        dist = np.abs(IM[s, 4])
        # market's implied "sigma" if it used the TWAP scale:  dist/sqrt(varM)
        sM = IM[s, 1]
        print(f"  T={TT:>3} n={int(s.sum()):>4}  median implied sigma={np.median(sM):.3e}  "
              f"sigma*sqrt(20)={np.median(sig)*np.sqrt(20):.3e}  "
              f"sigma*sqrt(300-T)={np.median(sig)*np.sqrt(300-TT):.3e}  "
              f"ratio(mkt/TWAPscale)={np.median(sM)/(np.median(sig)*np.sqrt(20)):.3f}")

    # ---- 4. trade p_correct vs the market ----
    print("\n=== 4) trading p_correct vs the market (walk-forward, depth+spread filters) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    nw = len(W)
    ALL = np.column_stack([mkt, zc, zn, ask, ask - bid])
    for name, cols in {"market only": [0], "market+z_corr": [0, 1], "z_corr only": [1], "all": [0, 1, 2, 3, 4]}.items():
        X = ALL[:, cols]
        line = []
        for frac in (0.4, 0.5, 0.6):
            cut = int(nw * frac)
            tr = wi < cut; te = wi >= cut
            mu_, sd_ = X[tr].mean(0), X[tr].std(0); sd_[sd_ == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=4000).fit((X[tr] - mu_) / sd_, y[tr])
            p = lr.predict_proba((X[te] - mu_) / sd_)[:, 1]
            ll_m = log_loss(y[te], np.clip(mkt[te], 1e-6, 1 - 1e-6))
            ll_c = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            line.append(f"cut{frac}: mkt={ll_m:.4f} mod={ll_c:.4f}{' MODEL' if ll_c<ll_m else ''}")
        print(f"  {name:<16} " + " | ".join(line))

    # direct: trade where p_correct disagrees with the market's ask-implied prob
    print("\n  --- direct p_correct vs market (no model) ---")
    pc_all = norm_cdf(zc)
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        te = wi >= cut
        for margin in (0.05, 0.10, 0.20):
            for direc in (1, -1):
                sel = np.where(te & (direc * (pc_all - mkt) > margin))[0]
                if len(sel) < 25:
                    continue
                a = ask[sel]
                sh = 10.0 / a
                fee = FEE(a, sh)
                pnl = sh * y[sel] - 10.0 - fee
                w2 = wi[sel]
                uq = np.unique(w2)
                s_ = np.array([pnl[w2 == w].sum() for w in uq])
                c_ = np.array([(w2 == w).sum() for w in uq])
                tst = (s_.sum() / c_.sum()) / (s_.std(ddof=1) / np.sqrt(len(uq)))
                print(f"    cut{frac} dir{'+' if direc>0 else '-'} margin>{margin:.2f}: N={len(sel):>4} w={len(uq):>3} "
                      f"total=${pnl.sum():+8.2f} avg=${pnl.mean():+.3f} cluster-t={tst:+.2f} "
                      f"win={y[sel].mean():.4f} meanAsk={a.mean():.4f}")


if __name__ == "__main__":
    main()
