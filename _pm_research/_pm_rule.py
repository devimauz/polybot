"""Stage 24: DOES THE MARKET PRICE THE TWAP RULE OR THE SPOT RULE?

BREAKTHROUGH FROM STAGE 23:
  My model's |z_mkt|/|z_corr| ratios by time bucket were
      T=60: 0.298   T=120: 0.368   T=180: 0.508   T=240: 1.178
  and sqrt(20 / Var_true(T)) is
      T=60: 0.316   T=120: 0.378   T=180: 0.500   T=240: 1.000
  -> an almost EXACT match.  The market is using the correct conditional
  variance of the 60s-TWAP settlement.  My stage-23 variance was missing the
  TRANSPORT term (the price must travel from T to 240 before the settlement
  window even opens):
      Var[Q|F_T] = s^2 * (1/60^2) * int_240^300 int_240^300 min(u-T, v-T) du dv
                 = s^2 * 2/3600 * [b^3/6 - b a^2/2 + a^3/3],  a=240-T, b=300-T
  With that fixed, z_corr and z_mkt should agree, and the MARKET IS EFFICIENT.

  BUT the same structure gives a second, sharper hypothesis (H-rule):
  if the market priced the SPOT rule (final price vs strike) instead of the
  TWAP rule, its z would be
      z_spot = (log X_T - log S) / (s * sqrt(300-T))
  which uses a DIFFERENT variance AND a different mean.  Testing which of the
  two explains z_mkt better is a clean, falsifiable discrimination.

THIS SCRIPT:
  1. Correct Var[Q|F_T] (transport term included) for T<=240.
  2. Regress z_mkt on z_twap vs z_spot; compare R^2 / residual RMSE.
  3. Where the market's z disagrees with the TWAP rule, build the residual
     signal and test whether it predicts the outcome better than the market.
  4. Honest walk-forward + cluster-robust trading of the residual.
"""
import os, pickle, json
import numpy as np
from scipy.special import ndtri

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


def var_twap(s2, T):
    """Var[Q|F_T]/s^2 for the 60s-TWAP settlement over [240,300)."""
    if T >= 240.0:
        rem = 300.0 - T
        return rem ** 3 / 10800.0          # (1/3600)*int_0^rem u^2 du
    a = 240.0 - T
    b = 300.0 - T
    return 2.0 / 3600.0 * (b ** 3 / 6.0 - b * a * a / 2.0 + a ** 3 / 3.0)


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
        cum_rv = np.cumsum(r * r)
        cum_t = np.maximum(t, 1e-9)
        sig2 = cum_rv / cum_t
        j240 = int(np.searchsorted(t, 240.0))
        for T in (30, 60, 90, 120, 150, 180, 210, 230, 240, 250, 260, 270, 280, 290):
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
            # ---- TWAP rule ----
            if T <= 240:
                eQ = cb[i]
            else:
                obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
                rem = 300.0 - T
                eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
            vQ = var_twap(s2, T)
            if vQ <= 0:
                continue
            z_twap = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
            # ---- spot rule (naive: final price vs strike) ----
            v_spot = s2 * (300.0 - T)
            z_spot = (np.log(cb[i]) - np.log(strike)) / np.sqrt(v_spot) if v_spot > 0 else np.nan
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
                sgn = 1.0 if side == "Up" else -1.0
                rows.append((wi, float(T), float(ask), float(bid), float(mkt_side),
                             sgn * z_twap, sgn * z_spot, float(z_mkt), float(ask - bid),
                             1 if d["_winner"] == side else 0))
    A = np.array(rows, dtype=float)
    wi, T, ask, bid, mkt, zt, zs, zm = (A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4],
                                         A[:, 5], A[:, 6], A[:, 7])
    y = A[:, 9].astype(int)
    print(f"rows {len(A)}")

    # ---------- 1. which rule explains z_mkt? ----------
    print("\n=== 1) which rule explains the market's z? (regress z_mkt on each) ===")
    print(f"  {'T':>4} {'n':>5} | {'TWAP rule':^26} | {'SPOT rule':^26}")
    for TT in (30, 60, 90, 120, 150, 180, 210, 240, 250, 270):
        s = T == TT
        if s.sum() < 50:
            continue
        cells = []
        for nm, x in (("twap", zt), ("spot", zs)):
            m = s & np.isfinite(x)
            if m.sum() < 40 or x[m].std() < 1e-12:
                cells.append(f"{nm}: n/a"); continue
            sl, ic = np.polyfit(x[m], zm[m], 1)
            pred = sl * x[m] + ic
            rmse = np.sqrt(((zm[m] - pred) ** 2).mean())
            cells.append(f"slope{sl:>6.3f} rmse{rmse:>7.4f}")
        print(f"  {TT:>4} {int(s.sum()):>5} | {cells[0]:>26} | {cells[1]:>26}")
    print("  (slope 1.0 + smallest rmse = the market uses that rule)")

    # pooled, and residual structure
    print("\n  pooled (all rows):")
    for nm, x in (("twap", zt), ("spot", zs)):
        m = np.isfinite(x)
        sl, ic = np.polyfit(x[m], zm[m], 1)
        pred = sl * x[m] + ic
        rmse = np.sqrt(((zm[m] - pred) ** 2).mean())
        r2 = 1 - ((zm[m] - pred) ** 2).sum() / ((zm[m] - zm[m].mean()) ** 2).sum()
        print(f"    {nm}: slope={sl:.4f} intercept={ic:+.4f} rmse={rmse:.4f} R2={r2:.4f}")

    # ---------- 2. residual as a signal ----------
    print("\n=== 2) residual (z_mkt - fitted z_twap) as a trading signal ===")
    m = np.isfinite(zt)
    sl, ic = np.polyfit(zt[m], zm[m], 1)
    resid = zm - (sl * zt + ic)
    # is the residual predictive of the outcome beyond the market's own price?
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    nw = len(W)
    X = np.column_stack([zm, resid, zt])
    for frac in (0.4, 0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut; te = wi >= cut
        for nm, cols in (("mkt only", [0]), ("mkt+resid", [0, 1]), ("mkt+resid+z", [0, 1, 2])):
            Xs = X[:, cols]
            mu_, sd_ = Xs[tr].mean(0), Xs[tr].std(0); sd_[sd_ == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=4000).fit((Xs[tr] - mu_) / sd_, y[tr])
            p = lr.predict_proba((Xs[te] - mu_) / sd_)[:, 1]
            ll = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            llm = log_loss(y[te], np.clip(mkt[te], 1e-6, 1 - 1e-6))
            print(f"    cut{frac} {nm:<12} LL model={ll:.5f}  LL market={llm:.5f}  "
                  f"{'MODEL WINS' if ll < llm else ''}")
    print("\n  trading the residual directly (OOS):")
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        te = wi >= cut
        for margin in (0.15, 0.3, 0.5):
            for direc in (1, -1):
                sel = np.where(te & (direc * resid > margin))[0]
                if len(sel) < 25:
                    continue
                a = ask[sel]; sh = 10.0 / a
                pnl = sh * y[sel] - 10.0 - FEE(a, sh)
                m_, tst, nww = cluster_t(wi[sel], pnl)
                print(f"    cut{frac} dir{'+' if direc>0 else '-'} resid>{margin:.2f}: N={len(sel):>5} w={nww:>3} "
                      f"total=${pnl.sum():+8.2f} avg=${pnl.mean():+.3f} cluster-t={tst:+.2f} "
                      f"win={y[sel].mean():.4f} meanAsk={a.mean():.4f}")

    with open(os.path.join(OUT, "rule_test.json"), "w") as fh:
        json.dump({"twap_slope": float(sl), "rmse": float(rmse)}, fh, indent=2)


if __name__ == "__main__":
    main()
