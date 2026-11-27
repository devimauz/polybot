"""Stage 18: THE RESOLUTION-VOL SCALING EDGE  (the decisive test).

RESOLUTION RULE (confirmed, Polymarket 5-min crypto, since 2026-08-07):
  * The PRICE TO BEAT  = Chainlink 60s TWAP value at the open
                         = mean(price over the 60s BEFORE the window).
  * The SETTLEMENT    = Chainlink 60s TWAP value at the close
                         = mean(price over the 60s BEFORE the close, i.e. [240,300]).
  * Up iff settlement >= strike.

WHY THIS CREATES A MODELLING EDGE:
  The settlement is an AVERAGE of 60 prices, not a single price. For a random
  walk with per-second vol s, Var(mean of L prices) = s^2 * L/3, so
      std(settlement) = s * sqrt(60/3) = s * 4.47
  whereas a single price 60s away has std s*sqrt(60) = s*7.75.
  The distance to the strike must be measured in units of s*sqrt(60/3).
  A bot (or market maker) that normalises by SPOT vol, or by s*sqrt(T), is
  mis-scaled by a factor of ~1.7 and will be systematically over- or
  under-confident near the strike -> tradeable mispricing.

THIS SCRIPT:
  1. Stitch adjacent windows to recover the 60s BEFORE each window (the strike
     region) from the previous window's log. Validate the reconstructed
     resolution metric against the recorded resolution.
  2. Build z_correct = (E[settlement | info] - strike) / (s*sqrt(60/3))
     and z_naive   = (spot - strike) / (s*sqrt(remaining))
     plus the market's own implied probability (SIDE-CORRECT).
  3. Walk-forward: can a model beat the MARKET's own probability OOS?
  4. Trade the disagreement, cluster-robust over windows.
"""
import os, pickle, json
import numpy as np

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


def main():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    print(f"windows {len(W)}  ws range {W[0]['ws']}..{W[-1]['ws']}")

    # ---------- 1. stitch: recover the 60s BEFORE each window ----------
    # price series per window: (t_rel, price).  To get the 60s before ws we need
    # the previous window's tail (t_rel in [240,300] of ws-300).
    def price_series(d):
        t = d["t"]
        cb = ffill(d["cb_px"])
        return t, cb

    pre = {}   # ws -> (times, prices) covering [ws-60, ws)
    for d in W:
        t, cb = price_series(d)
        prev = by_ws.get(d["ws"] - 300)
        if prev is None:
            continue
        pt, pc = price_series(prev)
        m = (pt >= 240) & np.isfinite(pc)
        if m.sum() < 10:
            continue
        # times relative to THIS window start
        pre[d["ws"]] = (pt[m] - 300.0, pc[m])

    def strike_of(d):
        """Estimated Chainlink 60s TWAP at the open = mean of the 60s before."""
        ws = d["ws"]
        if ws in pre:
            tt, pp = pre[ws]
            m = tt >= -60.0
            if m.sum() >= 10:
                return float(np.mean(pp[m])), "prev_window"
        t, cb = price_series(d)
        i0 = int(np.searchsorted(t, 0.0))
        i0 = min(max(i0, 0), len(t) - 1)
        return float(cb[i0]), "first_tick"

    def settlement_of(d):
        """Estimated Chainlink 60s TWAP at the close = mean of the last 60s."""
        t, cb = price_series(d)
        m = (t >= 240.0) & np.isfinite(cb)
        if m.sum() < 10:
            return np.nan
        return float(np.mean(cb[m]))

    # ---------- 2. validate the resolution metric ----------
    print("\n=== 1) reconstructed resolution metric vs the recorded resolution ===")
    n_ok = n = 0
    n_ok_spot = 0
    for d in W:
        k, src = strike_of(d)
        s = settlement_of(d)
        if not np.isfinite(k) or not np.isfinite(s) or k <= 0:
            continue
        n += 1
        n_ok += int((s > k) == (d["_winner"] == "Up"))
        t, cb = price_series(d)
        n_ok_spot += int((cb[-1] > k) == (d["_winner"] == "Up"))
    print(f"  n={n}")
    print(f"  settlement=mean(last 60s) vs strike=mean(prev 60s): {n_ok}/{n} = {n_ok/max(n,1):.4f}")
    print(f"  settlement=final spot    vs same strike            : {n_ok_spot}/{n} = {n_ok_spot/max(n,1):.4f}")

    # how good is the strike estimate? compare sources
    both = [(d, strike_of(d)) for d in W]
    nsrc = {}
    for d, (k, src) in both:
        nsrc[src] = nsrc.get(src, 0) + 1
    print(f"  strike sources: {nsrc}")

    # ---------- 3. build the feature table with SIDE-CORRECT market prob ----------
    print("\n=== 2) side-correct market probability + z_correct vs z_naive ===")
    rows = []
    for wi, d in enumerate(W):
        t, cb = price_series(d)
        n = len(t)
        strike, _ = strike_of(d)
        if not np.isfinite(strike) or strike <= 0:
            continue
        r = np.zeros(n)
        r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        s2 = np.cumsum(r * r) / np.maximum(np.arange(1, n + 1), 1)
        sd = np.sqrt(s2)
        settle = settlement_of(d)
        for T in (60, 120, 180, 240, 270):
            i = int(np.searchsorted(t, T))
            if i >= n:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.001 < p_up < 0.999):
                continue
            # per-second vol over the window so far
            s = sd[i] if np.isfinite(sd[i]) and sd[i] > 0 else np.nan
            if not np.isfinite(s) or s <= 0:
                continue
            # settlement = mean of last 60s.  At time T, remaining unknown part.
            # E[settlement] = known mean of [240,300] portion + future drift 0.
            m_last = (t >= 240.0) & np.isfinite(cb)
            if T <= 240:
                # nothing of the settlement window observed yet
                e_settle = cb[i]                      # best guess = current price
                var_known = 0.0
                rem = 60
            else:
                j0 = int(np.searchsorted(t, 240.0))
                obs = cb[j0:i + 1]
                e_settle = float(np.mean(obs)) if len(obs) else cb[i]
                rem = 300 - T
                var_known = 0.0
            # var of the settlement conditional on info at T:
            #   mean of L=60 prices; known part contributes 0 extra var
            #   future part: s^2 * rem/3 * (60/60)^2 ... use s^2*rem/3
            var_settle = s * s * rem / 3.0
            z_corr = (e_settle - strike) / np.sqrt(var_settle) if var_settle > 0 else np.nan
            z_naive = (cb[i] - strike) / (s * np.sqrt(max(300 - T, 1)))
            if not np.isfinite(z_corr):
                continue
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                if not (np.isfinite(ask) and np.isfinite(bid)) or not (0 < bid <= ask < 1):
                    continue
                mkt_side = p_up if side == "Up" else 1.0 - p_up
                rows.append((wi, T, 1.0 if side == "Up" else 0.0, float(ask), float(bid), float(mkt_side),
                             float(z_corr), float(z_naive), float(T),
                             1 if d["_winner"] == side else 0))
    print(f"  rows {len(rows)}")
    A = np.array(rows, dtype=float)
    wi, T, side, ask, bid, mkt, zc, zn = (A[:, 0], A[:, 1], A[:, 2], A[:, 3],
                                          A[:, 4], A[:, 5], A[:, 6], A[:, 7])
    y = A[:, 9].astype(int)
    for T in (60, 120, 180, 240, 270):
        s = T == T
        s = A[:, 1] == T
        if s.sum() < 50:
            continue
        print(f"    T={T:>3}: n={int(s.sum()):>6}  corr(z_correct,y)={np.corrcoef(zc[s], y[s])[0,1]:+.4f} "
              f" corr(z_naive,y)={np.corrcoef(zn[s], y[s])[0,1]:+.4f} "
              f" corr(mkt,y)={np.corrcoef(mkt[s], y[s])[0,1]:+.4f}")

    # ---------- 4. walk-forward: model vs market (SIDE-CORRECT) ----------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    print("\n=== 3) walk-forward logloss: model vs the market's own probability ===")
    nw = len(W)
    ALL = np.column_stack([mkt, zc, zn, ask, ask - bid])
    FEATS = {
        "market only": [0],
        "market+z_corr": [0, 1],
        "market+z_naive": [0, 2],
        "z_corr only": [1],
        "z_naive only": [2],
        "all": [0, 1, 2, 3, 4],
    }
    for name, cols in FEATS.items():
        X = ALL[:, cols]
        line = []
        for frac in (0.4, 0.5, 0.6):
            cut = int(nw * frac)
            tr = wi < cut
            te = wi >= cut
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=4000).fit((X[tr] - mu) / sd, y[tr])
            p = lr.predict_proba((X[te] - mu) / sd)[:, 1]
            ll_m = log_loss(y[te], np.clip(mkt[te], 1e-6, 1 - 1e-6))
            ll_c = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            line.append(f"cut{frac}: mkt={ll_m:.5f} mod={ll_c:.5f} {'MODEL' if ll_c<ll_m else 'mkt'}")
        print(f"  {name:<18} " + " | ".join(line))

    # ---------- 5. trade the disagreement ----------
    print("\n=== 4) trading the model's disagreement with the market (OOS) ===")
    X = ALL
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut
        te = wi >= cut
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        lr = LogisticRegression(C=1.0, max_iter=4000).fit((X[tr] - mu) / sd, y[tr])
        p = lr.predict_proba((X[te] - mu) / sd)[:, 1]
        for margin in (0.03, 0.05, 0.10):
            for direc in (1, -1):
                sel = np.where(direc * (p - mkt[te]) > margin)[0]
                if len(sel) < 30:
                    continue
                a = ask[te][sel]
                sh = 10.0 / a
                fee = FEE(a, sh)
                pnl = sh * y[te][sel] - 10.0 - fee
                w2 = wi[te][sel]
                uq = np.unique(w2)
                s = np.array([pnl[w2 == w].sum() for w in uq])
                c = np.array([(w2 == w).sum() for w in uq])
                tst = (s.sum() / c.sum()) / (s.std(ddof=1) / np.sqrt(len(uq)))
                print(f"  cut{frac} dir{'+' if direc>0 else '-'} margin>{margin:.2f}: "
                      f"N={len(sel):>5} w={len(uq):>3} total=${pnl.sum():+9.2f} avg=${pnl.mean():+.3f} "
                      f"cluster-t={tst:+.2f} win={y[te][sel].mean():.4f} meanAsk={a.mean():.4f}")

    with open(os.path.join(OUT, "res_weight.json"), "w") as fh:
        json.dump({"resolution_agreement": n_ok / max(n, 1), "n": n}, fh, indent=2)


if __name__ == "__main__":
    main()
