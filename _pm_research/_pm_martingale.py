"""Stage 14: THE MARTINGALE TEST (definitive, max power).

If the market's mid is a martingale (no predictable drift), then NO price-based
strategy can have positive EV, and the only remaining edges are:
  (a) a model with EXTRA information (a better probability estimate), or
  (b) execution/structural tricks (maker fees, pair fills, latency).

Test: does E[mid_{t+Δ} - mid_t | state_t] differ from 0 in a way that a
trader could exploit? Fit a small linear model of the future mid change on
current state (mid, mid change over the last k ticks, imbalance, coinbase
return), then check whether the predicted change exceeds the round-trip cost
(half-spread in + half-spread out + 2 taker fees) and survives out-of-sample.

This uses EVERY snapshot (289k rows) and is by far the highest-power test
available: it does not depend on any entry rule.
"""
import os, pickle
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"


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
        out.append(d)
    out.sort(key=lambda d: int(d["ws"][0]))
    return out


def ffill(a):
    a = np.array(a, dtype=float).copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


def main():
    W = load_all()
    print(f"windows {len(W)}")

    # Build a flat table of (window, i, mid, bid, ask, imb, cb_ret, dmid_recent)
    X = []
    for wi, d in enumerate(W):
        t = d["t"]
        mu, md = ffill(d["Up_mid"]), ffill(d["Down_mid"])
        mid = 0.5 * (mu + (1 - md))
        ub, ua = ffill(d["Up_bid"]), ffill(d["Up_ask"])
        db, da = ffill(d["Down_bid"]), ffill(d["Down_ask"])
        cb = ffill(d["cb_px"])
        n = len(t)
        if n < 40:
            continue
        dmid = np.zeros(n)
        dmid[1:] = np.diff(mid)
        dmid[~np.isfinite(dmid)] = 0
        cbr = np.zeros(n)
        cbr[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        cbr[~np.isfinite(cbr)] = 0
        bb = np.nansum(d["Up_bids_px"] * d["Up_bids_sz"], axis=1)
        aa = np.nansum(d["Up_asks_px"] * d["Up_asks_sz"], axis=1)
        imb = np.where((bb + aa) > 0, (bb - aa) / (bb + aa), 0.0)
        imb[~np.isfinite(imb)] = 0
        # rolling dmid over 10 snapshots (~3s)
        roll = np.zeros(n)
        csum = np.cumsum(dmid)
        for i in range(n):
            j = max(0, i - 10)
            roll[i] = csum[i] - csum[j]
        for i in range(30, n - 30):
            if not np.isfinite(mid[i]):
                continue
            if not (0.005 < mid[i] < 0.995):
                continue
            X.append((wi, i, mid[i], ua[i], ub[i], da[i], db[i], imb[i], cbr[i], roll[i], t[i]))
    X = np.array(X, dtype=float)
    print(f"rows: {len(X)}")
    wi = X[:, 0].astype(int)
    i = X[:, 1].astype(int)
    mid = X[:, 2]
    up_ask, up_bid, dn_ask, dn_bid = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
    imb, cbr, roll, tt = X[:, 7], X[:, 8], X[:, 9], X[:, 10]

    # Map rows back to windows to compute future mid at t+delta
    win_of = {id(d): k for k, d in enumerate(W)}

    # --- future mid at several horizons ---
    fut = {}
    for hz in (10, 30, 60, 120):
        arr = np.full(len(X), np.nan)
        # per window, compute mid series once
        for k, d in enumerate(W):
            mu, md = ffill(d["Up_mid"]), ffill(d["Down_mid"])
            m = 0.5 * (mu + (1 - md))
            m[~np.isfinite(m)] = np.nan
            sel = np.where(wi == k)[0]
            if len(sel) == 0:
                continue
            idxs = i[sel].astype(int)
            j = idxs + hz
            ok = j < len(m)
            arr[sel[ok]] = m[j[ok]]
        fut[hz] = arr
    print("built future series")

    # --- regression: future dmid ~ current state ---
    from sklearn.linear_model import LinearRegression
    for hz in (10, 30, 60, 120):
        f = fut[hz]
        ok = np.isfinite(f)
        y = f[ok] - mid[ok]
        feats = np.column_stack([mid[ok], roll[ok], imb[ok], cbr[ok]])
        # chronological split
        nw = len(W)
        cut = int(nw * 0.6)
        wtr = wi[ok] < cut
        wte = wi[ok] >= cut
        lr = LinearRegression().fit(feats[wtr], y[wtr])
        pred_te = lr.predict(feats[wte])
        yte = y[wte]
        # how often does the predicted move exceed the round-trip cost?
        # cost for buying UP  = ask_up - bid_up (spread) + 2 fees at mid
        cost = (up_ask[ok] - up_bid[ok]) + 2 * 0.07 * mid[ok] * (1 - mid[ok])
        cost_dn = (dn_ask[ok] - dn_bid[ok]) + 2 * 0.07 * (1 - mid[ok]) * mid[ok]
        cost = np.minimum(cost, cost_dn)
        cost_te = cost[wte]
        # take the trade with the larger predicted move in the direction we predict
        dirn = np.sign(pred_te)
        gross = np.abs(pred_te)
        edge = gross - cost_te
        n_act = int((edge > 0).sum())
        if n_act:
            avg_edge = edge[edge > 0].mean()
            print(f"  hz={hz:>3}s  n_test={int(wte.sum()):>6}  R2={lr.score(feats[wte], yte):+.5f} "
                  f"| predicted |move| mean={gross.mean():.5f}  cost mean={cost_te.mean():.5f} "
                  f"| trades with edge>0: {n_act:>6} ({n_act/max(wte.sum(),1):.2%}) avg edge/share={avg_edge:+.5f}")
        else:
            print(f"  hz={hz:>3}s  n_test={int(wte.sum()):>6}  R2={lr.score(feats[wte], yte):+.5f} "
                  f"| predicted |move| mean={gross.mean():.5f}  cost mean={cost_te.mean():.5f} | NO trade clears cost")

    # --- direct: is the mid a martingale? mean future change conditional on buckets ---
    print("\n  --- martingale check: E[fwd dmid | current dmid bucket] ---")
    f = fut[30]
    ok = np.isfinite(f)
    y = (f - mid)[ok]
    d10 = roll[ok]
    for lo, hi in [(-0.2, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, 0.2)]:
        m = (d10 >= lo) & (d10 < hi)
        if m.sum() < 50:
            continue
        print(f"    recent 3s dmid in [{lo:+.2f},{hi:+.2f}): n={int(m.sum()):>6} "
              f"E[next 8s dmid]={y[m].mean():+.5f} (se {y[m].std()/np.sqrt(m.sum()):.5f})")


if __name__ == "__main__":
    main()
