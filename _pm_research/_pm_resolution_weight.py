"""Stage 17: THE RESOLUTION-WEIGHTING EDGE (the last structural candidate).

MECHANISM (why the market can be wrong in a way a model can exploit):
  Polymarket 5-min crypto markets resolve Up iff the window's resolution
  metric >= the open price. The metric is built from Chainlink's 60-second TWAP
  stream AVERAGED OVER THE WINDOW (per the market's published rule).
  Averaging a 60s-TWAP over the window gives prices a RAMPED weight:
  early prices count ~half, recent prices count fully.
  => By t=250 the resolution metric is ~80% ALREADY DETERMINED, while the
     market's price reacts to the CURRENT SPOT. If the market underweights the
     already-realized average, it misprices windows where the recent tick
     diverges from the window average.

TEST:
 1. Build several resolution-metric proxies from the logged coinbase price and
    score each against the recorded resolution (which proxy explains the
    outcome best?).
 2. Build the "partial metric" available at time t (weighted average of the
    window's 60s-TWAP readings so far, with the ramp weight) and compare the
    market's implied probability with a calibrated model of that partial
    metric.
 3. Trade the disagreement: buy the side where the model > the ask, walk-forward.
 4. Report with cluster-robust stats (windows are the iid unit).
"""
import os, pickle
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"


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
        out.append(d)
    out.sort(key=lambda d: int(d["ws"][0]))
    return out


def twap60_series(t, px):
    """At each sample, the trailing 60s average price (Chainlink-style 60s TWAP)."""
    n = len(t)
    out = np.full(n, np.nan)
    csum = np.cumsum(px)
    tsum = np.cumsum(t)
    for i in range(n):
        j = int(np.searchsorted(t, t[i] - 60.0))
        if i - j < 2 or t[i] - t[j] < 5:
            continue
        out[i] = (csum[i] - csum[j]) / (i - j + 1)
    return out


def main():
    W = load_all()
    print(f"windows {len(W)}")

    # ---------- 1. which resolution metric explains the outcome best? ----------
    print("\n=== 1) resolution-metric proxies vs the recorded resolution ===")
    metrics = {
        "final_spot": lambda t, cb, tw: cb[-1],
        "twap60_last": lambda t, cb, tw: tw[-1],
        "mean_twap60_over_window": lambda t, cb, tw: np.nanmean(tw),
        "mean_twap60_first240": lambda t, cb, tw: np.nanmean(tw[t <= 240]),
        "mean_twap60_last120": lambda t, cb, tw: np.nanmean(tw[t >= 180]),
        "mean_cb_over_window": lambda t, cb, tw: np.nanmean(cb),
    }
    for name, fn in metrics.items():
        ok = 0
        n = 0
        for d in W:
            t = d["t"]
            cb = ffill(d["cb_px"])
            tw = twap60_series(t, cb)
            p_open = cb[0]
            v = fn(t, cb, tw)
            if not np.isfinite(v) or not np.isfinite(p_open) or p_open <= 0:
                continue
            n += 1
            ok += int((v > p_open) == (d["_winner"] == "Up"))
        print(f"  {name:<28} {ok:>3}/{n:<4} = {ok/max(n,1):.4f}")

    # ---------- 2. weighted ("ramped") window metric ----------
    # weight of the 60s-TWAP reading at time u, when the window ends at 300:
    #   w(u) = min(1, (u - (u-60) overlap)) / ... -> for a 60s TWAP, the reading
    #   at u covers [u-60, u]; a price point p is counted by every reading whose
    #   window contains it, so its total weight is proportional to the number of
    #   readings covering it. For a continuous stream this is a linear ramp
    #   from 0 at t=0 to 1 at t>=60.
    print("\n=== 2) ramped-window metric: how much of the outcome is known at time t? ===")
    # For each window compute, at t=250, the ramped partial metric using the
    # ramp weight w(u) = clamp(u/60, 0, 1) applied to the trailing-60s TWAP.
    agree = {}
    for T in (150, 200, 240, 270):
        ok = n = 0
        for d in W:
            t = d["t"]
            cb = ffill(d["cb_px"])
            tw = twap60_series(t, cb)
            p_open = cb[0]
            m = (t <= T) & np.isfinite(tw)
            if m.sum() < 5:
                continue
            w = np.clip(t[m] / 60.0, 0.0, 1.0)
            v = np.sum(w * tw[m]) / np.sum(w)
            if not np.isfinite(v):
                continue
            n += 1
            ok += int((v > p_open) == (d["_winner"] == "Up"))
        agree[T] = (ok, n)
        print(f"  T={T:>3}s  ramped partial metric agrees {ok}/{n} = {ok/max(n,1):.4f}")

    # ---------- 3. does the market price the partial metric or the spot? ----------
    print("\n=== 3) what does the market price? (agreement with each candidate at t=240) ===")
    T = 240
    cands = {"final_spot": 0, "spot_at_T": 0, "ramped_partial": 0, "mean_cb_so_far": 0}
    conf = 0
    for d in W:
        t = d["t"]
        cb = ffill(d["cb_px"])
        tw = twap60_series(t, cb)
        i = int(np.searchsorted(t, T))
        if i >= len(t):
            continue
        mu, md = d["Up_mid"][i], d["Down_mid"][i]
        if not (np.isfinite(mu) and np.isfinite(md)):
            continue
        p_up = 0.5 * (mu + (1 - md))
        if not (0.85 < p_up < 0.995):
            continue
        conf += 1
        p_open = cb[0]
        m = (t <= T) & np.isfinite(tw)
        w = np.clip(t[m] / 60.0, 0.0, 1.0)
        vals = {
            "final_spot": cb[-1],
            "spot_at_T": cb[i],
            "ramped_partial": (np.sum(w * tw[m]) / np.sum(w)) if m.sum() else np.nan,
            "mean_cb_so_far": np.nanmean(cb[:i + 1]),
        }
        for k, v in vals.items():
            if np.isfinite(v):
                cands[k] += int((v > p_open) == (p_up > 0.5))
    print(f"  confident windows (p_up>0.85 at t={T}): n={conf}")
    for k, v in sorted(cands.items(), key=lambda x: -x[1]):
        print(f"    market agrees with {k:<18} {v}/{conf} = {v/max(conf,1):.4f}")

    # ---------- 4. trade the model-vs-market disagreement ----------
    print("\n=== 4) trading a model built on the ramped partial metric (walk-forward) ===")
    from sklearn.linear_model import LogisticRegression
    FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]
        cb = ffill(d["cb_px"])
        tw = twap60_series(t, cb)
        p_open = cb[0]
        n = len(t)
        for T in (120, 180, 240):
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            m = (t <= i) & np.isfinite(tw)
            if m.sum() < 5:
                continue
            w = np.clip(t[m] / 60.0, 0.0, 1.0)
            partial = np.sum(w * tw[m]) / np.sum(w)
            if not np.isfinite(partial):
                continue
            rel = (partial - p_open) / p_open
            sd = np.std(cb[:i + 1])
            if not np.isfinite(sd) or sd <= 0:
                continue
            zpart = rel / (sd * np.sqrt(max(i, 1)) / p_open)
            rel_spot = (cb[i] - p_open) / p_open
            zspot = rel_spot / (sd * np.sqrt(max(i, 1)) / p_open)
            for side, ask, bid in (("Up", d["Up_ask"][i], d["Up_bid"][i]),
                                   ("Down", d["Down_ask"][i], d["Down_bid"][i])):
                if not (np.isfinite(ask) and np.isfinite(bid)) or not (0 < bid <= ask < 1):
                    continue
                mkt = 0.5 * (mu + (1 - md))
                mkt_side = mkt if side == "Up" else 1 - mkt
                rows.append((wi, T, side, float(ask), float(mkt_side), float(zpart), float(zspot),
                             1 if d["_winner"] == side else 0))
    X = np.array([(r[5], r[6], r[4]) for r in rows])
    y = np.array([r[7] for r in rows])
    wi = np.array([r[0] for r in rows])
    ask = np.array([r[3] for r in rows])
    mkt = np.array([r[4] for r in rows])
    print(f"  rows {len(rows)}")
    nw = len(W)
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut
        te = wi >= cut
        for cols, nm in (([0], "z_part"), ([1], "z_spot"), ([0, 1], "z_part+z_spot"), ([0, 1, 2], "+market")):
            Xs = X[:, cols]
            mu, sd = Xs[tr].mean(0), Xs[tr].std(0); sd[sd == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=3000).fit((Xs[tr] - mu) / sd, y[tr])
            p = lr.predict_proba((Xs[te] - mu) / sd)[:, 1]
            for margin in (0.04, 0.08, 0.12):
                sel = np.where((p - mkt[te]) > margin)[0]
                if len(sel) < 30:
                    continue
                shares = 10.0 / ask[te][sel]
                fee = FEE(ask[te][sel], shares)
                pnl = shares * y[te][sel] - 10.0 - fee
                uq = np.unique(wi[te][sel])
                s = np.array([pnl[wi[te][sel] == w].sum() for w in uq])
                c = np.array([(wi[te][sel] == w).sum() for w in uq])
                tstat = (s.sum() / c.sum()) / (s.std(ddof=1) / np.sqrt(len(uq)))
                print(f"    cut{frac} {nm:<14} margin>{margin:.2f}: N={len(sel):>5} w={len(uq):>3} "
                      f"total=${pnl.sum():+8.2f} avg=${pnl.mean():+.3f} cluster-t={tstat:+.2f}")


if __name__ == "__main__":
    main()
