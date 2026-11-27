"""Stage 2: book integrity + model-free edges on the Polymarket BTC 5m book.

Edges tested (in order of strength - model-free first):
  A) COMPLETE-SET ARB: buy Up + Down when ask_up + ask_down < 1 -> guaranteed $1 payout.
     Needs no predictive model, so it is the gold standard if it exists.
  B) BOOK IMBALANCE / MICROPRICE: does L2 depth predict the resolution better
     than the market's own mid?
  C) COINBASE TREND: does the spot path predict the resolution vs. market price?
"""
import os, glob, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)
FEE_RATE = 0.05  # taker fee: 0.05 * min(p,1-p) * shares


def eff_book(bid, ask, bid_px, bid_sz, ask_px, ask_sz):
    """Return a robust (bid, ask, mid) per row from raw levels.
    Falls back to book level 0 when the top-of-book field is missing."""
    n = len(bid)
    eb, ea = bid.copy(), ask.copy()
    # fill missing top-of-book from the raw arrays
    for i in range(n):
        if not np.isfinite(eb[i]) and np.isfinite(bid_px[i, 0]):
            eb[i] = bid_px[i, 0]
        if not np.isfinite(ea[i]) and np.isfinite(ask_px[i, 0]):
            ea[i] = ask_px[i, 0]
    return eb, ea


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        out.append(d)
    return out


def main():
    data = load_all()
    print(f"windows {len(data)}")

    # ---------- book integrity ----------
    inv = 0
    tot = 0
    spread_sum = 0.0
    spread_n = 0
    for d in data:
        for s in ("Up", "Down"):
            b, a = d[s + "_bid"], d[s + "_ask"]
            ok = np.isfinite(b) & np.isfinite(a)
            inv += int((ok & (b > a + 1e-9)).sum())
            tot += int(ok.sum())
            sp = a[ok] - b[ok]
            spread_sum += float(sp.sum())
            spread_n += len(sp)
    print(f"inverted books: {inv}/{tot} ({inv/max(tot,1):.4%})  mean spread {spread_sum/max(spread_n,1):.4f}")

    # ---------- A) COMPLETE-SET ARB ----------
    print("\n=== A) COMPLETE-SET ARB (ask_up + ask_down < 1) ===")
    arb_rows = []
    depth_hist = []
    for d in data:
        t = d["t"]
        au, ad = d["Up_ask"], d["Down_ask"]
        su, sd = d["Up_asksz"], d["Down_asksz"]
        ok = np.isfinite(au) & np.isfinite(ad)
        if not ok.any():
            continue
        ssum = np.where(ok, au + ad, np.nan)
        i = int(np.nanargmin(ssum))
        min_sum = float(ssum[i])
        depth_hist.append((min_sum, float(au[i]), float(ad[i]), float(su[i]) if np.isfinite(su[i]) else np.nan,
                           float(sd[i]) if np.isfinite(sd[i]) else np.nan, float(t[i])))
    dh = np.array(depth_hist, dtype=np.float64)
    print(f"windows with a usable book: {len(dh)}")
    print(f"  min(ask_up+ask_down) per window: p0={np.nanmin(dh[:,0]):.4f} p5={np.nanpercentile(dh[:,0],5):.4f} "
          f"p25={np.nanpercentile(dh[:,0],25):.4f} p50={np.nanpercentile(dh[:,0],50):.4f}")
    for thr in (1.0, 0.99, 0.98, 0.97, 0.96, 0.95):
        m = dh[:, 0] < thr
        print(f"  windows with min sum < {thr}: {int(m.sum())}/{len(dh)}")
    # depth at the best arb point for those under 0.98
    m = dh[:, 0] < 0.98
    if m.any():
        print("  [sum<0.98] ask_up/ask_down/ask_size_up/ask_size_down/t:")
        for r in dh[m][:15]:
            print("   ", np.round(r, 4))

    # how often does the pair sum dip below 1 at ANY time with both sizes >= $20?
    print("\n  -- pair-sum crossing stats (both legs, ask sizes) --")
    cross1 = 0
    cross1_depth = 0
    for d in data:
        au, ad = d["Up_ask"], d["Down_ask"]
        su, sd = d["Up_asksz"], d["Down_asksz"]
        ok = np.isfinite(au) & np.isfinite(ad) & np.isfinite(su) & np.isfinite(sd)
        if not ok.any():
            continue
        ssum = np.where(ok, au + ad, np.nan)
        if np.nanmin(ssum) < 1.0:
            cross1 += 1
            # at that point is there >=$20 of size on both legs?
            i = int(np.nanargmin(ssum))
            if su[i] * au[i] >= 20 and sd[i] * ad[i] >= 20:
                cross1_depth += 1
    print(f"  windows whose pair sum ever went < 1.00: {cross1}  (of which both legs had >=$20 depth: {cross1_depth})")

    # ---------- B) MICROPRICE / IMBALANCE ----------
    print("\n=== B) L2 MICROPRICE vs MARKET MID ===")
    # microprice = (ask*bid_size + bid*ask_size)/(bid_size+ask_size) weighted toward the thin side
    recs = []
    for d in data:
        if not d["_winner"]:
            continue
        t = d["t"]
        for T in (60, 120, 180, 240):
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            bu, au = d["Up_bid"][i], d["Up_ask"][i]
            bs, asz = d["Up_bidsz"][i], d["Up_asksz"][i]
            if not all(np.isfinite(v) for v in (bu, au, bs, asz)) or (bs + asz) <= 0:
                continue
            micro = (au * bs + bu * asz) / (bs + asz)
            mid = (bu + au) / 2
            # full book imbalance: sum of bid sizes vs ask sizes over top levels
            bb = np.nansum(d["Up_bids_px"][i] * d["Up_bids_sz"][i])
            aa = np.nansum(d["Up_asks_px"][i] * d["Up_asks_sz"][i])
            imb = (bb - aa) / (bb + aa) if (bb + aa) > 0 else np.nan
            recs.append((T, mid, micro, imb, 1 if d["_winner"] == "Up" else 0))
    R = np.array(recs, dtype=np.float64)
    for T in (60, 120, 180, 240):
        s = R[R[:, 0] == T]
        if len(s) < 20:
            continue
        mid, micro, imb, y = s[:, 1], s[:, 2], s[:, 3], s[:, 4]
        # correlation of each feature with outcome
        def corr(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10 or np.std(a[m]) == 0:
                return np.nan
            return float(np.corrcoef(a[m], b[m])[0, 1])
        # implied win prob from each feature, calibrated by isotonic-free binning
        print(f"  T={T}s n={len(s)}  corr(mid,y)={corr(mid,y):+.3f}  corr(micro,y)={corr(micro,y):+.3f}  "
              f"corr(imbalance,y)={corr(imb,y):+.3f}  corr(micro-mid,y)={corr(micro-mid,y):+.3f}")
        # edge of microprice vs mid
        print(f"       mean(micro-mid)={np.nanmean(micro-mid):+.4f}  "
              f"P(micro>mid & wins)={np.mean((micro>mid)&(y==1)):.3f} vs P(micro>mid)={np.mean(micro>mid):.3f}")

    # ---------- C) COINBASE TREND ----------
    print("\n=== C) COINBASE MOMENTUM AS A PROBABILITY MODEL ===")
    for T in (60, 120, 180, 240, 270):
        xs, ys, mids = [], [], []
        for d in data:
            if not d["_winner"]:
                continue
            t, cb = d["t"], d["cb_px"]
            fin = np.isfinite(cb)
            if fin.sum() < 5:
                continue
            idx = np.where(fin)[0]
            i0 = idx[0]
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            if not np.isfinite(cb[i]) or not np.isfinite(cb[i0]):
                continue
            # normalize move by the window's own realized vol (from coinbase ticks)
            seg = cb[idx]
            ticks = np.diff(seg)
            vol = np.std(ticks) * np.sqrt(len(ticks)) if len(ticks) > 2 else np.nan
            if not np.isfinite(vol) or vol <= 0:
                continue
            z = (cb[i] - cb[i0]) / vol
            mi = int(np.searchsorted(t, T))
            if mi >= len(t) or not np.isfinite(d["Up_mid"][mi]):
                continue
            xs.append(z)
            ys.append(1 if d["_winner"] == "Up" else 0)
            mids.append(d["Up_mid"][mi])
        xs, ys, mids = np.array(xs), np.array(ys), np.array(mids)
        if len(xs) < 20:
            continue
        # logistic fit
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0).fit(xs.reshape(-1, 1), ys)
            p = lr.predict_proba(xs.reshape(-1, 1))[:, 1]
        except Exception:
            # simple univariate normal approximation
            p = 0.5 * (1 + np.tanh(xs / 2.0))
        ll_mkt = np.mean(ys * np.log(np.clip(mids, 1e-6, 1 - 1e-6)) + (1 - ys) * np.log(np.clip(1 - mids, 1e-6, 1)))
        ll_mod = np.mean(ys * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - ys) * np.log(np.clip(1 - p, 1e-6, 1)))
        # where do they disagree, and what happens there?
        dis = (p - mids) > 0.05
        disn = (mids - p) > 0.05
        print(f"  T={T}s n={len(xs)}  logloss market={ll_mkt:.4f} model={ll_mod:.4f}  "
              f"model better: {ll_mod < ll_mkt}")
        if dis.sum() > 5:
            print(f"      model >> market (p>m+0.05): n={int(dis.sum())} model winrate={ys[dis].mean():.3f} "
                  f"mean ask-proxy mid={mids[dis].mean():.3f}  edge/contract={ys[dis].mean()-mids[dis].mean():+.3f}")
        if disn.sum() > 5:
            print(f"      market >> model (m>p+0.05): n={int(disn.sum())} model winrate={ys[disn].mean():.3f} "
                  f"mean mid={mids[disn].mean():.3f}  edge/contract={ys[disn].mean()-mids[disn].mean():+.3f}")


if __name__ == "__main__":
    main()
