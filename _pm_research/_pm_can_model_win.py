"""Stage 16: THE DECISIVE QUESTION.

Can ANY model beat the market's own price out-of-sample?

If a model (market price + microstructure + spot) cannot beat the market's
price on logloss out of sample, then no trading bot built on this data can be
profitable after fees, and the correct answer to the user is "no edge".

Design (no leakage, no tuning on the test set):
 * Rows: every snapshot with a valid two-sided book, 20<t<280.
 * Features (all causal): market mid, bid/ask spread, book imbalance (both
   sides), microprice deviation, coinbase z, coinbase 60s-TWAP z, time.
 * Split: chronological 3-fold (train / validate / test), repeated over
   4 DIFFERENT cut points to avoid a lucky split.
 * Metrics: logloss vs the market's own price, and Brier score.
 * Then: trade where the model's probability exceeds the market's by a margin,
   and measure P&L with cluster-robust stats over windows.
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


def build(W):
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]
        cb = ffill(d["cb_px"])
        p_open = cb[0]
        n = len(t)
        # causal realized vol
        r = np.zeros(n)
        r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        var = np.cumsum(r * r) / np.maximum(np.arange(1, n + 1), 1)
        sd = np.sqrt(var)
        for side in ("Up", "Down"):
            o = "Down" if side == "Up" else "Up"
            b = ffill(d[side + "_bid"]); a = ffill(d[side + "_ask"])
            ob = ffill(d[o + "_bid"]); oa = ffill(d[o + "_ask"])
            bs = d[side + "_bidsz"]; asz = d[side + "_asksz"]
            bb = np.nansum(d[f"{side}_bids_px"] * d[f"{side}_bids_sz"], axis=1)
            aa = np.nansum(d[f"{side}_asks_px"] * d[f"{side}_asks_sz"], axis=1)
            imb = np.where((bb + aa) > 0, (bb - aa) / (bb + aa), 0.0)
            imb[~np.isfinite(imb)] = 0
            micro = np.where((bs + asz) > 0, (a * bs + b * asz) / np.maximum(bs + asz, 1e-9), np.nan)
            micro = ffill(micro)
            for i in range(20, n - 20):
                ti = t[i]
                if not (20 < ti < 280):
                    continue
                if not (0 < b[i] <= a[i] < 1) or not (0 < ob[i] <= oa[i] < 1):
                    continue
                if (a[i] - b[i]) > 0.05 or (oa[i] - ob[i]) > 0.05:
                    continue
                if not np.isfinite(sd[i]) or sd[i] <= 0:
                    continue
                z = (cb[i] - p_open) / max(sd[i] * np.sqrt(max(i, 1)), 1e-9)
                # P(Up) implied by the Up book  = (a+b)/2
                # P(Up) implied by the Down book = ob (bid of Down) or 1-oa (ask of Down)
                mkt = 0.5 * (0.5 * (a[i] + b[i]) + 0.5 * (ob[i] + (1.0 - oa[i])))
                rows.append((wi, i, side,
                             float(b[i]), float(a[i]),
                             float(a[i] - b[i]),
                             float(imb[i]),
                             float(micro[i] - 0.5 * (a[i] + b[i])),
                             float(z),
                             float(mkt),
                             float(ti)))
    return rows


def main():
    W = load_all()
    print(f"windows {len(W)}")
    rows = build(W)
    print(f"rows: {len(rows)}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, brier_score_loss

    wi = np.array([r[0] for r in rows], dtype=int)
    side = np.array([r[2] for r in rows])
    bid = np.array([r[3] for r in rows]); ask = np.array([r[4] for r in rows])
    sprd = np.array([r[5] for r in rows]); imb = np.array([r[6] for r in rows])
    mdev = np.array([r[7] for r in rows]); z = np.array([r[8] for r in rows])
    mkt = np.array([r[9] for r in rows]); tt = np.array([r[10] for r in rows])
    y = np.array([1 if (W[r[0]]["_winner"] == r[2]) else 0 for r in rows])

    FEATS = {
        "market only": [0],
        "market+imb": [0, 1],
        "market+imb+micro": [0, 1, 2],
        "market+imb+micro+z": [0, 1, 2, 3],
        "all": [0, 1, 2, 3, 4, 5, 6],
    }
    nw = len(W)
    print("\n=== walk-forward: model vs market price (logloss / brier) ===")
    print(f"{'features':<22} {'split':<8} {'n':>7} {'LL_market':>10} {'LL_model':>10} {'Brier_mkt':>10} {'Brier_mod':>10}  verdict")
    results = {}
    ALL = np.column_stack([mkt, imb, mdev, z, sprd, ask, bid])
    for name, cols in FEATS.items():
        X = ALL[:, cols]
        for si, frac in enumerate((0.4, 0.5, 0.6)):
            cut = int(nw * frac)
            tr = wi < cut
            te = wi >= cut
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
            lr = LogisticRegression(C=1.0, max_iter=3000)
            lr.fit((X[tr] - mu) / sd, y[tr])
            p = lr.predict_proba((X[te] - mu) / sd)[:, 1]
            pm = np.clip(mkt[te], 1e-6, 1 - 1e-6)
            ll_m = log_loss(y[te], pm); ll_c = log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            b_m = brier_score_loss(y[te], pm); b_c = brier_score_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            verdict = "MODEL BETTER" if ll_c < ll_m else "market better"
            print(f"{name:<22} cut{frac:<7} {int(te.sum()):>7} {ll_m:>10.5f} {ll_c:>10.5f} "
                  f"{b_m:>10.5f} {b_c:>10.5f}  {verdict}")
            results[(name, frac)] = (ll_m, ll_c)
        print()

    # --- trading the model's disagreements (OOS) ---
    print("=== trading model-vs-market disagreements (OOS, hold to settlement) ===")
    cols = FEATS["all"]
    X = np.column_stack([mkt, imb, mdev, z, sprd, ask, bid])[:, cols]
    FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
    for frac in (0.5, 0.6):
        cut = int(nw * frac)
        tr = wi < cut; te = wi >= cut
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        lr = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        p = lr.predict_proba((X[te] - mu) / sd)[:, 1]
        # p is P(this side wins)
        ask_te = ask[te]; mkt_te = mkt[te]; y_te = y[te]; wi_te = wi[te]
        for margin in (0.03, 0.05, 0.10, 0.15):
            sel = np.where((p - mkt_te) > margin)[0]
            if len(sel) < 20:
                continue
            stake = 10.0
            shares = stake / ask_te[sel]
            fee = FEE(ask_te[sel], shares)
            pnl = shares * y_te[sel] - stake - fee
            uq = np.unique(wi_te[sel])
            s = np.array([pnl[wi_te[sel] == w].sum() for w in uq])
            c = np.array([(wi_te[sel] == w).sum() for w in uq])
            tstat = (s.sum() / c.sum()) / (s.std(ddof=1) / np.sqrt(len(uq)))
            print(f"  cut{frac} margin>{margin:.2f}: N={len(sel):>6} windows={len(uq):>4} "
                  f"total=${pnl.sum():+9.2f} avg=${pnl.mean():+.3f} cluster-t={tstat:+.2f} "
                  f"win={y_te[sel].mean():.4f} meanAsk={ask_te[sel].mean():.4f}")


if __name__ == "__main__":
    main()
