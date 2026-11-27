"""Stage 10: THE DECISIVE TEST.

Question 1 (max power): is the market's price miscalibrated, and in which
direction? Use EVERY snapshot (not just first entries) and a WINDOW-CLUSTERED
bootstrap, because Up and Down entries within a window are dependent.

Question 2: is there a resolution-aligned model (coinbase 60s-TWAP vs open
price, which is how Polymarket actually resolves) that beats the market's price
after fees, OUT OF SAMPLE?

Question 3: does any of it survive the fee schedule? (and what does a
fee-free version look like -> is the edge real or just fee-shaped?)

Everything is reported with clustered CIs. Windows are the iid unit.
"""
import os, pickle, math, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

FEE_FNS = {
    "doc_0.07": lambda p, sh: 0.07 * p * (1 - p) * sh,
    "pred_0.25sq": lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh,
    "repo_0.05min": lambda p, sh: 0.05 * min(p, 1 - p) * sh,
    "zero": lambda p, sh: 0.0,
}
STAKE = 10.0


def load_windows():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    wins = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        wins.append(d)
    wins.sort(key=lambda d: int(d["ws"][0]))
    return wins


def cluster_boot(win_ids, values, n_boot=3000, seed=11):
    """Bootstrap the MEAN of per-window aggregates, resampling WINDOWS."""
    win_ids = np.asarray(win_ids)
    values = np.asarray(values, dtype=float)
    uq = np.unique(win_ids)
    sums = np.array([values[win_ids == w].sum() for w in uq])
    cnts = np.array([(win_ids == w).sum() for w in uq])
    tot_n = cnts.sum()
    if len(uq) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uq), size=(n_boot, len(uq)))
    means = sums[idx].sum(1) / tot_n
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ======================================================================== Q1
def q1_calibration(W):
    print("=" * 100)
    print("Q1: IS THE MARKET MISCALIBRATED?  (all snapshots, window-clustered bootstrap)")
    print("=" * 100)
    # For each (window, snapshot, side): mid, ask, outcome
    # Calibration curve: P(win) vs MID, then the spread-adjusted edge vs ASK.
    buckets = [(0.02, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
               (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.96),
               (0.96, 1.00)]
    print(f"\n{'mid band':>12} {'n_snap':>8} {'n_win':>7} {'P(win)':>8} "
          f"{'meanMID':>8} {'edge_mid':>9} {'meanASK':>8} {'edge_ask':>9}  {'CI(edge_mid)':>18}")
    out = []
    for lo, hi in buckets:
        mids, asks, ys, wids = [], [], [], []
        for wi, d in enumerate(W):
            t = d["t"]
            up_mid, dn_mid = d["Up_mid"], d["Down_mid"]
            up_bid, up_ask = d["Up_bid"], d["Up_ask"]
            dn_bid, dn_ask = d["Down_bid"], d["Down_ask"]
            for i in range(len(t)):
                if t[i] < 20 or t[i] > 280:
                    continue
                for side, mid, ask in (("Up", up_mid[i], up_ask[i]),
                                       ("Down", dn_mid[i], dn_ask[i])):
                    if not (np.isfinite(mid) and np.isfinite(ask)):
                        continue
                    if not (lo <= mid < hi):
                        continue
                    mids.append(mid)
                    asks.append(ask)
                    ys.append(1 if d["_winner"] == side else 0)
                    wids.append(wi)
        if not mids:
            continue
        mids = np.array(mids); asks = np.array(asks); ys = np.array(ys, dtype=float)
        wids = np.array(wids)
        p = ys.mean()
        edge_mid = p - mids.mean()
        edge_ask = p - asks.mean()
        ci = cluster_boot(wids, ys - mids)
        print(f"{lo:.2f}-{hi:.2f} {len(mids):>8} {int(ys.sum()):>7} {p:>8.4f} "
              f"{mids.mean():>8.4f} {edge_mid:>+9.4f} {asks.mean():>8.4f} {edge_ask:>+9.4f}  "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}]")
        out.append(dict(lo=lo, hi=hi, n=len(mids), p=float(p), mean_mid=float(mids.mean()),
                        mean_ask=float(asks.mean()), edge_mid=float(edge_mid),
                        edge_ask=float(edge_ask), ci=ci))
    return out


# ======================================================================== Q2
def prep_feats(d):
    """Resolution-aligned features, causal."""
    t = d["t"]
    cb = d["cb_px"].copy()
    for i in range(1, len(cb)):
        if not np.isfinite(cb[i]) and np.isfinite(cb[i - 1]):
            cb[i] = cb[i - 1]
    for i in range(len(cb) - 2, -1, -1):
        if not np.isfinite(cb[i]) and np.isfinite(cb[i + 1]):
            cb[i] = cb[i + 1]
    n = len(t)
    p_open = cb[0]
    ret = np.zeros(n)
    ret[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
    ret[~np.isfinite(ret)] = 0
    var = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += ret[i] * ret[i]
        var[i] = s / max(i, 1)
    sd = np.sqrt(var)
    # trailing 60s TWAP (resolution uses a 60s TWAP stream)
    twap60 = np.full(n, np.nan)
    for i in range(n):
        j = int(np.searchsorted(t, t[i] - 60.0))
        st, sp = t[j:i + 1], cb[j:i + 1]
        if len(st) >= 3 and st[-1] > st[0]:
            twap60[i] = np.trapezoid(sp, st) / (st[-1] - st[0])
    # z of twap vs open, scaled by realized vol of the window
    sigma = np.nanstd(np.diff(cb[np.isfinite(cb)])) * np.sqrt(len(cb)) if np.isfinite(cb).sum() > 10 else np.nan
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(ret) * np.sqrt(n) or 1e-6
    z_twap = (twap60 - p_open) / sigma
    z_spot = (cb - p_open) / sigma
    return dict(t=t, cb=cb, z_twap=z_twap, z_spot=z_spot, twap60=twap60, p_open=p_open)


def q2_model(W):
    print("\n" + "=" * 100)
    print("Q2: RESOLUTION-ALIGNED MODEL vs MARKET PRICE (walk-forward, OOS only)")
    print("=" * 100)
    F = [prep_feats(d) for d in W]
    nW = len(W)
    # dataset rows: (window, t) samples
    TIMES = list(range(30, 281, 15))
    rows = []
    for wi, (d, f) in enumerate(zip(W, F)):
        t = d["t"]
        for T in TIMES:
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            up_b, up_a = d["Up_bid"][i], d["Up_ask"][i]
            dn_b, dn_a = d["Down_bid"][i], d["Down_ask"][i]
            if not (np.isfinite(up_b) and np.isfinite(up_a) and np.isfinite(dn_b) and np.isfinite(dn_a)):
                continue
            if up_b <= 0 or up_a >= 1 or dn_b <= 0 or dn_a >= 1:
                continue
            if (up_a - up_b) > 0.05 or (dn_a - dn_b) > 0.05:
                continue
            mkt_up = 0.5 * (d["Up_mid"][i] + (1 - d["Down_mid"][i])) if np.isfinite(d["Up_mid"][i]) and np.isfinite(d["Down_mid"][i]) else np.nan
            if not np.isfinite(mkt_up):
                continue
            zt = f["z_twap"][i]
            zs = f["z_spot"][i]
            if not (np.isfinite(zt) and np.isfinite(zs)):
                continue
            rows.append((wi, T, i, zt, zs, mkt_up, up_a, up_b, dn_a, dn_b))
    print(f"  samples: {len(rows)}  windows: {nW}")

    X = np.array([[r[3], r[4], r[5]] for r in rows])   # z_twap, z_spot, mkt_up
    y = np.array([1 if W[r[0]]["_winner"] == "Up" else 0 for r in rows])
    wids = np.array([r[0] for r in rows])
    times = np.array([r[1] for r in rows])

    # chronological split: train on first 60%, validate next 20%, test last 20%
    order = np.argsort(wids)
    ntr = int(0.6 * len(rows)); nva = int(0.2 * len(rows))
    tr = order[:ntr]; va = order[ntr:ntr + nva]; te = order[ntr + nva:]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    def fit(idx, feats):
        mu, sd = X[idx][:, feats].mean(0), X[idx][:, feats].std(0)
        sd[sd == 0] = 1
        lr = LogisticRegression(C=1.0, max_iter=3000)
        lr.fit((X[idx][:, feats] - mu) / sd, y[idx])
        return mu, sd, lr

    def pred(model, idx, feats):
        mu, sd, lr = model
        return lr.predict_proba((X[idx][:, feats] - mu) / sd)[:, 1]

    for name, feats in [("z_twap only", [0]), ("z_spot only", [1]),
                        ("z_twap+z_spot", [0, 1]), ("z+market", [0, 1, 2])]:
        m = fit(tr, feats)
        for split, idx in (("train", tr), ("valid", va), ("test", te)):
            p = pred(m, idx, feats)
            p_mkt = X[idx][:, 2]
            ll_m = log_loss(y[idx], np.clip(p_mkt, 1e-6, 1 - 1e-6))
            ll_c = log_loss(y[idx], np.clip(p, 1e-6, 1 - 1e-6))
            print(f"  {name:<16} {split:<6} n={len(idx):>5} logloss market={ll_m:.5f} "
                  f"model={ll_c:.5f}  {'MODEL WINS' if ll_c < ll_m else 'market wins'}")

    # ---- trade the OOS (valid+test) rows: model prob vs ask ----
    print("\n  --- trading the OOS rows (valid+test), model prob vs the ASK we pay ---")
    oos = np.concatenate([va, te])
    m = fit(tr, [0, 1])
    p_model = pred(m, oos, [0, 1])
    oos_rows = [rows[i] for i in oos]
    for fee_name, fee_fn in FEE_FNS.items():
        for margin in (0.0, 0.02, 0.05, 0.08, 0.12):
            pnls, wids_l = [], []
            ntrades = 0
            per_window = {}
            for j, r in enumerate(oos_rows):
                wi, T, i, zt, zs, mkt_up, up_a, up_b, dn_a, dn_b = r
                pu = p_model[j]
                # choose side with the larger model-vs-ask edge
                e_up = pu - up_a
                e_dn = (1 - pu) - dn_a
                if e_up >= e_dn:
                    side, ask = "Up", up_a
                else:
                    side, ask = "Down", dn_a
                edge = max(e_up, e_dn)
                if edge < margin:
                    continue
                if ask <= 0 or ask >= 1:
                    continue
                shares = STAKE / ask
                fee = fee_fn(ask, shares)
                won = (W[wi]["_winner"] == side)
                pnl = shares * (1.0 if won else 0.0) - STAKE - fee
                pnls.append(pnl); wids_l.append(wi)
                per_window.setdefault(wi, []).append(pnl)
            if not pnls:
                continue
            lo_ci, hi_ci = cluster_boot(wids_l, pnls)
            print(f"    fee={fee_name:<13} margin={margin:.2f}  N={len(pnls):>4} "
                  f"net=${sum(pnls):+8.2f} avg=${np.mean(pnls):+.3f} "
                  f"CI[{lo_ci:+.3f},{hi_ci:+.3f}]  windows={len(per_window)}")
    return rows, X, y


if __name__ == "__main__":
    W = load_windows()
    print(f"windows {len(W)}")
    q1 = q1_calibration(W)
    q2_model(W)
    with open(os.path.join(OUTDIR, "q1_calibration.json"), "w") as fh:
        json.dump(q1, fh, indent=2, default=float)
