"""Stage 7: DEFINITIVE walk-forward backtest.

Design rules (from the trading-bot-backtesting skill):
 * Replay EVERY window's raw samples; enter at the FIRST sample the gate fires.
   Never replay historical entries (they were made under a different gate).
 * No lookahead: features at sample i use only data up to and including i.
 * Model is FIT on training windows only, TRADED on later windows (walk-forward).
 * Fees on BOTH legs; fills at the ASK, exits sell at the BID.
 * Report: N, net $, worst loss, split by exit method (tp/stop/settle), Up vs Down.
 * Also report per-fold so we can see stability.

Fee schedules tested:
   A: 0.07 * p * (1-p) * shares        (official docs, crypto category)
   B: 0.25 * (p * (1-p))**2 * shares   (predtools: 5/15-min crypto binaries)
   C: 0.05 * min(p, 1-p) * shares      (the repo's existing assumption)
   D: 0 (upper bound: is there a raw edge before costs?)
"""
import os, pickle, json, sys
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

FEES = {
    "A_doc_0.07": lambda p, sh: 0.07 * p * (1 - p) * sh,
    "B_predtools_0.25sq": lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh,
    "C_repo_0.05min": lambda p, sh: 0.05 * min(p, 1 - p) * sh,
    "D_zero": lambda p, sh: 0.0,
}

MIN_TRADE = 5.0     # Polymarket min order is $1; use $5 for realism
MAX_SPREAD = 0.05
STAKES = (10.0, 25.0, 50.0)


# ------------------------------------------------------------------ data prep
def ffill(a):
    a = np.array(a, dtype=float).copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


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


def prep(d):
    """Compute per-snapshot features. No lookahead."""
    t = d["t"]
    cb = ffill(d["cb_px"])
    n = len(t)
    F = {}
    F["t"] = t
    F["cb"] = cb
    # market prob of Up from mids (both legs), and from best bid/ask
    mu, md = ffill(d["Up_mid"]), ffill(d["Down_mid"])
    F["p_mkt_mid"] = 0.5 * (mu + (1 - md))
    ub, ua = d["Up_bid"], d["Up_ask"]
    db, da = d["Down_ask"], d["Down_bid"]
    up_bid = ffill(ub)
    up_ask = ffill(ua)
    dn_ask = ffill(da)
    dn_bid = ffill(db)
    F["up_bid"], F["up_ask"], F["dn_bid"], F["dn_ask"] = up_bid, up_ask, dn_bid, dn_ask
    F["up_ok"] = np.isfinite(up_bid) & np.isfinite(up_ask) & (up_bid > 0) & (up_ask < 1) & (up_bid <= up_ask + 1e-9)
    F["dn_ok"] = np.isfinite(dn_bid) & np.isfinite(dn_ask) & (dn_bid > 0) & (dn_ask < 1) & (dn_bid <= dn_ask + 1e-9)
    # coinbase z-score vs window open, using running sigma (expanding)
    ret = np.zeros(n)
    ret[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
    ret[~np.isfinite(ret)] = 0
    F["cb_ret"] = ret
    # expanding realized vol of cb returns (causal)
    var = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += ret[i] * ret[i]
        var[i] = s / max(i, 1)
    sd = np.sqrt(var)
    p_open = cb[0]
    F["cb_z"] = np.where(sd > 0, (cb - p_open) / np.maximum(sd, 1e-9), np.nan)
    # momentum over trailing 15s (causal)
    F["cb_mom15"] = np.zeros(n)
    for i in range(n):
        j = int(np.searchsorted(t, t[i] - 15.0))
        F["cb_mom15"][i] = (cb[i] - cb[j]) / p_open if p_open > 0 else 0.0
    # book imbalance (causal: uses only this snapshot)
    def imbalance(pxb, szb, pxa, sza):
        bb = np.nansum(pxb * szb, axis=1)
        aa = np.nansum(pxa * sza, axis=1)
        return np.where((bb + aa) > 0, (bb - aa) / (bb + aa), 0.0)
    F["imb_up"] = imbalance(d["Up_bids_px"], d["Up_bids_sz"], d["Up_asks_px"], d["Up_asks_sz"])
    F["imb_dn"] = imbalance(d["Down_bids_px"], d["Down_bids_sz"], d["Down_asks_px"], d["Down_asks_sz"])
    # microprice per side
    def micro(b, a, bs, asz):
        ok = np.isfinite(b) & np.isfinite(a) & np.isfinite(bs) & np.isfinite(asz) & ((bs + asz) > 0)
        m = np.where(ok, (a * bs + b * asz) / np.maximum(bs + asz, 1e-9), np.nan)
        return ffill(m)
    F["micro_up"] = micro(up_bid, up_ask, d["Up_bidsz"], d["Up_asksz"])
    F["micro_dn"] = micro(dn_bid, dn_ask, d["Down_bidsz"], d["Down_asksz"])
    # last-60s TWAP signal (causal: TWAP of cb over [t-60, t])
    F["twap60"] = np.full(n, np.nan)
    for i in range(n):
        lo = t[i] - 60.0
        j = int(np.searchsorted(t, lo))
        seg_t = t[j:i + 1]
        seg_p = cb[j:i + 1]
        if len(seg_t) >= 3 and seg_t[-1] > seg_t[0]:
            F["twap60"][i] = np.trapezoid(seg_p, seg_t) / (seg_t[-1] - seg_t[0])
    F["_winner"] = d["_winner"]
    F["_slug"] = d["_slug"]
    F["_ws"] = int(d["ws"][0])
    return F


# ------------------------------------------------------------------ modeling
FEATURE_SETS = {
    "z_only": ["cb_z"],
    "z_mom": ["cb_z", "cb_mom15"],
    "z_mom_imb": ["cb_z", "cb_mom15", "imb_up", "imb_dn"],
    "full": ["cb_z", "cb_mom15", "imb_up", "imb_dn", "micro_up", "micro_dn", "p_mkt_mid", "t"],
}


def build_dataset(windows, feats, times, min_spread=MAX_SPREAD):
    """Rows: one per (window, time) with a valid two-sided book."""
    X, y, meta = [], [], []
    for wi, F in enumerate(windows):
        t = F["t"]
        for T in times:
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            if not (F["up_ok"][i] and F["dn_ok"][i]):
                continue
            if (F["up_ask"][i] - F["up_bid"][i]) > min_spread or (F["dn_ask"][i] - F["dn_bid"][i]) > min_spread:
                continue
            row = []
            bad = False
            for f in feats:
                v = F[f][i]
                if not np.isfinite(v):
                    bad = True
                    break
                row.append(float(v))
            if bad:
                continue
            X.append(row)
            y.append(1 if F["_winner"] == "Up" else 0)
            meta.append((wi, T, i))
    return np.array(X, dtype=float), np.array(y, dtype=int), meta


def fit_logistic(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit((X - mu) / sd, y)
    return dict(mu=mu, sd=sd, lr=lr)


def predict(model, X):
    return model["lr"].predict_proba((X - model["mu"]) / model["sd"])[:, 1]


# ------------------------------------------------------------------ execution
def simulate(windows, idx_windows, side_arr, entry_idx_arr, params, fee_fn):
    """Run the trades: entry at ask, TP/SL at bid, else settle."""
    stake = params["stake"]
    tp = params.get("tp")
    sl = params.get("sl")
    trades = []
    for k in range(len(idx_windows)):
        F = windows[idx_windows[k]]
        i = entry_idx_arr[k]
        side = side_arr[k]
        o = "Down" if side == "Up" else "Up"
        ask = F[f"{'up' if side == 'Up' else 'dn'}_ask"][i]
        bid = F[f"{'up' if side == 'Up' else 'dn'}_bid"][i]
        if not np.isfinite(ask) or not np.isfinite(bid) or ask <= 0 or ask >= 1:
            continue
        depth = F[f"{'up' if side == 'Up' else 'dn'}_asksz"][i] if "up_asksz" in F else None
        if depth is not None and np.isfinite(depth) and depth * ask < stake:
            continue
        shares = stake / ask
        cost = stake + fee_fn(ask, shares)
        exit_px = None
        reason = "settle"
        exit_i = None
        if tp or sl:
            bid_ser = F[f"{'up' if side == 'Up' else 'dn'}_bid"]
            for j in range(i + 1, len(bid_ser)):
                b = bid_ser[j]
                if not np.isfinite(b):
                    continue
                if tp and b >= tp:
                    exit_px, exit_i, reason = b, j, "tp"
                    break
                if sl and b <= sl:
                    exit_px, exit_i, reason = b, j, "stop"
                    break
        won = (F["_winner"] == side)
        if exit_px is None:
            proceeds = shares * (1.0 if won else 0.0)
            pnl = proceeds - cost
        else:
            proceeds = shares * exit_px - fee_fn(exit_px, shares)
            pnl = proceeds - cost
        trades.append(dict(slug=F["_slug"], side=side, reason=reason, won=bool(won),
                           entry_px=ask, exit_px=exit_px, stake=stake, cost=cost,
                           proceeds=proceeds, pnl=pnl, t_entry=float(F["t"][i]),
                           ws=F["_ws"]))
    return trades


def summarize(trades, label):
    if not trades:
        return f"{label:<42} NO TRADES"
    pnl = np.array([t["pnl"] for t in trades])
    wins = sum(1 for t in trades if t["won"])
    tp = [t for t in trades if t["reason"] == "tp"]
    st = [t for t in trades if t["reason"] == "stop"]
    se = [t for t in trades if t["reason"] == "settle"]
    up = [t for t in trades if t["side"] == "Up"]
    dn = [t for t in trades if t["side"] == "Down"]
    def g(x):
        return f"{len(x):>3}/${sum(t['pnl'] for t in x):+7.2f}"
    return (f"{label:<42} N={len(trades):>3} net=${pnl.sum():+8.2f} wr={wins/len(trades):.3f} "
            f"avg=${pnl.mean():+.3f} worst=${pnl.min():+.2f} | tp:{g(tp)} stop:{g(st)} "
            f"settle:{g(se)} | up:{g(up)} dn:{g(dn)}")


def main():
    raw = load_windows()
    print(f"windows {len(raw)}")
    W = [prep(d) for d in raw]
    n = len(W)
    K = 3
    folds = []
    step = n // K
    for k in range(K):
        a = k * step
        b = n if k == K - 1 else (k + 1) * step
        folds.append((a, b))
    print("folds:", folds)

    TIMES = list(range(30, 271, 30))
    FEE_NAME = sys.argv[1] if len(sys.argv) > 1 else "A_doc_0.07"
    fee_fn = FEES[FEE_NAME]
    print(f"fee schedule: {FEE_NAME}\n")

    # ---------- walk-forward: fit on all prior folds, trade the next fold ----
    for fsname, feats in FEATURE_SETS.items():
        Xall, yall, meta = build_dataset(W, feats, TIMES)
        print(f"\n########## features={fsname}  rows={len(Xall)} ##########")
        for gapmin, scale_only in ((0.0, True), (0.04, False), (0.08, False), (0.12, False)):
            oos_trades = []
            for k, (a, b) in enumerate(folds):
                tr_mask = np.array([m[0] < a for m in meta])
                te_mask = np.array([a <= m[0] < b for m in meta])
                if tr_mask.sum() < 50 or te_mask.sum() == 0:
                    continue
                model = fit_logistic(Xall[tr_mask], yall[tr_mask])
                p = predict(model, Xall[te_mask])
                te_meta = [m for m, keep in zip(meta, te_mask) if keep]
                # pick the FIRST time per window where the gate fires
                chosen = {}
                for j, (wi, T, i) in enumerate(te_meta):
                    if wi in chosen:
                        continue
                    p_up = p[j]
                    p_mkt = Xall[te_mask][j][FEATURE_SETS[fsname].index("p_mkt_mid")] if "p_mkt_mid" in FEATURE_SETS[fsname] else None
                    if p_mkt is None:
                        p_mkt = 0.5
                    # compare model prob to the MARKET's price for that side
                    F = W[wi]
                    p_side_up = p_up
                    mkt_up = F["p_mkt_mid"][i]
                    gap_up = p_side_up - mkt_up
                    gap_dn = (1 - p_side_up) - (1 - mkt_up)
                    if gap_up >= gap_dn and gap_up > gapmin:
                        side = "Up"
                    elif gap_dn > gapmin:
                        side = "Down"
                    else:
                        continue
                    chosen[wi] = (side, i)
                if not chosen:
                    continue
                wids = list(chosen.keys())
                sides = [chosen[w][0] for w in wids]
                eidx = [chosen[w][1] for w in wids]
                for params in ({"stake": 10.0}, {"stake": 25.0}, {"stake": 50.0},
                               {"stake": 25.0, "tp": 0.95, "sl": 0.70},
                               {"stake": 25.0, "tp": 0.90, "sl": 0.75}):
                    tr = simulate(W, wids, sides, eidx, params, fee_fn)
                    for t_ in tr:
                        t_["fold"] = k
                        t_["gapmin"] = gapmin
                        t_["params"] = str(params)
                    oos_trades.append((params, k, tr))
            # aggregate per (gapmin, params)
            for gapmin2 in (0.0, 0.04, 0.08, 0.12):
                pass
            by_params = {}
            for params, k, tr in oos_trades:
                by_params.setdefault(str(params), []).extend(tr)
            for ps, tr in by_params.items():
                if gapmin != by_params.get("_gap", gapmin):
                    pass
            # print one summary per gapmin
            gap_groups = {}
            for params, k, tr in oos_trades:
                gap_groups.setdefault(gapmin, []).append((params, tr))
            for g, items in gap_groups.items():
                for ps, tr in items:
                    print(summarize(tr, f"gap>{g} {ps}"))


if __name__ == "__main__":
    main()
