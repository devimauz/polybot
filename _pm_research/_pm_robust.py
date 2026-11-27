"""Stage 13: ROBUSTNESS of the longshot edge. Is it real or a few lucky windows?

The edge appeared only in the SECOND HALF of the sample (+$2.98/trade) vs the
first half (+$0.02/trade). That is the classic signature of either (a) a
regime change or (b) a handful of lucky windows. Distinguish them:

 1. Bootstrap CI clustered by window (windows are iid).
 2. Drop-the-best-k test: remove the k most profitable trades; if the edge dies
    with k=3, it was luck, not a strategy.
 3. Trades per window: are we counting the same window many times?
 4. Cumulative P&L curve in chronological order (a lucky stretch shows as a
    step; a real edge shows as a steady slope).
 5. Jackknife: leave-one-block-out, does the edge survive?
 6. Compare against the CALIBRATED NULL: if the market is fair, the expected
    win rate for entries at mean ask m is m. Test observed win rate vs m
    with a clustered bootstrap.
 7. Per-window detail dump for the biggest winners, with their timestamps,
    so we can see whether they cluster in time (a data glitch) or are spread.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

BAND = (0.06, 0.20)
T_LO, T_HI = 30, 150
STAKE = 10.0
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


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


def build(W, band=BAND, t_lo=T_LO, t_hi=T_HI, stake=STAKE):
    out = []
    for wi, d in enumerate(W):
        for side in ("Up", "Down"):
            t, bid, ask, asz = d["t"], d[side + "_bid"], d[side + "_ask"], d[side + "_asksz"]
            for i in range(len(t)):
                ti = t[i]
                if ti < t_lo:
                    continue
                if ti > t_hi:
                    break
                b, a = bid[i], ask[i]
                if not (np.isfinite(b) and np.isfinite(a)):
                    continue
                if b <= 0 or a >= 1 or b > a + 1e-9 or (a - b) > 0.05:
                    continue
                if a < band[0] or a > band[1]:
                    continue
                if not np.isfinite(asz[i]) or asz[i] * a < stake:
                    continue
                shares = stake / a
                fee = FEE(a, shares)
                won = (d["_winner"] == side)
                pnl = shares * (1.0 if won else 0.0) - STAKE - fee
                out.append(dict(wi=wi, slug=d["_slug"], ws=int(d["ws"][0]), side=side,
                                t=float(ti), ask=float(a), won=won, pnl=pnl,
                                shares=shares))
                break
    return out


def cluster_boot(wi, vals, n=4000, seed=13):
    wi = np.asarray(wi); vals = np.asarray(vals, dtype=float)
    uq = np.unique(wi)
    s = np.array([vals[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uq), size=(n, len(uq)))
    means = s[idx].sum(1) / c.sum()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    W = load_windows()
    tr = build(W)
    pnl = np.array([t["pnl"] for t in tr])
    wi = np.array([t["wi"] for t in tr])
    won = np.array([t["won"] for t in tr], dtype=float)
    ask = np.array([t["ask"] for t in tr])
    print(f"trades {len(tr)}  windows {len(set(wi.tolist()))}  total ${pnl.sum():+.2f}  avg ${pnl.mean():+.3f}")

    # ---- 1. clustered bootstrap ----
    lo, hi = cluster_boot(wi, pnl)
    print(f"\n1) clustered bootstrap CI of mean P&L: [{lo:+.3f}, {hi:+.3f}]  (mean {pnl.mean():+.3f})")

    # ---- 2. drop the best k ----
    print("\n2) DROP-THE-BEST-k (if the edge dies, it was luck):")
    order = np.argsort(-pnl)
    for k in (0, 1, 2, 3, 5, 8, 12):
        keep = np.ones(len(tr), dtype=bool)
        keep[order[:k]] = False
        p = pnl[keep]
        w = won[keep]
        a = ask[keep]
        print(f"   drop top {k:>2}: N={len(p):>3} total=${p.sum():+8.2f} avg=${p.mean():+.3f} "
              f"win={w.mean():.3f} meanAsk={a.mean():.3f} edge={w.mean()-a.mean():+.4f}")

    # ---- 3. trades per window ----
    from collections import Counter
    c = Counter(wi.tolist())
    hist = Counter(c.values())
    print(f"\n3) trades per window: {dict(sorted(hist.items()))}")
    multi = [w for w, n in c.items() if n > 1]
    print(f"   windows with >1 trade: {len(multi)}")

    # ---- 4. cumulative P&L curve ----
    print("\n4) cumulative P&L by trade index (looking for a step vs a slope):")
    cum = np.cumsum(pnl)
    step = max(1, len(cum) // 12)
    for i in range(0, len(cum), step):
        print(f"   trades {i:>3}-{min(i+step,len(cum)):>3}: cum ${cum[min(i+step,len(cum))-1]:+8.2f}")

    # ---- 5. leave-one-block-out ----
    print("\n5) LEAVE-ONE-BLOCK-OUT (8 blocks):")
    wss = sorted(set(t["ws"] for t in tr))
    edges = np.linspace(wss[0], wss[-1], 9)
    for b in range(8):
        sel = np.array([not (edges[b] <= t["ws"] <= edges[b + 1]) for t in tr])
        print(f"   drop block {b+1}: N={int(sel.sum()):>3} total=${pnl[sel].sum():+8.2f} avg=${pnl[sel].mean():+.3f}")

    # ---- 6. calibrated null ----
    print("\n6) CALIBRATED NULL: if the market is fair, E[win] = mean ask")
    print(f"   observed win rate {won.mean():.4f} vs mean ask {ask.mean():.4f}  -> edge {won.mean()-ask.mean():+.4f}")
    lo2, hi2 = cluster_boot(wi, won - ask)
    print(f"   clustered CI of (win - ask): [{lo2:+.4f}, {hi2:+.4f}]")
    # binomial: total wins vs sum of asks
    exp_wins = ask.sum()
    k = int(won.sum())
    from math import comb
    # normal approx with overdispersion
    p = ask.mean()
    se = np.sqrt(p * (1 - p) / len(ask))
    print(f"   wins {k} vs expected {exp_wins:.1f} (simple binomial z={((k-exp_wins)/np.sqrt(exp_wins*(1-p))):+.2f})")
    print(f"   iid binomial z={((k-exp_wins)/(len(ask)*se)):+.2f}  (ignores clustering)")
    # cluster-robust: treat window as unit
    per_win = []
    for w in set(wi.tolist()):
        m = wi == w
        per_win.append(won[m].sum() - ask[m].sum())
    per_win = np.array(per_win)
    tstat = per_win.mean() / (per_win.std(ddof=1) / np.sqrt(len(per_win)))
    print(f"   CLUSTER-ROBUST t (window as unit): {tstat:+.2f}  (n_windows={len(per_win)})")

    # ---- 7. biggest winners detail ----
    print("\n7) TOP 12 WINNERS (timestamps -> do they cluster?):")
    top = np.argsort(-pnl)[:12]
    for i in top:
        t = tr[i]
        import datetime
        print(f"   {t['slug']}  t={t['t']:>6.1f}s  side={t['side']:<4} ask={t['ask']:.3f} "
              f"pnl=${t['pnl']:+.2f}  time={datetime.datetime.utcfromtimestamp(t['ws'])}")
    print("\n   sum of top-12 P&L = $" + f"{pnl[top].sum():.2f} of total ${pnl.sum():.2f} "
          f"({pnl[top].sum()/pnl.sum()*100:.0f}%)")


if __name__ == "__main__":
    main()
