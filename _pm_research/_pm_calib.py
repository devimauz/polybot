"""Stage 36: HIGH-POWER CALIBRATION + PROPER EXIT SIMULATION.

The trade test (51 windows) has limited power.  The SAME claim - "the market
misprices the endgame" - can be tested with far more power using CALIBRATION:
over all ~2000 endgame quotes, does the market's implied probability match the
realised win rate?  If the market systematically underprices the favourite by
X points, then:
    realised(favourite | market says p) = p + X
which is a much more powerful test (hundreds of windows) than 51 trades.

Also: simulate EXITS properly (the earlier "stop" test wrongly applied the
stop to every trade).  A TP/stop must be checked against the actual snapshot
path after entry, using the executable bid.
"""
import os, pickle
import numpy as np
from scipy.stats import binom

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
STAKE = 10.0


def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]): last = a[i]
        elif np.isfinite(last): a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]): nxt = a[i]
        elif np.isfinite(nxt): a[i] = nxt
    return a


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]; d["ws"] = int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


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
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]; n = len(t)
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n: continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998): continue
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                if not (np.isfinite(o) and 0.001 < o < 0.999): continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), mkt=float(mkt),
                                 won=1 if d["_winner"] == side else 0))
    print(f"endgame rows (both sides live, 2c spread, $10 depth): {len(rows)}")

    # ---------------- 1. CALIBRATION of the favourite, by market price bucket ----
    print("\n=== 1) CALIBRATION: favourite, realised win rate vs market-implied ===")
    print("  bucket          n    mkt    realised    gap     t      bins_p")
    fav = [r for r in rows if r["mkt"] >= 0.5]
    buckets = [(0.50, 0.70), (0.70, 0.80), (0.80, 0.86), (0.86, 0.90),
               (0.90, 0.93), (0.93, 0.96), (0.96, 0.98), (0.98, 1.001)]
    tot_gap_num = 0.0; tot_gap_den = 0
    for lo, hi in buckets:
        s = [r for r in fav if lo <= r["mkt"] < hi]
        if len(s) < 10: continue
        w = np.array([r["won"] for r in s]); m = np.array([r["mkt"] for r in s])
        gap = w.mean() - m.mean()
        se = np.sqrt(max(m.mean() * (1 - m.mean()), 1e-9) / len(s))
        pv = float(binom.sf(int(w.sum()) - 1, len(w), m.mean()))
        print(f"  [{lo:.2f},{hi:.3f}) {len(s):>5} {m.mean():>7.4f} {w.mean():>9.4f} "
              f"{gap:+8.4f} {gap/se if se>0 else 0:>6.2f}  {pv:.4f}")

    # overall calibration with a window-clustered t on (realised - mkt)
    print("\n=== 2) POOLED calibration test: realised - mkt, per WINDOW ===")
    byw = {}
    for r in fav:
        byw.setdefault(r["wi"], []).append(r)
    gaps, wi = [], []
    for w_, rs in byw.items():
        gaps.append(np.mean([r["won"] for r in rs]) - np.mean([r["mkt"] for r in rs]))
        wi.append(w_)
    gaps = np.array(gaps); wi = np.array(wi)
    m, t, nwin = cluster_t(wi, gaps)
    print(f"  windows={nwin}  mean gap = {m:+.4f}  cluster-t = {t:+.2f}")
    print(f"  (if the market were efficient, the gap would be 0)")
    # binomial on the pooled trades
    allw = np.array([r["won"] for r in fav]); allm = np.array([r["mkt"] for r in fav])
    print(f"  pooled trades: n={len(fav)} expected wins if efficient="
          f"{allm.sum():.1f} observed={int(allw.sum())} "
          f"binom_p={float(binom.sf(int(allw.sum())-1, len(fav), allm.mean())):.5f}")

    # ---------------- 3. same for the UNDERDOG: must be symmetric --------------
    print("\n=== 3) SYMMETRY: underdog, realised vs market-implied ===")
    udg = [r for r in rows if r["mkt"] < 0.5]
    byw2 = {}
    for r in udg:
        byw2.setdefault(r["wi"], []).append(r)
    g2, w2i = [], []
    for w_, rs in byw2.items():
        g2.append(np.mean([r["won"] for r in rs]) - np.mean([r["mkt"] for r in rs]))
        w2i.append(w_)
    if g2:
        m2, t2, n2 = cluster_t(w2i, np.array(g2))
        print(f"  windows={n2}  mean gap = {m2:+.4f}  cluster-t = {t2:+.2f}")
        print("  -> favourite gap and underdog gap should be OPPOSITE (zero-sum market)")

    # ---------------- 4. calibration over the WHOLE window (not just endgame) --
    print("\n=== 4) is the mispricing ENDGAME-SPECIFIC? calibration by T bucket ===")
    print("  T     n_fav   mkt    realised    gap")
    for T in range(0, 300, 20):
        s = [r for r in rows if T <= r["T"] < T + 20 and r["mkt"] >= 0.5]
        if len(s) < 10: continue
        w = np.array([r["won"] for r in s]); m = np.array([r["mkt"] for r in s])
        print(f"  {T:>3} {len(s):>6} {m.mean():>7.4f} {w.mean():>9.4f} {w.mean()-m.mean():>+8.4f}")

    # ---------------- 5. PROPER EXIT SIMULATION -------------------------------
    print("\n=== 5) EXIT SIMULATION on the strategy trades (using the real bid path) ===")
    def first(rs):
        best = {}
        for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
            best.setdefault(r["wi"], r)
        return list(best.values())
    sel = first([r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5 and 0.80 <= r["ask"] < 0.99])
    byw3 = {d["ws"]: d for d in W}
    print(f"  strategy trades: N={len(sel)}")

    def sim(tp=None, stop=None):
        pnls = []
        for r in sel:
            d = byw3[r["ws"]]
            t = d["t"]; side = r["side"]
            bid = d[side + "_bid"]; ask_a = d[side + "_ask"]
            i0 = int(np.searchsorted(t, r["T"]))
            exit_px = None; exit_idx = None
            for i in range(i0 + 1, len(t)):
                b = bid[i]
                if not np.isfinite(b): continue
                if stop is not None and b <= stop:
                    exit_px = b; exit_idx = i; break
                if tp is not None and b >= tp:
                    exit_px = b; exit_idx = i; break
            if exit_px is not None:
                sh = STAKE / r["ask"]
                # sell at the bid (taker exit) - fee on the sale
                pnl = sh * exit_px - STAKE - FEE(r["ask"], sh) - FEE(exit_px, sh)
            else:
                sh = STAKE / r["ask"]
                pnl = sh * r["won"] - STAKE - FEE(r["ask"], sh)
            pnls.append(pnl)
        return np.array(pnls)

    for lbl, tp, stop in (("hold to settlement", None, None),
                          ("TP 0.95", 0.95, None), ("TP 0.97", 0.97, None),
                          ("TP 0.98", 0.98, None), ("TP 0.99", 0.99, None),
                          ("stop 0.60", None, 0.60), ("stop 0.50", None, 0.50),
                          ("TP 0.99 / stop 0.60", 0.99, 0.60),
                          ("TP 0.99 / stop 0.75", 0.99, 0.75)):
        p = sim(tp, stop)
        if p is None or len(p) == 0: continue
        m, t, w_ = cluster_t([r["wi"] for r in sel], p)
        worst = p.min()
        print(f"  {lbl:<22} tot=${p.sum():+8.2f} avg=${p.mean():+.3f} t={t:+6.2f} "
              f"worst=${worst:+.2f} wins={int((p>0).sum())}/{len(p)}")


if __name__ == "__main__":
    main()
