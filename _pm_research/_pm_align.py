"""Stage 35: ALIGNMENT + SYMMETRY PLACEBO (decisive sanity checks).

The endgame-favourite result (51 trades, 50 wins, t=+3.32) is the ONLY
positive result of this whole investigation.  Before shipping anything I
must rule out the mundane explanations:

  A. MISALIGNMENT: if the logged book snapshots were not actually aligned
     with the resolution (clock skew, wrong window, off-by-one), then
     "the favourite" could look right for a trivial reason.
     TEST: buy the UNDERDOG at exactly the same timestamps.  If the
     timestamps are aligned, the underdog must lose almost always.  If the
     underdog also wins, the alignment or the outcome mapping is broken.

  B. SYMMETRY: the claim is "the market UNDERPRICES the favourite".
     So the underdog is OVERPRICED and buying it must LOSE.  That is a
     second, independent placebo.

  C. TRADE-LEVEL: does the edge come from a handful of lucky windows?
     (block stability, PnL concentration, leave-one-out)
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


def pnl_of(ask, won):
    ask = np.asarray(ask, float); won = np.asarray(won, float)
    sh = STAKE / ask
    return sh * won - STAKE - FEE(ask, sh)


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
                                 mkt=float(mkt), won=1 if d["_winner"] == side else 0))

    def first(rs):
        best = {}
        for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
            best.setdefault(r["wi"], r)
        return list(best.values())

    TLO, AMIN = 260, 0.80
    fav = first([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5 and AMIN <= r["ask"] < 0.99])
    # underdog at the SAME timestamps: the other side of the same window/T
    udg = []
    for r in fav:
        cands = [x for x in rows if x["wi"] == r["wi"] and x["T"] == r["T"] and x["side"] != r["side"]]
        if cands: udg.append(cands[0])
    print(f"=== A/B) ALIGNMENT + SYMMETRY PLACEBO (T>={TLO}, fav ask>={AMIN}) ===")
    for lbl, sel in (("FAVOURITE (the strategy)", fav), ("UNDERDOG (same timestamps)", udg)):
        if not sel: continue
        ask = np.array([r["ask"] for r in sel]); won = np.array([r["won"] for r in sel])
        p = pnl_of(ask, won); m, t, w_ = cluster_t([r["wi"] for r in sel], p)
        pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
        print(f"  {lbl:<32} N={len(sel):>3} win={won.mean():.4f} ask={ask.mean():.4f} "
              f"tot=${p.sum():+8.2f} avg=${p.mean():+.3f} t={t:+6.2f} binom_p={pv:.4f}")
    print("  -> if the underdog also wins, the timestamps/outcomes are misaligned.")

    # A2: underdog bought at the SAME ask price (what a market maker would face)
    print("\n=== A2) underdog at the FAVOURITE's ask price (identical fill) ===")
    ask_f = np.array([r["ask"] for r in fav])
    won_u = np.array([1 - r["won"] for r in fav])
    p = pnl_of(ask_f, won_u); m, t, w_ = cluster_t([r["wi"] for r in fav], p)
    print(f"  underdog @ same price: win={won_u.mean():.4f} tot=${p.sum():+8.2f} t={t:+6.2f}")
    print("  (complementary P&L: the favourite and underdog P&L must sum to the fee+spread)")

    # C: leave-one-out on the winning strategy
    print("\n=== C) LEAVE-ONE-OUT + CONCENTRATION (favourite strategy) ===")
    ask = np.array([r["ask"] for r in fav]); won = np.array([r["won"] for r in fav])
    wi = np.array([r["wi"] for r in fav])
    p = pnl_of(ask, won)
    loo = []
    for i in range(len(fav)):
        keep = np.ones(len(fav), bool); keep[i] = False
        m, t, w_ = cluster_t(wi[keep], p[keep])
        loo.append(t)
    loo = np.array(loo)
    print(f"  full-sample t = {cluster_t(wi, p)[1]:+.2f}")
    print(f"  leave-one-out t: min={loo.min():+.2f} median={np.median(loo):+.2f} max={loo.max():+.2f}")
    print(f"  windows whose removal drops t below 2.0: {int((loo < 2.0).sum())}")
    srt = np.sort(p)[::-1]
    print(f"  top-1 trade = ${srt[0]:+.2f} ({srt[0]/p.sum()*100:.0f}% of total), "
          f"top-3 = {srt[:3].sum()/p.sum()*100:.0f}%")

    # D: what would the market have to believe?  probability-accounting
    print("\n=== D) PROBABILITY ACCOUNTING ===")
    print(f"  mean ask paid                 = {ask.mean():.4f}")
    print(f"  realised win rate              = {won.mean():.4f}")
    print(f"  gap (market underprices by)    = {won.mean()-ask.mean():+.4f}")
    print(f"  fee as fraction of price paid  = {0.07*ask.mean()*(1-ask.mean()):.4f}")
    print(f"  edge per share                 = {won.mean()-ask.mean()-0.07*ask.mean()*(1-ask.mean()):+.4f}")
    print(f"  break-even win rate            = "
          f"{(STAKE + FEE(ask.mean(), STAKE/ask.mean()))/(STAKE/ask.mean()):.4f}")

    # E: does the gap depend on the T at which we enter? (mechanism: lock-in)
    print("\n=== E) EDGE vs ENTRY TIME (favourite strategy, all qualifying T) ===")
    print("  T     n    mkt    realised   gap")
    for T in range(260, 300, 4):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and AMIN <= r["ask"] < 0.99]
        if len(s) < 5: continue
        s1 = first(s)
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        print(f"  {T:>3} {len(s1):>4} {m1.mean():>7.4f} {w1.mean():>10.4f} {w1.mean()-m1.mean():>+7.4f}")


if __name__ == "__main__":
    main()
