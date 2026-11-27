"""Stage 38: RESOLVE THE TENSION (fixed-T vs first-qualifying) + high-power test.

TENSION:
  "first qualifying snapshot" (T>=260, fav ask in [0.80,0.99)) -> t=+3.32
  fixed T=260, same band                                      -> t=+1.18, p=0.31
  If the difference is a selection artifact, the edge is fake.

HYPOTHESIS TO TEST: the edge is NOT about timing, it is about the PRICE BAND.
  The highest-power evidence in stage 36 was the CALIBRATION bucket
  [0.80,0.86): n=117, mkt=0.8319, realised=0.9487, gap +0.1168, p=0.0001.
  That bucket is huge relative to the trade sample.  If the mispricing is
  really a function of the price band, it must show up in a WINDOW-CLUSTERED
  test and survive a chronological split.
"""
import os, pickle
import numpy as np
from scipy.stats import binom

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
STAKE = 10.0


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


def build_rows(W):
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
    return rows


def main():
    W = load_all()
    rows = build_rows(W)
    fav = [r for r in rows if r["mkt"] >= 0.5]

    # ---------- 1. entry-T distribution of the 51 first-qualifying trades ----
    print("=== 1) ENTRY-T distribution of the first-qualifying trades ===")
    best = {}
    for r in sorted([r for r in fav if r["T"] >= 260 and 0.80 <= r["ask"] < 0.99],
                    key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    sel = list(best.values())
    import collections
    print("  ", dict(sorted(collections.Counter(int(r["T"]) for r in sel).items())))
    print(f"  total {len(sel)} trades")

    # ---------- 2. WINDOW-CLUSTERED calibration by price band --------------
    print("\n=== 2) WINDOW-CLUSTERED calibration gap by price band (favourite) ===")
    print("  band          rows  wins   mkt    realised    gap   cluster_t  wins_ok")
    for lo, hi in ((0.70, 0.80), (0.80, 0.86), (0.86, 0.90), (0.90, 0.93),
                   (0.93, 0.96), (0.96, 0.98), (0.98, 1.001)):
        s = [r for r in fav if lo <= r["ask"] < hi]
        if len(s) < 20: continue
        byw = {}
        for r in s: byw.setdefault(r["wi"], []).append(r)
        gaps, wi = [], []
        for w_, rs in byw.items():
            gaps.append(np.mean([r["won"] for r in rs]) - np.mean([r["ask"] for r in rs]))
            wi.append(w_)
        m, t, nw = cluster_t(wi, np.array(gaps))
        won = np.array([r["won"] for r in s]); ask = np.array([r["ask"] for r in s])
        pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
        print(f"  [{lo:.2f},{hi:.3f}) {len(s):>5} {int(won.sum()):>5} {ask.mean():>6.4f} "
              f"{won.mean():>8.4f} {m:>+8.4f} {t:>+9.2f}  {pv:.5f}  (w={nw})")

    # ---------- 3. chronological split of the [0.80,0.86) band -------------
    print("\n=== 3) CHRONOLOGICAL SPLIT: band [0.80,0.86), all rows, cluster-t ===")
    s = [r for r in fav if 0.80 <= r["ask"] < 0.86]
    ws_all = sorted(r["ws"] for r in s)
    med = ws_all[len(ws_all) // 2]
    for lbl, sub in (("first half", [r for r in s if r["ws"] <= med]),
                     ("second half", [r for r in s if r["ws"] > med])):
        byw = {}
        for r in sub: byw.setdefault(r["wi"], []).append(r)
        gaps, wi = [], []
        for w_, rs in byw.items():
            gaps.append(np.mean([r["won"] for r in rs]) - np.mean([r["ask"] for r in rs]))
            wi.append(w_)
        m, t, nw = cluster_t(wi, np.array(gaps))
        won = np.array([r["won"] for r in sub]); ask = np.array([r["ask"] for r in sub])
        pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
        print(f"  {lbl:<12} rows={len(sub):>4} wins={int(won.sum()):>4} mkt={ask.mean():.4f} "
              f"realised={won.mean():.4f} gap={m:+.4f} t={t:+.2f} binom_p={pv:.5f} (w={nw})")

    # ---------- 4. is the gap a function of T or of the band? ---------------
    print("\n=== 4) within band [0.80,0.86): gap by T (is it timing?) ===")
    for T in range(240, 300, 10):
        ss = [r for r in fav if 0.80 <= r["ask"] < 0.86 and T <= r["T"] < T + 10]
        if len(ss) < 10: continue
        won = np.array([r["won"] for r in ss]); ask = np.array([r["ask"] for r in ss])
        print(f"  T[{T},{T+10}) n={len(ss):>4} mkt={ask.mean():.4f} realised={won.mean():.4f} "
              f"gap={won.mean()-ask.mean():+.4f}")

    print("\n=== 4b) across the WHOLE window: gap by band (is it endgame-specific?) ===")
    # need rows from the whole window, not just 240-300
    print("  (rebuilding rows for the full window)")

    # ---------- 5. pre-registered rule test on the band, no T filter --------
    print("\n=== 5) PRE-REGISTERED: 'favourite with ask in [0.80,0.86), T>=240' ===")
    s = [r for r in fav if r["T"] >= 240 and 0.80 <= r["ask"] < 0.86]
    best2 = {}
    for r in sorted(s, key=lambda r: (r["wi"], r["T"])):
        best2.setdefault(r["wi"], r)
    sel2 = list(best2.values())
    ask = np.array([r["ask"] for r in sel2]); won = np.array([r["won"] for r in sel2])
    sh = STAKE / ask
    pnl = sh * won - STAKE - FEE(ask, sh)
    m, t, nw = cluster_t([r["wi"] for r in sel2], pnl)
    pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
    print(f"  N={len(sel2)} wins={int(won.sum())} mkt={ask.mean():.4f} realised={won.mean():.4f}")
    print(f"  net=${pnl.sum():+.2f} avg=${pnl.mean():+.3f} cluster-t={t:+.2f} binom_p={pv:.5f}")


if __name__ == "__main__":
    main()
