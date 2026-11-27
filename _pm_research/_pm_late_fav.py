"""Stage 15: LATE-WINDOW EXTREME-PRICE EDGE (the last remaining structural asymmetry).

Why this can exist even though the martingale test says "no taker edge":
  * The taker fee is 0.07*p*(1-p) per share. At p=0.99 that is 0.0007/share
    (0.07% of notional) -- essentially FREE. At p=0.50 it is 0.0175/share
    (3.5% of notional) -- prohibitive. So only the EXTREMES are tradeable.
  * Calibration is NOT time-invariant: over ALL times, ask 0.96-1.00 wins only
    96.2% (a loser). But in the last 30s, ask 0.99 won 37/37. A price of 0.99
    at t=250 means "nearly settled"; at t=30 it means "probably 99% right".
  => Buy the near-certain favourite LATE, hold to settlement: pay 0.95-0.99,
     collect $1, pay ~0 fees, and let the (rare) reversal be the only risk.

This is the classic "buy the 95-99c favourite in the endgame" trade.
Tests: holdout, cluster-robust t, block sign test, drop-best-k, price-band
plateau, time-window sweep, and the "how often does it flip" tail risk.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
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


def late_fav_trades(W, p_lo, p_hi, t_lo, t_hi, stake=STAKE, max_spread=0.05):
    """Buy the side with the HIGHER ask, if it is in [p_lo,p_hi], at the first
    qualifying snapshot in [t_lo,t_hi]. Hold to settlement."""
    out = []
    for wi, d in enumerate(W):
        t = d["t"]
        chosen = None
        for i in range(len(t)):
            ti = t[i]
            if ti < t_lo:
                continue
            if ti > t_hi:
                break
            for side in ("Up", "Down"):
                o = "Down" if side == "Up" else "Up"
                b, a = d[side + "_bid"][i], d[side + "_ask"][i]
                ob, oa = d[o + "_bid"][i], d[o + "_ask"][i]
                if not (np.isfinite(b) and np.isfinite(a) and np.isfinite(ob) and np.isfinite(oa)):
                    continue
                if not (0 < ob <= oa < 1) or not (0 < b <= a < 1):
                    continue
                if (a - b) > max_spread or (oa - ob) > max_spread:
                    continue
                if a < oa:      # only take the FAVOURITE side
                    continue
                if not (p_lo <= a <= p_hi):
                    continue
                asz = d[side + "_asksz"][i]
                if not np.isfinite(asz) or asz * a < stake:
                    continue
                chosen = (side, i, float(a), float(b))
                break
            if chosen:
                break
        if not chosen:
            continue
        side, i, a, b = chosen
        shares = stake / a
        fee = FEE(a, shares)
        won = (d["_winner"] == side)
        pnl = shares * (1.0 if won else 0.0) - stake - fee
        out.append(dict(wi=wi, slug=d["_slug"], ws=int(d["ws"][0]), side=side, t=float(t[i]),
                        ask=a, won=won, pnl=pnl, shares=shares))
    return out


def cluster_t(wi, vals):
    wi = np.asarray(wi); vals = np.asarray(vals, dtype=float)
    uq = np.unique(wi)
    s = np.array([vals[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq))
    return m, m / se if se > 0 else np.nan, len(uq)


def main():
    W = load_windows()
    print(f"windows {len(W)}\n")

    # ---- 1. time-window x price-band sweep (the core grid) ----
    print("=" * 104)
    print("1) LATE FAVOURITE: win rate vs ask, by time window and price band")
    print("=" * 104)
    TB = [(60, 120), (120, 180), (180, 220), (220, 250), (250, 270), (270, 300)]
    PB = [(0.80, 0.95), (0.90, 0.97), (0.93, 0.98), (0.95, 0.99), (0.97, 1.01)]
    for tlo, thi in TB:
        cells = []
        for plo, phi in PB:
            tr = late_fav_trades(W, plo, phi, tlo, thi)
            if len(tr) < 8:
                cells.append(f"n<8")
                continue
            w = np.mean([t["won"] for t in tr])
            ma = np.mean([t["ask"] for t in tr])
            p = np.mean([t["pnl"] for t in tr])
            cells.append(f"n{len(tr):>3} w{w:.3f} ask{ma:.3f} ev{p:+.3f}")
        print(f"  t[{tlo:>3},{thi:>3}]: " + " | ".join(f"{c:>22}" for c in cells))
    print("  (bands: 0.80-0.95 | 0.90-0.97 | 0.93-0.98 | 0.95-0.99 | 0.97-1.01)")

    # ---- 2. the pre-registered best cell, with full stats ----
    print("\n" + "=" * 104)
    print("2) PRE-REGISTERED: band 0.93-0.99, t[240,300], hold to settlement, $10")
    print("=" * 104)
    tr = late_fav_trades(W, 0.93, 0.99, 240, 300)
    pnl = np.array([t["pnl"] for t in tr])
    won = np.array([t["won"] for t in tr], dtype=float)
    ask = np.array([t["ask"] for t in tr])
    wi = np.array([t["wi"] for t in tr])
    print(f"  N={len(tr)}  total=${pnl.sum():+.2f}  avg=${pnl.mean():+.3f}  win={won.mean():.4f} "
          f"meanAsk={ask.mean():.4f}  worst=${pnl.min():+.2f}")
    m, tstat, nw = cluster_t(wi, pnl)
    print(f"  cluster-robust (window as unit): mean=${m:+.3f} t={tstat:+.2f} (n_windows={nw})")
    print(f"  edge per share: {won.mean()-ask.mean():+.4f}  (need > {0.07*ask.mean()*(1-ask.mean()):.4f} to clear the fee)")
    # holdout
    wss = sorted(set(t["ws"] for t in tr))
    cut = wss[len(wss) // 2]
    A = [t for t in tr if t["ws"] <= cut]
    B = [t for t in tr if t["ws"] > cut]
    for nm, S in (("first half ", A), ("SECOND half", B)):
        p = np.array([x["pnl"] for x in S])
        w = np.array([x["won"] for x in S], dtype=float)
        print(f"    {nm}: N={len(S):>3} total=${p.sum():+7.2f} avg=${p.mean():+.3f} win={w.mean():.4f}")
    # block sign test
    print("  block sign test (6 blocks):")
    edges = np.linspace(wss[0], wss[-1], 7)
    pos = 0
    for b in range(6):
        sel = [t for t in tr if edges[b] <= t["ws"] <= edges[b + 1]]
        if not sel:
            continue
        v = np.mean([t["pnl"] for t in sel])
        pos += int(v > 0)
        print(f"    block {b+1}: n={len(sel):>3} avg=${v:+.3f}")
    print(f"    positive blocks: {pos}/6")
    # drop-best-k
    order = np.argsort(-pnl)
    for k in (0, 1, 2, 3, 5):
        keep = np.ones(len(tr), dtype=bool); keep[order[:k]] = False
        p = pnl[keep]; w = won[keep]; a = ask[keep]
        print(f"    drop top {k}: N={len(p):>3} total=${p.sum():+7.2f} avg=${p.mean():+.3f} "
              f"win={w.mean():.4f} edge={w.mean()-a.mean():+.4f}")

    # ---- 3. tail risk: how often does the 0.93+ favourite lose? ----
    print("\n" + "=" * 104)
    print("3) TAIL RISK at the extreme (t[240,300])")
    print("=" * 104)
    for plo, phi in [(0.95, 0.99), (0.97, 1.01), (0.98, 1.01), (0.99, 1.01)]:
        tr = late_fav_trades(W, plo, phi, 240, 300)
        if not tr:
            continue
        losses = [t for t in tr if not t["won"]]
        pnl = np.array([t["pnl"] for t in tr])
        print(f"  ask {plo:.2f}-{phi:.2f}: N={len(tr):>3} win={1-len(losses)/len(tr):.4f} "
              f"total=${pnl.sum():+7.2f} avg=${pnl.mean():+.3f} | losses={len(losses)} "
              f"worst=${pnl.min():+.2f} | max consecutive loss risk: 1 loss costs {-pnl.min():.2f}")

    # ---- 4. stake scaling ----
    print("\n" + "=" * 104)
    print("4) STAKE SENSITIVITY (band 0.93-0.99, t[240,300])")
    print("=" * 104)
    for st in (5, 10, 25, 50, 100):
        tr = late_fav_trades(W, 0.93, 0.99, 240, 300, stake=st)
        if not tr:
            continue
        p = np.array([t["pnl"] for t in tr])
        print(f"  stake=${st:>3}: N={len(tr):>3} total=${p.sum():+7.2f} avg=${p.mean():+.3f} "
              f"worst=${p.min():+.2f} roi={p.sum()/(st*len(tr))*100:+.2f}%")

    with open(os.path.join(OUTDIR, "late_fav.json"), "w") as fh:
        json.dump({"n": len(tr), "total": float(pnl.sum()), "avg": float(pnl.mean()),
                   "win": float(won.mean())}, fh, indent=2)


if __name__ == "__main__":
    main()
