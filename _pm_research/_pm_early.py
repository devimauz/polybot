"""Stage 9: DECISIVE test of the early-window miscalibration hypothesis.

!! RE-TESTED ON THE CAUSAL CLOCK - REJECTED, DO NOT TRADE !!
This script uses the OPTIMISTIC logger clock (`d["t"]`).  Re-running the
same hypothesis on the source-aligned clock (`t_src`) gives:
    band[0.02,0.20) T[40,100)  N=139  net=+$381.27  t=+1.37  CI=[-1.15,+6.79]  top5=65%
    band[0.02,0.20) T[30,120)  N=171  net=+$420.81  t=+1.38  CI=[-0.79,+5.96]  top5=59%
    band[0.05,0.23) T[60,120)  N=195  net=+$323.39  t=+1.04  CI=[-1.35,+4.80]  top5=94%
    band[0.02,0.20) T[30,150)  N=217  net=+$311.73  t=+0.93  CI=[-1.49,+4.43]  top5=80%
Every CI includes zero and the top 5 trades carry 59-94% of the P&L.  The
large net figure rests on a handful of lottery fills, not a systematic edge.
See ENDGAME_EDGE.md, "what failed".

Hypothesis H: EARLY in the window (t < ~90s) the market's price is systematically
MISCALIBRATED, monotonically: longshots (ask<0.30) are UNDERPRICED and favourites
(ask>0.70) are OVERPRICED. The effect decays/vanishes later in the window.

This script:
 1. Measures calibration by (time bucket x ask band) with Wilson CIs.
 2. Tests the STRATEGY: buy longshots early (and the mirror: buy favourites early).
 3. Walk-forward: choose the band on the FIRST half of the data, trade the SECOND
    half (and vice versa). Reports OOS only.
 4. Bootstrap CI over windows (windows are the iid unit).
 5. Does the effect survive the fee schedules? (0.07 doc / 0.25sq predtools / zero)
 6. WHY: is it mean reversion of the market price toward the open? Measures the
    market's own price path after the entry (does the favourite's ask fall back?).
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


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


def find_entry(d, side, t_lo, t_hi, p_lo, p_hi, stake, max_spread=0.05):
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
        if b <= 0 or a >= 1 or b > a + 1e-9 or (a - b) > max_spread:
            continue
        if a < p_lo or a > p_hi:
            continue
        if not np.isfinite(asz[i]) or asz[i] * a < stake:
            continue
        return dict(i=i, t=float(ti), ask=float(a), bid=float(b))
    return None


def pnl_hold(entry, d, side, stake, fee_fn):
    a = entry["ask"]
    shares = stake / a
    fee = fee_fn(a, shares)
    won = (d["_winner"] == side)
    return shares * (1.0 if won else 0.0) - stake - fee, won


def boot_ci(vals, n_boot=4000, seed=7):
    v = np.asarray(vals, dtype=float)
    if len(v) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    W = load_windows()
    STAKE = 10.0
    print(f"windows {len(W)}\n")

    # ---------- 1. calibration table (early vs late) ----------
    print("=== calibration: win rate vs ask, by time bucket (Wilson CI) ===")
    BANDS = [(0.01, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.45), (0.45, 0.55),
             (0.55, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.96), (0.96, 1.01)]
    TB = [(30, 60), (60, 90), (90, 120), (120, 180), (180, 240), (240, 300)]
    print(f"{'band':>12} " + " ".join(f"{'t'+str(a)+'-'+str(b):>22}" for a, b in TB))
    for lo, hi in BANDS:
        cells = []
        for tlo, thi in TB:
            recs = []
            for d in W:
                for side in ("Up", "Down"):
                    e = find_entry(d, side, tlo, thi, lo, hi, STAKE)
                    if e is None:
                        continue
                    recs.append((e["ask"], d["_winner"] == side))
            if len(recs) < 10:
                cells.append(f"{'n<10':>22}")
                continue
            k = sum(1 for r in recs if r[1])
            n = len(recs)
            l, h = wilson(k, n)
            ma = np.mean([r[0] for r in recs])
            cells.append(f"{ma:.3f} w{k/n:.3f} n{n} [{l:.2f},{h:.2f}]")
        print(f"{lo:.2f}-{hi:.2f} " + " ".join(f"{c:>22}" for c in cells))

    # ---------- 2. the strategy, with walk-forward ----------
    print("\n=== strategy: buy LONG SHOTS early (band, time window) ===")
    print("    hold-to-settlement; P&L per $10 stake; boot CI over windows")
    cands = []
    for (lo, hi) in [(0.02, 0.30), (0.05, 0.25), (0.10, 0.30), (0.10, 0.40),
                     (0.15, 0.35), (0.02, 0.20), (0.20, 0.40), (0.30, 0.50)]:
        for (tlo, thi) in [(30, 60), (30, 90), (40, 100), (60, 120), (30, 120), (30, 150), (30, 300)]:
            recs = []
            for d in W:
                for side in ("Up", "Down"):
                    e = find_entry(d, side, tlo, thi, lo, hi, STAKE)
                    if e is None:
                        continue
                    p, won = pnl_hold(e, d, side, STAKE, FEE_FNS["doc_0.07"])
                    recs.append((d["_slug"], int(d["ws"][0]), side, e["ask"], p, won))
            if len(recs) < 25:
                continue
            # walk-forward: pick band on half A, evaluate on half B
            slugs = sorted(set(r[0] for r in recs))
            mid = slugs[len(slugs) // 2]
            A = [r for r in recs if r[0] in set(slugs[:len(slugs)//2])]
            B = [r for r in recs if r[0] in set(slugs[len(slugs)//2:])]
            res = dict(lo=lo, hi=hi, tlo=tlo, thi=thi, n=len(recs),
                       all_pnl=float(np.mean([r[4] for r in recs])),
                       A=float(np.mean([r[4] for r in A])) if A else np.nan,
                       B=float(np.mean([r[4] for r in B])) if B else np.nan,
                       nA=len(A), nB=len(B),
                       winrate=float(np.mean([r[5] for r in recs])))
            cands.append(res)
    cands.sort(key=lambda r: -r["all_pnl"])
    print(f"{'band':>12} {'t':>10} {'N':>5} {'win':>6} {'allP&L':>8} {'halfA':>8} {'halfB':>8}  {'CI(all)':>18}")
    for c in cands[:18]:
        recs_all = []
        for d in W:
            for side in ("Up", "Down"):
                e = find_entry(d, side, c["tlo"], c["thi"], c["lo"], c["hi"], STAKE)
                if e is None:
                    continue
                p, _ = pnl_hold(e, d, side, STAKE, FEE_FNS["doc_0.07"])
                recs_all.append(p)
        lo_ci, hi_ci = boot_ci(recs_all)
        print(f"{c['lo']:.2f}-{c['hi']:.2f} {c['tlo']:>4}-{c['thi']:<5} {c['n']:>5} {c['winrate']:.3f} "
              f"{c['all_pnl']:>+8.3f} {c['A']:>+8.3f} {c['B']:>+8.3f}  [{lo_ci:+.3f},{hi_ci:+.3f}]")

    # ---------- 3. fee sensitivity for the best few ----------
    print("\n=== fee sensitivity for the top candidates ===")
    for c in cands[:5]:
        line = f"  band {c['lo']:.2f}-{c['hi']:.2f} t[{c['tlo']},{c['thi']}]: "
        for fname, ff in FEE_FNS.items():
            vals = []
            for d in W:
                for side in ("Up", "Down"):
                    e = find_entry(d, side, c["tlo"], c["thi"], c["lo"], c["hi"], STAKE)
                    if e is None:
                        continue
                    p, _ = pnl_hold(e, d, side, STAKE, ff)
                    vals.append(p)
            l, h = boot_ci(vals)
            line += f"{fname}=${np.mean(vals):+.3f}[{l:+.2f},{h:+.2f}]  "
        print(line)

    # ---------- 4. WHY: does the market price revert after the entry? ----------
    print("\n=== mechanism: market price path after an early longshot entry ===")
    reverts = []
    for d in W:
        for side in ("Up", "Down"):
            e = find_entry(d, side, 30, 90, 0.10, 0.30, STAKE)
            if e is None:
                continue
            i = e["i"]
            ask_ser = d[side + "_ask"]
            # price of the same side 30s and 60s later
            j30 = int(np.searchsorted(d["t"], e["t"] + 30))
            j60 = int(np.searchsorted(d["t"], e["t"] + 60))
            p0 = e["ask"]
            p30 = ask_ser[j30] if j30 < len(ask_ser) else np.nan
            p60 = ask_ser[j60] if j60 < len(ask_ser) else np.nan
            reverts.append((p0, p30, p60, d["_winner"] == side))
    a30 = np.array([r[1] - r[0] for r in reverts if np.isfinite(r[1])])
    a60 = np.array([r[2] - r[0] for r in reverts if np.isfinite(r[2])])
    won = np.array([r[3] for r in reverts], dtype=float)
    print(f"  n={len(reverts)}  mean ask change +30s={np.nanmean(a30):+.4f}  +60s={np.nanmean(a60):+.4f}")
    print(f"  for LOSERS: +30s={np.nanmean(a30[won==0]):+.4f} (n={int((won==0).sum())})   "
          f"for WINNERS: +30s={np.nanmean(a30[won==1]):+.4f} (n={int((won==1).sum())})")
    print("  -> if losers' prices keep falling, the market was RIGHT to price them low;")
    print("     if winners' prices keep falling too, the market overreacted early.")

    with open(os.path.join(OUTDIR, "early_mispricing.json"), "w") as fh:
        json.dump(cands[:30], fh, indent=2, default=float)


if __name__ == "__main__":
    main()
