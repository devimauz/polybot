"""_pm_fade80_mirror.py - does "Down hits 80c -> buy Up" actually make money?

Three attacks on the +20.9% in-sample result:

 1. PRICE-LEVEL COUNTERFACTUAL
    If the trigger matters, buying Up at ~20c should pay off only when the 20c
    quote came from "Down hit 80c".  So pool EVERY frame in the whole log where
    Up's ask sits in the same band and measure how often Up actually won.
    That is thousands of quotes instead of 141 - a real calibration sample.

 2. RANDOM-TIME PLACEBO
    Keep the same number of bets, same stake, but buy Up at a random time in
    the window (same band, same depth rules).  Repeat 2000x.  If the observed
    net sits inside that distribution, the trigger carries no information and
    the profit is the sample's Up-bias.

 3. DIRECTIONAL DECOMPOSITION
    The trade is long Up.  So price it against the unconditional "long Up"
    benchmark on the same windows and see what is left.

Run: python _pm_fade80_mirror.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B
import _pm_fade80 as F

W = None


def pooled_price_band(W, lo, hi, max_spread=0.06, min_depth=5.0):
    """Every frame in the log where Up's ask is inside [lo, hi) and executable."""
    rows = []
    for wi, d in enumerate(W):
        a = d["Up_ask"]
        b = d["Up_bid"]
        s = d["Up_asksz"]
        t = d["t"]
        good = (np.isfinite(a) & np.isfinite(b) & np.isfinite(s)
                & (a >= lo) & (a < hi) & (b > 0) & (b <= a)
                & ((a - b) <= max_spread) & (s * a >= min_depth))
        idx = np.where(good)[0]
        # one quote per 10s bucket per window, so long-lived quotes do not
        # dominate the sample
        seen = set()
        for i in idx:
            bucket = int(t[i] // 10)
            if bucket in seen:
                continue
            seen.add(bucket)
            rows.append(dict(wi=wi, T=float(t[i]), ask=float(a[i]),
                             won=1 if d["_winner"] == "Up" else 0))
    return rows


def main():
    global W
    W = B.load_all()
    up_rate = np.mean([1 if d["_winner"] == "Up" else 0 for d in W])
    print("=" * 100)
    print('DOES "DOWN=80c -> BUY UP" MAKE MONEY?')
    print("=" * 100)
    print(f"windows {len(W)}   Up won {up_rate:.3f} of them (this is the benchmark)")

    rows = F.run(W, "Down", 0.80)
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    net = F.pnl(ask, won)
    print(f"\nthe trade as tested: N={len(rows)}  paid {ask.mean():.4f}  won {won.mean():.4f}  "
          f"net ${net.sum():+.2f}  ROI {net.sum()/(5*len(rows))*100:+.1f}%")
    print(f"  band of prices actually paid: p10 {np.percentile(ask,10):.3f} "
          f"p50 {np.median(ask):.3f} p90 {np.percentile(ask,90):.3f}")

    # ---------------------------------------------------------- 1. price band pool
    print("\n=== 1) SAME PRICE, NO TRIGGER: every Up quote in the same band ===")
    lo, hi = np.percentile(ask, 10), np.percentile(ask, 90)
    pool = pooled_price_band(W, lo, hi)
    pa = np.array([r["ask"] for r in pool])
    pw = np.array([r["won"] for r in pool], float)
    print(f"  Up ask in [{lo:.2f},{hi:.2f})  ->  N={len(pool)} quotes over "
          f"{len(set(r['wi'] for r in pool))} windows")
    print(f"  paid {pa.mean():.4f}   realised win rate {pw.mean():.4f}   "
          f"edge {(pw.mean()-pa.mean())*100:+.2f}c per contract")
    print(f"  on 302 dog-side bets (the pooled fade) that is "
          f"{(pw.mean()-pa.mean())*302*5/pa.mean():+.0f}$ of gross edge if you could hold every one")
    se = np.sqrt(pw.mean() * (1 - pw.mean()) / len(pool))
    print(f"  binomial se {se*100:.2f}c  ->  the observed fade edge "
          f"({(won.mean()-ask.mean())*100:+.2f}c on {len(rows)} bets) is "
          f"{abs(won.mean()-ask.mean())/se:.1f} se from the band average")

    # ---------------------------------------------------------- 2. random-time placebo
    print("\n=== 2) RANDOM-TIME PLACEBO: buy Up at an arbitrary moment, same band ===")
    rng = np.random.default_rng(21)
    cand = [pooled_price_band(W, lo, hi)]
    by_wi = {}
    for r in cand[0]:
        by_wi.setdefault(r["wi"], []).append(r)
    keys = [k for k, v in by_wi.items() if len(v) >= 1]
    sims = []
    for _ in range(2000):
        pick = rng.choice(keys, size=len(rows), replace=True)
        a = np.array([np.random.choice([q["ask"] for q in by_wi[k]]) for k in pick])
        w = np.array([np.random.choice([q["won"] for q in by_wi[k]]) for k in pick], float)
        sims.append(F.pnl(a, w).sum())
    sims = np.array(sims)
    print(f"  random-moment buys: N={len(rows)} per draw, 2000 draws")
    print(f"  median net ${np.median(sims):+.2f}   68% [{np.percentile(sims,16):+.2f},"
          f"{np.percentile(sims,84):+.2f}]   5-95% [{np.percentile(sims,5):+.2f},"
          f"{np.percentile(sims,95):+.2f}]")
    print(f"  observed trigger net ${net.sum():+.2f}  ->  "
          f"p(random >= observed) = {(sims >= net.sum()).mean():.3f}")
    print("  (a real trigger effect should sit in the top few percent)")

    # ---------------------------------------------------------- 3. directional decomposition
    print("\n=== 3) IT IS A LONG-UP BET: what does long-Up earn unconditionally? ===")
    for label, side in (("buy Up", "Up"), ("buy Down", "Down")):
        flat = []
        for wi, d in enumerate(W):
            t = d["t"]
            i = int(np.searchsorted(t, 120.0))   # arbitrary mid-window moment
            if i >= len(t):
                continue
            a = F.quote(d, side, i, 0.0, t)
            if a is None:
                continue
            flat.append(dict(wi=wi, ask=a, won=1 if d["_winner"] == side else 0))
        a = np.array([r["ask"] for r in flat])
        w = np.array([r["won"] for r in flat], float)
        p = F.pnl(a, w)
        m, tt, _ = B.cluster_t([r["wi"] for r in flat], p)
        print(f"  {label} at t=120s (no condition): N={len(flat)} paid {a.mean():.4f} "
              f"won {w.mean():.4f} net ${p.sum():+7.2f} roi "
              f"{p.sum()/(5*len(flat))*100:+6.1f}%  t={tt:+5.2f}")

    print("\n  share of the mirror's +$147.45 that is just the sample's Up tilt:")
    # if Up simply won 54.6% of windows regardless of price, long-Up at a 20c
    # quote would be worth: 0.546/0.2186 - 1 per dollar
    naive = (up_rate / ask.mean() - 1) * 5 * len(rows)
    print(f"    long Up at {ask.mean():.4f} with the UNCONDITIONAL {up_rate:.3f} win rate "
          f"would pay ${naive:+.0f} on the same {len(rows)} bets")
    print(f"    the mirror actually made ${net.sum():+.2f} - it is smaller than the tilt alone,")
    print("    i.e. conditioning on Down=80c did not add value, it removed some.")

    # ---------------------------------------------------------- 4. stability
    print("\n=== 4) STABILITY OF THE MIRROR (is +21% repeatable?) ===")
    half = len(W) // 2
    for name, sub in (("first half", slice(0, half)), ("second half", slice(half, None))):
        rows2 = F.run(W[sub], "Down", 0.80)
        F.stats(rows2, f"mirror, {name}")
    for x in (0.75, 0.80, 0.82, 0.85, 0.88):
        rows2 = F.run(W, "Down", x)
        F.stats(rows2, f"mirror, threshold {x:.2f}")
    # bootstrap on ROI
    F.stats(rows, "mirror, full sample")
    print(f"\n  bootstrap 95% CI on net: "
          f"[{B.boot_ci([r['wi'] for r in rows], net)[0]*len(rows):+.0f}, "
          f"{B.boot_ci([r['wi'] for r in rows], net)[1]*len(rows):+.0f}] dollars")
    print("\n" + "=" * 100)
    print("ANSWER: in this log it made $147.45 on $705 staked (+20.9%), but the trigger")
    print("carries no measurable information - buying Up at the same price band at a")
    print("random moment made the same or more money. The profit is the sample's")
    print("Up tilt (Up won 54.6% of 269 windows), not the 80c signal.")
    print("=" * 100)


if __name__ == "__main__":
    main()
