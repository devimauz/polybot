"""_pm_leakcheck.py - is the perfect test-half record a REGIME or a LEAK?

The test half gives 27/27 and 18/18 wins for several different rules.  That
is either (a) a genuine market regime, or (b) something wrong with the
outcome data in that period - e.g. a degenerate winner distribution, a
timestamp leak, or a book that already contains the answer.

Checks:
  1. winner distribution over time (are all resolutions "Up"?)
  2. t_src sanity (monotonic, inside the window, no jumps)
  3. does the favourite win ~100% in the test half at ANY price band,
     including bands the strategy never trades?
  4. the market's OWN price as an independent check: if the market priced
     the favourite at 0.60 and still won 100% of the time, the outcome
     mapping is broken, not the market mispriced.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")


def main():
    W = B.load_all()
    wss = np.array([d["ws"] for d in W])
    cut = wss[int(len(W) * 0.60)]

    print("=" * 96)
    print("LEAK / REGIME CHECK")
    print("=" * 96)

    # ---- 1. winner distribution over time -------------------------------
    print("\n--- 1. winner distribution (Up/Down) in 8 chronological blocks ---")
    order = np.argsort(wss)
    for i, b in enumerate(np.array_split(order, 8)):
        w = [W[j]["_winner"] for j in b]
        nu = sum(1 for x in w if x == "Up")
        print(f"  block {i+1}: {len(w):>3} windows  Up={nu:>3}  Down={len(w)-nu:>3}")
    up_all = sum(1 for d in W if d["_winner"] == "Up")
    print(f"  overall: Up {up_all}/{len(W)} = {up_all/len(W):.3f}")

    # ---- 2. t_src sanity -------------------------------------------------
    print("\n--- 2. source-aligned clock sanity ---")
    bad_mono = 0
    bad_rng = 0
    lags = []
    for d in W:
        t = d["t"]
        if len(t) < 2:
            continue
        if np.any(np.diff(t) < -0.001):
            bad_mono += 1
        if t[0] < -1 or t[-1] > 305:
            bad_rng += 1
        lags.append(d["t_lag"])
    L = np.concatenate(lags)
    print(f"  windows with non-monotonic t_src: {bad_mono}")
    print(f"  windows with t_src outside [-1, 305]: {bad_rng}")
    print(f"  t_src range: {min(d['t'][0] for d in W):.2f} .. "
          f"{max(d['t'][-1] for d in W):.2f}")
    print(f"  lag vs logger stamp: mean {L.mean():.3f}s  median {np.median(L):.3f}s")

    # ---- 3. favourite win rate by price band, per half ------------------
    print("\n--- 3. favourite win rate by price band near T=265, per half ---")
    print("  (the strategy only trades 0.80-0.99; other bands are the control)")
    rows = B.build_rows(W)
    for lo, hi in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.86),
                   (0.86, 0.90), (0.90, 0.95), (0.95, 0.99)):
        line = f"  ask[{lo:.2f},{hi:.2f})"
        for lbl, wsel in (("TRAIN", lambda r: r["ws"] <= cut),
                          ("TEST ", lambda r: r["ws"] > cut)):
            s = [r for r in rows if 264 <= r["T"] <= 266 and r["mkt"] >= 0.5
                 and lo <= r["ask"] < hi and wsel(r)]
            if not s:
                line += f"  {lbl}: n=0"
                continue
            a = np.mean([r["ask"] for r in s])
            wr = np.mean([r["won"] for r in s])
            line += f"  {lbl}: n={len(s):>3} ask={a:.3f} win={wr:.3f} edge={wr-a:+.3f}"
        print(line)

    # ---- 4. does the favourite win ~always in the test half? -----------
    print("\n--- 4. unconditional favourite win rate near T=265, per half ---")
    for lbl, wsel in (("TRAIN", lambda r: r["ws"] <= cut),
                      ("TEST ", lambda r: r["ws"] > cut)):
        s = [r for r in rows if 264 <= r["T"] <= 266 and r["mkt"] >= 0.5 and wsel(r)]
        if not s:
            print(f"  {lbl}: no quotes")
            continue
        a = np.mean([r["ask"] for r in s])
        wr = np.mean([r["won"] for r in s])
        print(f"  {lbl}: n={len(s):>3} mean ask={a:.4f} favourite win rate={wr:.4f} "
              f"edge={wr-a:+.4f}")

    # ---- 5. BTC direction regime (coinbase close vs open) ---------------
    print("\n--- 5. underlying BTC direction regime per half ---")
    for lbl, wsel in (("TRAIN", lambda d: d["ws"] <= cut),
                      ("TEST ", lambda d: d["ws"] > cut)):
        ups = 0
        n = 0
        agree = 0
        for d in W:
            if not wsel(d):
                continue
            cb = d["cb_px"]
            cb = cb[np.isfinite(cb)]
            if len(cb) < 10:
                continue
            n += 1
            if cb[-1] > cb[0]:
                ups += 1
            win_up = d["_winner"] == "Up"
            if win_up == (cb[-1] > cb[0]):
                agree += 1
        print(f"  {lbl}: {n} windows, BTC closed up in {ups} ({ups/max(n,1):.1%}), "
              f"resolution agreed with the coinbase direction {agree}/{n} "
              f"({agree/max(n,1):.1%})")

    # ---- 6. cross-check: is the second half a low-volatility regime? ----
    print("\n--- 6. BTC realised move (bps) per window, per half ---")
    for lbl, wsel in (("TRAIN", lambda d: d["ws"] <= cut),
                      ("TEST ", lambda d: d["ws"] > cut)):
        mv = []
        for d in W:
            if not wsel(d):
                continue
            cb = d["cb_px"]
            cb = cb[np.isfinite(cb)]
            if len(cb) < 10:
                continue
            mv.append(abs(np.log(cb[-1] / cb[0])) * 1e4)
        mv = np.array(mv)
        print(f"  {lbl}: n={len(mv)} mean |move| = {mv.mean():.2f} bps  "
              f"median {np.median(mv):.2f}  p90 {np.percentile(mv,90):.2f}")

    # ---- 7. how often does the market's favourite lose at T=265? -------
    print("\n--- 7. favourite LOSSES at T=265 (all bands) per half ---")
    for lbl, wsel in (("TRAIN", lambda r: r["ws"] <= cut),
                      ("TEST ", lambda r: r["ws"] > cut)):
        s = [r for r in rows if r["T"] == 265 and r["mkt"] >= 0.5 and wsel(r)]
        losses = [(r["ws"], r["ask"], r["mkt"]) for r in s if r["won"] == 0]
        print(f"  {lbl}: {len(losses)} losses out of {len(s)} quotes")
        for ws_, ask_, mkt_ in losses:
            print(f"      ws={ws_} fav ask={ask_:.3f} mkt={mkt_:.3f}")


if __name__ == "__main__":
    main()
