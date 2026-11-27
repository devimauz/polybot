"""_pm_maker.py - the only structural avenue left: be the maker, not the taker.

Every test so far paid the spread AND the 0.07*p*(1-p) taker fee, which is why
nothing survived: the overround (~1.2c) is somebody's income, and the fee is
1.4-5.6% of the stake.  A maker posts on the passive side of the book and, on
Polymarket's crypto binaries, pays no taker fee.

Fill model (deliberately pessimistic):
  * post a BID at the current best bid on the side the calibrated model favours
  * the bid fills only if a later frame in the same window shows the ASK at or
    below that price - i.e. somebody actually crossed down to your level.  This
    is adverse by construction: you get filled when the price is falling.
  * hold whatever fills to settlement (no exit skill assumed)
  * fee: 0 as a maker (stress test with the taker fee is printed too)

Protocol: side/price/time-band chosen on the design slice, one holdout run.

Run: python _pm_maker.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402
import _pm_lastlook as L           # noqa: E402

STAKE = 5.0
FEE = S.FEE
K = L.K
DESIGN_FRAC = S.DESIGN_FRAC


def maker_trades(wins, t_lo, t_hi, p_min, hold=True):
    """Post a bid at the touch, fill only on a later downward cross."""
    rows = []
    for w in wins:
        d = w["d"]
        t = np.asarray(d["t"], float)
        frames = L.frames_tail(w, t_lo, t_hi, 1.0)
        if not frames:
            continue
        for fr in frames:
            side = "Up" if fr["p_up"] >= 0.5 else "Down"
            p = fr[side + "_p"]
            if p < p_min:
                continue
            i = fr["i"]
            bid = d[side + "_bid"][i]
            if not (np.isfinite(bid) and 0.01 < bid < 0.99):
                continue
            if bid + 0.01 >= 1.0:
                continue
            q = float(bid)          # join the bid, do not pay up
            # fill: some later frame must show the ask at or below q
            j0 = i + 1
            n = len(t)
            filled = None
            for j in range(j0, n):
                a = d[side + "_ask"][j]
                if not np.isfinite(a):
                    continue
                if a <= q + 1e-9:
                    filled = j
                    break
            if filled is None:
                continue
            rows.append(dict(wi=w["ws"], T=float(t[filled]), side=side, ask=q,
                             p=float(p), won=1 if d["_winner"] == side else 0,
                             wait=float(t[filled] - t[i])))
            break               # one position per window
    return rows


def show(rows, label, fee=FEE):
    if not rows:
        print(f"  {label:<40} no fills")
        return None
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    sh = STAKE / ask
    p = sh * won - STAKE - sh * fee * ask * (1 - ask)
    wi = [r["wi"] for r in rows]
    m, tstat, _ = B.cluster_t(wi, p)
    lo, hi = B.boot_ci(wi, p)
    st = STAKE * len(rows)
    print(f"  {label:<40} N={len(rows):>3} w={int(won.sum()):>3} pay={ask.mean():.4f} "
          f"real={won.mean():.4f} net=${p.sum():+7.2f} roi={p.sum()/st*100:+6.1f}% "
          f"t={tstat:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] "
          f"median wait {np.median([r['wait'] for r in rows]):.1f}s")
    return dict(n=len(rows), net=p.sum(), t=tstat, lo=lo, hi=hi, p=p, rows=rows)


def main():
    W = B.load_all()
    wins = S.prep(W)
    wins.sort(key=lambda w: w["ws"])
    cut = int(len(wins) * DESIGN_FRAC)
    design, hold = wins[:cut], wins[cut:]
    print("=" * 116)
    print("MAKER SIDE: bid at the touch, fill only on a downward cross, hold to settlement")
    print("=" * 116)
    print(f"windows {len(wins)}   design {len(design)}   holdout {len(hold)}")

    print("\n=== DESIGN (fee-free as a maker) ===")
    cells = []
    for t_lo, t_hi in ((240, 270), (270, 290), (240, 299), (0, 240)):
        for pmin in (0.55, 0.60, 0.70, 0.80):
            r = maker_trades(design, t_lo, t_hi, pmin)
            res = show(r, f"T {t_lo}-{t_hi}s  p>={pmin}")
            if res:
                cells.append(((t_lo, t_hi, pmin), res))
    print("\n  with the TAKER fee applied (stress):")
    for k, _ in sorted(cells, key=lambda c: -c[1]["net"])[:4]:
        show(maker_trades(design, k[0], k[1], k[2]), f"T {k[0]}-{k[1]}s p>={k[2]} (fee)", fee=FEE)
    return wins, design, hold, cells


if __name__ == "__main__":
    main()
