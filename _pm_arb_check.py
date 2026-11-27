"""_pm_arb_check.py - is there a RISKLESS two-legged trade left in these books?

Buying both legs of the same 5-minute market costs ask_up + ask_down and pays $1
whatever happens.  So it is free money whenever

    ask_up + ask_down + fee(ask_up) + fee(ask_down) < 1

No prediction, no holdout needed - it either happens in the recorded books or it
does not.  Fees are C*0.07*p*(1-p) per share, and a $5/$5 pair needs the book to
cover 5 shares on each side.

Run: python _pm_arb_check.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402

FEE = S.FEE


def main():
    W = B.load_all()
    tot, best, execs = [], [], []
    for d in W:
        ua, da = d["Up_ask"], d["Down_ask"]
        ub, db = d["Up_bid"], d["Down_bid"]
        us, ds = d["Up_asksz"], d["Down_asksz"]
        m = np.isfinite(ua) & np.isfinite(da) & (ua > 0) & (da > 0) & (ua < 1) & (da < 1)
        if not m.any():
            continue
        s = ua[m] + da[m]
        tot.append(s)
        best.append(float(s.min()))
        # executable pair: $5 a side needs 5/ask shares available at that ask
        need_u, need_d = 5.0 / ua[m], 5.0 / da[m]
        ok = (np.isfinite(us[m]) & np.isfinite(ds[m])
              & (us[m] >= need_u) & (ds[m] >= need_d))
        fees = FEE * ua[m] * (1 - ua[m]) + FEE * da[m] * (1 - da[m])
        net = 1.0 - s - fees
        ex = ok & (net > 0)
        if ex.any():
            i = int(np.argmax(np.where(ex, net, -9)))
            execs.append((d["_slug"], float(net[i]), float(s[i]), float(d["t"][i])))

    tot = np.concatenate(tot)
    best = np.array(best)
    print("=" * 96)
    print("TWO-LEGGED COMPLETE-SET CHECK (buy Up and Down, get $1 back)")
    print("=" * 96)
    print(f"windows {len(best)}   frames {len(tot)}")
    print(f"ask_up + ask_down over all frames: min {tot.min():.4f}  p0.1 {np.percentile(tot,0.1):.4f}  "
          f"median {np.median(tot):.4f}")
    print(f"best (lowest) two-legged cost per window: min {best.min():.4f}  "
          f"median {np.median(best):.4f}")
    print(f"frames with cost < 1.000: {int((tot < 1).sum())}  ({100*(tot < 1).mean():.3f}%)")
    print(f"frames with cost + fees < 1: {int((tot < 1 - 0.035).sum())}")
    print(f"windows with an executable, fee-positive pair: {len(execs)}")
    for e in execs[:10]:
        print("   ", e)
    if best.min() > 0.99:
        print("\n  -> the book never offers a complete set below 0.99. There is no")
        print("     riskless two-legged trade in this log; the overround is always paid.")
    return execs


if __name__ == "__main__":
    main()
