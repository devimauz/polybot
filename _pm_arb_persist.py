"""_pm_arb_persist.py - does the two-legged discount survive a fill delay?

_pm_arb_check.py found 126/819 windows with a frame where ask_up + ask_down + fees
< 1 with enough depth for $5 a side.  A single frame is not tradeable: the cheap
leg is often a stale print.  So this test re-prices BOTH legs 1 s later and only
counts a pair that is still below 1 after fees at that later frame - i.e. it
survives the round trip that a real order would take.

Two legs bought for c pay exactly $1 at settlement, so the trade is market
neutral: it does not care which side wins, only that both fills happen.

Run: python _pm_arb_persist.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402
import _pm_twap_strategy as S      # noqa: E402

FEE = S.FEE
STAKE = 5.0
MAX_SPREAD = 0.06


def leg_ask(d, side, i, stake=STAKE):
    a, b, s = d[side + "_ask"][i], d[side + "_bid"][i], d[side + "_asksz"][i]
    if not all(np.isfinite(v) for v in (a, b, s)):
        return None
    if not (0 < b <= a < 1) or (a - b) > MAX_SPREAD:
        return None
    if s * a < stake:
        return None
    return float(a)


def pair_net(ua, da, stake=STAKE):
    """net $ per $1 payout for a complete set bought at ua and da (same stake)."""
    sh_u, sh_d = stake / ua, stake / da
    # equalise shares so the pair pays exactly $1 per share-set
    sh = min(sh_u, sh_d)
    cost = sh * ua + sh * da
    fee = sh * FEE * (ua * (1 - ua) + da * (1 - da))
    return sh * 1.0 - cost - fee, sh


def main():
    W = B.load_all()
    print("=" * 100)
    print("TWO-LEGGED DISCOUNT: persistence test with a 1 s fill delay")
    print("=" * 100)
    trades = []
    for d in W:
        t = np.asarray(d["t"], float)
        n = len(t)
        for i in range(n):
            ua = leg_ask(d, "Up", i)
            da = leg_ask(d, "Down", i)
            if ua is None or da is None:
                continue
            net0, sh0 = pair_net(ua, da)
            if net0 <= 0.004:
                continue
            j = int(np.searchsorted(t, t[i] + 1.0))
            if j >= n:
                continue
            ua2 = leg_ask(d, "Up", j)
            da2 = leg_ask(d, "Down", j)
            if ua2 is None or da2 is None:
                continue
            net1, sh1 = pair_net(ua2, da2)
            if net1 <= 0.002:
                continue          # the discount was stale - it evaporated
            trades.append(dict(wi=d["ws"], T=float(t[i]), ua=ua, da=da,
                               cost=ua + da, net=net1, sh=sh1,
                               net0=net0, t_fill=float(t[j])))
            break             # one pair per window
    print(f"windows {len(W)}   tradeable pairs after a 1 s delay: {len(trades)}")
    if not trades:
        print("  -> nothing survives the delay; the discount is a stale print.")
        return []
    net = np.array([x["net"] for x in trades])
    cost = np.array([x["cost"] for x in trades])
    sh = np.array([x["sh"] for x in trades])
    cap = cost * sh
    print(f"  cost of the pair (up+down): mean {cost.mean():.4f}  min {cost.min():.4f}")
    print(f"  capital committed per pair: mean ${cap.mean():.2f}  max ${cap.max():.2f}")
    print(f"  net profit per pair: total ${net.sum():+.2f}  mean ${net.mean():+.3f}  "
          f"min ${net.min():+.3f}")
    print(f"  return on capital: {net.sum()/cap.sum()*100:+.2f}%")
    m, tstat, nwin = B.cluster_t([x["wi"] for x in trades], net)
    lo, hi = B.boot_ci([x["wi"] for x in trades], net)
    print(f"  cluster-t {tstat:+.2f}   bootstrap CI95 [${lo:+.3f}, ${hi:+.3f}] per pair")
    print(f"  trades per day of data: {len(trades)/ (365.7):.1f}")
    times = np.array([x["T"] for x in trades])
    print(f"  entry time: med {np.median(times):.0f}s  (0-240s: {int((times<240).sum())}, "
          f"240+: {int((times>=240).sum())})")
    print(f"  worst pairs:", np.sort(net)[:3])
    return trades


if __name__ == "__main__":
    main()
