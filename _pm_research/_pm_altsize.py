"""Can we make MORE profit from the SAME edge by changing the SIZING?

The shipped rule's edge per trade is roughly constant, so scaling the stake
up or down moves dollars but not ROI.  The interesting question is whether
the edge is CONDITIONAL - i.e. bigger in some sub-populations - so that a
conditional stake extracts more profit from the same 38 opportunities.

Conditional sizing levers tested (all decided at entry time, from data
available then - no lookahead):
  A. flat $10                                   (the shipped baseline)
  B. stake by price bucket - cheaper favourite = more stake
  C. stake by spread tightness
  D. stake by remaining time (later = more stake)
  E. stake by displayed depth at the ask
  F. fractional Kelly on the empirical win rate of the price bucket
  G. MULTIPLE trades per window (the rule fires more than once when the
     price stays in band) - this INCREASES the number of opportunities

G is the one that can genuinely raise total profit rather than re-slice it.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B
import _pm_altsearch as A

W = A.W
NW = A.NW
CUT = A.CUT
TAKE = 10.0
FEE = 0.07

rows_all = B.build_rows(W)


def shipped_rows():
    return [r for r in rows_all if r["T"] >= 262 and r["mkt"] >= 0.5
            and 0.80 <= r["ask"] < 0.96]


def pnl_for(sel, stakes):
    """sel and stakes are parallel lists."""
    out = []
    for r, st in zip(sel, stakes):
        a = r["ask"]
        sh = st / a
        fee = sh * FEE * a * (1 - a)
        out.append(sh * r["won"] - st - fee)
    return np.array(out)


def report(name, sel, stakes):
    p = pnl_for(sel, stakes)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    staked = sum(stakes)
    o = np.argsort(p)
    tot = p.sum()
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    return dict(name=name, n=len(sel), net=float(p.sum()), staked=float(staked),
                roi=float(p.sum()) / staked, t=float(t), lo=float(lo), hi=float(hi),
                avg=float(p.mean()), worst=float(p.min()), c5=c5)


def halves(sel, stakes):
    out = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        idx = [i for i, r in enumerate(sel) if lo <= r["wi"] < hi]
        if idx:
            out.append(pnl_for([sel[i] for i in idx], [stakes[i] for i in idx]).sum())
        else:
            out.append(0.0)
    return out


def bucket_stake(sel, keyfn, buckets):
    """Stake = bucket value, computed from the bucket's TRAIN win rate."""
    # fit win rate per bucket on TRAIN only
    tr = [r for r in sel if r["wi"] < CUT]
    rates = {}
    for b in buckets:
        rs = [r for r in tr if keyfn(r) == b]
        rates[b] = (sum(r["won"] for r in rs) / len(rs)) if rs else None
    stakes = []
    for r in sel:
        b = keyfn(r)
        rate = rates.get(b)
        if rate is None:
            stakes.append(TAKE)
            continue
        a = r["ask"]
        # fractional Kelly on a binary paying 1: f = (p - price)/(1 - price)
        edge = (rate - a) / (1 - a) if a < 1 else 0.0
        stakes.append(max(2.0, min(30.0, TAKE * max(0.0, 0.5 * edge / 0.1))))
    return stakes, rates


if __name__ == "__main__":
    sel = B.first_per_window(shipped_rows())
    print(f"baseline opportunities: {len(sel)} (shipped rule)\n")
    print(f"{'variant':<42} {'N':>4} {'staked':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} {'worst':>7} {'top5':>5}")
    print("-" * 120)

    # A. flat
    A_ = report("A. flat $10 (baseline)", sel, [TAKE] * len(sel))
    for v in (A_,):
        print(f"{v['name']:<42} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
              f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['worst']:>+7.2f} {v['c5']*100:>4.0f}%")

    # B. stake by price bucket (cheaper = more)
    for lo, hi, mult in ((0.80, 0.85, 20.0), (0.80, 0.85, 30.0), (0.85, 0.90, 15.0),
                          (0.90, 0.96, 10.0)):
        st = [mult if (lo <= r["ask"] < hi) else TAKE for r in sel]
        v = report(f"B. ${mult:.0f} if ask in [{lo:.2f},{hi:.2f}) else $10", sel, st)
        print(f"{v['name']:<42} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
              f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['worst']:>+7.2f} {v['c5']*100:>4.0f}%")

    # F. fractional Kelly by price bucket, fit on TRAIN
    buckets = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.96)]
    keyfn = lambda r: next(b for b in buckets if b[0] <= r["ask"] < b[1])
    stakes, rates = bucket_stake(sel, keyfn, buckets)
    v = report("F. fractional Kelly by price bucket (train-fit)", sel, stakes)
    print(f"{v['name']:<42} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
          f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['worst']:>+7.2f} {v['c5']*100:>4.0f}%")
    print(f"     train win rates per bucket: " +
          ", ".join(f"{b}={rates[b]:.3f}" if rates[b] is not None else f"{b}=n/a" for b in buckets))

    # G. MULTIPLE trades per window: every qualifying frame, not just the first
    print()
    for tmin in (262, 260, 258):
        for band in ((0.80, 0.96), (0.80, 0.95), (0.85, 0.96)):
            multi = [r for r in rows_all if r["T"] >= tmin and r["mkt"] >= 0.5
                     and band[0] <= r["ask"] < band[1]]
            if not multi:
                continue
            st = [TAKE] * len(multi)
            v = report(f"G. EVERY qualifying quote, T>={tmin} band[{band[0]:.2f},{band[1]:.2f})",
                       multi, st)
            h = halves(multi, st)
            print(f"{v['name']:<42} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
                  f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['worst']:>+7.2f} {v['c5']*100:>4.0f}%"
                  f"  halves[{h[0]:+.1f},{h[1]:+.1f},{h[2]:+.1f}]")
    print("\nDONE")
