"""OOS validation of the conditional-sizing variants.

The Kelly variant (F) fit on TRAIN win rates of 1.000, so it must be judged
on the untouched OOS half.  Protocol:
  * stakes are computed from TRAIN statistics only
  * evaluate the OOS half (windows >= CUT) with those frozen stakes
  * compare OOS $ per unit staked against the flat-$10 baseline
  * report the ROI (scale-free) and the concentration
  * a variant that only wins because it concentrates 3x the capital in a
    lucky TRAIN bucket will show a WORSE OOS ROI
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

BUCKETS = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.96)]


def bucket_of(r):
    for b in BUCKETS:
        if b[0] <= r["ask"] < b[1]:
            return b
    return BUCKETS[-1]


def pnl_for(sel, stakes):
    out = []
    for r, st in zip(sel, stakes):
        a = r["ask"]
        sh = st / a
        fee = sh * FEE * a * (1 - a)
        out.append(sh * r["won"] - st - fee)
    return np.array(out)


def kelly_stakes(sel, cap=30.0, frac=0.5):
    """Fractional Kelly using TRAIN-only win rates per price bucket."""
    tr = [r for r in sel if r["wi"] < CUT]
    rates = {}
    for b in BUCKETS:
        rs = [r for r in tr if bucket_of(r) == b]
        rates[b] = (sum(r["won"] for r in rs) / len(rs)) if rs else None
    st = []
    for r in sel:
        rate = rates.get(bucket_of(r))
        if rate is None:
            st.append(TAKE)
            continue
        a = r["ask"]
        edge = (rate - a) / (1 - a) if a < 1 else 0.0
        st.append(max(2.0, min(cap, TAKE * max(0.0, frac * edge / 0.1))))
    return st, rates


def score(name, sel, stakes, lo_wi, hi_wi):
    idx = [i for i, r in enumerate(sel) if lo_wi <= r["wi"] < hi_wi]
    if not idx:
        return None
    s = [sel[i] for i in idx]
    st = [stakes[i] for i in idx]
    p = pnl_for(s, st)
    wi = [r["wi"] for r in s]
    t = B.cluster_t(wi, p)[1]
    clo, chi = B.boot_ci(wi, p)
    staked = sum(st)
    o = np.argsort(p)
    tot = p.sum()
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    return dict(name=name, n=len(s), net=float(p.sum()), staked=float(staked),
                roi=float(p.sum()) / staked, t=float(t), lo=float(clo), hi=float(chi),
                worst=float(p.min()), c5=c5, avg=float(p.mean()))


sel = B.first_per_window([r for r in rows_all if r["T"] >= 262 and r["mkt"] >= 0.5
                          and 0.80 <= r["ask"] < 0.96])
print(f"shipped-rule opportunities: {len(sel)}   TRAIN 0..{CUT-1}  OOS {CUT}..{NW-1}\n")

k_st, rates = kelly_stakes(sel)
print("TRAIN win rates per price bucket (fitted on TRAIN only):")
for b in BUCKETS:
    trs = [r for r in sel if r["wi"] < CUT and bucket_of(r) == b]
    tes = [r for r in sel if r["wi"] >= CUT and bucket_of(r) == b]
    trw = (sum(r["won"] for r in trs) / len(trs)) if trs else float("nan")
    tew = (sum(r["won"] for r in tes) / len(tes)) if tes else float("nan")
    meanask = np.mean([r["ask"] for r in trs]) if trs else float("nan")
    print(f"  ask {b}: train n={len(trs):>2} win={trw:.3f} (mean ask {meanask:.3f})   "
          f"OOS n={len(tes):>2} win={tew:.3f}")

variants = [
    ("flat $10", [TAKE] * len(sel)),
    ("Kelly cap$30 frac0.5", k_st),
    ("$20 if ask<0.85 else $10", [20.0 if r["ask"] < 0.85 else TAKE for r in sel]),
    ("$30 if ask<0.85 else $10", [30.0 if r["ask"] < 0.85 else TAKE for r in sel]),
]

for split_name, lo_wi, hi_wi in (("TRAIN", 0, CUT), ("OOS", CUT, NW), ("FULL", 0, NW)):
    print(f"\n{'='*104}\n{split_name}  (windows {lo_wi}..{hi_wi-1})\n{'='*104}")
    print(f"  {'variant':<26} {'N':>4} {'staked':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} {'worst':>7} {'top5':>5}")
    for name, st in variants:
        v = score(name, sel, st, lo_wi, hi_wi)
        if not v:
            continue
        print(f"  {v['name']:<26} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
              f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['worst']:>+7.2f} {v['c5']*100:>4.0f}%")

# Is the cheap bucket really better, or is it the train fit talking?
print(f"\n{'='*104}\nIS THE CHEAP BUCKET GENUINELY BETTER?  (edge = realised win rate - mean ask paid)\n{'='*104}")
for b in BUCKETS:
    for split_name, lo_wi, hi_wi in (("TRAIN", 0, CUT), ("OOS", CUT, NW)):
        s = [r for r in sel if lo_wi <= r["wi"] < hi_wi and bucket_of(r) == b]
        if not s:
            continue
        w = np.array([r["won"] for r in s], dtype=float)
        a = np.array([r["ask"] for r in s])
        print(f"  ask {b} {split_name:<5}: n={len(s):>2} realised WR={w.mean():.3f} "
              f"mean ask={a.mean():.3f}  edge={w.mean()-a.mean():+.4f}")

# and the decisive one: does the CHEAP bucket survive in OOS?
print("\nOOS-only check of the $30-if-cheap rule vs flat, on ROI (scale-free):")
for name, st in variants:
    v = score(name, sel, st, CUT, NW)
    if v:
        print(f"  {name:<26} OOS ROI {v['roi']*100:+.2f}%   net ${v['net']:+.2f} on ${v['staked']:.0f} staked   t={v['t']:+.2f}")
