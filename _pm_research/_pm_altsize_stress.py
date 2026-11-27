"""Stress the conditional-sizing result: is it real, or a fit to 38 trades?

Everything here targets the strongest alternative found so far:
  * "$30 if the favourite's ask < 0.85, else $10"  (no fitting at all)
  * fractional Kelly on the price bucket (fitted on TRAIN only)
both applied to the SHIPPED rule's 38 opportunities.

Tests:
  1. SWAP-SPLIT: fit the bucket win rates on OOS, test on TRAIN (and the
     reverse).  If the bucket structure only exists in one half, it is noise.
  2. PERMUTATION on the bucket effect: shuffle which trades land in which
     price bucket and see how often a spread that large appears by chance.
  3. STRESS: 1s execution latency, 4x fee, 1-cent spread.
  4. THE THRESHOLD: is 0.85 a plateau or a knife edge?
  5. CAPITAL: what does each variant actually require in the bankroll, and
     what is the drawdown?
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


def shipped(sel_rows):
    return B.first_per_window([r for r in sel_rows if r["T"] >= 262 and r["mkt"] >= 0.5
                               and 0.80 <= r["ask"] < 0.96])


def pnl_for(sel, stakes, fee=FEE):
    out = []
    for r, st in zip(sel, stakes):
        a = r["ask"]
        sh = st / a
        f = sh * fee * a * (1 - a)
        out.append(sh * r["won"] - st - f)
    return np.array(out)


def score(sel, stakes, fee=FEE):
    if not sel:
        return None
    p = pnl_for(sel, stakes, fee)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    clo, chi = B.boot_ci(wi, p)
    staked = sum(stakes)
    o = np.argsort(p)
    tot = p.sum()
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    # max drawdown on the cumulative equity curve (in chronological order)
    order = np.argsort(wi)
    eq = np.cumsum(p[order])
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    return dict(n=len(sel), net=float(p.sum()), staked=float(staked),
                roi=float(p.sum()) / staked, t=float(t), lo=float(clo), hi=float(chi),
                worst=float(p.min()), c5=c5, dd=dd, p=p)


def kelly(sel, fit_lo, fit_hi, cap=30.0, frac=0.5):
    tr = [r for r in sel if fit_lo <= r["wi"] < fit_hi]
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


sel = shipped(rows_all)
print(f"opportunities: {len(sel)}   TRAIN 0..{CUT-1}   OOS {CUT}..{NW-1}\n")

# ---------------------------------------------------------------- 1. swap split
print("=" * 104)
print("1. SWAP-SPLIT: does the bucket structure hold when the fit and the test swap?")
print("=" * 104)
print(f"  {'variant':<34} {'fit on':<7} {'testN':>6} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18}")
for label, fit_lo, fit_hi, te_lo, te_hi in (
        ("Kelly", 0, CUT, CUT, NW), ("Kelly", CUT, NW, 0, CUT)):
    st, rates = kelly(sel, fit_lo, fit_hi)
    idx = [i for i, r in enumerate(sel) if te_lo <= r["wi"] < te_hi]
    s = [sel[i] for i in idx]
    stk = [st[i] for i in idx]
    v = score(s, stk)
    print(f"  {'Kelly cap$30 frac0.5':<34} {'TRAIN' if fit_lo==0 else 'OOS':<7} {v['n']:>6} "
          f"{v['net']:>+9.2f} {v['roi']*100:>+7.2f}% {v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}]")
    st, rates = kelly(sel, fit_lo, fit_hi)
    flat = score(s, [TAKE] * len(s))
    print(f"  {'  (flat $10 for reference)':<34} {'-':<7} {flat['n']:>6} "
          f"{flat['net']:>+9.2f} {flat['roi']*100:>+7.2f}% {flat['t']:>+6.2f} [{flat['lo']:>+6.3f},{flat['hi']:>+6.3f}]")

# ---------------------------------------------------------------- 2. permutation
print("\n" + "=" * 104)
print("2. PERMUTATION on the bucket effect: is 'cheap favourites win more' luck?")
print("=" * 104)
a = np.array([r["ask"] for r in sel])
w = np.array([r["won"] for r in sel], dtype=float)
cheap = a < 0.85
obs_gap = w[cheap].mean() - a[cheap].mean() - (w[~cheap].mean() - a[~cheap].mean())
obs_wr = w[cheap].mean() - w[~cheap].mean()
rng = np.random.default_rng(99)
null_gap, null_wr = [], []
for _ in range(20000):
    p = rng.permutation(len(w))
    cc = w[p][cheap]
    nc = w[p][~cheap]
    if len(cc) == 0 or len(nc) == 0:
        continue
    null_wr.append(cc.mean() - nc.mean())
    null_gap.append((cc.mean() - a[cheap].mean()) - (nc.mean() - a[~cheap].mean()))
null_wr = np.array(null_wr); null_gap = np.array(null_gap)
print(f"  cheap bucket n={int(cheap.sum())} win={w[cheap].mean():.3f} | "
      f"other n={int((~cheap).sum())} win={w[~cheap].mean():.3f}")
print(f"  observed win-rate gap  = {obs_wr:+.4f}   null 95th pct {np.percentile(null_wr,95):+.4f}  "
      f"p = {float((null_wr>=obs_wr).mean()):.4f}")
print(f"  observed edge gap      = {obs_gap:+.4f}   null 95th pct {np.percentile(null_gap,95):+.4f}  "
      f"p = {float((null_gap>=obs_gap).mean()):.4f}")

# ---------------------------------------------------------------- 3. stress
print("\n" + "=" * 104)
print("3. STRESS: latency, 4x fee, tight spread")
print("=" * 104)
print(f"  {'variant':<30} {'scenario':<16} {'N':>4} {'net$':>9} {'ROI':>8} {'t':>6} {'worst':>7} {'maxDD':>7}")
for label, stake_fn in (("flat $10", lambda r: TAKE),
                        ("$30 if ask<0.85 else $10", lambda r: 30.0 if r["ask"] < 0.85 else TAKE)):
    for scen, lat, fee, spread in (("baseline", 0.0, FEE, 0.04),
                                   ("1s latency", 1.0, FEE, 0.04),
                                   ("4x fee", 0.0, FEE * 4, 0.04)):
        rr = B.build_rows(W, latency_s=lat)
        if spread != 0.04:
            rr = [r for r in rr if r["ask"] - r["bid"] <= spread]
        s = shipped(rr)
        if not s:
            continue
        st = [stake_fn(r) for r in s]
        v = score(s, st, fee=fee)
        print(f"  {label:<30} {scen:<16} {v['n']:>4} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
              f"{v['t']:>+6.2f} {v['worst']:>+7.2f} {v['dd']:>+7.2f}")

# ---------------------------------------------------------------- 4. threshold
print("\n" + "=" * 104)
print("4. THRESHOLD: is the cheap-bucket cut a plateau or a knife edge?")
print("=" * 104)
print(f"  {'cut':>6} {'big$':>6} {'N_big':>6} {'net$':>9} {'ROI':>8} {'t':>6}  {'halves(T1,T2,OOS)':>28}")
for cut in (0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90):
    st = [30.0 if r["ask"] < cut else TAKE for r in sel]
    v = score(sel, st)
    h = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        idx = [i for i, r in enumerate(sel) if lo <= r["wi"] < hi]
        p = pnl_for([sel[i] for i in idx], [st[i] for i in idx]) if idx else np.array([0.0])
        h.append(p.sum())
    nb = sum(1 for r in sel if r["ask"] < cut)
    print(f"  {cut:>6.2f} {30.0:>6.0f} {nb:>6} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% {v['t']:>+6.2f}  "
          f"[{h[0]:>+8.2f},{h[1]:>+8.2f},{h[2]:>+8.2f}]")

# ---------------------------------------------------------------- 5. capital
print("\n" + "=" * 104)
print("5. CAPITAL / DRAWDOWN: what does each variant need, and how bad is the drawdown?")
print("=" * 104)
print(f"  {'variant':<30} {'N':>4} {'staked':>8} {'peak$':>8} {'maxDD$':>8} {'net$':>9} {'ROI':>8}")
for label, stake_fn in (("flat $10", lambda r: TAKE),
                        ("$20 if ask<0.85 else $10", lambda r: 20.0 if r["ask"] < 0.85 else TAKE),
                        ("$30 if ask<0.85 else $10", lambda r: 30.0 if r["ask"] < 0.85 else TAKE)):
    st = [stake_fn(r) for r in sel]
    v = score(sel, st)
    peak = float(max(stake_fn(r) for r in sel))
    print(f"  {label:<30} {v['n']:>4} {v['staked']:>8.0f} {peak:>8.0f} {v['dd']:>+8.2f} "
          f"{v['net']:>+9.2f} {v['roi']*100:>+7.2f}%")
print("\nDONE")
