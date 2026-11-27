"""LAST CANDIDATE: sizing by price bucket - tested properly.

Why this one still matters: the earlier permutation test asked the WRONG
question.  It tested "do cheap favourites win MORE often?" (p=0.66 - no).
But the Kelly argument does not need a higher win rate.  It needs the win
rate to be roughly CONSTANT across the band, in which case buying at 0.82
simply has more ROI per dollar than buying at 0.93:
    EV per dollar at ask a, win rate p:  (p - a) / a
With p=0.974:  a=0.82 -> 18.8%   a=0.93 -> 4.7%
So concentrating stake on the cheap end can be right even with NO win-rate
gap - provided the win rate is at least as high there.

Tests:
  1. simple fixed ratios (2x, 3x) on the cheap sub-band - no fitting, so no
     overfitting risk
  2. shrinkage-Kelly: shrink the bucket win rate toward the band mean
     (n=13 is far too few to trust p=1.000)
  3. BOOTSTRAP THE SIZING DECISION: resample windows, refit, rescore.  Does
     the sizing beat flat on average, or was the win a lucky resample?
  4. SENSITIVITY: what cheap-bucket win rate would make the aggressive
     sizing lose to flat?  (the honest way to state the risk)
  5. 1s LATENCY on the sized variants
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
rng = np.random.default_rng(2024)

CUT_PRICE = 0.85


def sized_pnl(sel, stake_fn, fee=FEE):
    out = []
    for r in sel:
        st = stake_fn(r)
        a = r["ask"]
        sh = st / a
        out.append(sh * r["won"] - st - sh * fee * a * (1 - a))
    return np.array(out)


def score(sel, stake_fn, fee=FEE):
    p = sized_pnl(sel, stake_fn, fee)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    staked = sum(stake_fn(r) for r in sel)
    o = np.argsort(p)
    tot = p.sum()
    return dict(n=len(sel), net=float(p.sum()), staked=staked,
                roi=float(p.sum()) / staked, t=float(t), lo=float(lo), hi=float(hi),
                c5=float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan"),
                worst=float(p.min()), p=p)


def thirds(sel, stake_fn, fee=FEE):
    o = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        o.append(sized_pnl(s, stake_fn, fee).sum() if s else 0.0)
    return o


sel = A.select(262, 0.80, 0.96, "fav")
print(f"shipped-rule opportunities: {len(sel)}\n")

# --------------------------------------------------------------- 1. fixed ratios
print("=" * 108)
print("1. FIXED RATIOS on the cheap sub-band (no fitting at all)")
print("=" * 108)
print(f"  {'variant':<32} {'N':>4} {'staked':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} "
      f"{'OOS net':>9} {'OOS t':>6}  {'thirds'}")
variants = [
    ("flat $10", lambda r: TAKE),
    ("$20 if ask<0.85 else $10", lambda r: 20.0 if r["ask"] < CUT_PRICE else TAKE),
    ("$30 if ask<0.85 else $10", lambda r: 30.0 if r["ask"] < CUT_PRICE else TAKE),
    ("$15 if ask<0.85 else $10", lambda r: 15.0 if r["ask"] < CUT_PRICE else TAKE),
]
for name, fn in variants:
    v = score(sel, fn)
    te = score([r for r in sel if r["wi"] >= CUT], fn)
    h = thirds(sel, fn)
    print(f"  {name:<32} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% "
          f"{v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {te['net']:>+9.2f} {te['t']:>+6.2f}  "
          f"[{h[0]:>+6.1f},{h[1]:>+6.1f},{h[2]:>+6.1f}]")

# --------------------------------------------------------------- 2. shrinkage Kelly
print("\n" + "=" * 108)
print("2. SHRINKAGE-KELLY: shrink each bucket's win rate toward the band mean")
print("=" * 108)


def shrunk_kelly(sel, fit_lo, fit_hi, cap=30.0, frac=0.5, prior_w=8.0):
    fit = [r for r in sel if fit_lo <= r["wi"] < fit_hi]
    prior_p = (sum(r["won"] for r in fit) / len(fit)) if fit else 0.5
    buckets = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.96)]

    def bk(r):
        for b in buckets:
            if b[0] <= r["ask"] < b[1]:
                return b
        return buckets[-1]
    rates = {}
    for b in buckets:
        rs = [r for r in fit if bk(r) == b]
        if rs:
            w = sum(r["won"] for r in rs)
            rates[b] = (w + prior_w * prior_p) / (len(rs) + prior_w)
        else:
            rates[b] = prior_p

    def fn(r):
        a = r["ask"]
        p = rates[bk(r)]
        edge = (p - a) / (1 - a) if a < 1 else 0.0
        return max(2.0, min(cap, TAKE * max(0.0, frac * edge / 0.10)))
    return fn, rates


for pw in (0.0, 4.0, 8.0, 20.0):
    fn, rates = shrunk_kelly(sel, 0, CUT, prior_w=pw)
    v = score(sel, fn)
    te = score([r for r in sel if r["wi"] >= CUT], fn)
    print(f"  prior weight {pw:>5.1f}: rates " +
          ", ".join(f"{b[0]:.2f}={rates[b]:.3f}" for b in rates) +
          f"  -> net ${v['net']:+.2f} ROI {v['roi']*100:+.2f}% t {v['t']:+.2f}  OOS ${te['net']:+.2f} t {te['t']:+.2f}")

# --------------------------------------------------------------- 3. bootstrap
print("\n" + "=" * 108)
print("3. BOOTSTRAP THE SIZING DECISION (resample windows, refit, rescore)")
print("=" * 108)
wins_by_win = {}
for wi in range(NW):
    wins_by_win[wi] = [r for r in sel if r["wi"] == wi]
active = [wi for wi in wins_by_win if wins_by_win[wi]]
# vectorise: per-window arrays so the bootstrap is a numpy operation
W_ASK = {wi: np.array([r["ask"] for r in wins_by_win[wi]]) for wi in active}
W_WON = {wi: np.array([r["won"] for r in wins_by_win[wi]], dtype=float) for wi in active}
A_ARR = np.array([W_ASK[wi] for wi in active], dtype=object)
V_ARR = np.array([W_WON[wi] for wi in active], dtype=object)
N_ACT = len(active)
IDX = np.arange(N_ACT)


def boot_net_roi(ratio, n_draw=2000, seed=1000):
    """Resample windows with replacement, compute net$ and ROI for flat and
    for the ratio-sized variant on the SAME resample."""
    r2 = np.random.default_rng(seed)
    fn_nets = [0.0, 0.0]
    fn_rois = [0.0, 0.0]
    fn_stak = [0.0, 0.0]
    for _ in range(n_draw):
        picks = r2.integers(0, N_ACT, N_ACT)
        asks = np.concatenate([A_ARR[i] for i in picks])
        wons = np.concatenate([V_ARR[i] for i in picks])
        for k, ratio_k in enumerate((None, ratio)):
            mult = 1.0 if ratio_k is None else ratio_k
            st = np.where(asks < CUT_PRICE, TAKE * mult, TAKE).astype(float)
            sh = st / asks
            p = sh * wons - st - sh * FEE * asks * (1 - asks)
            fn_nets[k] += p.sum()
            fn_rois[k] += p.sum() / st.sum()
            fn_stak[k] += st.sum()
    n_draw = float(n_draw)
    return (fn_nets[0] / n_draw, fn_rois[0] / n_draw,
            fn_nets[1] / n_draw, fn_rois[1] / n_draw)


print(f"  {'ratio':<12} {'flat mean net$':>16} {'flat mean ROI':>15} | {'sized mean net$':>17} {'sized mean ROI':>16}")
for ratio in (1.5, 2.0, 3.0):
    fn_net, fn_roi, sz_net, sz_roi = boot_net_roi(ratio, n_draw=400)
    print(f"  {ratio:<12.1f} {fn_net:>+16.2f} {fn_roi*100:>+14.2f}% | {sz_net:>+17.2f} {sz_roi*100:>+15.2f}%")
print("  (a positive difference in BOTH net$ and ROI across 400 window resamples")
print("   means the sizing is not an artefact of one lucky resample)")

# --------------------------------------------------------------- 4. sensitivity
print("\n" + "=" * 108)
print("4. SENSITIVITY: what cheap-bucket win rate would make the sizing LOSE to flat?")
print("=" * 108)
cheap = [r for r in sel if r["ask"] < CUT_PRICE]
dear = [r for r in sel if r["ask"] >= CUT_PRICE]
print(f"  cheap n={len(cheap)} (realised {sum(r['won'] for r in cheap)/len(cheap):.3f}), "
      f"dear n={len(dear)} (realised {sum(r['won'] for r in dear)/len(dear):.3f})")
print(f"  {'assumed cheap p':>17} {'$30-if-cheap net$':>19} {'vs flat +$38.13':>17} {'verdict':>10}")
for p_cheap in (1.00, 0.95, 0.90, 0.85, 0.83, 0.80, 0.75):
    # re-price the cheap bucket at the assumed win rate, dear bucket unchanged
    tot = 0.0
    for r in sel:
        a = r["ask"]
        st = 30.0 if a < CUT_PRICE else TAKE
        won = p_cheap if a < CUT_PRICE else r["won"]
        sh = st / a
        tot += sh * won - st - sh * FEE * a * (1 - a)
    print(f"  {p_cheap:>17.2f} {tot:>+19.2f} {tot-38.13:>+17.2f} "
          f"{'BETTER' if tot > 38.13 else 'WORSE':>10}")

# --------------------------------------------------------------- 5. latency
print("\n" + "=" * 108)
print("5. 1-SECOND LATENCY on the sized variants")
print("=" * 108)
print(f"  {'variant':<32} {'base net':>10} {'base CI':>17} | {'1s net':>9} {'1s t':>6} {'1s CI':>17} {'survives':>9}")
for name, fn in variants:
    rr = B.build_rows(W, latency_s=1.0)
    c = [r for r in rr if r["T"] >= 262 and r["mkt"] >= 0.5 and 0.80 <= r["ask"] < 0.96]
    s1 = B.first_per_window(c)
    if not s1:
        continue
    v0 = score(sel, fn)
    v1 = score(s1, fn)
    print(f"  {name:<32} {v0['net']:>+10.2f} [{v0['lo']:>+5.2f},{v0['hi']:>+5.2f}] | "
          f"{v1['net']:>+9.2f} {v1['t']:>+6.2f} [{v1['lo']:>+5.2f},{v1['hi']:>+5.2f}] "
          f"{'YES' if v1['lo']>0 else 'NO':>9}")
print("\nDONE")
