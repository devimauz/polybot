"""Is the wider lower band a REAL parameter effect or selection noise?

The band family showed a JAGGED pattern: lo=0.70 strong, lo=0.65 weak,
lo=0.75 weak, lo=0.80 strong.  A real parameter effect would be smooth
(monotone-ish in the lower bound).  A jagged pattern means the differences
come from WHICH FRAME you happen to enter in a handful of windows - i.e.
selection noise.

The test that settles it: WALK-FORWARD over the lower bound.  Instead of one
60/40 split, use K chronological folds; for each fold, pick the best lower
bound on the data BEFORE the fold, then score it on the fold.  If the
effect is real, the out-of-fold score tracks the in-fold score.  If it is
noise, the out-of-fold score is random.

Also: decompose the mechanism.  How many trades actually CHANGE entry frame
when the lower bound moves, and how much P&L do those specific trades carry?
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
LOS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def stat(sel):
    if not sel:
        return None
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    wi = [r["wi"] for r in sel]
    sh = TAKE / a
    p = sh * w - TAKE - sh * FEE * a * (1 - a)
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    return dict(n=len(sel), net=float(p.sum()), t=float(t), lo=float(lo), hi=float(hi),
                c5=float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan"),
                meanask=float(a.mean()))


# ---------------------------------------------------------------- 1. walk-forward
print("=" * 110)
print("1. WALK-FORWARD over the lower bound (5 chronological folds)")
print("=" * 110)
print("   For each fold: choose the lower bound on ALL data before the fold,")
print("   then score that choice on the fold.  Real effect -> OOF tracks IN.")
K = 5
bounds = [int(NW * k / K) for k in range(K + 1)]
print(f"\n  {'fold':<6} {'windows':<18} {'chosen lo':>10} {'IN net':>9} {'IN t':>6} | "
      f"{'OOF net':>9} {'OOF t':>6} {'OOF n':>6}")
oof_nets, oof_ts = [], []
for k in range(K):
    lo_w, hi_w = bounds[k], bounds[k + 1]
    train = [r for r in range(NW) if r < lo_w]
    if len(train) < 40:
        continue
    best = None
    for L in LOS:
        s = A.select(262, L, 0.96, "fav")
        tr = [r for r in s if r["wi"] < lo_w]
        v = stat(tr)
        if not v or v["n"] < 12:
            continue
        if best is None or v["net"] > best[1]:
            best = (L, v["net"], v)
    if best is None:
        continue
    L = best[0]
    s = A.select(262, L, 0.96, "fav")
    oof = [r for r in s if lo_w <= r["wi"] < hi_w]
    vo = stat(oof)
    if not vo:
        continue
    oof_nets.append(vo["net"]); oof_ts.append(vo["t"])
    print(f"  fold{k}   [{lo_w},{hi_w}){'':<6} {L:>10.2f} {best[1]:>+9.2f} {best[2]['t']:>+6.2f} | "
          f"{vo['net']:>+9.2f} {vo['t']:>+6.2f} {vo['n']:>6}")
print(f"\n  OOF net: {['%+.2f' % x for x in oof_nets]}  -> mean {np.mean(oof_nets):+.2f}")
print(f"  OOF t:   {['%+.2f' % x for x in oof_ts]}")
pos = sum(1 for x in oof_nets if x > 0)
print(f"  folds with positive OOF: {pos}/{len(oof_nets)}")

# what would the SHIPPED band have scored on the same folds?
print("\n  the SHIPPED band [0.80,0.96) on the same folds:")
ship_oof = []
for k in range(K):
    lo_w, hi_w = bounds[k], bounds[k + 1]
    s = A.select(262, 0.80, 0.96, "fav")
    oof = [r for r in s if lo_w <= r["wi"] < hi_w]
    v = stat(oof)
    if v:
        ship_oof.append(v["net"])
        print(f"  fold{k}   OOF net {v['net']:>+9.2f}  t {v['t']:>+6.2f}  n {v['n']:>3}")
print(f"  SHIPPED OOF mean: {np.mean(ship_oof):+.2f}   "
      f"(walk-forward-chosen: {np.mean(oof_nets):+.2f})")

# ---------------------------------------------------------------- 2. mechanism
print("\n" + "=" * 110)
print("2. MECHANISM: how much does the lower bound actually change?")
print("=" * 110)
base = A.select(262, 0.80, 0.96, "fav")
base_map = {r["wi"]: r for r in base}
for L in LOS:
    s = A.select(262, L, 0.96, "fav")
    m = {r["wi"]: r for r in s}
    changed = [wi for wi in m if wi not in base_map or m[wi]["ask"] != base_map[wi]["ask"]]
    # P&L of trades present in THIS set but with a different entry vs shipped
    extra = [wi for wi in m if wi not in base_map]
    print(f"  lo={L:.2f}: N={len(s):>3} net={stat(s)['net']:>+8.2f} t={stat(s)['t']:>+5.2f} | "
          f"windows with a DIFFERENT entry: {len(changed):>3} | windows not in shipped: {len(extra):>2} "
          f"| mean ask {stat(s)['meanask']:.4f}")

# per-window: what is the P&L contribution of the windows that only the
# wide band trades?
print("\n  the windows that ONLY the wide band trades (lo=0.70 vs shipped 0.80):")
w70 = {r["wi"]: r for r in A.select(262, 0.70, 0.96, "fav")}
only = [wi for wi in w70 if wi not in base_map]
for wi in only:
    r = w70[wi]
    sh = TAKE / r["ask"]
    pnl = sh * r["won"] - TAKE - sh * FEE * r["ask"] * (1 - r["ask"])
    print(f"    window {wi}: ask {r['ask']:.3f} T={r['T']:.0f} side {r['side']} "
          f"-> {'WON' if r['won'] else 'LOST'}  P&L {pnl:+.2f}")
print(f"    ({len(only)} window(s); their total P&L = "
      f"{sum(TAKE/r['ask']*r['won']-TAKE-(TAKE/r['ask'])*FEE*r['ask']*(1-r['ask']) for r in [w70[w] for w in only]):+.2f})")

# ---------------------------------------------------------------- 3. robustness
print("\n" + "=" * 110)
print("3. ROBUSTNESS: is the lo=0.70 advantage stable under latency and fees?")
print("=" * 110)
print(f"  {'band':<16} {'scenario':<12} {'N':>4} {'net$':>9} {'t':>6} {'CI95':>17}")
for lo in (0.70, 0.80):
    for scen, lat, fee in (("baseline", 0.0, FEE), ("1s latency", 1.0, FEE), ("4x fee", 0.0, FEE * 4)):
        rr = B.build_rows(W, latency_s=lat)
        c = [r for r in rr if r["T"] >= 262 and r["mkt"] >= 0.5 and lo <= r["ask"] < 0.96]
        s = B.first_per_window(c)
        a = np.array([r["ask"] for r in s])
        w = np.array([r["won"] for r in s], dtype=float)
        sh = TAKE / a
        p = sh * w - TAKE - sh * fee * a * (1 - a)
        t = B.cluster_t([r["wi"] for r in s], p)[1]
        lo_, hi_ = B.boot_ci([r["wi"] for r in s], p)
        print(f"  [{lo:.2f},0.96){'':<4} {scen:<12} {len(s):>4} {p.sum():>+9.2f} {t:>+6.2f} "
              f"[{lo_:>+5.2f},{hi_:>+5.2f}]")
print("\nDONE")
