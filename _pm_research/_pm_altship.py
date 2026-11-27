"""FINAL VERIFICATION of the conditional-sizing candidate.

The candidate: SAME rule as shipped (T>=262, favourite ask [0.80,0.96)),
but the stake is $30 when the favourite's ask < 0.85 and $10 otherwise.

Why this can work even with NO win-rate gap: with win rate p constant
across the band, EV per dollar = (p - ask)/ask, so a lower ask simply
carries more ROI per dollar.  Buy 0.82 with p=0.97 -> 18.3%/dollar;
buy 0.93 with p=0.97 -> 4.3%/dollar.  The Kelly logic says: bet more where
the price is lower, PROVIDED p is not lower there too.

Final checks:
  1. the rule is EXACTLY the shipped rule plus a stake multiplier
  2. it is not a fitting artefact: the threshold is swept and the OOS half
     is checked at every threshold
  3. the sizing survives: latency, fees, depth, min-order-size
  4. the PLACEBO: reassign real outcomes, keep the sizing, does the
     advantage disappear?  (it must, or the sizing is picking up the
     13/13 win rate of one bucket)
  5. the walk-forward: choose the threshold on TRAIN, score on OOS
  6. capital: max drawdown, worst trade, $ needed
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
rng = np.random.default_rng(606)

sel = A.select(262, 0.80, 0.96, "fav")


def pnl_of(sel, fn, fee=FEE):
    out = []
    for r in sel:
        st = fn(r)
        a = r["ask"]
        sh = st / a
        out.append(sh * r["won"] - st - sh * fee * a * (1 - a))
    return np.array(out)


def sc(sel, fn, fee=FEE):
    if not sel:
        return None
    p = pnl_of(sel, fn, fee)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    staked = sum(fn(r) for r in sel)
    o = np.argsort(p)
    tot = p.sum()
    order = np.argsort(wi)
    eq = np.cumsum(p[order])
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(n=len(sel), net=float(p.sum()), staked=staked,
                roi=float(p.sum()) / staked, t=float(t), lo=float(lo), hi=float(hi),
                c5=float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan"),
                worst=float(p.min()), dd=dd)


flat = lambda r: TAKE
cheap30 = lambda r: 30.0 if r["ask"] < 0.85 else TAKE

print("=" * 108)
print("1. THE CANDIDATE")
print("=" * 108)
print("   Rule: IDENTICAL to the shipped rule (T>=262, favourite ask [0.80,0.96), both legs alive,")
print("   spread<=0.04, depth>=$10, hold to settlement).  ONLY the stake changes:")
print("       $30 when the favourite's executable ask < 0.85, else $10.")
for nm, fn in (("flat $10", flat), ("$30 if ask<0.85", cheap30)):
    v = sc(sel, fn)
    te = sc([r for r in sel if r["wi"] >= CUT], fn)
    tr = sc([r for r in sel if r["wi"] < CUT], fn)
    print(f"   {nm:<18} N={v['n']} net=${v['net']:+.2f} ROI={v['roi']*100:+.2f}% t={v['t']:+.2f} "
          f"CI[{v['lo']:+.2f},{v['hi']:+.2f}] top5={v['c5']*100:.0f}% maxDD=${v['dd']:.2f} | "
          f"train ${tr['net']:+.2f} (t {tr['t']:+.2f})  OOS ${te['net']:+.2f} (t {te['t']:+.2f})")

print("\n" + "=" * 108)
print("2. THRESHOLD SWEEP - is 0.85 a plateau, and does the OOS half agree at every threshold?")
print("=" * 108)
print(f"  {'cut':>6} {'n_cheap':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} {'train$':>9} {'OOS$':>9} {'OOS t':>6}")
for cut in (0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.92):
    fn = (lambda c: (lambda r: 30.0 if r["ask"] < c else TAKE))(cut)
    v = sc(sel, fn)
    tr = sc([r for r in sel if r["wi"] < CUT], fn)
    te = sc([r for r in sel if r["wi"] >= CUT], fn)
    nc = sum(1 for r in sel if r["ask"] < cut)
    print(f"  {cut:>6.2f} {nc:>8} {v['net']:>+9.2f} {v['roi']*100:>+7.2f}% {v['t']:>+6.2f} "
          f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}] {tr['net']:>+9.2f} {te['net']:>+9.2f} {te['t']:>+6.2f}")

print("\n" + "=" * 108)
print("3. STRESS: latency, fees, min-order-size")
print("=" * 108)
for scen, lat, fee in (("baseline", 0.0, FEE), ("1s latency", 1.0, FEE), ("2s latency", 2.0, FEE),
                       ("4x fee", 0.0, FEE * 4), ("8x fee", 0.0, FEE * 8)):
    rr = B.build_rows(W, latency_s=lat)
    c = [r for r in rr if r["T"] >= 262 and r["mkt"] >= 0.5 and 0.80 <= r["ask"] < 0.96]
    s1 = B.first_per_window(c)
    v0, v1 = sc(sel, cheap30, fee), sc(s1, cheap30, fee)
    print(f"   {scen:<12} base ${v0['net']:>+7.2f} t {v0['t']:>+5.2f} CI[{v0['lo']:>+5.2f},{v0['hi']:>+5.2f}] | "
          f"stressed N={v1['n']:>2} ${v1['net']:>+7.2f} t {v1['t']:>+5.2f} CI[{v1['lo']:>+5.2f},{v1['hi']:>+5.2f}]")
print("   min-order-size: $30 at ask 0.81 = 37 shares (need >=5) OK;  $10 at ask 0.95 = 10.5 shares OK")

print("\n" + "=" * 108)
print("4. PLACEBO: reassign real outcomes across windows, KEEP the sizing rule")
print("=" * 108)
rows_all = B.build_rows(W)
fav = [r for r in rows_all if r["mkt"] >= 0.5]
by_wi = {}
for r in fav:
    by_wi.setdefault(r["wi"], []).append(r)
wi_list = sorted(by_wi)
obs = sc(sel, cheap30)
diffs = []
for _ in range(1500):
    perm = list(wi_list)
    rng.shuffle(perm)
    wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
    c = []
    for src_wi in wi_list:
        for r in by_wi[src_wi]:
            if r["T"] < 262 or not (0.80 <= r["ask"] < 0.96):
                continue
            r2 = dict(r)
            r2["won"] = 1 if W[wperm[src_wi]]["_winner"] == r["side"] else 0
            c.append(r2)
    c = B.first_per_window(c)
    if len(c) < 5:
        continue
    ps = pnl_of(c, cheap30)
    pf = pnl_of(c, flat)
    diffs.append((ps.sum() - pf.sum()) / abs(pf.sum()) if pf.sum() else 0.0)
diffs = np.array(diffs)
print(f"   observed: sized ${obs['net']:+.2f} vs flat ${sc(sel,flat)['net']:+.2f}  "
      f"(+{obs['net']-sc(sel,flat)['net']:.2f}, +{(obs['net']/sc(sel,flat)['net']-1)*100:.0f}%)")
print(f"   placebo relative-gain null: mean {diffs.mean()*100:+.1f}%  95th pct {np.percentile(diffs,95)*100:+.1f}%  "
      f"max {diffs.max()*100:+.1f}%")
print(f"   P(placebo beats the observed gain) = {float((diffs >= (obs['net']/sc(sel,flat)['net']-1)).mean()):.4f}")

print("\n" + "=" * 108)
print("5. WALK-FORWARD: pick the threshold on TRAIN, score on the untouched fold")
print("=" * 108)
K = 5
bounds = [int(NW * k / K) for k in range(K + 1)]
res = []
for k in range(K):
    lo_w, hi_w = bounds[k], bounds[k + 1]
    if lo_w < 40:
        continue
    best = None
    for cut in (0.81, 0.83, 0.85, 0.87, 0.89, 0.91):
        fn = (lambda c: (lambda r: 30.0 if r["ask"] < c else TAKE))(cut)
        tr = sc([r for r in sel if r["wi"] < lo_w], fn)
        if tr and tr["n"] >= 8:
            if best is None or tr["net"] > best[1]:
                best = (cut, tr["net"], tr)
    if best is None:
        continue
    cut = best[0]
    fn = (lambda c: (lambda r: 30.0 if r["ask"] < c else TAKE))(cut)
    oof = sc([r for r in sel if lo_w <= r["wi"] < hi_w], fn)
    oof_flat = sc([r for r in sel if lo_w <= r["wi"] < hi_w], flat)
    if not oof:
        continue
    res.append(oof["net"] - oof_flat["net"])
    print(f"   fold{k}: chosen cut {cut:.2f} (train net {best[1]:+.2f}) -> OOF sized ${oof['net']:+.2f} "
          f"vs flat ${oof_flat['net']:+.2f}  (delta {oof['net']-oof_flat['net']:+.2f})")
print(f"   mean OOF delta vs flat: {np.mean(res):+.2f}   folds positive: {sum(1 for x in res if x>0)}/{len(res)}")

print("\n" + "=" * 108)
print("6. CAPITAL AND RISK")
print("=" * 108)
for nm, fn in (("flat $10", flat), ("$30 if ask<0.85", cheap30)):
    v = sc(sel, fn)
    print(f"   {nm:<18} staked ${v['staked']:.0f}  net ${v['net']:+.2f}  ROI {v['roi']*100:+.2f}%  "
          f"worst ${v['worst']:+.2f}  maxDD ${v['dd']:+.2f}  top5 {v['c5']*100:.0f}%")
print("\nDONE")
