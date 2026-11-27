"""Final adjudication of the three "more profit" candidates.

  A. A DIFFERENT ENTRY RULE      - 1023-candidate grid; does anything else
                                   survive train-selection -> OOS -> placebo?
  B. MORE TRADES per window      - take every qualifying quote, not just the
                                   first. More raw dollars, weaker edge?
  C. CONDITIONAL SIZING          - bet more when the favourite is cheap.
                                   Is the price-bucket effect real, or is it
                                   just "the single loss happened to sit in
                                   the expensive band"?

C is the subtle one: with 37 wins out of 38, ANY split of the one loss
looks like a price effect.  A permutation that shuffles the wins across the
trades while keeping the prices fixed is the honest test.
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
rng = np.random.default_rng(4242)


def pnl_of_stakes(sel, stakes, fee=FEE):
    out = []
    for r, st in zip(sel, stakes):
        a = r["ask"]
        sh = st / a
        out.append(sh * r["won"] - st - sh * fee * a * (1 - a))
    return np.array(out)


def stat(sel, stakes, fee=FEE):
    p = pnl_of_stakes(sel, stakes, fee)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    staked = sum(stakes)
    o = np.argsort(p)
    tot = p.sum()
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    order = np.argsort(wi)
    eq = np.cumsum(p[order])
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    return dict(n=len(sel), net=float(p.sum()), staked=float(staked),
                roi=float(p.sum()) / staked, t=float(t), lo=float(lo), hi=float(hi),
                worst=float(p.min()), c5=c5, dd=dd, p=p)


def thirds(sel, stakes):
    out = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        idx = [i for i, r in enumerate(sel) if lo <= r["wi"] < hi]
        out.append(pnl_of_stakes([sel[i] for i in idx], [stakes[i] for i in idx]).sum()
                   if idx else 0.0)
    return out


print("=" * 106)
print("C. IS THE PRICE-BUCKET EFFECT REAL?")
print("=" * 106)
sel = B.first_per_window([r for r in rows_all if r["T"] >= 262 and r["mkt"] >= 0.5
                          and 0.80 <= r["ask"] < 0.96])
a = np.array([r["ask"] for r in sel])
w = np.array([r["won"] for r in sel], dtype=float)
for cut in (0.85, 0.88, 0.90):
    cheap = a < cut
    obs_wr = w[cheap].mean() - w[~cheap].mean()
    obs_edge = (w[cheap].mean() - a[cheap].mean()) - (w[~cheap].mean() - a[~cheap].mean())
    nulls_wr, nulls_edge = [], []
    for _ in range(20000):
        p = rng.permutation(len(w))
        cc, nc = w[p][cheap], w[p][~cheap]
        nulls_wr.append(cc.mean() - nc.mean())
        nulls_edge.append((cc.mean() - a[cheap].mean()) - (nc.mean() - a[~cheap].mean()))
    nulls_wr = np.array(nulls_wr); nulls_edge = np.array(nulls_edge)
    print(f"  cut {cut:.2f}: cheap n={int(cheap.sum())} win={w[cheap].mean():.3f}  "
          f"other n={int((~cheap).sum())} win={w[~cheap].mean():.3f}")
    print(f"      win-rate gap  obs {obs_wr:+.4f}  null 95th {np.percentile(nulls_wr,95):+.4f}  "
          f"p={float((nulls_wr>=obs_wr).mean()):.4f}")
    print(f"      edge gap      obs {obs_edge:+.4f}  null 95th {np.percentile(nulls_edge,95):+.4f}  "
          f"p={float((nulls_edge>=obs_edge).mean()):.4f}")

# leave-one-out: how stable is "cheap wins more"?
print("\n  leave-one-out stability of the win-rate gap (cut 0.85):")
cheap = a < 0.85
gaps = []
for i in range(len(w)):
    m = np.ones(len(w), bool); m[i] = False
    g = w[m & cheap].mean() - w[m & ~cheap].mean()
    gaps.append(g)
gaps = np.array(gaps)
print(f"    gap = {w[cheap].mean() - w[~cheap].mean():+.4f};  leave-one-out range "
      f"[{gaps.min():+.4f}, {gaps.max():+.4f}];  "
      f"sign flips: {int((gaps<=0).sum())}/{len(gaps)}")
print("    -> the gap survives only because ONE trade carries it")

print("\n" + "=" * 106)
print("B. MORE TRADES PER WINDOW (every qualifying quote) - does it survive?")
print("=" * 106)
print(f"  {'variant':<34} {'N':>4} {'staked':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} {'top5':>5} {'thirds':>28}")
multi = [r for r in rows_all if r["T"] >= 262 and r["mkt"] >= 0.5 and 0.80 <= r["ask"] < 0.96]
v = stat(multi, [TAKE] * len(multi))
h = thirds(multi, [TAKE] * len(multi))
print(f"  {'every quote, T>=262 [0.80,0.96)':<34} {v['n']:>4} {v['staked']:>8.0f} {v['net']:>+9.2f} "
      f"{v['roi']*100:>+7.2f}% {v['t']:>+6.2f} [{v['lo']:>+6.3f},{v['hi']:>+6.3f}] {v['c5']*100:>4.0f}%  "
      f"[{h[0]:>+7.1f},{h[1]:>+7.1f},{h[2]:>+7.1f}]")
one = B.first_per_window(multi)
v2 = stat(one, [TAKE] * len(one))
h2 = thirds(one, [TAKE] * len(one))
print(f"  {'first quote only (SHIPPED)':<34} {v2['n']:>4} {v2['staked']:>8.0f} {v2['net']:>+9.2f} "
      f"{v2['roi']*100:>+7.2f}% {v2['t']:>+6.2f} [{v2['lo']:>+6.3f},{v2['hi']:>+6.3f}] {v2['c5']*100:>4.0f}%  "
      f"[{h2[0]:>+7.1f},{h2[1]:>+7.1f},{h2[2]:>+7.1f}]")

# placebo for the multi-trade variant: reassign outcomes across windows
def placebo_multi(n_draw=800):
    fav = [r for r in rows_all if r["mkt"] >= 0.5]
    by_wi = {}
    for r in fav:
        by_wi.setdefault(r["wi"], []).append(r)
    wi_list = sorted(by_wi)
    ts = []
    for _ in range(n_draw):
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
        if len(c) < 5:
            continue
        p = B.pnl_of([r["ask"] for r in c], [r["won"] for r in c])
        ts.append(B.cluster_t([r["wi"] for r in c], p)[1])
    return np.array(ts)

null = placebo_multi()
obs = stat(multi, [TAKE] * len(multi))["t"]
print(f"\n  PLACEBO for the multi-trade variant: obs t={obs:+.2f}  null mean {null.mean():+.2f} "
      f"95th {np.percentile(null,95):+.2f}  p={float((null>=obs).mean()):.4f}")

print("\n" + "=" * 106)
print("A. THE GRID: does any DIFFERENT rule beat the shipped one, end to end?")
print("=" * 106)
recs = np.load("_pm_alt_recs.npy", allow_pickle=True)
recs = list(recs)
# how many candidates are positive OOS with a CI excluding zero?
good = [r for r in recs if r["te"]["lo"] > 0]
print(f"  candidates with an OOS CI95 that EXCLUDES zero: {len(good)} of {len(recs)}")
for r in sorted(good, key=lambda x: -x["te"]["net"])[:12]:
    te = r["te"]
    print(f"    {r['name']:<32} OOS n={te['n']:>3} net={te['net']:>+8.2f} t={te['t']:>+6.2f} "
          f"CI=[{te['lo']:>+6.2f},{te['hi']:>+6.2f}]  train net={r['tr']['net']:>+8.2f} "
          f"train t={r['tr']['t']:>+5.2f}")
# of those, how many also have a positive TRAIN half and all three thirds?
if good:
    robust = [r for r in good if r["tr"]["net"] > 0]
    print(f"\n  of those, also positive on TRAIN: {len(robust)}")
    for r in sorted(robust, key=lambda x: -x["te"]["net"])[:10]:
        te = r["te"]
        print(f"    {r['name']:<32} OOS n={te['n']:>3} net={te['net']:>+8.2f} t={te['t']:>+6.2f} "
              f"CI=[{te['lo']:>+6.2f},{te['hi']:>+6.2f}]  train net={r['tr']['net']:>+8.2f}")
print("\nDONE")
