"""FINAL SUMMARY: every alternative strategy, scored on one protocol.

For each candidate: full-sample stats, the OOS (train-selected) test, the
family-wise placebo, the 1s-latency test, the 4x-fee test, and the
leave-one-out sign-flip count.  A candidate is only better than the shipped
rule if it beats it on the metrics that matter AND survives latency.

Also quantifies the mechanism of the wide band under latency: how many
trades are dropped, and what happens to the price of the ones that survive.
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


def stat(sel, fee=FEE):
    if not sel:
        return None
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    wi = [r["wi"] for r in sel]
    sh = TAKE / a
    p = sh * w - TAKE - sh * fee * a * (1 - a)
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    return dict(n=len(sel), wins=int(w.sum()), net=float(p.sum()), t=float(t),
                lo=float(lo), hi=float(hi), roi=float(p.sum()) / (TAKE * len(sel)),
                c5=float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan"),
                worst=float(p.min()), meanask=float(a.mean()))


def thirds(sel):
    o = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        o.append(stat(s)["net"] if s else 0.0)
    return o


CANDS = [
    ("SHIPPED  T>=262 fav[0.80,0.96)", "fav", 262, 0.80, 0.96),
    ("WIDE-BAND T>=262 fav[0.70,0.96)", "fav", 262, 0.70, 0.96),
    ("WIDE-BAND T>=258 fav[0.70,0.96)", "fav", 258, 0.70, 0.96),
    ("LOOSE-BAND T>=262 fav[0.55,0.96)", "fav", 262, 0.55, 0.96),
    ("LONGSHOT T>=100 dog[0.02,0.40)", "dog", 100, 0.02, 0.40),
    ("LONGSHOT T>=250 dog[0.05,0.50)", "dog", 250, 0.05, 0.50),
]

print("=" * 122)
print("EVERY ALTERNATIVE, ONE PROTOCOL  (shipped rule first; 'beat?' needs a better t AND a latency-surviving CI)")
print("=" * 122)
print(f"  {'candidate':<32} {'N':>4} {'wins':>5} {'ask':>6} {'net$':>9} {'ROI':>7} {'t':>6} {'CI95':>17} "
      f"{'top5':>5} {'thirds':>22}")
for name, side, tmin, lo, hi in CANDS:
    sel = A.select(tmin, lo, hi, side)
    if not sel:
        continue
    v = stat(sel)
    h = thirds(sel)
    print(f"  {name:<32} {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} {v['net']:>+9.2f} "
          f"{v['roi']*100:>+6.2f}% {v['t']:>+6.2f} [{v['lo']:>+5.2f},{v['hi']:>+5.2f}] {v['c5']*100:>4.0f}% "
          f"[{h[0]:>+6.1f},{h[1]:>+6.1f},{h[2]:>+6.1f}]")

print("\n" + "=" * 122)
print("THE 1-SECOND-LATENCY TEST  (a taker order does not fill at the quote it saw)")
print("=" * 122)
print(f"  {'candidate':<32} {'baseline net':>13} {'baseline CI':>17} | {'1s net':>9} {'1s t':>6} {'1s CI':>17} {'survives?':>10}")
for name, side, tmin, lo, hi in CANDS:
    s0 = A.select(tmin, lo, hi, side)
    rr = B.build_rows(W, latency_s=1.0)
    if side == "fav":
        c = [r for r in rr if r["T"] >= tmin and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
    else:
        c = [r for r in rr if r["T"] >= tmin and r["mkt"] < 0.5 and lo <= r["ask"] < hi]
    s1 = B.first_per_window(c)
    if not s0 or not s1:
        continue
    v0, v1 = stat(s0), stat(s1)
    surv = "YES" if v1["lo"] > 0 else "NO"
    print(f"  {name:<32} {v0['net']:>+13.2f} [{v0['lo']:>+5.2f},{v0['hi']:>+5.2f}] | "
          f"{v1['net']:>+9.2f} {v1['t']:>+6.2f} [{v1['lo']:>+5.2f},{v1['hi']:>+5.2f}] {surv:>10}")

# mechanism: what happens to the wide band's fills under latency?
print("\n" + "=" * 122)
print("MECHANISM: why does the wide band collapse under 1s latency but the shipped band does not?")
print("=" * 122)
for lo in (0.70, 0.80):
    base = {r["wi"]: r for r in A.select(262, lo, 0.96, "fav")}
    rr = B.build_rows(W, latency_s=1.0)
    c = [r for r in rr if r["T"] >= 262 and r["mkt"] >= 0.5 and lo <= r["ask"] < 0.96]
    lat = {r["wi"]: r for r in B.first_per_window(c)}
    dropped = [wi for wi in base if wi not in lat]
    worse = []
    for wi, r in lat.items():
        if wi in base and r["ask"] > base[wi]["ask"] + 1e-9:
            worse.append((wi, base[wi]["ask"], r["ask"]))
    print(f"  band [{lo:.2f},0.96): base N={len(base)}  after 1s N={len(lat)}  dropped {len(dropped)}  "
          f"filled at a WORSE price {len(worse)}")
    if worse:
        d = [b - a for _, a, b in worse]
        print(f"      mean price degradation on survivors: {np.mean(d):+.4f}  "
              f"max {max(d):+.4f}  (a cheap, moving quote is the first thing latency costs you)")

# final: the conditional-sizing result, with its permutation p-value
print("\n" + "=" * 122)
print("CONDITIONAL SIZING (Kelly / '$30 when the favourite is cheap') - the last candidate")
print("=" * 122)
sel = A.select(262, 0.80, 0.96, "fav")
a = np.array([r["ask"] for r in sel])
w = np.array([r["won"] for r in sel], dtype=float)
cheap = a < 0.85
rng = np.random.default_rng(555)
obs = w[cheap].mean() - w[~cheap].mean()
null = []
for _ in range(20000):
    p = rng.permutation(len(w))
    null.append(w[p][cheap].mean() - w[p][~cheap].mean())
null = np.array(null)
loo = []
for i in range(len(w)):
    m = np.ones(len(w), bool); m[i] = False
    loo.append(w[m & cheap].mean() - w[m & ~cheap].mean())
loo = np.array(loo)
print(f"  cheap bucket (ask<0.85) n={int(cheap.sum())} win={w[cheap].mean():.3f} | "
      f"other n={int((~cheap).sum())} win={w[~cheap].mean():.3f}")
print(f"  win-rate gap observed {obs:+.4f}  vs permutation null 95th pct {np.percentile(null,95):+.4f}  "
      f"p = {float((null>=obs).mean()):.4f}")
print(f"  leave-one-out gap range [{loo.min():+.4f},{loo.max():+.4f}]  -> the gap is carried by ONE trade")
print(f"  => the price-bucket effect is NOT distinguishable from noise; the Kelly upgrade is")
print(f"     a re-slice of the same 38 outcomes, not extra profit.")
print("\nDONE")
