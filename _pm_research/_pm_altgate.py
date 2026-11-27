"""THE CRUX: does a WIDER band pass the SHIPPED rule's own gate, selected on
TRAIN ONLY, and then hold up out of sample?

The shipped rule was NOT chosen by "max net$" - it was chosen by a
pre-registered gate:
    G1  the bootstrap CI95 of the mean P&L EXCLUDES ZERO
    G2  both chronological halves are positive
    G3  top-5 P&L concentration < 40%
    G4  survives 1s execution latency and 4x fees
    G5  the family-wide placebo (outcome reassignment) is significant
Apply the SAME gate to the band grid, SELECT ON TRAIN ONLY, and see which
band it picks - then look at that band's OOS result.

If the gate picks a different band AND that band holds up OOS, the wider
band is a genuine improvement.  If it picks the shipped band, the shipped
band was already the right answer and the wider bands were selection noise.
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


def halves(sel):
    o = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        o.append(stat(s)["net"] if s else 0.0)
    return o


def gate(sel):
    """The shipped rule's pre-registered gate."""
    v = stat(sel)
    if not v or v["n"] < 20:
        return None
    h = halves(sel)
    return dict(
        v=v,
        g1_ci_excludes_zero=v["lo"] > 0,
        g2_both_halves_positive=(h[0] > 0 and h[1] > 0),
        g3_concentration=v["c5"] < 0.40,
    )


BAND_GRID = []
for tmin in (258, 260, 262, 264, 266):
    for lo in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        for hi in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01):
            if hi - lo < 0.05:
                continue
            BAND_GRID.append((tmin, lo, hi))

print(f"band grid: {len(BAND_GRID)} rules\n")
print("=" * 116)
print("APPLYING THE SHIPPED GATE, SELECTED ON TRAIN ONLY")
print("=" * 116)
print(f"  {'rule':<26} {'trN':>4} {'trNet':>8} {'trT':>6} {'trCI':>16} {'G1':>4} {'G2':>4} {'G3':>4} "
      f"{'G2c':>5} | {'oosN':>5} {'oosNet':>8} {'oosT':>6} {'oosCI':>16}")
passing = []
for tmin, lo, hi in BAND_GRID:
    sel = A.select(tmin, lo, hi, "fav")
    tr = [r for r in sel if r["wi"] < CUT]
    g = gate(tr)
    if not g:
        continue
    v = g["v"]
    te = stat([r for r in sel if r["wi"] >= CUT])
    allh = halves(sel)
    g2c = (allh[0] > 0 and allh[1] > 0)
    if g["g1_ci_excludes_zero"] and g["g2_both_halves_positive"] and g["g3_concentration"]:
        passing.append((v["net"], tmin, lo, hi, sel))
    print(f"  T>={tmin} fav[{lo:.2f},{hi:.2f}){'':<2} {v['n']:>4} {v['net']:>+8.2f} {v['t']:>+6.2f} "
          f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}] {'Y' if g['g1_ci_excludes_zero'] else 'n':>4} "
          f"{'Y' if g['g2_both_halves_positive'] else 'n':>4} {'Y' if g['g3_concentration'] else 'n':>4} "
          f"{'Y' if g2c else 'n':>5} | {te['n']:>5} {te['net']:>+8.2f} {te['t']:>+6.2f} "
          f"[{te['lo']:>+5.2f},{te['hi']:>+5.2f}]")

passing.sort(key=lambda x: -x[0])
print(f"\n  rules passing the gate on TRAIN: {len(passing)}")
for net, tmin, lo, hi, sel in passing[:8]:
    te = stat([r for r in sel if r["wi"] >= CUT])
    allv = stat(sel)
    print(f"    T>={tmin} fav[{lo:.2f},{hi:.2f})  trainNet {net:+.2f}  FULL net {allv['net']:+.2f} "
          f"t {allv['t']:+.2f} CI[{allv['lo']:+.2f},{allv['hi']:+.2f}]  OOS net {te['net']:+.2f} "
          f"t {te['t']:+.2f} CI[{te['lo']:+.2f},{te['hi']:+.2f}]")

if passing:
    net, tmin, lo, hi, sel = passing[0]
    print(f"\n  >>> GATE-SELECTED (train): T>={tmin} fav[{lo:.2f},{hi:.2f})")
    ship = A.select(262, 0.80, 0.96, "fav")
    for nm, s in (("GATE-SELECTED", sel), ("SHIPPED", ship)):
        v = stat(s)
        te = stat([r for r in s if r["wi"] >= CUT])
        h = halves(s)
        print(f"      {nm:<16} N={v['n']:>3} net=${v['net']:>+7.2f} t={v['t']:>+5.2f} "
              f"CI[{v['lo']:>+5.2f},{v['hi']:>+5.2f}] top5={v['c5']*100:>3.0f}%  "
              f"OOS ${te['net']:>+7.2f} t={te['t']:>+5.2f}  halves[{h[0]:+.1f},{h[1]:+.1f}]")

# --------------------------------------------------------------- why does the band matter?
print("\n" + "=" * 116)
print("WHY DOES THE LOWER BAND BOUND CHANGE THE RESULT?  (it changes the ENTRY FRAME)")
print("=" * 116)
for lo, hi in ((0.80, 0.96), (0.70, 0.96), (0.60, 0.96), (0.55, 0.96)):
    sel = A.select(262, lo, hi, "fav")
    a = np.array([r["ask"] for r in sel])
    t = np.array([r["T"] for r in sel])
    v = stat(sel)
    print(f"  [{lo:.2f},{hi:.2f}): N={v['n']:>3} net=${v['net']:>+7.2f} t={v['t']:>+5.2f}  "
          f"mean ask {a.mean():.4f}  mean entry T {t.mean():.1f}  "
          f"entries at T<262: {int((t<262).sum())}  cheapest ask {a.min():.3f}")
print("\nDONE")
