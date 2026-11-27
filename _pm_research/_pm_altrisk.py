"""RISK ANALYSIS of the conditional-sizing overlay, before shipping it.

The sizing works for a reason that does NOT require a win-rate gap: with a
roughly CONSTANT win rate p across the band, EV per dollar = (p - ask)/ask,
so a cheaper entry carries more ROI per dollar.  Verify that arithmetic
directly, then quantify the risk:

  1. EV-per-dollar curve at constant p (the mechanism)
  2. BREAK-EVEN win rate for each stake multiplier - the number that
     decides whether the overlay is safe
  3. CONFIDENCE on the cheap bucket's win rate (13/13 observed)
  4. DRAWDOWN if the loss lands in the CHEAP bucket (it happened to land in
     the dear one - do not pretend that is structural)
  5. the conservative multiplier: which one keeps the best risk/return?
"""
import sys, numpy as np
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B
import _pm_altsearch as A

W = A.W
CUT = A.CUT
TAKE = 10.0
FEE = 0.07
sel = A.select(262, 0.80, 0.96, "fav")
CUT_PRICE = 0.85

print("=" * 104)
print("1. THE MECHANISM: EV per dollar at a CONSTANT win rate (no gap needed)")
print("=" * 104)
print(f"  {'ask':>6} " + " ".join(f"p={p:.2f}" for p in (0.90, 0.95, 0.9737, 0.99)))
for a in (0.81, 0.83, 0.85, 0.88, 0.91, 0.94):
    row = []
    for p in (0.90, 0.95, 0.9737, 0.99):
        ev = (p * (1 - a) - (1 - p) * a) / a - FEE * (1 - a)   # net per dollar, fee incl.
        row.append(f"{ev*100:+6.2f}%")
    print(f"  {a:>6.2f} " + " ".join(row))
print("   -> at a constant p, ROI per dollar rises steeply as the price falls.")
print("      That is the whole reason to size up on cheap entries.")

print("\n" + "=" * 104)
print("2. BREAK-EVEN: the cheap-bucket win rate at which each multiplier ties flat $10")
print("=" * 104)
cheap = [r for r in sel if r["ask"] < CUT_PRICE]
dear = [r for r in sel if r["ask"] >= CUT_PRICE]
print(f"   cheap n={len(cheap)} observed {sum(r['won'] for r in cheap)}/{len(cheap)}  "
      f"mean ask {np.mean([r['ask'] for r in cheap]):.4f}")
print(f"   dear  n={len(dear)} observed {sum(r['won'] for r in dear)}/{len(dear)}  "
      f"mean ask {np.mean([r['ask'] for r in dear]):.4f}")
print(f"\n   {'multiplier':>11} {'stake':>7} {'breakeven cheap p':>20} {'margin vs observed':>20}")
for mult in (1.5, 2.0, 2.5, 3.0):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = 0.0
        for r in sel:
            a = r["ask"]
            st = TAKE * mult if a < CUT_PRICE else TAKE
            won = mid if a < CUT_PRICE else r["won"]
            sh = st / a
            tot += sh * won - st - sh * FEE * a * (1 - a)
        if tot > 38.13:
            hi = mid
        else:
            lo = mid
    be = (lo + hi) / 2
    obs = sum(r["won"] for r in cheap) / len(cheap)
    print(f"   {mult:>11.1f} {TAKE*mult:>7.0f} {be:>20.4f} {obs-be:>+20.4f}")

print("\n" + "=" * 104)
print("3. CONFIDENCE on the cheap bucket's win rate (13/13 observed)")
print("=" * 104)
n, k = len(cheap), sum(r["won"] for r in cheap)
# Clopper-Pearson one-sided 95% lower bound
lo = 0.0 if k == 0 else (0.05 ** (1.0 / n))
hi = 1.0 if k == n else (1 - (1 - 0.05) ** (1.0 / n))
print(f"   observed {k}/{n} = {k/n:.3f}")
print(f"   95% one-sided CI: [{lo:.3f}, {hi:.3f}]")
print(f"   P(13/13 wins | true p = 0.90) = {0.90**n:.3f}  <- a 13/13 run is ORDINARY if p=0.90")
print(f"   P(13/13 wins | true p = 0.80) = {0.80**n:.3f}")
print("   -> the cheap bucket's 13/13 is consistent with a constant p; it is NOT")
print("      evidence of a win-rate gap, and it does not need to be.")

print("\n" + "=" * 104)
print("4. DRAWDOWN if the loss lands in the CHEAP bucket (it landed in the dear one)")
print("=" * 104)
for mult in (1.5, 2.0, 3.0):
    fn = (lambda m: (lambda r: TAKE * m if r["ask"] < CUT_PRICE else TAKE))(mult)
    pnl = []
    for r in sel:
        st = fn(r)
        a = r["ask"]
        sh = st / a
        pnl.append((r["wi"], sh * r["won"] - st - sh * FEE * a * (1 - a)))
    pnl.sort()
    eq = np.cumsum([x[1] for x in pnl])
    dd = float((eq - np.maximum.accumulate(eq)).min())
    worst_cheap = min((x[1] for x in pnl if any(r["wi"] == x[0] and r["ask"] < CUT_PRICE for r in sel)),
                       default=0.0)
    print(f"   mult {mult:.1f}: realised maxDD ${dd:+.2f}   worst-cheap-bucket trade ${worst_cheap:+.2f} "
          f"(this is the number to plan for)")

print("\n" + "=" * 104)
print("5. RISK/RETURN OF EACH MULTIPLIER")
print("=" * 104)
print(f"   {'mult':>6} {'staked':>8} {'net$':>9} {'ROI':>8} {'t':>6} {'CI95':>18} {'maxDD':>8} {'breakeven p':>12}")
for mult in (1.0, 1.5, 2.0, 2.5, 3.0):
    fn = (lambda m: (lambda r: TAKE * m if r["ask"] < CUT_PRICE else TAKE))(mult)
    p = []
    for r in sel:
        st = fn(r)
        a = r["ask"]
        sh = st / a
        p.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
    p = np.array(p)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    eq = np.cumsum(p[np.argsort(wi)])
    dd = float((eq - np.maximum.accumulate(eq)).min())
    staked = sum(fn(r) for r in sel)
    # breakeven
    lo2, hi2 = 0.0, 1.0
    for _ in range(60):
        mid = (lo2 + hi2) / 2
        tot = 0.0
        for r in sel:
            a = r["ask"]
            st = TAKE * mult if a < CUT_PRICE else TAKE
            won = mid if a < CUT_PRICE else r["won"]
            sh = st / a
            tot += sh * won - st - sh * FEE * a * (1 - a)
        if tot > 38.13:
            hi2 = mid
        else:
            lo2 = mid
    print(f"   {mult:>6.1f} {staked:>8.0f} {p.sum():>+9.2f} {p.sum()/staked*100:>+7.2f}% {t:>+6.2f} "
          f"[{lo:>+5.2f},{hi:>+5.2f}] {dd:>+8.2f} {(lo2+hi2)/2:>12.4f}")
print("\nDONE")
