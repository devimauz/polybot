"""FINAL RISK CHECK on the sizing overlay + the exact numbers for the report.

The overlay's justification is: at a CONSTANT win rate p across the band,
EV per dollar = (p - ask)/ask, so cheap entries carry more ROI per dollar
and deserve more capital.  It does NOT require a win-rate gap (the placebo
found none: p=0.66).

The real risks to quantify:
  1. CONSTANT-p SENSITIVITY: if the true p is p0 (not 0.974) and constant
     across the band, at what p0 does the sized portfolio tie the flat one?
     This is the honest "how wrong can I be" number.
  2. NESTED WALK-FORWARD: choose BOTH the threshold and the multiplier on
     train, score on the untouched fold.
  3. Is the cheap bucket's 13/13 carried by ONE window?
  4. Exact wide-band vs shipped under 1s latency, for the report.
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
sel = A.select(262, 0.80, 0.96, "fav")
CUT_PRICE = 0.85


def port(sel, fn, p0=None, fee=FEE):
    """Portfolio P&L.  p0 overrides the win rate of EVERY trade (constant-p world)."""
    tot = 0.0
    staked = 0.0
    for r in sel:
        st = fn(r)
        a = r["ask"]
        sh = st / a
        won = r["won"] if p0 is None else p0
        tot += sh * won - st - sh * fee * a * (1 - a)
        staked += st
    return tot, staked


flat = lambda r: TAKE
mk = lambda m, c: (lambda r: TAKE * m if r["ask"] < c else TAKE)

print("=" * 100)
print("1. CONSTANT-p SENSITIVITY: sized (3x cheap) vs flat, in a world where the")
print("   true win rate is p0 for EVERY trade and the band has no edge at all")
print("=" * 100)
print(f"   {'p0':>6} {'flat net$':>11} {'sized net$':>12} {'flat ROI':>10} {'sized ROI':>11} {'winner':>9}")
for p0 in (0.975, 0.97, 0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80):
    f, fs = port(sel, flat, p0)
    s, ss = port(sel, mk(3.0, CUT_PRICE), p0)
    print(f"   {p0:>6.3f} {f:>+11.2f} {s:>+12.2f} {f/fs*100:>+9.2f}% {s/ss*100:>+10.2f}% "
          f"{'sized' if s > f else 'flat':>9}")
print("   (this is a STRESS TEST, not a forecast: it assumes p0 applies to every")
print("    trade, i.e. that the whole edge is a constant-p artefact)")

print("\n" + "=" * 100)
print("2. NESTED WALK-FORWARD: choose threshold AND multiplier on TRAIN, score OOS")
print("=" * 100)
K = 5
bounds = [int(NW * k / K) for k in range(K + 1)]
GRID = [(c, m) for c in (0.83, 0.85, 0.87, 0.89, 0.91) for m in (1.5, 2.0, 2.5, 3.0)]
deltas, nfold = [], 0
for k in range(K):
    lo_w, hi_w = bounds[k], bounds[k + 1]
    if lo_w < 40:
        continue
    best = None
    for c, m in GRID:
        tr = [r for r in sel if r["wi"] < lo_w]
        if len(tr) < 8:
            continue
        net, _ = port(tr, mk(m, c))
        if best is None or net > best[0]:
            best = (net, c, m)
    if best is None:
        continue
    _, c, m = best
    oof = [r for r in sel if lo_w <= r["wi"] < hi_w]
    fo, _ = port(oof, flat)
    so, _ = port(oof, mk(m, c))
    deltas.append(so - fo)
    nfold += 1
    print(f"   fold{k}: chose cut {c:.2f} mult {m:.1f} on train -> OOF sized ${so:+.2f} vs flat ${fo:+.2f} "
          f"(delta {so-fo:+.2f})")
print(f"   mean OOF delta {np.mean(deltas):+.2f}  positive folds {sum(1 for d in deltas if d>0)}/{nfold}")

print("\n" + "=" * 100)
print("3. IS THE CHEAP BUCKET'S 13/13 CARRIED BY ONE WINDOW?")
print("=" * 100)
cheap = [r for r in sel if r["ask"] < CUT_PRICE]
print(f"   cheap trades: {len(cheap)}, wins {sum(r['won'] for r in cheap)}")
print(f"   distinct windows: {len(set(r['wi'] for r in cheap))} "
      f"(one trade per window, so no single window can carry two)")
print(f"   asks: {sorted(round(r['ask'],3) for r in cheap)}")
print(f"   entry T: {sorted(r['T'] for r in cheap)}")

print("\n" + "=" * 100)
print("4. WIDE BAND vs SHIPPED under 1s latency (exact numbers for the report)")
print("=" * 100)
print(f"   {'rule':<30} {'base net':>10} {'base t':>7} {'base CI':>16} | {'1s net':>9} {'1s t':>6} {'1s CI':>16} {'survives':>9}")
for nm, tmin, lo, hi in (("SHIPPED T>=262 [0.80,0.96)", 262, 0.80, 0.96),
                         ("WIDE    T>=262 [0.70,0.96)", 262, 0.70, 0.96),
                         ("WIDE    T>=262 [0.55,0.96)", 262, 0.55, 0.96)):
    s0 = A.select(tmin, lo, hi, "fav")
    rr = B.build_rows(W, latency_s=1.0)
    c = [r for r in rr if r["T"] >= tmin and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
    s1 = B.first_per_window(c)
    a0 = np.array([r["ask"] for r in s0]); w0 = np.array([r["won"] for r in s0], float)
    p0_ = (TAKE / a0) * w0 - TAKE - (TAKE / a0) * FEE * a0 * (1 - a0)
    t0 = B.cluster_t([r["wi"] for r in s0], p0_)[1]; l0, h0 = B.boot_ci([r["wi"] for r in s0], p0_)
    a1 = np.array([r["ask"] for r in s1]); w1 = np.array([r["won"] for r in s1], float)
    p1_ = (TAKE / a1) * w1 - TAKE - (TAKE / a1) * FEE * a1 * (1 - a1)
    t1 = B.cluster_t([r["wi"] for r in s1], p1_)[1]; l1, h1 = B.boot_ci([r["wi"] for r in s1], p1_)
    print(f"   {nm:<30} {p0_.sum():>+10.2f} {t0:>+7.2f} [{l0:>+5.2f},{h0:>+5.2f}] | "
          f"{p1_.sum():>+9.2f} {t1:>+6.2f} [{l1:>+5.2f},{h1:>+5.2f}] {'YES' if l1 > 0 else 'NO':>9}")
print("\nDONE")
