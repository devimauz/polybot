"""Check what a $5 stake (and the $15 cheap-bucket stake) can actually fill.

Two things must hold at every entry:
  1. the 5-SHARE exchange minimum: stake / worst_price >= 5
  2. the DISPLAYED DEPTH at the ask must cover the stake, or the FOK
     order is rejected for insufficient size (not a share-count error).

The bot's depth filter (MIN_DEPTH_USD) is a property of the RULE (it decides
which trades qualify, and is part of the validated 38-trade trade set), so it
stays at $10.  But the SIZING must be coverable by the book at the entry.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import btc5m_endgame_backtest as B

W = B.load_all()
rows = B.build_rows(W)
sel = B.first_per_window([r for r in rows if r["T"] >= 262 and r["mkt"] >= 0.5
                          and 0.80 <= r["ask"] < 0.96])
sel.sort(key=lambda r: r["ws"])

print("=" * 108)
print("1. DISPLAYED DEPTH AT THE ENTRY vs THE STAKE")
print("=" * 108)
print(f"   {'#':>3} {'ask':>6} {'depth $ at ask':>14} {'$5 stake':>9} {'$15 stake':>10} {'$30 stake':>10}  {'ok@$15':>7}")
depths = []
for i, r in enumerate(sel, 1):
    a = r["ask"]
    d = r.get("asksz", 0) * a if r.get("asksz") is not None else float("nan")
    depths.append(d)
    print(f"   {i:>3} {a:>6.3f} {d:>14.2f} {5.0/a:>9.2f} {15.0/a:>10.2f} {30.0/a:>10.2f}  "
          f"{'YES' if d >= 15.0 else 'NO':>7}")
depths = np.array(depths, float)
print(f"\n   min depth ${np.nanmin(depths):.2f}   median ${np.nanmedian(depths):.2f}   "
      f"max ${np.nanmax(depths):.2f}")
for st in (5.0, 15.0, 30.0):
    print(f"   entries where depth < ${st:.0f}: {int((depths < st).sum())}/{len(depths)}")

print("\n" + "=" * 108)
print("2. THE 5-SHARE EXCHANGE MINIMUM at $5 and $15")
print("=" * 108)
worst = 0.96
print(f"   at the band cap (ask {worst}, +0.03 slippage cushion -> {worst+0.03:.2f}):")
for st in (5.0, 10.0, 15.0, 30.0):
    sh = st / (worst + 0.03)
    print(f"     ${st:>5.0f} buys {sh:>6.2f} shares  -> {'OK' if sh >= 5 else 'REJECTED'}")
print(f"   smallest safe stake at the cap: ${5*(worst+0.03):.2f}")
print(f"   at the DEAR end of the band the bot actually uses (ask 0.95): "
      f"$5 buys {5.0/0.98:.2f} shares -> OK")
print(f"   at the CHEAP end (ask 0.81): $5 buys {5.0/0.84:.2f} shares, "
      f"$15 buys {15.0/0.84:.2f} shares -> OK")

print("\n" + "=" * 108)
print("3. THE $5-SCALED STRATEGY (base $5, cheap bucket 3x = $15)")
print("=" * 108)
FEE = 0.07
for label, stake_fn in (("flat $5", lambda a: 5.0),
                        ("sized: $15 if ask<0.85 else $5",
                         lambda a: 15.0 if a < 0.85 else 5.0)):
    p = []
    staked = 0.0
    for r in sel:
        a = r["ask"]
        st = stake_fn(a)
        sh = st / a
        p.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
        staked += st
    p = np.array(p)
    m, t, _ = B.cluster_t([r["wi"] for r in sel], p)
    lo, hi = B.boot_ci([r["wi"] for r in sel], p)
    print(f"   {label:<32} staked ${staked:>6.0f}  net ${p.sum():>+7.2f}  "
          f"ROI {p.sum()/staked*100:>+6.2f}%  per trade ${p.mean():>+6.3f}")
    print(f"   {'':<32} cluster-t {t:>+5.2f}  CI95 [{lo:>+5.2f},{hi:>+5.2f}]  "
          f"worst ${p.min():>+6.2f}  best ${p.max():>+6.2f}")

print("\n" + "=" * 108)
print("4. SCALE INVARIANCE CHECK: is $5 exactly 1/2 of $10?")
print("=" * 108)
for label, stake_fn in (("flat", lambda a: 5.0), ("sized", lambda a: 15.0 if a < 0.85 else 5.0)):
    p5 = []
    for r in sel:
        a = r["ask"]
        st = stake_fn(a)
        sh = st / a
        p5.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
    p5 = np.array(p5)
    mult = 2.0 if label == "flat" else None
    if mult:
        print(f"   flat: $5 net ${p5.sum():+.2f}  x2 = ${p5.sum()*2:+.2f}  "
              f"(the $10 result was +$38.13)")
    else:
        p10 = []
        for r in sel:
            a = r["ask"]
            st = 30.0 if a < 0.85 else 10.0
            sh = st / a
            p10.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
        print(f"   sized: $5/$15 net ${p5.sum():+.2f}  x2 = ${p5.sum()*2:+.2f}  "
              f"(the $10/$30 result was +$89.45)")
print("   -> the fee is proportional, so P&L scales exactly with the stake:")
print("      ROI, t-statistic and win rate are IDENTICAL at every stake level.")
