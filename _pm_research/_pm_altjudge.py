"""Adjudicate the alternatives the grid surfaced: are any of them REAL?

Tests applied to every candidate:
  1. implied-vs-realised win rate (does the realised win rate beat what the
     price implies? that IS the edge, in the cleanest possible form)
  2. PnL concentration (top-1/3/5 share) - lottery wins must not pass as edge
  3. chronological halves (all three must be positive)
  4. a CORRECT placebo: reassign REAL outcomes across windows using the
     backtest's own permutation machinery, then re-select the rule from the
     permuted data (so the placebo also re-optimises the rule)
  5. family-wise max-t over the whole 1023-candidate grid
"""
import sys, numpy as np
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B
import _pm_altsearch as A

W = A.W
NW = A.NW
CUT = A.CUT
rng = np.random.default_rng(20261021)


def realized_vs_implied(sel):
    """The edge in its cleanest form: realised win rate vs the price paid."""
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    n = len(sel)
    implied = a.mean()                       # fair win prob implied by the price
    realized = w.mean()
    # binomial SE on the realised rate
    se = np.sqrt(realized * (1 - realized) / n) if 0 < realized < 1 else 0.0
    z = (realized - implied) / se if se > 0 else 0.0
    return implied, realized, z


def placebo_rule(tmin, lo, hi, side, n_draw=3000):
    """Reassign real outcomes across windows, RE-SELECT the rule, return the
    null distribution of its t.  Mirrors the backtest's own method."""
    rows = B.build_rows(W)
    fav = [r for r in rows if (r["mkt"] >= 0.5) == (side == "fav")]
    by_wi = {}
    for r in fav:
        by_wi.setdefault(r["wi"], []).append(r)
    wi_list = sorted(by_wi)
    if not wi_list:
        return np.array([]), 0.0
    ts = []
    for _ in range(n_draw):
        perm = list(wi_list)
        rng.shuffle(perm)
        wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
        c = []
        for src_wi in wi_list:
            for r in by_wi[src_wi]:
                if r["T"] < tmin or not (lo <= r["ask"] < hi):
                    continue
                r2 = dict(r)
                r2["won"] = 1 if W[wperm[src_wi]]["_winner"] == r["side"] else 0
                c.append(r2)
        c = B.first_per_window(c)
        if len(c) < 5:
            continue
        p = B.pnl_of([r["ask"] for r in c], [r["won"] for r in c])
        ts.append(B.cluster_t([r["wi"] for r in c], p)[1])
    return np.array(ts), 0.0


CANDS = [
    ("SHIPPED T>=262 fav[0.80,0.96)", "fav", 262, 0.80, 0.96),
    ("GRID#1 dog T>=100 ask[0.02,0.40)", "dog", 100, 0.02, 0.40),
    ("dog T>=100 ask[0.02,0.30)", "dog", 100, 0.02, 0.30),
    ("dog T>=100 ask[0.02,0.60)", "dog", 100, 0.02, 0.60),
    ("dog T>=250 ask[0.05,0.50)", "dog", 250, 0.05, 0.50),
    ("LOOSE fav T>=262 ask[0.60,0.99)", "fav", 262, 0.60, 0.99),
    ("WIDE-T fav T>=200 ask[0.80,0.96)", "fav", 200, 0.80, 0.96),
    ("fav T>=200 ask[0.60,0.98)", "fav", 200, 0.60, 0.98),
]

print(f"{'candidate':<34} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'z':>6} "
      f"{'net$':>9} {'t':>6} {'top1':>5} {'top3':>5} {'top5':>5}  {'halves'}")
print("-" * 132)
for name, side, tmin, lo, hi in CANDS:
    sel = A.select(tmin, lo, hi, side)
    if not sel:
        print(f"{name:<34}  no trades")
        continue
    st = A.stats(sel)
    imp, real, z = realized_vs_implied(sel)
    h = A.halves(sel)
    print(f"{name:<34} {st['n']:>4} {st['wins']:>5} {imp:>6.4f} {real:>7.4f} {z:>+6.2f} "
          f"{st['net']:>+9.2f} {st['t']:>+6.2f} {st['c1']*100:>4.0f}% {st['c3']*100:>4.0f}% "
          f"{st['c5']*100:>4.0f}%  [{h[0]:>+8.2f},{h[1]:>+8.2f},{h[2]:>+8.2f}]")

# ---------------------------------------------------------------- placebo
print("\n" + "=" * 100)
print("PLACEBO (correct version): reassign real outcomes across windows, RE-SELECT the rule")
print("=" * 100)
for name, side, tmin, lo, hi in CANDS[:5]:
    sel = A.select(tmin, lo, hi, side)
    if not sel:
        continue
    st = A.stats(sel)
    null, _ = placebo_rule(tmin, lo, hi, side, n_draw=1500)
    if len(null) == 0:
        print(f"  {name:<34} placebo too few draws")
        continue
    p = float((null >= st["t"]).mean())
    print(f"  {name:<34} obs t={st['t']:>+6.2f}  null mean {null.mean():>+6.2f} "
          f"95th {np.percentile(null,95):>+6.2f}  max {null.max():>+6.2f}  p={p:.4f}")
