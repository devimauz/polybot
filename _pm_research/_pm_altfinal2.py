"""FINAL: is [0.70,0.96) really better than the shipped [0.80,0.96)?

The band plateau showed a peak at lower bound 0.70 that beats the shipped
rule on net$, t, and CI.  But I FOUND that peak by looking at 85 bands, so
the selection must be redone honestly:

  STEP 1  Define the candidate grid FIRST (t_min x lower x upper bound).
  STEP 2  Select on TRAIN ONLY: max train net, subject to a minimum train
          sample and a positive train t.  No peeking at OOS.
  STEP 3  Report the selected rule's OOS result.  If the train-selected rule
          beats the shipped rule OOS, the wider band is real.
  STEP 4  Full battery on the winner: thirds, concentration, leave-one-out,
          latency, fee, minimum-order-size feasibility, capital.
  STEP 5  Family-wise max-t over the WHOLE grid, so the comparison itself is
          inside the multiple-testing correction.
  STEP 6  The trade lists, side by side, to see WHY it differs.
"""
import sys, numpy as np
from datetime import datetime, timezone
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B
import _pm_altsearch as A

W = A.W
NW = A.NW
CUT = A.CUT
TAKE = 10.0
FEE = 0.07
rng = np.random.default_rng(88)

# ---- STEP 1: the grid, defined before any selection -------------------------
GRID = []
for tmin in (258, 260, 262, 264, 266, 268):
    for lo in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        for hi in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01):
            if hi - lo < 0.05:
                continue
            GRID.append((tmin, lo, hi))
print(f"STEP 1  candidate grid: {len(GRID)} rules "
      f"(t_min x band lower x band upper, favourite side, T>=258)\n")


def stat(sel, fee=FEE, stake=TAKE):
    if not sel:
        return None
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    wi = [r["wi"] for r in sel]
    sh = stake / a
    p = sh * w - stake - sh * fee * a * (1 - a)
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    return dict(n=len(sel), wins=int(w.sum()), net=float(p.sum()), t=float(t),
                lo=float(lo), hi=float(hi), roi=float(p.sum()) / (stake * len(sel)),
                c1=float(p[o[-1:]].sum() / tot) if tot > 0 else float("nan"),
                c3=float(p[o[-3:]].sum() / tot) if tot > 0 else float("nan"),
                c5=float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan"),
                worst=float(p.min()), meanask=float(a.mean()))


def thirds(sel, **kw):
    o = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        o.append(stat(s, **kw)["net"] if s else 0.0)
    return o


# ---- STEP 2: select on TRAIN ONLY ------------------------------------------
cands = []
for tmin, lo, hi in GRID:
    sel = A.select(tmin, lo, hi, "fav")
    if len(sel) < 20:
        continue
    tr = [r for r in sel if r["wi"] < CUT]
    vtr = stat(tr)
    if not vtr or vtr["n"] < 12 or vtr["t"] <= 0:
        continue
    cands.append((vtr["net"], vtr["t"], tmin, lo, hi, sel))
cands.sort(key=lambda x: (-x[0], -x[1]))
print("STEP 2  selection on TRAIN ONLY (max train net$, require train t>0, train n>=12)")
print(f"  {'rule':<26} {'trN':>4} {'trNet':>8} {'trT':>6} {'trCI':>16} | {'oosN':>5} {'oosNet':>8} {'oosT':>6} {'oosCI':>16}")
best = None
for net, t, tmin, lo, hi, sel in cands[:10]:
    te = stat([r for r in sel if r["wi"] >= CUT])
    vtr = stat([r for r in sel if r["wi"] < CUT])
    print(f"  T>={tmin} fav[{lo:.2f},{hi:.2f}){'':<2} {vtr['n']:>4} {vtr['net']:>+8.2f} {vtr['t']:>+6.2f} "
          f"[{vtr['lo']:>+5.2f},{vtr['hi']:>+5.2f}] | {te['n']:>5} {te['net']:>+8.2f} {te['t']:>+6.2f} "
          f"[{te['lo']:>+5.2f},{te['hi']:>+5.2f}]")
    if best is None:
        best = (tmin, lo, hi, sel)

SEL_T, SEL_LO, SEL_HI, SEL = best
ship = A.select(262, 0.80, 0.96, "fav")
print(f"\n  >>> TRAIN-SELECTED: T>={SEL_T} fav[{SEL_LO:.2f},{SEL_HI:.2f})")
v_s = stat(SEL); v_h = stat(ship)
print(f"  >>> vs SHIPPED      : T>=262 fav[0.80,0.96)")
print(f"      OOS  selected {stat([r for r in SEL if r['wi']>=CUT])['net']:+.2f} "
      f"vs shipped {stat([r for r in ship if r['wi']>=CUT])['net']:+.2f}")
print(f"      FULL selected net ${v_s['net']:+.2f} t={v_s['t']:+.2f} CI[{v_s['lo']:+.2f},{v_s['hi']:+.2f}] "
      f"| shipped net ${v_h['net']:+.2f} t={v_h['t']:+.2f} CI[{v_h['lo']:+.2f},{v_h['hi']:+.2f}]")

# ---- STEP 4: full battery ---------------------------------------------------
print("\n" + "=" * 108)
print("STEP 4  FULL BATTERY on the train-selected rule vs the shipped rule")
print("=" * 108)
print(f"  {'':<26} {'SELECTED':>16} {'SHIPPED':>16}")
for label, fn in (("trades", lambda s: s["n"]), ("wins", lambda s: s["wins"]),
                   ("mean ask", lambda s: f"{s['meanask']:.4f}"),
                   ("realised win rate", lambda s: f"{s['wins']/s['n']:.4f}"),
                   ("net $", lambda s: f"{s['net']:+.2f}"),
                   ("ROI", lambda s: f"{s['roi']*100:+.2f}%"),
                   ("cluster-t", lambda s: f"{s['t']:+.2f}"),
                   ("CI95", lambda s: f"[{s['lo']:+.2f},{s['hi']:+.2f}]"),
                   ("top-1 P&L share", lambda s: f"{s['c1']*100:.0f}%"),
                   ("top-3 P&L share", lambda s: f"{s['c3']*100:.0f}%"),
                   ("top-5 P&L share", lambda s: f"{s['c5']*100:.0f}%"),
                   ("worst trade", lambda s: f"{s['worst']:+.2f}")):
    print(f"  {label:<26} {fn(v_s):>16} {fn(v_h):>16}")
print(f"  {'thirds':<26} {str([f'{x:+.1f}' for x in thirds(SEL)]):>16} "
      f"{str([f'{x:+.1f}' for x in thirds(ship)]):>16}")

print("\n  leave-one-out:")
for name, sel in (("SELECTED", SEL), ("SHIPPED", ship)):
    ts = np.array([stat(sel[:i] + sel[i+1:])["t"] for i in range(len(sel)) if len(sel) > 1])
    print(f"    {name:<10} t={stat(sel)['t']:+.2f}  LOO [{ts.min():+.2f},{ts.max():+.2f}]  "
          f"sign flips {int((ts<=0).sum())}/{len(ts)}")

print("\n  stress:")
for scen, lat, fee in (("baseline", 0.0, FEE), ("1s latency", 1.0, FEE), ("4x fee", 0.0, FEE * 4),
                       ("8x fee", 0.0, FEE * 8)):
    out = []
    for tmin, lo, hi, nm in ((SEL_T, SEL_LO, SEL_HI, "SEL"), (262, 0.80, 0.96, "SHIP")):
        rr = B.build_rows(W, latency_s=lat)
        c = [r for r in rr if r["T"] >= tmin and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
        s = B.first_per_window(c)
        v = stat(s, fee=fee)
        out.append(f"{nm} n={v['n']:>2} net={v['net']:>+7.2f} t={v['t']:>+5.2f} "
                   f"CI[{v['lo']:>+5.2f},{v['hi']:>+5.2f}]")
    print(f"    {scen:<12} {'   |   '.join(out)}")

# minimum order size feasibility
print("\n  Polymarket 5-share minimum feasibility (stake must buy 5 shares at ask+0.03):")
for nm, sel, lo_b, hi_b in (("SELECTED", SEL, SEL_LO, SEL_HI), ("SHIPPED", ship, 0.80, 0.96)):
    need = 5 * (hi_b + 0.03)
    for stake in (5.0, 10.0):
        ok = stake >= need
        print(f"    {nm:<9} stake ${stake:.0f} -> needs ${need:.2f}  {'OK' if ok else 'REJECTED'}")

# ---- STEP 5: family-wise ----------------------------------------------------
print("\n" + "=" * 108)
print("STEP 5  FAMILY-WISE max-t over the whole grid (the comparison is inside the correction)")
print("=" * 108)
rows_all = B.build_rows(W)
fav = [r for r in rows_all if r["mkt"] >= 0.5]
by_wi = {}
for r in fav:
    by_wi.setdefault(r["wi"], []).append(r)
wi_list = sorted(by_wi)
obs_t = {}
for tmin, lo, hi in GRID:
    s = A.select(tmin, lo, hi, "fav")
    if len(s) >= 20:
        obs_t[(tmin, lo, hi)] = stat(s)["t"]
omax = max(obs_t.values())
orule = max(obs_t, key=obs_t.get)
print(f"  observed max t over {len(obs_t)} rules: {omax:+.2f}  at {orule}")
null_max = []
for _ in range(400):
    perm = list(wi_list)
    rng.shuffle(perm)
    wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
    ts = []
    for tmin, lo, hi in GRID:
        c = []
        for src_wi in wi_list:
            for r in by_wi[src_wi]:
                if r["T"] < tmin or not (lo <= r["ask"] < hi):
                    continue
                r2 = dict(r)
                r2["won"] = 1 if W[wperm[src_wi]]["_winner"] == r["side"] else 0
                c.append(r2)
        c = B.first_per_window(c)
        if len(c) < 20:
            continue
        p = B.pnl_of([r["ask"] for r in c], [r["won"] for r in c])
        ts.append(B.cluster_t([r["wi"] for r in c], p)[1])
    if ts:
        null_max.append(max(ts))
null_max = np.array(null_max)
print(f"  permutation null (400 outcome reassignments): mean {null_max.mean():+.2f}, "
      f"95th pct {np.percentile(null_max,95):+.2f}")
print(f"  FAMILY-WISE p-value: {float((null_max>=omax).mean()):.4f}")

# ---- STEP 6: the trade lists ------------------------------------------------
print("\n" + "=" * 108)
print("STEP 6  THE TRADE LISTS SIDE BY SIDE - why do they differ?")
print("=" * 108)
def dump(sel, label):
    print(f"\n  {label}  (N={len(sel)})")
    print(f"    {'#':>3} {'entry UTC':<16} {'T':>4} {'side':>5} {'ask':>6} {'shares':>7} {'result':>7} {'P&L':>7}")
    tot = 0.0
    for i, r in enumerate(sel):
        d = W[r["wi"]]
        et = datetime.fromtimestamp(d["ws"] + r["T"], tz=timezone.utc)
        sh = TAKE / r["ask"]
        fee = sh * FEE * r["ask"] * (1 - r["ask"])
        pnl = sh * r["won"] - TAKE - fee
        tot += pnl
        print(f"    {i+1:>3} {et.strftime('%m-%d %H:%M:%S'):<16} {r['T']:>4.0f} {r['side']:>5} "
              f"{r['ask']:>6.3f} {sh:>7.2f} {'WON' if r['won'] else 'LOST':>7} {pnl:>+7.2f}")
    print(f"    {'':>3} {'':16} {'':>4} {'':>5} {'':>6} {'':>7} {'':>7} {tot:>+7.2f}")
dump(SEL, f"SELECTED  T>={SEL_T} fav[{SEL_LO:.2f},{SEL_HI:.2f})")
dump(ship, "SHIPPED   T>=262 fav[0.80,0.96)")
print("\nDONE")
