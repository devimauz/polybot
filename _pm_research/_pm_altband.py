"""The decisive test: is a WIDER favourite band a better strategy than the
shipped [0.80,0.96) rule?

The wider bands found more raw dollars (+$59.90 vs +$38.13) but a lower
t-statistic.  Before believing that, apply the same protocol the shipped
rule had to pass:

  1. does the wider band beat the shipped rule on TRAIN alone?
  2. is the band a PLATEAU (many nearby bands work) or a knife edge?
  3. family-wise max-t over the whole band family, with the shipped rule
     INCLUDED as a candidate (so the family test covers the comparison)
  4. chronological thirds - all three must be positive
  5. latency + fee stress
  6. the comparison the user actually asked: which makes more PROFIT, and
     what does it cost in reliability?
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
rng = np.random.default_rng(2026)


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


# ---------------------------------------------------------------- 1 & 2
print("=" * 112)
print("1+2. THE BAND PLATEAU: every favourite band at T>=262, train and OOS")
print("=" * 112)
print(f"  {'band':<16} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'net$':>9} {'ROI':>7} {'t':>6} "
      f"{'CI95':>17} {'trNet':>8} {'trT':>6} {'oosNet':>8} {'oosT':>6}  {'thirds'}")
bands = []
for lo in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    for hi in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.01):
        if hi - lo < 0.05:
            continue
        bands.append((lo, hi))
for lo, hi in bands:
    sel = A.select(262, lo, hi, "fav")
    if len(sel) < 12:
        continue
    v = stat(sel)
    tr = stat([r for r in sel if r["wi"] < CUT])
    te = stat([r for r in sel if r["wi"] >= CUT])
    if not (v and tr and te):
        continue
    h = thirds(sel)
    flag = "  <== SHIPPED" if (lo, hi) == (0.80, 0.96) else ""
    print(f"  [{lo:.2f},{hi:.2f}){'':<4} {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} "
          f"{v['wins']/v['n']:>7.4f} {v['net']:>+9.2f} {v['roi']*100:>+6.2f}% {v['t']:>+6.2f} "
          f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}] {tr['net']:>+8.2f} {tr['t']:>+6.2f} "
          f"{te['net']:>+8.2f} {te['t']:>+6.2f}  [{h[0]:>+7.1f},{h[1]:>+7.1f},{h[2]:>+7.1f}]{flag}")

# ---------------------------------------------------------------- 3
print("\n" + "=" * 112)
print("3. FAMILY-WISE max-t over the band family (the shipped rule INCLUDED)")
print("=" * 112)
rows_all = B.build_rows(W)
fav = [r for r in rows_all if r["mkt"] >= 0.5]
by_wi = {}
for r in fav:
    by_wi.setdefault(r["wi"], []).append(r)
wi_list = sorted(by_wi)

obs = {}
for lo, hi in bands:
    s = A.select(262, lo, hi, "fav")
    if len(s) >= 12:
        obs[(lo, hi)] = stat(s)["t"]
obs_max = max(obs.values())
obs_rule = max(obs, key=obs.get)
print(f"  observed max t over {len(obs)} bands: {obs_max:+.2f}  (rule {obs_rule})")

null_max = []
for _ in range(600):
    perm = list(wi_list)
    rng.shuffle(perm)
    wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
    ts = []
    for lo, hi in bands:
        c = []
        for src_wi in wi_list:
            for r in by_wi[src_wi]:
                if r["T"] < 262 or not (lo <= r["ask"] < hi):
                    continue
                r2 = dict(r)
                r2["won"] = 1 if W[wperm[src_wi]]["_winner"] == r["side"] else 0
                c.append(r2)
        c = B.first_per_window(c)
        if len(c) < 12:
            continue
        p = B.pnl_of([r["ask"] for r in c], [r["won"] for r in c])
        ts.append(B.cluster_t([r["wi"] for r in c], p)[1])
    if ts:
        null_max.append(max(ts))
null_max = np.array(null_max)
p_fw = float((null_max >= obs_max).mean())
print(f"  permutation null (600 outcome reassignments): mean {null_max.mean():+.2f}, "
      f"95th pct {np.percentile(null_max,95):+.2f}")
print(f"  FAMILY-WISE p-value: {p_fw:.4f}")

# how many bands are positive OOS?  (the family sign test)
pos_oos = 0
tot_oos = 0
for lo, hi in bands:
    s = A.select(262, lo, hi, "fav")
    if len(s) < 12:
        continue
    te = stat([r for r in s if r["wi"] >= CUT])
    if not te or te["n"] < 5:
        continue
    tot_oos += 1
    if te["net"] > 0:
        pos_oos += 1
print(f"  bands positive OOS: {pos_oos} of {tot_oos}")

# ---------------------------------------------------------------- 5
print("\n" + "=" * 112)
print("5. STRESS on the two finalists")
print("=" * 112)
print(f"  {'band':<16} {'scenario':<12} {'N':>4} {'net$':>9} {'t':>6} {'CI95':>17}")
for lo, hi in ((0.80, 0.96), (0.50, 0.98), (0.50, 0.96)):
    for scen, lat, fee in (("baseline", 0.0, FEE), ("1s latency", 1.0, FEE), ("4x fee", 0.0, FEE * 4),
                           ("8x fee", 0.0, FEE * 8)):
        rr = B.build_rows(W, latency_s=lat)
        c = [r for r in rr if r["T"] >= 262 and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
        s = B.first_per_window(c)
        if not s:
            continue
        v = stat(s, fee=fee)
        print(f"  [{lo:.2f},{hi:.2f}){'':<2} {scen:<12} {v['n']:>4} {v['net']:>+9.2f} {v['t']:>+6.2f} "
              f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}]")

# ---------------------------------------------------------------- 6
print("\n" + "=" * 112)
print("6. THE HEAD-TO-HEAD: more profit vs more reliable")
print("=" * 112)
print(f"  {'':<20} {'SHIPPED [0.80,0.96)':>20} {'WIDER [0.50,0.98)':>18}")
for label, fn in (("trades", lambda s: s["n"]), ("wins", lambda s: s["wins"]),
                   ("mean ask", lambda s: f"{s['meanask']:.4f}"),
                   ("realised win rate", lambda s: f"{s['wins']/s['n']:.4f}"),
                   ("net $ (at $10)", lambda s: f"{s['net']:+.2f}"),
                   ("ROI", lambda s: f"{s['roi']*100:+.2f}%"),
                   ("cluster-t", lambda s: f"{s['t']:+.2f}"),
                   ("CI95", lambda s: f"[{s['lo']:+.2f},{s['hi']:+.2f}]"),
                   ("top-5 P&L share", lambda s: f"{s['c5']*100:.0f}%"),
                   ("worst trade", lambda s: f"{s['worst']:+.2f}")):
    a = stat(A.select(262, 0.80, 0.96, "fav"))
    b = stat(A.select(262, 0.50, 0.98, "fav"))
    print(f"  {label:<20} {fn(a):>20} {fn(b):>18}")
print("\nDONE")
