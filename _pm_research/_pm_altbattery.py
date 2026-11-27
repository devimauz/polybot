"""Decisive test of the finalists that survived the first pass.

Finalists:
  1. dog T>=100 ask[0.20,0.40)      - the big-dollar longshot
  2. fav T>=262 ask[0.65,0.98)      - looser band, more trades
  3. fav T>=262 ask[0.50,0.98)      - looser band, more trades
  4. SHIPPED fav T>=262 ask[0.80,0.96)
  5. multi-trade (every qualifying quote)

Each gets the FULL battery:
  a. realised win rate vs the price implied  (the edge, cleanly)
  b. PnL concentration (top-1/3/5)
  c. three chronological thirds - all must be positive
  d. PLACEBO: reassign real outcomes across windows, RE-SELECT the rule
  e. a swap-split: fit on one half, test on the other
  f. 1s execution latency + 4x fee
  g. leave-one-out: can ONE trade carry the whole result?
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
rng = np.random.default_rng(31337)

FINALISTS = [
    ("SHIPPED fav T>=262 [0.80,0.96)", "fav", 262, 0.80, 0.96),
    ("dog T>=100 [0.20,0.40)", "dog", 100, 0.20, 0.40),
    ("fav T>=262 [0.65,0.98)", "fav", 262, 0.65, 0.98),
    ("fav T>=262 [0.50,0.98)", "fav", 262, 0.50, 0.98),
    ("fav T>=262 [0.65,0.94)", "fav", 262, 0.65, 0.94),
]


def sel_of(side, tmin, lo, hi):
    return A.select(tmin, lo, hi, side)


def pnl_of(sel, stake=TAKE, fee=FEE):
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    sh = stake / a
    return sh * w - stake - sh * fee * a * (1 - a)


def stat(sel, stake=TAKE, fee=FEE):
    p = pnl_of(sel, stake, fee)
    wi = [r["wi"] for r in sel]
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    c1 = float(p[o[-1:]].sum() / tot) if tot > 0 else float("nan")
    c3 = float(p[o[-3:]].sum() / tot) if tot > 0 else float("nan")
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    return dict(n=len(sel), net=float(p.sum()), t=float(t), lo=float(lo), hi=float(hi),
                roi=float(p.sum()) / (stake * len(sel)), c1=c1, c3=c3, c5=c5,
                worst=float(p.min()), p=p, wins=int(sum(r["won"] for r in sel)),
                meanask=float(np.mean([r["ask"] for r in sel])))


def thirds(sel, stake=TAKE, fee=FEE):
    out = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        out.append(pnl_of(s, stake, fee).sum() if s else 0.0)
    return out


def placebo(side, tmin, lo, hi, n_draw=1500):
    fav = [r for r in rows_all if (r["mkt"] >= 0.5) == (side == "fav")]
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
    return np.array(ts)


print("=" * 112)
print("THE FINALISTS")
print("=" * 112)
print(f"  {'candidate':<32} {'N':>4} {'wins':>5} {'ask':>6} {'realWR':>7} {'edge':>8} {'net$':>9} {'ROI':>7} "
      f"{'t':>6} {'CI95':>17} {'top1':>5} {'top3':>5} {'top5':>5}  {'thirds'}")
print("  " + "-" * 108)
for name, side, tmin, lo, hi in FINALISTS:
    sel = sel_of(side, tmin, lo, hi)
    if not sel:
        continue
    v = stat(sel)
    h = thirds(sel)
    realwr = v["wins"] / v["n"]
    print(f"  {name:<32} {v['n']:>4} {v['wins']:>5} {v['meanask']:>6.4f} {realwr:>7.4f} "
          f"{realwr - v['meanask']:>+8.4f} {v['net']:>+9.2f} {v['roi']*100:>+6.2f}% {v['t']:>+6.2f} "
          f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}] {v['c1']*100:>4.0f}% {v['c3']*100:>4.0f}% {v['c5']*100:>4.0f}%  "
          f"[{h[0]:>+8.1f},{h[1]:>+8.1f},{h[2]:>+8.1f}]")

print("\n" + "=" * 112)
print("PLACEBO: reassign real outcomes across windows, RE-SELECT the rule")
print("=" * 112)
for name, side, tmin, lo, hi in FINALISTS:
    sel = sel_of(side, tmin, lo, hi)
    if not sel:
        continue
    v = stat(sel)
    null = placebo(side, tmin, lo, hi)
    p = float((null >= v["t"]).mean())
    print(f"  {name:<32} obs t={v['t']:>+6.2f}  null mean {null.mean():>+6.2f} "
          f"95th {np.percentile(null,95):>+6.2f}  max {null.max():>+6.2f}  p={p:.4f}")

print("\n" + "=" * 112)
print("SWAP-SPLIT: the rule's own parameters are fixed, but fit the WIN-RATE on one half")
print("=" * 112)
print("  (for the dog rule: does the profitable price band replicate in both halves?)")
for name, side, tmin, lo, hi in FINALISTS[:3]:
    for split, lo_w, hi_w in (("TRAIN", 0, CUT), ("OOS", CUT, NW)):
        s = [r for r in sel_of(side, tmin, lo, hi) if lo_w <= r["wi"] < hi_w]
        if not s:
            continue
        v = stat(s)
        h3 = thirds(s)
        print(f"  {name:<32} {split:<5} n={v['n']:>3} wins={v['wins']:>3} net={v['net']:>+8.2f} "
              f"t={v['t']:>+5.2f} CI=[{v['lo']:>+5.2f},{v['hi']:>+5.2f}]  thirds[{h3[0]:>+7.1f},{h3[1]:>+7.1f},{h3[2]:>+7.1f}]")

print("\n" + "=" * 112)
print("STRESS: 1s execution latency and 4x fee")
print("=" * 112)
print(f"  {'candidate':<32} {'scenario':<12} {'N':>4} {'net$':>9} {'t':>6} {'CI95':>17}")
for name, side, tmin, lo, hi in FINALISTS:
    for scen, lat, fee in (("baseline", 0.0, FEE), ("1s latency", 1.0, FEE), ("4x fee", 0.0, FEE * 4)):
        rr = B.build_rows(W, latency_s=lat)
        if side == "fav":
            c = [r for r in rr if r["T"] >= tmin and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
        else:
            c = [r for r in rr if r["T"] >= tmin and r["mkt"] < 0.5 and lo <= r["ask"] < hi]
        s = B.first_per_window(c)
        if not s:
            continue
        v = stat(s, fee=fee)
        print(f"  {name:<32} {scen:<12} {v['n']:>4} {v['net']:>+9.2f} {v['t']:>+6.2f} "
              f"[{v['lo']:>+5.2f},{v['hi']:>+5.2f}]")

print("\n" + "=" * 112)
print("LEAVE-ONE-OUT: can ONE trade carry the whole result?")
print("=" * 112)
for name, side, tmin, lo, hi in FINALISTS:
    sel = sel_of(side, tmin, lo, hi)
    if not sel:
        continue
    ts = []
    for i in range(len(sel)):
        s = sel[:i] + sel[i + 1:]
        if len(s) < 5:
            continue
        ts.append(stat(s)["t"])
    ts = np.array(ts)
    print(f"  {name:<32} t={stat(sel)['t']:>+6.2f}  LOO range [{ts.min():>+6.2f},{ts.max():>+6.2f}]  "
          f"sign flips {int((ts<=0).sum())}/{len(ts)}")
print("\nDONE")
