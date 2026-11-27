"""_pm_maker_design.py - crossing-fill MAKER edge surface, design slice only.

Rule family: at frame i (T on a 2s grid), rest a BUY limit for floor($5/B) shares
(min 5) at the CURRENT BEST BID B of one side. Hold to settlement. The order is
provably filled iff the displayed best bid later drops STRICTLY below B (the whole
queue at B, including us, was swept). No fill = no position, no P&L.

Output: edge table by T-band x bid-band on the EARLY chronological slice (design).
Per-cell stats: N, mean paid, realised win-rate on FILLED events, gross edge c/sh,
net edge after taker-formula fee (conservative), halves check (skill #10).
Also: complete-set maker diagnostic (bid BOTH legs at one frame).

Design slice = first 450 resolved windows. The last 369 are UNTOUCHED.
"""
import os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

CAP = 5.0
FEE = 0.07

def fee_of(p, sh):
    return sh * FEE * p * (1.0 - p)

W = B.load_all()
print("resolved windows:", len(W), flush=True)

# ---- settlement sanity: chainlink TWAP end-vs-open vs recorded winner
agree = tot = 0
for d in W:
    cl = d.get("cl_px")
    if cl is None:
        continue
    f = np.isfinite(cl)
    if f.sum() > 10:
        win = "Up" if cl[f][-1] > cl[f][0] else "Down"
        tot += 1
        agree += (win == d["_winner"])
print(f"chainlink end-vs-open == winner: {agree}/{tot} windows with usable TWAP series", flush=True)

N_DESIGN = 450
design = W[:N_DESIGN]

TBANDS = [(60, 120), (120, 180), (180, 240), (240, 270), (270, 300)]
BBANDS = [(0.55, 0.70), (0.70, 0.80), (0.80, 0.88), (0.88, 0.94), (0.94, 0.99)]

rows = []          # maker events
pairdiag = []      # complete-set diagnostic, earliest attempt per window
t0 = 0.0
for wi, d in enumerate(design):
    t = d["t"]; n = len(t)
    win = d["_winner"]
    grid = np.arange(60, 300, 2.0)
    idx = np.searchsorted(t, grid)          # first frame at or after each grid T
    side_cross = {}
    for side in ("Up", "Down"):
        bid = d[side + "_bid"]; bsz = d[side + "_bidsz"]; ask = d[side + "_ask"]
        won = 1.0 if win == side else 0.0
        alive = np.isfinite(bid) & np.isfinite(ask) & (bid > 0.01) & (bid < 0.99) & (ask >= bid)
        side_cross[side] = (bid, alive, won)
        seen_band = set()
        for k in range(len(grid)):
            i = idx[k]
            if i >= n - 3 or not alive[i]:
                continue
            Tg = float(t[i])
            Bd = float(bid[i])
            sh = int(np.floor(CAP / Bd))
            if sh < 5:
                continue
            # first strict crossing below Bd after i
            later = bid[i + 1:]
            cross = later < Bd
            j = int(np.argmax(cross)) if cross.any() else -1
            if j >= 0:
                fillT = float(t[i + 1 + j])
                # gross edge per share = won - Bd ; fee charged at maker fill conservatively
                cost = sh * Bd
                g_net = sh * won - cost - fee_of(Bd, sh)
                rows.append(dict(wi=wi + 1, slug=d["_slug"], side=side, T=Tg, B=Bd, sh=sh,
                                 won=won, fillT=fillT, hold=fillT - Tg,
                                 net=g_net, gross=sh * won - cost))
            # cell bookkeeping: keep EARLIEST placement per (side, band-cell) per window
            tb = next((a for a in TBANDS if a[0] <= Tg < a[1]), None)
            bb = next((a for a in BBANDS if a[0] <= Bd < a[1]), None)
            if tb and bb and (tb, bb) not in seen_band:
                seen_band.add((tb, bb))
    # ---- complete-set diagnostic at the first frame >=120 with both legs sane
    for k in range(len(grid)):
        i = idx[k]
        if i >= n - 3:
            continue
        bu, au, wu = side_cross["Up"]; bd, ad, wd = side_cross["Down"]
        if not (au[i] and ad[i]):
            continue
        su = int(np.floor(CAP / float(bu[i]))); sd = int(np.floor(CAP / float(bd[i])))
        if su < 5 or sd < 5:
            continue
        cu = (bu[i + 1:] < bu[i]); cd = (bd[i + 1:] < bd[i])
        fu, fd = cu.any(), cd.any()
        if fu or fd:
            pairdiag.append(dict(wi=wi + 1, both=bool(fu and fd), only_up=bool(fu and not fd),
                                 only_down=bool(fd and not fu),
                                 cost=float(bu[i]) + float(bd[i]) if (fu and fd) else np.nan,
                                 net=(1.0 - (su * float(bu[i]) + sd * float(bd[i])) - fee_of(float(bu[i]), su) - fee_of(float(bd[i]), sd)) if (fu and fd) else np.nan))
            break

os.makedirs("_pm_out", exist_ok=True)
with open("_pm_out/maker_design_events.csv", "w", newline="", encoding="utf-8") as fh:
    wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wcsv.writeheader(); wcsv.writerows(rows)

R = {k: np.array([r[k] for r in rows], float) for k in ("wi", "T", "B", "sh", "won", "hold", "net", "gross")}
R["side"] = np.array([r["side"] for r in rows])
half = R["wi"] <= (N_DESIGN // 2)
print(f"\nFILLED maker events on design: {len(rows)} over {len(set(R['wi']))} windows",
      f"(fill rate differs by band - see table)", flush=True)

def cell_str(sub, subh1, subh2):
    if len(sub) == 0:
        return "  n=0"
    paid = np.average(sub["B"], weights=sub["sh"])
    wr = sub["won"].mean()
    edg = wr - paid
    net_ps = sub["net"].sum() / sub["sh"].sum()
    s1 = sub["net"][subh1].sum() if subh1.any() else np.nan
    s2 = sub["net"][subh2].sum() if subh2.any() else np.nan
    return (f"n={len(sub):4d} w={wr:.3f} paid={paid:.4f} edge={edg*100:+5.2f}c "
            f"net/sh={net_ps*100:+5.2f}c $={sub['net'].sum():+7.2f} "
            f"halves=${s1:+6.2f}/${s2:+6.2f}")

print("\n=== MAKER crossing-fill edge surface, DESIGN (first 450 windows) ===")
for (tb, tlo, thi) in [(a, a[0], a[1]) for a in TBANDS]:
    for (bb, blo, bhi) in [(a, a[0], a[1]) for a in BBANDS]:
        m = (R["T"] >= tlo) & (R["T"] < thi) & (R["B"] >= blo) & (R["B"] < bhi)
        if not m.any():
            continue
        sub = {k: R[k][m] for k in ("B", "sh", "won", "net")}
        h1 = half[m]; h2 = ~half[m]
        print(f"T[{tlo:3d},{thi:3d}) B[{blo:.2f},{bhi:.2f}) pooled | " + cell_str(sub, h1, h2))
        for sd in ("Up", "Down"):
            ms = m & (R["side"] == sd)
            if ms.any():
                subs = {k: R[k][ms] for k in ("B", "sh", "won", "net")}
                print(f"                  {sd:4s}    | " + cell_str(subs, half[ms], (~half)[ms]))
    print()

pd = pairdiag
both = [p for p in pd if p["both"]]; ou = [p for p in pd if p["only_up"]]; od = [p for p in pd if p["only_down"]]
print(f"=== complete-set maker (bid BOTH legs at first sane frame >=120s), design: "
      f"{len(pd)} windows attempted: both-filled {len(both)}, up-only {len(ou)}, down-only {len(od)}")
if both:
    nets = np.array([p["net"] for p in both]); costs = np.array([p["cost"] for p in both])
    print(f"both-fill: mean bid-sum={costs.mean():.4f} net mean=${nets.mean():+.3f} total=${nets.sum():+.2f} (fee-modelled)")
print("elapsed ok - design slice only; holdout untouched", flush=True)
