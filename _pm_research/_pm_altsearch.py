"""Search for a MORE PROFITABLE strategy than the shipped endgame favourite.

Built directly on the backtest's primitives so it CANNOT drift from it:
  * B.load_all()  - the one source-aligned loader (t_src, never the logger stamp)
  * B.build_rows() - the one row builder (spread, depth, alive-opposite-leg)
  * B.first_per_window() - one trade per window, earliest qualifying frame
Selection is TRAIN-only (first 60% of windows); the untouched last 40% is
the out-of-sample test.

The question: is there a strategy family that makes MORE profit than the
shipped rule (T>=262, favourite ask [0.80,0.96), $10 stake)?
"""
import sys, numpy as np
sys.path.insert(0, '.')
import btc5m_endgame_backtest as B

W = B.load_all()
NW = len(W)
CUT = int(NW * 0.60)
TAKE = 10.0


def select(tmin, lo, hi, side="fav"):
    """side: 'fav' = the side the market prices higher (mkt>=0.5),
             'dog' = the other side (mkt<0.5)."""
    rows = B.build_rows(W)
    if side == "fav":
        c = [r for r in rows if r["T"] >= tmin and r["mkt"] >= 0.5
             and lo <= r["ask"] < hi]
    else:
        c = [r for r in rows if r["T"] >= tmin and r["mkt"] < 0.5
             and lo <= r["ask"] < hi]
    return B.first_per_window(c)


def stats(sel, stake=TAKE):
    if not sel:
        return None
    a = np.array([r["ask"] for r in sel])
    w = np.array([r["won"] for r in sel], dtype=float)
    wi = [r["wi"] for r in sel]
    p = B.pnl_of(a, w, stake)
    t = B.cluster_t(wi, p)[1]
    lo, hi = B.boot_ci(wi, p)
    o = np.argsort(p)
    tot = p.sum()
    c1 = float(p[o[-1:]].sum() / tot) if tot > 0 else float("nan")
    c3 = float(p[o[-3:]].sum() / tot) if tot > 0 else float("nan")
    c5 = float(p[o[-5:]].sum() / tot) if tot > 0 else float("nan")
    return dict(n=len(sel), wins=int(w.sum()), net=float(p.sum()),
                roi=float(p.sum()) / (stake * len(sel)), t=float(t),
                lo=float(lo), hi=float(hi), avg=float(p.mean()),
                worst=float(p.min()), ask=float(a.mean()),
                c1=c1, c3=c3, c5=c5)


def split(sel):
    return ([r for r in sel if r["wi"] < CUT],
            [r for r in sel if r["wi"] >= CUT])


def halves(sel):
    out = []
    for lo, hi in ((0, CUT // 2), (CUT // 2, CUT), (CUT, NW)):
        s = [r for r in sel if lo <= r["wi"] < hi]
        out.append(B.pnl_of(np.array([r["ask"] for r in s]),
                             np.array([r["won"] for r in s], dtype=float)).sum()
                   if s else 0.0)
    return out


if __name__ == "__main__":
    print(f"windows={NW}  train=0..{CUT-1}  OOS={CUT}..{NW-1}  stake=${TAKE:.0f}\n")
    # sanity: the shipped rule must reproduce the backtest exactly
    ship = select(B.SHIP_T_MIN, B.SHIP_BAND[0], B.SHIP_BAND[1], "fav")
    st = stats(ship)
    print(f"SANITY shipped rule: N={st['n']} (backtest says 38) wins={st['wins']} "
          f"net=${st['net']:+.2f} t={st['t']:+.2f} CI=[{st['lo']:+.2f},{st['hi']:+.2f}]\n")

    grid = []
    for tmin in (180, 200, 220, 240, 250, 255, 258, 260, 262, 264, 266, 268, 270, 275, 280, 285):
        for lo in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            for hi in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
                if hi - lo < 0.05:
                    continue
                grid.append(("fav", tmin, lo, hi))
    for tmin in (100, 150, 200, 250, 262):
        for lo in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
            for hi in (0.25, 0.30, 0.40, 0.45, 0.50, 0.60):
                if hi - lo < 0.05:
                    continue
                grid.append(("dog", tmin, lo, hi))

    recs = []
    for side, tmin, lo, hi in grid:
        sel = select(tmin, lo, hi, side)
        if not sel:
            continue
        tr, te = split(sel)
        s_tr, s_te = stats(tr), stats(te)
        if s_tr is None or s_te is None or s_tr["n"] < 8 or s_te["n"] < 5:
            continue
        recs.append(dict(side=side, tmin=tmin, lo=lo, hi=hi,
                         name=f"{side} T>={tmin} ask[{lo:.2f},{hi:.2f})",
                         tr=s_tr, te=s_te, full=stats(sel)))

    print(f"grid: {len(grid)} candidates, {len(recs)} with enough train AND oos trades")
    recs.sort(key=lambda r: (-r["tr"]["net"], -r["tr"]["t"]))
    print(f"\n--- top 15 by TRAIN net (selection step) ---")
    print(f"  {'rule':<32} {'trN':>4} {'trNet':>8} {'trT':>6} | {'oosN':>5} {'oosNet':>8} {'oosT':>6} {'oosCI':>17} {'fullN':>5} {'fullNet':>8}")
    for r in recs[:15]:
        te = r["te"]
        print(f"  {r['name']:<32} {r['tr']['n']:>4} {r['tr']['net']:>+8.2f} {r['tr']['t']:>+6.2f} | "
              f"{te['n']:>5} {te['net']:>+8.2f} {te['t']:>+6.2f} [{te['lo']:>+5.2f},{te['hi']:>+5.2f}] "
              f"{r['full']['n']:>5} {r['full']['net']:>+8.2f}")

    ship_tr = stats([r for r in ship if r["wi"] < CUT])
    rank = sum(1 for r in recs if r["tr"]["net"] > ship_tr["net"])
    print(f"\n  shipped rule on TRAIN: net=${ship_tr['net']:+.2f} t={ship_tr['t']:+.2f} "
          f"-> {rank} of {len(recs)} candidates beat it on train alone")
    np.save("_pm_alt_recs.npy", np.array(recs, dtype=object), allow_pickle=True)
    print("\nDONE")
