"""Stage 21: RIGOROUS endgame-favourite test (all filters + honest walk-forward).

The t[255,275) x ask[0.95,1.01) cell showed 43/43 wins, but WITHOUT a depth
filter: at the end of a window one side's book empties out, so the ask may be
0.99 with $0.03 of size behind it. Such a "fill" is not obtainable.

This script redoes it properly:
  * depth filter      : ask_size * ask >= stake
  * spread filter     : ask - bid <= max_spread
  * sane book filter  : 0 < bid <= ask < 1 on BOTH sides
  * one trade per (window, T) -> the favourite only
  * fee_A (official 0.07 p(1-p)) and fee_B (0.25 (p(1-p))^2)
  * WALK-FORWARD with strategy SELECTION done on the training fold only,
    so the reported test number is not contaminated by grid search
  * tail risk, drawdown, and the "how many distinct windows" check
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUT, exist_ok=True)

FEE_A = lambda p, sh: 0.07 * p * (1 - p) * sh
FEE_B = lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh


def ffill(a):
    a = np.array(a, dtype=float).copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


def cluster_t(wi, pnl):
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se > 0 else np.nan), len(uq)


def build_candidates(W, stake=10.0, max_spread=0.04):
    """For each (window, T) and each side, record a candidate entry."""
    out = []
    for wi, d in enumerate(W):
        t = d["t"]
        n = len(t)
        for T in (240, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5:
                continue
            for side in ("Up", "Down"):
                o = "Down" if side == "Up" else "Up"
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                oask, obid = d[o + "_ask"][i], d[o + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not (np.isfinite(ask) and np.isfinite(bid)):
                    continue
                if not (0 < bid <= ask < 1):
                    continue
                if not (np.isfinite(oask) and np.isfinite(obid)) or not (0 < obid <= oask < 1):
                    continue
                if (ask - bid) > max_spread:
                    continue
                if not np.isfinite(asz) or asz * ask < stake:
                    continue
                out.append(dict(wi=wi, ws=int(d["ws"]), T=T, side=side, ask=float(ask),
                                bid=float(bid), sz=float(asz),
                                won=(d["_winner"] == side)))
    return out


def run(cands, t_lo, t_hi, p_lo, p_hi, stake=10.0, fee=FEE_A):
    """Favourite entry in [t_lo,t_hi) with ask in [p_lo,p_hi). One trade per
    (window, T): the higher-ask side only."""
    by = {}
    for c in cands:
        if not (t_lo <= c["T"] < t_hi):
            continue
        if not (p_lo <= c["ask"] < p_hi):
            continue
        k = (c["wi"], c["T"])
        if k not in by or c["ask"] > by[k]["ask"]:
            by[k] = c
    trades = list(by.values())
    for tr in trades:
        sh = stake / tr["ask"]
        tr["pnl"] = sh * (1.0 if tr["won"] else 0.0) - stake - fee(tr["ask"], sh)
        tr["sh"] = sh
    return trades


def main():
    W = load_all()
    print(f"windows {len(W)}")
    cands = build_candidates(W)
    print(f"candidate entries (with depth+spread filters): {len(cands)}")

    # ---------- grid with all filters ----------
    print("\n=== grid (all filters, $10, fee_A) ===")
    TB = [(240, 260), (250, 270), (255, 275), (260, 280), (265, 290), (270, 300), (250, 300)]
    PB = [(0.90, 0.97), (0.93, 0.98), (0.95, 1.01), (0.97, 1.01), (0.90, 1.01)]
    grid = []
    for tlo, thi in TB:
        cells = []
        for plo, phi in PB:
            tr = run(cands, tlo, thi, plo, phi)
            if len(tr) < 6:
                cells.append("n<6"); continue
            pnl = np.array([x["pnl"] for x in tr])
            w = np.mean([x["won"] for x in tr])
            a = np.mean([x["ask"] for x in tr])
            m, tst, nww = cluster_t([x["wi"] for x in tr], pnl)
            cells.append(f"n{len(tr):>3} w{w:.3f} a{a:.3f} ${pnl.sum():+6.1f} t{tst:+5.2f}")
            grid.append((tlo, thi, plo, phi, tr, pnl, tst))
        print(f"  t[{tlo},{thi}): " + " | ".join(f"{c:>24}" for c in cells))
    print("    (bands: 0.90-0.97 | 0.93-0.98 | 0.95-1.01 | 0.97-1.01 | 0.90-1.01)")

    # ---------- WALK-FORWARD with selection on train ----------
    print("\n=== walk-forward: strategy selected on the TRAIN fold only ===")
    nw = len(W)
    wss = sorted(set(d["ws"] for d in W))
    cut = wss[len(wss) // 2]
    for fold_name, sel_fn in (("fold1->fold2", lambda w: w <= cut), ("fold2->fold1", lambda w: w > cut)):
        best = None
        for (tlo, thi, plo, phi, tr, pnl, tst) in grid:
            tr_s = [x for x in tr if sel_fn(x["ws"])]
            if len(tr_s) < 6:
                continue
            p = np.array([x["pnl"] for x in tr_s])
            if best is None or p.sum() > best[0]:
                best = (p.sum(), tlo, thi, plo, phi, tr, tr_s)
        if best is None:
            continue
        _, tlo, thi, plo, phi, tr, tr_s = best
        tr_o = [x for x in tr if not sel_fn(x["ws"])]
        p_o = np.array([x["pnl"] for x in tr_o]) if tr_o else np.array([])
        p_s = np.array([x["pnl"] for x in tr_s])
        m, tst_o, nww = cluster_t([x["wi"] for x in tr_o], p_o) if tr_o else (0, 0, 0)
        print(f"  {fold_name}: selected t[{tlo},{thi}) ask[{plo},{phi})")
        print(f"     TRAIN n={len(tr_s)} total=${p_s.sum():+.2f} avg=${p_s.mean():+.3f} win={np.mean([x['won'] for x in tr_s]):.4f}")
        if len(p_o):
            print(f"     TEST  n={len(tr_o)} total=${p_o.sum():+.2f} avg=${p_o.mean():+.3f} "
                  f"win={np.mean([x['won'] for x in tr_o]):.4f} cluster-t={tst_o:+.2f} windows={nww}")

    # ---------- detail on the headline cell ----------
    print("\n=== detail: t[255,275) ask[0.95,1.01) ===")
    tr = run(cands, 255, 275, 0.95, 1.01)
    pnl = np.array([x["pnl"] for x in tr])
    w = np.array([x["won"] for x in tr], float)
    a = np.array([x["ask"] for x in tr])
    m, tst, nww = cluster_t([x["wi"] for x in tr], pnl)
    print(f"  n={len(tr)} windows={nww} win={w.mean():.4f} meanAsk={a.mean():.4f} "
          f"total=${pnl.sum():+.2f} avg=${pnl.mean():+.3f} cluster-t={tst:+.2f} worst=${pnl.min():+.2f}")
    losses = [x for x in tr if not x["won"]]
    print(f"  losses: {len(losses)}")
    for x in losses:
        print(f"     ws={x['ws']} T={x['T']} ask={x['ask']:.3f} sz={x['sz']:.1f} pnl={x['pnl']:+.2f}")
    # fee B
    trB = run(cands, 255, 275, 0.95, 1.01, fee=FEE_B)
    print(f"  with fee_B: total=${sum(x['pnl'] for x in trB):+.2f} avg=${np.mean([x['pnl'] for x in trB]):+.3f}")
    # drawdown
    order = sorted(tr, key=lambda x: x["ws"])
    cum = np.cumsum([x["pnl"] for x in order])
    dd = (np.maximum.accumulate(cum) - cum).max()
    print(f"  max drawdown: ${dd:.2f}   final: ${cum[-1]:+.2f}")
    # distinct windows
    print(f"  distinct windows touched: {len(set(x['ws'] for x in tr))} of {len(W)}")

    # ---------- does the edge appear in EVERY half? ----------
    print("\n=== stability of the headline cell across time ===")
    ws_list = sorted(set(x["ws"] for x in tr))
    edges = np.linspace(ws_list[0], ws_list[-1], 7)
    pos = 0
    for b in range(6):
        sel = [x for x in tr if edges[b] <= x["ws"] <= edges[b + 1]]
        if not sel:
            continue
        v = np.mean([x["pnl"] for x in sel])
        pos += int(v > 0)
        print(f"    block {b+1}: n={len(sel):>3} avg=${v:+.3f} win={np.mean([x['won'] for x in sel]):.3f}")
    print(f"    positive blocks: {pos}/6")

    with open(os.path.join(OUT, "endgame_rigorous.json"), "w") as fh:
        json.dump({"n": len(tr), "total": float(pnl.sum()), "avg": float(pnl.mean()),
                   "win": float(w.mean()), "cluster_t": float(tst), "losses": len(losses),
                   "max_dd": float(dd)}, fh, indent=2)


if __name__ == "__main__":
    main()
