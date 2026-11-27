"""Stage 1 analysis: is there ANY exploitable structure in the Polymarket BTC 5m book?

Key tests:
 1. Book sanity: bid<=ask? spread distribution? stale/inverted books?
 2. Market calibration: actual win rate vs. market mid, binned. (The bar to beat.)
 3. Coinbase price move vs. resolution: how well does the underlying predict?
 4. Order-book imbalance as a predictor.
"""
import os, glob, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    wins = []
    for m in man:
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        wins.append((m, {k: z[k] for k in z.files if k != "_winner"}))
    return wins


def main():
    data = load_all()
    print(f"windows: {len(data)}")

    # ---- 1. book sanity -------------------------------------------------
    inv_up = inv_dn = 0
    n_sp = []
    ask_sz_up = []
    for m, d in data:
        for side, cnt in (("Up", "inv_up"), ("Down", "inv_dn")):
            b, a = d[side + "_bid"], d[side + "_ask"]
            ok = np.isfinite(b) & np.isfinite(a)
            bad = ok & ((b > a) | (b <= 0) | (a >= 1))
            if side == "Up":
                inv_up += int(bad.sum())
            else:
                inv_dn += int(bad.sum())
        sp = d["Up_spread"]
        n_sp.append(sp[np.isfinite(sp)])
    n_sp = np.concatenate(n_sp)
    print(f"inverted/illegal Up rows: {inv_up}, Down rows: {inv_dn}")
    print(f"spread pct: p10={np.nanpercentile(n_sp,10):.3f} p50={np.nanpercentile(n_sp,50):.3f} "
          f"p90={np.nanpercentile(n_sp,90):.3f} p99={np.nanpercentile(n_sp,99):.3f}")

    # ---- 2. calibration: market mid -> actual win rate --------------------
    # Sample at fixed times-in-window.
    for T in (30, 60, 120, 180, 240, 280):
        rows = []
        for m, d in data:
            if not m["_winner"]:
                continue
            t = d["t"]
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            mid = d["Up_mid"][i]
            if not np.isfinite(mid):
                continue
            rows.append((mid, 1 if m["_winner"] == "Up" else 0))
        if not rows:
            continue
        arr = np.array(rows)
        # binned calibration
        edges = np.arange(0.0, 1.001, 0.05)
        bi = np.clip(np.digitize(arr[:, 0], edges) - 1, 0, len(edges) - 2)
        out = []
        for b in range(len(edges) - 1):
            m_ = bi == b
            if m_.sum() >= 3:
                out.append((round(edges[b], 2), round(edges[b + 1], 2), int(m_.sum()),
                            round(float(arr[m_, 1].mean()), 3), round(float(arr[m_, 1].mean() - arr[m_, 0].mean()), 3)))
        print(f"\n=== T={T}s  calibration (Up mid -> win rate)  n={len(arr)} ===")
        print(f"{'bucket':>14} {'n':>5} {'winrate':>8} {'edge':>7}")
        for o in out:
            print(f"{str(o[0])+'-'+str(o[1]):>14} {o[2]:>5} {o[3]:>8} {o[4]:>+7}")

    # ---- 3. coinbase price move vs resolution ---------------------------
    for T in (30, 60, 120, 180, 240, 270):
        agree = n = 0
        for m, d in data:
            if not m["_winner"]:
                continue
            t, cb = d["t"], d["cb_px"]
            idx = np.isfinite(cb)
            if idx.sum() < 2:
                continue
            i0 = int(np.argmax(idx))
            i = int(np.searchsorted(t, T))
            if i >= len(t) or not np.isfinite(cb[i]) or not np.isfinite(cb[i0]):
                continue
            up = cb[i] > cb[i0]
            n += 1
            agree += int(up == (m["_winner"] == "Up"))
        if n:
            print(f"T={T:>4}s  coinbase-move direction agrees with resolution: {agree}/{n} = {agree/n:.3f}")

    # ---- 4. what does the FINAL book look like vs resolution ------------
    rows = []
    for m, d in data:
        if not m["_winner"]:
            continue
        t = d["t"]
        i = len(t) - 1
        mid = d["Up_mid"][i]
        if not np.isfinite(mid):
            continue
        rows.append((mid, 1 if m["_winner"] == "Up" else 0, d["Up_bid"][i], d["Up_ask"][i]))
    arr = np.array(rows)
    print(f"\n=== FINAL snapshot (last logged): n={len(arr)} ===")
    edges = np.arange(0.0, 1.001, 0.05)
    bi = np.clip(np.digitize(arr[:, 0], edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m_ = bi == b
        if m_.sum() >= 3:
            print(f"  mid {edges[b]:.2f}-{edges[b+1]:.2f}  n={m_.sum():>3}  win={arr[m_,1].mean():.3f}  edge={arr[m_,1].mean()-arr[m_,0].mean():+.3f}")
    # what fraction of windows end with mid > 0.9 but lose
    for thr in (0.9, 0.95, 0.98):
        sel = arr[:, 0] > thr
        if sel.sum():
            print(f"  final mid>{thr}: n={int(sel.sum())} winrate={arr[sel,1].mean():.3f} "
                  f"edge_vs_mid={arr[sel,1].mean()-arr[sel,0].mean():+.4f} "
                  f"edge_vs_ask={arr[sel,1].mean()-arr[sel,3].mean():+.4f}")


if __name__ == "__main__":
    main()
