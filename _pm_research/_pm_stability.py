"""Stage 11: STABILITY of the longshot-underpriced edge.

The only positive signal so far: longshot ask bands (0.10-0.30) win more often
than their price, and favourite bands (0.70-0.90) win less often.
BUT the halfA/halfB split was -0.66 vs +4.83 -> suspicious.

Tests:
 A) Break the 269 windows into 6 chronological blocks; show the edge per block.
    An edge that flips sign across blocks is NOT a real edge.
 B) Placebo test: apply the same "buy the band" rule to a BAND THAT SHOULD BE
    ZERO-EDGE (0.40-0.60) and to a SHUFFLED-winner control. If the placebo
    also "works", the method is broken.
 C) The decisive one: TRADE THE OTHER SIDE. If longshots are genuinely
    underpriced, then favourites (ask 0.70-0.90) must be OVERPRICED and
    shorting them (buying the opposite side) must profit. A real, two-sided
    arbitrage-free inefficiency shows up on BOTH legs.
 D) Robustness: does the edge survive at other fee schedules and other stakes?
 E) Sub-band resolution: is the edge concentrated in a narrow price slice
    (a knife edge = overfit) or spread across the band (a plateau = real)?
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

FEE_FNS = {
    "doc_0.07": lambda p, sh: 0.07 * p * (1 - p) * sh,
    "pred_0.25sq": lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh,
    "repo_0.05min": lambda p, sh: 0.05 * min(p, 1 - p) * sh,
    "zero": lambda p, sh: 0.0,
}
STAKE = 10.0


def load_windows():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    wins = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        wins.append(d)
    wins.sort(key=lambda d: int(d["ws"][0]))
    return wins


def find_entry(d, side, t_lo, t_hi, p_lo, p_hi, stake, max_spread=0.05):
    t, bid, ask, asz = d["t"], d[side + "_bid"], d[side + "_ask"], d[side + "_asksz"]
    for i in range(len(t)):
        ti = t[i]
        if ti < t_lo:
            continue
        if ti > t_hi:
            break
        b, a = bid[i], ask[i]
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        if b <= 0 or a >= 1 or b > a + 1e-9 or (a - b) > max_spread:
            continue
        if a < p_lo or a > p_hi:
            continue
        if not np.isfinite(asz[i]) or asz[i] * a < stake:
            continue
        return dict(i=i, t=float(ti), ask=float(a), bid=float(b))
    return None


def trade_band(W, side_filter, t_lo, t_hi, p_lo, p_hi, fee_fn, stake=STAKE, seed=None):
    """side_filter: 'long' (buy the side in band), 'short' (buy the OPPOSITE side)."""
    out = []
    for wi, d in enumerate(W):
        w = d["_winner"]
        for side in ("Up", "Down"):
            e = find_entry(d, side, t_lo, t_hi, p_lo, p_hi, stake)
            if e is None:
                continue
            if side_filter == "short":
                o = "Down" if side == "Up" else "Up"
                oask = d[o + "_ask"][e["i"]]
                obid = d[o + "_bid"][e["i"]]
                if not (np.isfinite(oask) and np.isfinite(obid)):
                    continue
                if oask <= 0 or oask >= 1 or obid <= 0 or obid > oask + 1e-9:
                    continue
                if (oask - obid) > 0.05:
                    continue
                oasz = d[o + "_asksz"][e["i"]]
                if not np.isfinite(oasz) or oasz * oask < stake:
                    continue
                e = dict(e, ask=float(oask), bid=float(obid))
                buy_side = o
            else:
                buy_side = side
            a = e["ask"]
            shares = stake / a
            fee = fee_fn(a, shares)
            won = (w == buy_side)
            pnl = shares * (1.0 if won else 0.0) - stake - fee
            out.append(dict(wi=wi, slug=d["_slug"], ws=int(d["ws"][0]), side=buy_side,
                            ask=a, t=e["t"], pnl=pnl, won=won))
    return out


def block_report(tr, nblocks=6, label=""):
    if not tr:
        return f"{label}: no trades"
    wss = sorted(set(t["ws"] for t in tr))
    edges = np.linspace(wss[0], wss[-1], nblocks + 1)
    lines = []
    for b in range(nblocks):
        lo, hi = edges[b], edges[b + 1]
        sel = [t for t in tr if lo <= t["ws"] <= hi]
        if not sel:
            lines.append("     n=0")
            continue
        lines.append(f"n={len(sel):>3} ${np.mean([t['pnl'] for t in sel]):+6.2f}")
    return f"{label}: N={len(tr)} total=${sum(t['pnl'] for t in tr):+.2f} avg=${np.mean([t['pnl'] for t in tr]):+.3f} | blocks: " + " | ".join(lines)


def main():
    W = load_windows()
    print(f"windows {len(W)}\n")
    F = FEE_FNS["doc_0.07"]

    # ---------------- A. block stability ----------------
    print("=" * 110)
    print("A) BLOCK STABILITY (6 chronological blocks) -- longshot band 0.10-0.30, t 30-120")
    print("=" * 110)
    for band in [(0.10, 0.30), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.70, 0.90)]:
        tr = trade_band(W, "long", 30, 120, band[0], band[1], F)
        print("  LONG  " + block_report(tr, 6, f"band {band[0]:.2f}-{band[1]:.2f}"))

    # ---------------- B. placebo ----------------
    print("\n" + "=" * 110)
    print("B) PLACEBO / CONTROL")
    print("=" * 110)
    for band in [(0.10, 0.30), (0.40, 0.60), (0.70, 0.90)]:
        tr = trade_band(W, "long", 30, 120, band[0], band[1], F)
        real = np.mean([t["pnl"] for t in tr]) if tr else 0
        # control: shuffle winners across windows (destroy any real signal)
        rng = np.random.default_rng(3)
        W2 = [dict(d) for d in W]
        perm = rng.permutation(len(W2))
        for i, d in enumerate(W2):
            d["_winner"] = W[perm[i]]["_winner"]
        tr2 = trade_band(W2, "long", 30, 120, band[0], band[1], F)
        ctrl = np.mean([t["pnl"] for t in tr2]) if tr2 else 0
        print(f"  band {band[0]:.2f}-{band[1]:.2f}: real avg=${real:+.3f}   shuffled-winner control=${ctrl:+.3f}")

    # ---------------- C. two-sided check ----------------
    print("\n" + "=" * 110)
    print("C) TWO-SIDED CHECK: if longshots are underpriced, favourites must be overpriced")
    print("=" * 110)
    for band in [(0.05, 0.20), (0.10, 0.30), (0.20, 0.40)]:
        lg = trade_band(W, "long", 30, 120, band[0], band[1], F)
        sh = trade_band(W, "short", 30, 120, band[0], band[1], F)
        print(f"  band {band[0]:.2f}-{band[1]:.2f}:  BUY band   avg=${np.mean([t['pnl'] for t in lg]):+.3f} (N={len(lg)})"
              f"   |  BUY opposite avg=${np.mean([t['pnl'] for t in sh]):+.3f} (N={len(sh)})")
    for band in [(0.70, 0.90), (0.75, 0.95)]:
        lg = trade_band(W, "long", 30, 120, band[0], band[1], F)
        sh = trade_band(W, "short", 30, 120, band[0], band[1], F)
        print(f"  band {band[0]:.2f}-{band[1]:.2f}:  BUY band   avg=${np.mean([t['pnl'] for t in lg]):+.3f} (N={len(lg)})"
              f"   |  BUY opposite avg=${np.mean([t['pnl'] for t in sh]):+.3f} (N={len(sh)})")

    # ---------------- D. sub-band plateau ----------------
    print("\n" + "=" * 110)
    print("D) SUB-BAND RESOLUTION (is it a plateau or a knife edge?)  t 30-120, doc fee")
    print("=" * 110)
    for lo in (0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38):
        tr = trade_band(W, "long", 30, 120, lo, lo + 0.04, F)
        if len(tr) < 12:
            print(f"  {lo:.2f}-{lo+0.04:.2f}: n={len(tr)}")
            continue
        k = sum(1 for t in tr if t["won"])
        ma = np.mean([t["ask"] for t in tr])
        print(f"  {lo:.2f}-{lo+0.04:.2f}: n={len(tr):>3} win={k/len(tr):.3f} meanAsk={ma:.3f} "
              f"edge={k/len(tr)-ma:+.4f} P&L/trade=${np.mean([t['pnl'] for t in tr]):+.3f}")

    # ---------------- E. fee & stake robustness ----------------
    print("\n" + "=" * 110)
    print("E) FEE / STAKE ROBUSTNESS for band 0.10-0.30 t[30,120]")
    print("=" * 110)
    for fname, ff in FEE_FNS.items():
        tr = trade_band(W, "long", 30, 120, 0.10, 0.30, ff)
        print(f"  {fname:<14} avg=${np.mean([t['pnl'] for t in tr]):+.3f} total=${sum(t['pnl'] for t in tr):+.2f} N={len(tr)}")
    for stake in (5, 10, 25, 50, 100):
        tr = trade_band(W, "long", 30, 120, 0.10, 0.30, F, stake=stake)
        if tr:
            print(f"  stake=${stake:>3}: avg=${np.mean([t['pnl'] for t in tr]):+.3f} N={len(tr)}")


if __name__ == "__main__":
    main()
