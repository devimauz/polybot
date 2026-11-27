"""Stage 8: DEFINITIVE conditional-EV analysis.

Question: for a trade entered at ASK price p in a 5m BTC window, what is the
realized win rate, and is it > p + fees? (p is the price we PAY, not the mid.)

For each window and each side, take the FIRST snapshot in [t_lo, t_hi] where
the ask lies in the band and the book is tradeable (depth >= stake, spread sane).
One trade per (window, side) -> windows are the iid unit.

Outputs:
 * conditional win rate by ask band, with Wilson 95% CI
 * EV per share = winrate - ask - fee(ask)   [hold to settlement]
 * EV with TP/SL exits
 * the same split by time bucket
 * bootstrap CI over windows for the chosen strategy
"""
import os, pickle, math, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

# Official Polymarket crypto taker fee: fee = C * 0.07 * p * (1-p)
def fee_doc(p, shares):
    return 0.07 * p * (1 - p) * shares

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


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
    """First tradeable snapshot in the window/time/price band. Returns dict or None."""
    t = d["t"]
    bid = d[side + "_bid"]
    ask = d[side + "_ask"]
    bsz = d[side + "_bidsz"]
    asz = d[side + "_asksz"]
    n = len(t)
    for i in range(n):
        ti = t[i]
        if ti < t_lo:
            continue
        if ti > t_hi:
            break
        b, a = bid[i], ask[i]
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        if b <= 0 or a >= 1 or b > a + 1e-9:
            continue
        if a < p_lo or a > p_hi:
            continue
        if (a - b) > max_spread:
            continue
        if not np.isfinite(asz[i]) or asz[i] * a < stake:
            continue
        return dict(i=i, t=float(ti), ask=float(a), bid=float(b), asz=float(asz[i]),
                    bsz=float(bsz[i]) if np.isfinite(bsz[i]) else 0.0)
    return None


def ev_hold(entry, d, side, stake, fee_fn=fee_doc):
    a = entry["ask"]
    shares = stake / a
    fee = fee_fn(a, shares)
    won = (d["_winner"] == side)
    proceeds = shares * (1.0 if won else 0.0)
    return proceeds - stake - fee


def ev_with_exits(entry, d, side, stake, tp, sl, fee_fn=fee_doc):
    a = entry["ask"]
    shares = stake / a
    fee = fee_fn(a, shares)
    bid = d[side + "_bid"]
    bsz = d[side + "_bidsz"]
    t = d["t"]
    i = entry["i"]
    for j in range(i + 1, len(t)):
        b = bid[j]
        if not np.isfinite(b):
            continue
        if tp and b >= tp:
            if np.isfinite(bsz[j]) and bsz[j] >= shares:
                return shares * b - fee_fn(b, shares) - stake - fee, "tp"
            break
        if sl and b <= sl:
            if np.isfinite(bsz[j]) and bsz[j] >= shares:
                return shares * b - fee_fn(b, shares) - stake - fee, "stop"
            break
    won = (d["_winner"] == side)
    proceeds = shares * (1.0 if won else 0.0)
    return proceeds - stake - fee, "settle"


def main():
    W = load_windows()
    print(f"windows {len(W)}\n")
    STAKE = 10.0

    # ---------------- 1. conditional win rate by ask band (hold to settlement)
    print("=== conditional win rate by ASK band (all times 30-290s, first fill per window/side) ===")
    bands = [(0.01, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
             (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95),
             (0.95, 0.99), (0.99, 1.01)]
    rows = []
    for lo, hi in bands:
        recs = []
        for d in W:
            for side in ("Up", "Down"):
                e = find_entry(d, side, 30, 290, lo, hi, STAKE)
                if e is None:
                    continue
                won = (d["_winner"] == side)
                pnl = ev_hold(e, d, side, STAKE)
                recs.append((d["_slug"], side, e["t"], e["ask"], won, pnl))
        if not recs:
            print(f"  {lo:.2f}-{hi:.2f}: n=0")
            continue
        k = sum(1 for r in recs if r[4])
        n = len(recs)
        lo_ci, hi_ci = wilson(k, n)
        mean_ask = np.mean([r[3] for r in recs])
        mean_fee = np.mean([fee_doc(r[3], STAKE / r[3]) for r in recs])
        wr = k / n
        ev = wr - mean_ask - mean_fee
        # EV measured per trade (actual pnl)
        ev_actual = np.mean([r[5] for r in recs])
        print(f"  {lo:.2f}-{hi:.2f}: n={n:>4} win={wr:.3f} [{lo_ci:.3f},{hi_ci:.3f}] "
              f"meanAsk={mean_ask:.3f} fee={mean_fee:.4f} EV/sh={ev:+.4f} "
              f"actual P&L/trade=${ev_actual:+.3f}")
        rows.append(dict(lo=lo, hi=hi, n=n, win=wr, ci=(lo_ci, hi_ci), mean_ask=float(mean_ask),
                         fee=float(mean_fee), ev=float(ev), pnl=float(ev_actual)))

    # ---------------- 2. same, split by time bucket
    print("\n=== conditional win rate by ask band x time bucket ===")
    tbands = [(30, 90), (90, 150), (150, 210), (210, 270), (270, 300)]
    for tlo, thi in tbands:
        print(f"  -- t in [{tlo},{thi}] --")
        for lo, hi in bands:
            recs = []
            for d in W:
                for side in ("Up", "Down"):
                    e = find_entry(d, side, tlo, thi, lo, hi, STAKE)
                    if e is None:
                        continue
                    won = (d["_winner"] == side)
                    recs.append((e["ask"], won))
            if len(recs) < 8:
                continue
            k = sum(1 for r in recs if r[1])
            n = len(recs)
            lo_ci, hi_ci = wilson(k, n)
            mean_ask = np.mean([r[0] for r in recs])
            mean_fee = np.mean([fee_doc(r[0], STAKE / r[0]) for r in recs])
            ev = k / n - mean_ask - mean_fee
            print(f"     {lo:.2f}-{hi:.2f}: n={n:>4} win={k/n:.3f} [{lo_ci:.3f},{hi_ci:.3f}] "
                  f"ask={mean_ask:.3f} EV/sh={ev:+.4f}")

    # ---------------- 3. the favourite-late strategy, walked forward ------------
    print("\n=== strategy: buy favourite (ask band), TP/SL variants, walk-forward over time ===")
    # For each window: pick the side whose ask is higher, if it is in the band.
    def strat_fav(d, p_lo, p_hi, t_lo, t_hi, stake):
        best = None
        for side in ("Up", "Down"):
            o = "Down" if side == "Up" else "Up"
            e = find_entry(d, side, t_lo, t_hi, p_lo, p_hi, stake)
            if e is None:
                continue
            oask = d[o + "_ask"][e["i"]]
            if not np.isfinite(oask):
                continue
            if best is None or e["ask"] > best[1]["ask"]:
                best = (side, e)
        return best

    for (p_lo, p_hi) in [(0.90, 0.96), (0.90, 0.99), (0.93, 0.99), (0.95, 1.0),
                         (0.85, 0.99), (0.80, 0.99), (0.70, 0.99)]:
        for (t_lo, t_hi) in [(150, 300), (210, 300), (240, 300)]:
            for exits in (None, (0.98, 0.80), (0.95, 0.85)):
                tr = []
                for d in W:
                    r = strat_fav(d, p_lo, p_hi, t_lo, t_hi, STAKE)
                    if r is None:
                        continue
                    side, e = r
                    if exits:
                        pnl, reason = ev_with_exits(e, d, side, STAKE, exits[0], exits[1])
                    else:
                        pnl, reason = ev_hold(e, d, side, STAKE), "settle"
                    tr.append((d["_slug"], int(d["ws"][0]), side, e["ask"], pnl, reason, d["_winner"] == side))
                if len(tr) < 10:
                    continue
                pnls = np.array([t[4] for t in tr])
                k = sum(1 for t in tr if t[6])
                # split by chronological half for stability
                wss = sorted(set(t[1] for t in tr))
                half = wss[len(wss) // 2]
                p1 = np.mean([t[4] for t in tr if t[1] <= half])
                p2 = np.mean([t[4] for t in tr if t[1] > half])
                n1 = sum(1 for t in tr if t[1] <= half)
                n2 = len(tr) - n1
                ex = "hold" if exits is None else f"tp{exits[0]}/sl{exits[1]}"
                print(f"  band {p_lo:.2f}-{p_hi:.2f} t[{t_lo},{t_hi}] {ex:<14} "
                      f"N={len(tr):>3} net=${pnls.sum():+8.2f} wr={k/len(tr):.3f} "
                      f"avg=${pnls.mean():+.3f} worst=${pnls.min():+.2f} | "
                      f"1sthalf n={n1} avg=${p1:+.3f} | 2ndhalf n={n2} avg=${p2:+.3f}")

    with open(os.path.join(OUTDIR, "cond_ev.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
