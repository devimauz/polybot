"""Stage 5: LATENCY edge. Does the Polymarket book lag the underlying?

If the market's mid is a lagged/filtered function of spot, then when spot has
just moved and the market has NOT yet repriced, the market is stale -> buy the
side the market hasn't caught up on.

Method: for each window, build a series of (coinbase return over dt, market mid
change over dt) pairs. Regress mid-change on coinbase return at various LAGS.
If the market lags by L seconds, the coefficient at lag L will be high and the
book will still be 'wrong' right after a spot jump.

Then, the tradable version: at time t, compute the model's fair prob from spot
and compare with the book's mid. If the gap is large AND the gap arose from a
recent (within L seconds) spot move, the book has not repriced yet -> edge.
"""
import os, pickle
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"


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
        out.append(d)
    return out


def ffill(a):
    a = a.copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


def main():
    data = load_all()
    print(f"windows {len(data)}\n")

    # Build paired series: for each snapshot i, dt to next.
    # cb_ret[i] = log(cb[i]/cb[i-1]) ; mid_ret[i] = mid[i]-mid[i-1]
    # For lag L: mid_ret over [i, i+L] vs cb_ret over [i-L, i]
    lags = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
    acc = {L: [] for L in lags}
    for d in data:
        cb = ffill(d["cb_px"])
        mid = ffill(0.5 * (d["Up_mid"] + (1 - d["Down_mid"])))  # market prob of Up
        n = len(cb)
        dt = np.diff(d["t"])
        cbr = np.zeros(n)
        cbr[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        cbr[~np.isfinite(cbr)] = 0
        midr = np.zeros(n)
        midr[1:] = np.diff(mid)
        midr[~np.isfinite(midr)] = 0
        for L in lags:
            # market move from i to i+L ; spot move from i-L to i
            for i in range(L, n - L):
                mm = midr[i:i + L].sum()
                sm = cbr[i - L:i].sum()
                acc[L].append((sm, mm))

    print("=== market-mid response to spot move, by lag (slope = dmid/dspot) ===")
    print(f"{'lag':>5} {'n':>8} {'slope':>9} {'corr':>7}")
    for L in lags:
        arr = np.array(acc[L])
        if len(arr) < 100:
            continue
        sm, mm = arr[:, 0], arr[:, 1]
        sl = np.polyfit(sm, mm, 1)[0]
        c = np.corrcoef(sm, mm)[0, 1]
        print(f"{L:>5} {len(arr):>8} {sl:>9.3f} {c:>7.3f}")

    # ---- staleness trade: after a spot jump, is the book behind? ----
    print("\n=== staleness check: book 'wrongness' right after a spot jump ===")
    for THR in (0.0015, 0.0025, 0.004):
        for LOOK in (1, 2, 3, 5):
            gaps = []
            for d in data:
                cb = ffill(d["cb_px"])
                mid = ffill(0.5 * (d["Up_mid"] + (1 - d["Down_mid"])))
                n = len(cb)
                for i in range(LOOK, n):
                    p_now, p_then = cb[i], cb[i - LOOK]
                    if p_then <= 0:
                        continue
                    jump = (p_now - p_then) / p_then
                    if abs(jump) < THR:
                        continue
                    # if the market had fully caught up, mid would be near 0/1
                    # measure how far mid still is from the jump direction
                    gaps.append((jump, mid[i] - mid[i - LOOK]))
            if not gaps:
                continue
            g = np.array(gaps)
            up = g[g[:, 0] > 0]
            dn = g[g[:, 0] < 0]
            # for up jumps: has mid risen as much as it should? use sign agreement
            sign_agree = np.mean(np.sign(up[:, 1]) == np.sign(up[:, 0])) if len(up) else np.nan
            print(f"  jump>{THR:.4f} look={LOOK}: n={len(g):>5} up-jumps={len(up):>4} "
                  f"mid direction agrees with jump in {sign_agree:.3f} of up-jumps, "
                  f"median mid move on up-jump={np.median(up[:,1]):+.4f}")


if __name__ == "__main__":
    main()
