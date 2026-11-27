"""Stage 6: RESOLUTION MECHANICS.

Polymarket 5-min crypto markets resolve Up iff the time-weighted average
price of the Chainlink 60-second TWAP stream over the window >= the price at
window start. That is NOT the same as "final spot > open spot".

If the market converges to FINAL SPOT but resolution is TWAP-based, then late in
the window the market can be systematically wrong -> a structural edge.

Tests:
 1. How well does each candidate rule predict the recorded resolution?
      R1 final_spot > open_spot
      R2 TWAP(window) > open_spot
      R3 TWAP over [60,300] > open_spot
      R4 TWAP over [120,300] > open_spot
      R5 last-60s-TWAP > open_spot
 2. Is the market's own late price calibrated to the RESOLUTION rule or to the
    final-spot rule? (i.e. what is the market pricing?)
 3. Where in the window does the market stop moving? (convergence point)
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
    out.sort(key=lambda d: d["ws"][0])
    return out


def twap(t, px, lo, hi):
    """Time-weighted average price over [lo,hi] seconds."""
    m = np.isfinite(px) & (t >= lo) & (t <= hi)
    if m.sum() < 3:
        return np.nan
    tt = t[m]
    pp = px[m]
    if len(tt) < 2:
        return np.nan
    return float(np.trapezoid(pp, tt) / (tt[-1] - tt[0]))


def main():
    data = load_all()
    print(f"windows {len(data)}\n")

    # ---------- 1. which rule predicts the resolution? ----------
    rules = {}
    for name in ("final_spot", "twap_full", "twap_60_300", "twap_120_300", "twap_last60",
                 "twap_30_300", "twap_0_240"):
        rules[name] = [0, 0]  # agree, n

    for d in data:
        t, cb = d["t"], d["cb_px"]
        fin = np.isfinite(cb)
        if fin.sum() < 10:
            continue
        idx = np.where(fin)[0]
        p_open = cb[idx[0]]
        if not np.isfinite(p_open) or p_open <= 0:
            continue
        # final spot = last available in the window
        p_final = cb[idx[-1]]
        cands = {
            "final_spot": p_final,
            "twap_full": twap(t, cb, 0, 300),
            "twap_60_300": twap(t, cb, 60, 300),
            "twap_120_300": twap(t, cb, 120, 300),
            "twap_30_300": twap(t, cb, 30, 300),
            "twap_0_240": twap(t, cb, 0, 240),
            "twap_last60": twap(t, cb, 240, 300),
        }
        for k, v in cands.items():
            if not np.isfinite(v):
                continue
            rules[k][0] += int((v > p_open) == (d["_winner"] == "Up"))
            rules[k][1] += 1

    print("=== rule agreement with recorded resolution ===")
    for k, (a, n) in sorted(rules.items(), key=lambda x: -x[1][0] / max(x[1][1], 1)):
        print(f"  {k:<16} {a:>3}/{n:<4} = {a/max(n,1):.4f}")

    # ---------- 2. what is the market pricing? ----------
    print("\n=== market late-window price vs. each rule's implied prob ===")
    # For each window: at T=240 compute the market prob of Up, and the
    # "implied prob" from each rule (as a hard 0/1), then see which rule the
    # market's price AGREES with when it is confident.
    T = 240
    agree = {k: 0 for k in rules}
    nconf = 0
    for d in data:
        t, cb = d["t"], d["cb_px"]
        i = int(np.searchsorted(t, T))
        if i >= len(t):
            continue
        mid_u = d["Up_mid"][i]
        mid_d = d["Down_mid"][i]
        if not (np.isfinite(mid_u) and np.isfinite(mid_d)):
            continue
        p_mkt_up = 0.5 * (mid_u + (1 - mid_d))
        if p_mkt_up < 0.85:
            continue
        nconf += 1
        fin = np.isfinite(cb)
        idx = np.where(fin)[0]
        p_open = cb[idx[0]]
        cands = {
            "final_spot": cb[idx[-1]],
            "twap_full": twap(t, cb, 0, 300),
            "twap_60_300": twap(t, cb, 60, 300),
            "twap_120_300": twap(t, cb, 120, 300),
            "twap_30_300": twap(t, cb, 30, 300),
            "twap_0_240": twap(t, cb, 0, 240),
            "twap_last60": twap(t, cb, 240, 300),
        }
        for k, v in cands.items():
            if not np.isfinite(v):
                continue
            up = v > p_open
            # market says Up (confidently). If rule says Up too -> agreement.
            agree[k] += int(up == (p_mkt_up > 0.5))
    print(f"  confident windows (mkt p_up>0.85 at T={T}s): n={nconf}")
    for k, a in sorted(agree.items(), key=lambda x: -x[1]):
        print(f"    market agrees with {k:<16} {a:>3}/{nconf} = {a/max(nconf,1):.4f}")

    # ---------- 3. convergence: when does the market stop moving? ----------
    print("\n=== market |p_up - 0.5| by time (mean over windows) ===")
    for T in (10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 290):
        vals = []
        for d in data:
            t = d["t"]
            i = int(np.searchsorted(t, T))
            if i >= len(t):
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            vals.append(abs(0.5 * (mu + (1 - md)) - 0.5))
        if vals:
            print(f"  T={T:>4}s  mean|p-0.5|={np.mean(vals):.4f}  median={np.median(vals):.4f}  n={len(vals)}")

    # ---------- 4. KEY: late-window mispricing vs the TWAP rule ----------
    # Strategy: at T, if the market is confident but the TWAP rule says the
    # other side, take the other side. Measure P&L.
    print("\n=== candidate: fade the market's late price using the TWAP rule ===")
    for T in (180, 210, 240, 270):
        for RULE in ("final_spot", "twap_60_300", "twap_120_300"):
            wins = 0
            n = 0
            pnl = 0.0
            for d in data:
                t, cb = d["t"], d["cb_px"]
                i = int(np.searchsorted(t, T))
                if i >= len(t):
                    continue
                mu, md = d["Up_mid"][i], d["Down_mid"][i]
                if not (np.isfinite(mu) and np.isfinite(md)):
                    continue
                p_mkt_up = 0.5 * (mu + (1 - md))
                if not (0.85 < p_mkt_up < 0.99):
                    continue
                fin = np.isfinite(cb)
                idx = np.where(fin)[0]
                p_open = cb[idx[0]]
                v = {"final_spot": cb[idx[-1]], "twap_60_300": twap(t, cb, 60, 300),
                     "twap_120_300": twap(t, cb, 120, 300)}[RULE]
                if not np.isfinite(v):
                    continue
                rule_up = v > p_open
                if rule_up == (p_mkt_up > 0.5):
                    continue  # no disagreement -> no trade
                n += 1
                won = (rule_up == (d["_winner"] == "Up"))
                wins += int(won)
                # we buy the RULE's side at the market ask for that side
                if rule_up:
                    ask = d["Up_ask"][i]
                else:
                    ask = d["Down_ask"][i]
                if not np.isfinite(ask) or ask <= 0 or ask >= 1:
                    continue
                stake = 10.0
                shares = stake / ask
                fee = 0.07 * ask * (1 - ask) * shares
                p = shares * (1.0 if won else 0.0) - stake - fee
                pnl += p
            print(f"  T={T:>3} rule={RULE:<12} disagreements={n:>3} rule_correct={wins:>3} "
                  f"win%={wins/max(n,1):.3f} P&L=${pnl:+.2f}")


if __name__ == "__main__":
    main()
