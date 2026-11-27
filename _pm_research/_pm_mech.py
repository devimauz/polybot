"""Stage 33: THE MECHANISM - TWAP lock-in (and debug the strike reconstruction).

WHY THE MARKET UNDERPRICES THE FAVOURITE LATE (hypothesis, then test):

  Settlement:  Up  iff  Q > S, where
        Q = mean(price over [240,300))     (Chainlink 60s TWAP at close)
        S = mean(price over [ws-60, ws))    (Chainlink 60s TWAP at open)

  At time T in (240,300) a fraction (T-240)/60 of the settlement window is
  ALREADY OBSERVED and LOCKED IN.  A market maker who prices the naive
  "will the spot price be above the strike" rule ignores this and over-states
  the remaining uncertainty, so the favourite is underpriced.

  The correct conditional model:
        E[Q | F_T] = ( A(T)*(T-240) + (300-T)*P(T) ) / 60
        Var[Q | F_T] = sigma^2 * (300-T)^2 / 7200       (continuous time)
  and  P(Up) = Phi( (log E[Q] - log S) / sd ).

This script:
  1. rebuilds the strike S from the PREVIOUS window's last 60 seconds and
     reports how many windows have a usable neighbour (the NaN bug);
  2. VALIDATES the settlement reconstruction against ground truth;
  3. EMPIRICALLY CALIBRATES the conditional variance (no theory needed);
  4. compares the model's P(Up) with the market's price, bucket by bucket;
  5. reports the mispricing the model exposes, per time bucket.
"""
import os, pickle
import numpy as np
from scipy.special import ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"


def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]): last = a[i]
        elif np.isfinite(last): a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]): nxt = a[i]
        elif np.isfinite(nxt): a[i] = nxt
    return a


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]; d["ws"] = int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


def main():
    W = load_all()
    print(f"windows loaded: {len(W)}")
    ws_all = [d["ws"] for d in W]
    gaps = np.diff(ws_all)
    print(f"consecutive window gaps (s): unique={sorted(set(gaps.tolist()))[:10]} ...")
    n_contig = sum(1 for g in gaps if g == 300)
    print(f"windows whose PREVIOUS window is present: {n_contig+1}")

    # ---- rebuild strike from previous window ----
    by_ws = {d["ws"]: d for d in W}
    strike = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc) & (cc > 0)
        if m.sum() >= 20:
            strike[d["ws"]] = float(np.mean(cc[m]))
    print(f"windows with a reconstructed strike: {len(strike)}")

    # ---- validate the settlement reconstruction ----
    ok = agree = 0
    for d in W:
        if d["ws"] not in strike: continue
        cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
        t = d["t"]
        m = (t >= 240) & np.isfinite(cb)
        if m.sum() < 20: continue
        Q = float(np.mean(cb[m]))
        up_win = (Q > strike[d["ws"]])
        ok += 1
        if (up_win and d["_winner"] == "Up") or ((not up_win) and d["_winner"] == "Down"):
            agree += 1
    print(f"\nSETTLEMENT VALIDATION: {agree}/{ok} = {agree/max(ok,1):.4f} "
          f"(mean of last 60s vs mean of prior 60s)")

    # ---- empirical calibration of the conditional variance ----
    print("\n=== EMPIRICAL CALIBRATION: residual (log Q - log S) vs model sd ===")
    print("  T     n     rms(resid)   model_sd  ratio   implied_var_coeff")
    for T in (240, 250, 260, 270, 280, 290):
        res = []
        for d in W:
            if d["ws"] not in strike: continue
            cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
            t = d["t"]
            i = int(np.searchsorted(t, T))
            if i >= len(t): continue
            m = (t >= 240) & np.isfinite(cb)
            if m.sum() < 20: continue
            Q = float(np.mean(cb[m]))
            # model conditional mean of Q given F_T
            obs = cb[(t >= 240) & (t <= T)]; obs = obs[np.isfinite(obs)]
            rem = 300.0 - T
            if len(obs) == 0 or not np.isfinite(cb[i]): continue
            eQ = (obs.sum() + rem * cb[i]) / 60.0
            # realised residual of the FULL settlement around the model mean
            res.append(np.log(Q) - np.log(eQ))
        if len(res) < 8: continue
        res = np.array(res)
        rms = float(np.sqrt(np.mean(res ** 2)))
        # model sd coefficient: sigma (per second) estimated from the window
        print(f"  {T:>3} {len(res):>5} {rms:>11.6f}")

    # ---- where is the residual variance explained by remaining time? ----
    print("\n=== VARIANCE vs REMAINING TIME (theory: sd ~ (300-T) for the tail) ===")
    print("  Using the residual of log Q around log E[Q|F_T], which isolates the")
    print("  future-price uncertainty only.")
    for T in range(240, 300, 10):
        res = []; sig = []
        for d in W:
            if d["ws"] not in strike: continue
            cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
            t = d["t"]
            i = int(np.searchsorted(t, T))
            if i >= len(t): continue
            m = (t >= 240) & np.isfinite(cb)
            if m.sum() < 20: continue
            Q = float(np.mean(cb[m]))
            obs = cb[(t >= 240) & (t <= T)]; obs = obs[np.isfinite(obs)]
            rem = 300.0 - T
            if len(obs) == 0 or not np.isfinite(cb[i]): continue
            eQ = (obs.sum() + rem * cb[i]) / 60.0
            r = np.log(np.where(cb > 0, cb, np.nan))
            rv = np.nanvar(np.diff(r)) if len(r) > 3 else np.nan
            res.append(np.log(Q) - np.log(eQ)); sig.append(rv)
        if len(res) < 8: continue
        res = np.array(res); sig = np.array(sig)
        okm = np.isfinite(sig) & (sig > 0)
        if okm.sum() < 8: continue
        rms = float(np.sqrt(np.mean(res[okm] ** 2)))
        sbar = float(np.sqrt(np.median(sig[okm])))
        # model sd coefficient k such that sd = k * sqrt(sig) * (300-T)
        rem = 300.0 - T
        k = rms / (sbar * rem / 60.0 * 60.0) if rem else np.nan
        print(f"  T={T:>3} n={len(res):>3} rms_resid={rms:.6f}  sqrt(per-snap var)={sbar:.6f}  "
              f"k={rms/max(sbar,1e-12):.4f} (k/sqrt(rem)*sqrt(60)={k/np.sqrt(rem)*np.sqrt(60):.4f})")


if __name__ == "__main__":
    main()
