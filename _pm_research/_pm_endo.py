"""Stage 27: ENDOGENEITY AUDIT of the model-confirmed endgame rule.

The "model-confirmed T>=250" rule printed win=1.0000 over 48 windows.  That is
TOO good and MUST be audited before shipping.  Two possible explanations:

 (E1) REAL: the market genuinely underprices the favourite in the endgame, and
      my TWAP model spots it.  Then the rule should also work on data the
      model cannot see (held-out windows) and should survive a placebo.

 (E2) ENDOGENOUS: my "strike" is reconstructed from the SAME coinbase series
      that defines the outcome, so z_mine is partly built from the answer.
      If the market's mid and my z_mine are both derived from the same price
      path, selecting on z_mine - z_mkt can select windows where my
      reconstruction happens to agree with the resolution, inflating the
      measured win rate.

AUDITS PERFORMED HERE:
  A. PLACEBO on the signal: replace z_mine with a random z of the same
     distribution.  If the rule's edge survives, it is a price-band artefact;
     if it vanishes, the model carries real information.
  B. PLACEBO on the outcome: shuffle the winners across windows.  Any
     "profit" that survives is an artefact of the selection rule.
  C. SHUFFLED-STRIKE placebo: rebuild z_mine using a strike taken from a
     DIFFERENT (random) window.  This directly destroys any endogeneity
     between strike and outcome while keeping the market's prices intact.
  D. WINDOW SPLIT: first half vs second half of the sample.
  E. How much of the effect is just "ask >= 0.95" (no model at all)?
"""
import os, pickle
import numpy as np
from scipy.special import ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
RNG = np.random.default_rng(12345)


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


def var_mult(T):
    if T >= 240.0:
        return (300.0 - T) ** 3 / 10800.0
    a, b = 240.0 - T, 300.0 - T
    return 2.0 / 3600.0 * (b ** 3 / 6.0 - b * a * a / 2.0 + a ** 3 / 3.0)


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]; d["ws"] = int(d["ws"][0])
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


def main():
    W = load_all()
    nw = len(W)
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])

    # ---- build rows; also record a RANDOM-STRIKE variant for audit C ----
    all_strikes = [float(np.mean(ffill(by_ws[w]["cb_px"])[-40:])) for w in sorted(by_ws)]
    rows = []
    for wi, d in enumerate(W):
        t, cb, n = d["t"], ffill(d["cb_px"]), len(d["t"])
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        fake_strike = float(all_strikes[RNG.integers(len(all_strikes))])
        if not np.isfinite(strike) or strike <= 0: continue
        r = np.zeros(n); r[1:] = np.diff(np.log(np.where(cb > 0, cb, np.nan)))
        r[~np.isfinite(r)] = 0
        sig2 = np.cumsum(r * r) / np.maximum(t, 1e-9)
        j240 = int(np.searchsorted(t, 240.0))
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5: continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998): continue
            s2 = float(sig2[i])
            if not np.isfinite(s2) or s2 <= 0: continue
            obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
            rem = 300.0 - T
            eQ = (obs.sum() + rem * cb[i]) / 60.0 if len(obs) else cb[i]
            vQ = var_mult(T) * s2
            if vQ <= 0: continue
            for strike_use, tag in ((strike, "real"), (fake_strike, "fake")):
                z_up = (np.log(eQ) - np.log(strike_use)) / np.sqrt(vQ)
                for side in ("Up", "Down"):
                    ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                    asz = d[side + "_asksz"][i]
                    if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                    if not (0 < bid <= ask < 1): continue
                    if (ask - bid) > 0.04: continue
                    if asz * ask < 10.0: continue
                    mkt_side = p_up if side == "Up" else 1.0 - p_up
                    z_mkt = ndtri(min(max(mkt_side, 1e-6), 1 - 1e-6))
                    sgn = 1.0 if side == "Up" else -1.0
                    rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                     mkt=float(mkt_side), z_mkt=float(z_mkt),
                                     z_mine=float(sgn * z_up), tag=tag,
                                     won=1 if d["_winner"] == side else 0))
    print(f"rows {len(rows)}  (real-strike {sum(r['tag']=='real' for r in rows)}, "
          f"fake-strike {sum(r['tag']=='fake' for r in rows)})")

    def first_per_window(rs):
        best = {}
        for r in rs:
            k = r["wi"]
            if k not in best or r["T"] < best[k]["T"]:
                best[k] = r
        return list(best.values())

    def run(name, rs, keyfn, won_key="won", shuffle_winners=False, seed=0):
        rs = first_per_window(rs)
        if not rs: return
        wi = np.array([r["wi"] for r in rs])
        won = np.array([r["won"] for r in rs])
        if shuffle_winners:
            rng = np.random.default_rng(seed)
            won = rng.permutation(won)
        sel = np.array([keyfn(r) for r in rs])
        pnl = 10.0 / np.array([r["ask"] for r in rs]) * won - 10.0 \
              - FEE(np.array([r["ask"] for r in rs]), 10.0 / np.array([r["ask"] for r in rs]))
        m_, tst, w_ = cluster_t(wi, pnl)
        print(f"  {name:<46} N={len(rs):>3} w={w_:>3} tot=${pnl.sum():+7.2f} "
              f"avg=${pnl.mean():+.3f} t={tst:+6.2f} win={won.mean():.4f} "
              f"ask={np.mean([r['ask'] for r in rs]):.4f}")
        return rs, won

    real = [r for r in rows if r["tag"] == "real"]
    fake = [r for r in rows if r["tag"] == "fake"]

    print("\n=== A) PLACEBO: random z of the same distribution ===")
    base = [r for r in real if r["T"] >= 250 and r["mkt"] >= 0.5 and r["ask"] >= 0.90]
    run("real model  (z_mine - z_mkt > 0)", base, lambda r: r["z_mine"] - r["z_mkt"] > 0)
    rs = [dict(r) for r in base]
    for r in rs:
        r["z_rand"] = float(RNG.normal(0, 1)) * max(abs(r["z_mine"]), 0.1)
    run("PLACEBO random z", rs, lambda r: r["z_rand"] - r["z_mkt"] > 0)

    print("\n=== B) PLACEBO: shuffled winners ===")
    for s in range(3):
        run(f"real model, winners shuffled (seed {s})", base,
            lambda r: r["z_mine"] - r["z_mkt"] > 0, shuffle_winners=True, seed=s)

    print("\n=== C) PLACEBO: strike taken from a RANDOM window ===")
    fb = [r for r in fake if r["T"] >= 250 and r["mkt"] >= 0.5 and r["ask"] >= 0.90]
    run("FAKE strike, same rule", fb, lambda r: r["z_mine"] - r["z_mkt"] > 0)
    run("FAKE strike, winners shuffled", fb, lambda r: r["z_mine"] - r["z_mkt"] > 0,
        shuffle_winners=True, seed=1)

    print("\n=== D) real model, window split (first half vs second half) ===")
    cut = nw // 2
    for lbl, cond in (("first half", lambda r: r["wi"] < cut), ("second half", lambda r: r["wi"] >= cut)):
        run(f"real model, {lbl}", [r for r in base if cond(r)],
            lambda r: r["z_mine"] - r["z_mkt"] > 0)

    print("\n=== E) how much is just the price band (no model)? ===")
    run("price band only: T>=250 ask>=0.90", base, lambda r: True)
    run("price band only: T>=250 ask>=0.95", [r for r in base if r["ask"] >= 0.95], lambda r: True)

    print("\n=== F) the actual z-gap distribution (is the signal real?) ===")
    rs = first_per_window(base)
    gap = np.array([r["z_mine"] - r["z_mkt"] for r in rs])
    won = np.array([r["won"] for r in rs])
    print(f"  n={len(rs)}  mean gap={gap.mean():+.3f}  frac gap>0={np.mean(gap>0):.3f}")
    for lo, hi in ((-99, 0), (0, 1), (1, 2), (2, 99)):
        s = (gap >= lo) & (gap < hi)
        if s.sum() >= 3:
            print(f"    gap[{lo},{hi}) n={int(s.sum()):>3} win={won[s].mean():.4f} "
                  f"ask={np.mean([rs[i]['ask'] for i in np.where(s)[0]]):.4f}")


if __name__ == "__main__":
    main()
