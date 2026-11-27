"""Stage 26: FINAL VALIDATION of the endgame-favourite edge (theory-driven).

MECHANISM (established, not guessed):
  * Resolution: Up iff mean(price over [240,300)) > mean(price over [ws-60,ws)).
    Reconstructed from the logs at 97.8% accuracy (vs 91.1% for final spot).
  * The correct conditional z of the favourite is
        z_twap = (log E[Q|F_T] - log strike) / sqrt(Var[Q|F_T])
    with Var[Q|F_T] = s^2 * M(T), M(T) = (300-T)^3/10800 for T>240 and
    M(T) = 2/3600*(b^3/6 - b a^2/2 + a^3/3), a=240-T, b=300-T, for T<=240.
  * Stage 24 measured the market's implied z as a MULTIPLE of the naive SPOT
    z (sqrt(240-M) etc). The measured multiples were
        T=240: 1.720  (TWAP theory 1.732)  -> market already prices TWAP
        T=250: 1.983  (TWAP theory 2.079)  -> market still ~right
        T=270: 1.852  (TWAP theory 3.464)  -> market UNDERCONFIDENT by 1.9x
    => the mispricing grows sharply in the last ~40s, and the market's
    under-confidence means the FAVOURITE is underpriced exactly there.
  * Fee at 0.97 is only 0.07*0.97*0.03 = 0.0020/share, so a ~1.5c edge nets
    ~1.3c/share. That is the only place in this market where an edge can
    survive the fee.

CAVEAT THAT MUST BE HANDLED:
  The true settlement uses the CHAINLINK price; I have COINBASE. A basis
  error of a few bp swamps the late-window sd (~1.4bp), so z_mine is noisy
  here. The bot must therefore use the exact Chainlink inputs; this backtest
  uses Coinbase and reports the achievable-in-production number separately.

THIS SCRIPT:
  1. Direct test: at T>=260, how does the market's implied prob compare with
     the TWAP-model prob, and what is the realised win rate vs what we pay?
  2. Same for the pure price-band rule (no model) as a benchmark.
  3. Cluster bootstrap CI on the P&L (not just a t-stat).
  4. Power analysis: how many more windows are needed to confirm?
"""
import os, pickle
import numpy as np
from scipy.special import ndtri

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


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


def boot_ci(wi, pnl, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    tot = {w: pnl[wi == w].sum() for w in uq}
    cnt = {w: (wi == w).sum() for w in uq}
    keys = list(tot)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        s = sum(tot[k] for k in pick); c = sum(cnt[k] for k in pick)
        out.append(s / c)
    return np.percentile(out, [2.5, 97.5])


def main():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])

    rows = []
    for wi, d in enumerate(W):
        t, cb, n = d["t"], ffill(d["cb_px"]), len(d["t"])
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
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
            z_up = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                mkt_side = p_up if side == "Up" else 1.0 - p_up
                z_mkt = ndtri(min(max(mkt_side, 1e-6), 1 - 1e-6))
                sgn = 1.0 if side == "Up" else -1.0
                p_model = float(ndtri(max(-8.0, min(8.0, sgn * z_up))))
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), asz=float(asz), mkt=float(mkt_side),
                                 p_model=p_model, z_mkt=float(z_mkt), z_mine=float(sgn * z_up),
                                 won=1 if d["_winner"] == side else 0,
                                 mult_theory=float(np.sqrt((300.0 - T) / max(var_mult(T), 1e-12)))))
    print(f"candidate rows (T>=240): {len(rows)}")

    # ---------- 1. the core comparison ----------
    print("\n=== 1) market vs TWAP model, in the endgame ===")
    for T in (240, 250, 260, 270, 280, 290):
        s = [r for r in rows if r["T"] == T]
        if len(s) < 20: continue
        mkt = np.array([r["mkt"] for r in s]); mod = np.array([r["p_model"] for r in s])
        won = np.array([r["won"] for r in s]); ask = np.array([r["ask"] for r in s])
        fav = mkt >= 0.5
        print(f"  T={T:>3} n={len(s):>4} | favourite: mkt={mkt[fav].mean():.4f} "
              f"model={mod[fav].mean():.4f} REALISED={won[fav].mean():.4f} ask={ask[fav].mean():.4f} "
              f"| mult mkt/model={np.mean(mkt[fav]/np.maximum(mod[fav],1e-9)):.3f} "
              f"theory={s[0]['mult_theory']:.3f}")

    # ---------- 2. trading rules, first entry per window ----------
    print("\n=== 2) trading rules (first entry per window, hold to settlement) ===")
    def first_per_window(sel_rows):
        best = {}
        for r in sel_rows:
            k = r["wi"]
            if k not in best or r["T"] < best[k]["T"]:
                best[k] = r
        return list(best.values())

    def report(name, sel_rows):
        if not sel_rows: return
        rowsx = first_per_window(sel_rows)
        wi = np.array([r["wi"] for r in rowsx])
        pnl = np.array([10.0 / r["ask"] * r["won"] - 10.0 - FEE(r["ask"], 10.0 / r["ask"]) for r in rowsx])
        won = np.array([r["won"] for r in rowsx]); ask = np.array([r["ask"] for r in rowsx])
        m_, tst, w_ = cluster_t(wi, pnl)
        lo, hi = boot_ci(wi, pnl)
        print(f"  {name:<42} N={len(rowsx):>3} w={w_:>3} tot=${pnl.sum():+7.2f} "
              f"avg=${pnl.mean():+.3f} t={tst:+5.2f} CI95=[{lo:+.3f},{hi:+.3f}] "
              f"win={won.mean():.4f} ask={ask.mean():.4f} edge/sh={won.mean()-ask.mean():+.4f}")

    for TLO in (240, 250, 260, 270):
        for PLO in (0.90, 0.93, 0.95, 0.96):
            sel = [r for r in rows if r["T"] >= TLO and r["ask"] >= PLO and r["ask"] <= 1.001
                   and r["mkt"] >= 0.5 and r["asz"] * r["ask"] >= 10.0]
            report(f"price-band T>={TLO} ask>={PLO}", sel)

    print()
    for TLO in (250, 260, 270):
        for MARGIN in (0.0, 0.5, 1.0, 2.0):
            sel = [r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                   and r["ask"] >= 0.90 and r["asz"] * r["ask"] >= 10.0
                   and r["z_mine"] - r["z_mkt"] > MARGIN]
            report(f"model-confirmed T>={TLO} zgap>{MARGIN}", sel)

    # ---------- 3. what does the market mis-price most? ----------
    print("\n=== 3) realised win rate vs the price we PAY, endgame only ===")
    sel = [r for r in rows if r["mkt"] >= 0.5 and r["asz"] * r["ask"] >= 10.0]
    for lo, hi in ((0.90, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.001)):
        s = [r for r in sel if lo <= r["ask"] < hi]
        if not s: continue
        won = np.array([r["won"] for r in s]); ask = np.array([r["ask"] for r in s])
        fee = 0.07 * ask * (1 - ask)
        print(f"  ask[{lo:.2f},{hi:.3f}) n={len(s):>4} win={won.mean():.4f} ask={ask.mean():.4f} "
              f"gross/sh={won.mean()-ask.mean():+.4f} fee/sh={fee.mean():.4f} "
              f"net/sh={won.mean()-ask.mean()-fee.mean():+.4f} lossrate={1-won.mean():.4f}")

    # ---------- 4. power ----------
    print("\n=== 4) power / how much more data ===")
    s = [r for r in rows if r["T"] >= 250 and r["ask"] >= 0.95 and r["mkt"] >= 0.5
         and r["asz"] * r["ask"] >= 10.0]
    rowsx = first_per_window(s)
    w = len(rowsx); loss = sum(1 for r in rowsx if not r["won"])
    askm = np.mean([r["ask"] for r in rowsx])
    print(f"  windows used={w}  losses={loss}  mean ask={askm:.4f}")
    print(f"  break-even win rate (ask+fee) = {askm + 0.07*askm*(1-askm):.4f}")
    print(f"  observed win rate = {1-loss/max(w,1):.4f}")
    # what does the market's own price predict for the loss rate?
    print(f"  market-implied loss rate at mean ask = {1-askm:.4f}  -> expected losses in {w} windows = {w*(1-askm):.1f}, observed {loss}")
    for n in (100, 250, 500, 1000, 2500):
        p = 1 - askm
        se = np.sqrt(p * (1 - p) / n)
        print(f"    n={n:>5}: 95% CI on loss rate = [{max(0,p-1.96*se):.4f}, {p+1.96*se:.4f}]  "
              f"(detects a 50% relative cut at loss rate {p*0.5:.4f})")


if __name__ == "__main__":
    main()
