"""Stage 30: FINAL VALIDATION - correct max-t correction + executability checks.

Fixes two bugs from stage 29:
  * p_model was NaN (unresolved log/var) -> debugged here.
  * the max-t placebo compared window IDs to side names -> constant -9.00,
    so the multiple-testing correction was meaningless.  Now the winner
    permutation is applied correctly and the 16 grid rules are pre-selected
    once (selection depends only on PRE-TRADE data, so it is invariant).

Also checks EXECUTABILITY: does the edge survive realistic depth/spread
filters, and is it concentrated in thin books (i.e. untradeable)?
"""
import os, pickle
import numpy as np
from scipy.special import ndtri
from scipy.stats import binom

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
STAKE = 10.0


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
        if not m["_winner"]: continue
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


def boot_ci(wi, pnl, n=6000, seed=11):
    rng = np.random.default_rng(seed)
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    tot = {w: pnl[wi == w].sum() for w in uq}
    cnt = {w: (wi == w).sum() for w in uq}
    keys = list(tot); out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(sum(tot[k] for k in pick) / sum(cnt[k] for k in pick))
    return np.percentile(out, [2.5, 97.5])


def build(verbose=False):
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
    nbad = 0
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
            if not (np.isfinite(eQ) and eQ > 0) or vQ <= 0:
                nbad += 1
                p_model = float("nan")
            else:
                z_up = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
                p_model = float(ndtri(max(-8.0, min(8.0, z_up))))
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), asz=float(asz), mkt=float(mkt),
                                 p_model=p_model,
                                 won=1 if d["_winner"] == side else 0))
    if verbose:
        print(f"  [debug] rows={len(rows)}  non-finite model rows={nbad}")
    return W, rows


def first_per_window(rs):
    best = {}
    for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    return list(best.values())


def pnl_of(rs, won=None):
    won = np.array([r["won"] for r in rs]) if won is None else np.asarray(won, float)
    ask = np.array([r["ask"] for r in rs])
    sh = STAKE / ask
    return sh * won - STAKE - FEE(ask, sh)


def main():
    W, rows = build(verbose=True)
    nw = len(W)
    window_winner = {d["ws"]: d["_winner"] for d in W}
    ws_list = [d["ws"] for d in W]
    print(f"windows {nw}  rows {len(rows)}")

    # ---------------- pre-selected grid (selection is pre-trade => invariant) ----------------
    TLOs = (240, 250, 260, 270)
    PLOs = (0.50, 0.70, 0.80, 0.90)
    sels = {}
    for TLO in TLOs:
        for PLO in PLOs:
            key = (TLO, PLO)
            sels[key] = first_per_window([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                                          and r["ask"] >= PLO])
    sels = {k: v for k, v in sels.items() if len(v) >= 15}

    print("\n=== 1) OBSERVED GRID (cluster-t) ===")
    obs_t = {}
    for key, sel in sels.items():
        p = pnl_of(sel)
        _, tt, ww = cluster_t([r["wi"] for r in sel], p)
        obs_t[key] = tt
        print(f"  T>={key[0]} ask>={key[1]:.2f}: N={len(sel):>3} tot=${p.sum():+8.2f} t={tt:+5.2f}")
    best_key = max(obs_t, key=lambda k: obs_t[k])
    print(f"  best cell: {best_key}  t={obs_t[best_key]:+.2f}")

    print("\n=== 2) MAX-T MULTIPLE-TESTING CORRECTION (correct placebo) ===")
    null_max = []
    for s in range(3000):
        rng = np.random.default_rng(5000 + s)
        perm = rng.permutation(ws_list)
        remap = dict(zip(ws_list, perm))
        best = -9.0
        for key, sel in sels.items():
            nw2 = [1 if window_winner[remap[r["ws"]]] == r["side"] else 0 for r in sel]
            _, tt, _ = cluster_t([r["wi"] for r in sel], pnl_of(sel, nw2))
            best = max(best, tt)
        null_max.append(best)
    null_max = np.array(null_max)
    pmax = float((null_max >= obs_t[best_key]).mean())
    print(f"  observed max t = {obs_t[best_key]:+.2f}")
    print(f"  placebo max-t: mean={null_max.mean():+.2f} sd={null_max.std():.2f} "
          f"p95={np.percentile(null_max,95):+.2f} p99={np.percentile(null_max,99):+.2f}")
    print(f"  p(max placebo t >= observed) = {pmax:.4f}   "
          f"({'SIGNIFICANT at family-wise 5%' if pmax < 0.05 else 'NOT significant family-wise'})")

    print("\n=== 3) THE PRE-REGISTERED RULE, in detail ===")
    sel = sels[(260, 0.80)]
    wi = np.array([r["wi"] for r in sel]); won = np.array([r["won"] for r in sel])
    ask = np.array([r["ask"] for r in sel]); pnl = pnl_of(sel)
    k, n = int(won.sum()), len(won)
    pval = float(binom.sf(k - 1, n, ask.mean()))
    lo, hi = boot_ci(wi, pnl)
    print(f"  N={n} wins={k}  mean ask={ask.mean():.4f} realised={won.mean():.4f}")
    print(f"  market-efficiency binomial p = {pval:.5f}")
    print(f"  PnL=${pnl.sum():+.2f} avg=${pnl.mean():+.3f} cluster-t={obs_t[(260,0.80)]:+.2f} "
          f"CI95=[{lo:+.3f},{hi:+.3f}]")
    print(f"  break-even win rate at mean ask+fees = "
          f"{(STAKE + FEE(ask.mean(), STAKE/ask.mean()))/(STAKE/ask.mean()):.4f}")

    print("\n=== 4) EXECUTABILITY: depth / spread sensitivity ===")
    for dmin in (5, 10, 25, 50, 100):
        s = [r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5 and r["ask"] >= 0.80
             and r["asz"] * r["ask"] >= dmin]
        s = first_per_window(s)
        if len(s) < 10: continue
        p = pnl_of(s); _, tt, ww = cluster_t([r["wi"] for r in s], p)
        print(f"  depth>=${dmin:>3}: N={len(s):>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={tt:+5.2f}")
    print("  spread sensitivity:")
    for smax in (0.01, 0.02, 0.04, 0.06, 0.10):
        s = [r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5 and r["ask"] >= 0.80
             and (r["ask"] - r["bid"]) <= smax]
        s = first_per_window(s)
        if len(s) < 10: continue
        p = pnl_of(s); _, tt, ww = cluster_t([r["wi"] for r in s], p)
        print(f"  spread<={smax:.2f}: N={len(s):>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={tt:+5.2f}")

    print("\n=== 5) MECHANISM CHECK: does the favourite gap grow with T? ===")
    print("  (market-implied prob minus realised win rate; positive = market UNDERPRICES)")
    for T in range(240, 300, 4):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and r["ask"] >= 0.80]
        if len(s) < 8: continue
        s1 = first_per_window(s)
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        gap = w1.mean() - m1.mean()
        se = np.sqrt(max(m1.mean() * (1 - m1.mean()), 1e-9) / len(s1))
        print(f"  T={T:>3} n={len(s1):>3} mkt={m1.mean():.4f} realised={w1.mean():.4f} "
              f"gap={gap:+.4f} (t={gap/se if se>0 else 0:+.2f})")

    print("\n=== 6) WHY IT WORKS: strike-vs-settlement basis, by T ===")
    for T in (240, 250, 260, 270, 280, 290):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and r["ask"] >= 0.80]
        if len(s) < 8: continue
        s1 = first_per_window(s)
        m1 = np.array([r["mkt"] for r in s1]); w1 = np.array([r["won"] for r in s1])
        pm = np.array([r["p_model"] for r in s1])
        ok = np.isfinite(pm)
        print(f"  T={T:>3} n={len(s1):>3} market={m1.mean():.4f} realised={w1.mean():.4f} "
              f"twap_model={pm[ok].mean() if ok.any() else float('nan'):.4f}")


if __name__ == "__main__":
    main()
