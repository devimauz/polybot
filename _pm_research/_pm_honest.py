"""Stage 29: THE HONEST FINAL TEST of the endgame-favourite edge.

Fixes the broken placebo from stage 28 (permuting a win-vector preserves the
win COUNT, so PnL was unchanged - the placebo tested nothing).

This script:
  1. States the candidate rule ONCE, before looking at results.
  2. Tests it against the MARKET-EFFICIENCY null (wins ~ Binomial(N, ask)).
  3. Runs a CORRECT placebo: shuffle each window's winner across windows.
  4. Reports the full grid with a max-t multiple-testing correction.
  5. Does a pre-registered holdout (train = first half, test = second half).
  6. Checks block stability and PnL concentration.
  7. Dumps the individual trades for artifact inspection.

PRE-REGISTERED CANDIDATE (fixed now, before any result is read):
    Buy the favourite at the ASK, $10 stake, in the endgame,
    T >= 260 s into the 5-minute window,
    favourite's ask in [0.80, 0.99),
    hold to settlement.
    Rationale: the 60s-TWAP settlement locks in the observed part of the
    final minute, so the true probability of the favourite should exceed the
    spot-rule price the market quotes - the gap grows as T -> 300.
"""
import os, pickle, json
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


def boot_ci(wi, pnl, n=4000, seed=11):
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


def build():
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
                if asz * ask < 10.0: continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                sgn = 1.0 if side == "Up" else -1.0
                p_model = float(ndtri(max(-8.0, min(8.0, sgn * z_up))))
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), mkt=float(mkt), p_model=p_model,
                                 won=1 if d["_winner"] == side else 0))
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
    W, rows = build()
    print(f"rows {len(rows)}  windows {len(W)}")
    nw = len(W)
    window_winner = {d["ws"]: d["_winner"] for d in W}
    ws_list = [d["ws"] for d in W]

    # ================= 1. MARKET-EFFICIENCY NULL =================
    print("\n=== 1) PRE-REGISTERED RULE: T>=260, favourite, ask in [0.80,0.99), hold ===")
    cand = first_per_window([r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5
                             and 0.80 <= r["ask"] < 0.99])
    wi = np.array([r["wi"] for r in cand]); won = np.array([r["won"] for r in cand])
    ask = np.array([r["ask"] for r in cand])
    pnl = pnl_of(cand)
    m_, t_, w_ = cluster_t(wi, pnl)
    lo, hi = boot_ci(wi, pnl)
    k, n = int(won.sum()), len(won)
    exp = float(ask.sum())
    pval = float(binom.sf(k - 1, n, ask.mean()))
    print(f"  N={n} windows={w_} wins={k}  expected wins if market efficient = {exp:.2f}")
    print(f"  mean ask = {ask.mean():.4f}   realised win rate = {won.mean():.4f}")
    print(f"  binomial p(one-sided, market efficient) = {pval:.5f}")
    print(f"  PnL total = ${pnl.sum():+.2f}   avg = ${pnl.mean():+.3f}   cluster-t = {t_:+.2f}"
          f"   CI95 = [{lo:+.3f}, {hi:+.3f}]")

    # ================= 2. CORRECT PLACEBO =================
    print("\n=== 2) CORRECT PLACEBO: shuffle each window's winner across windows ===")
    tot_null = []
    for s in range(2000):
        rng = np.random.default_rng(s)
        perm = rng.permutation(ws_list)
        remap = dict(zip(ws_list, perm))
        newwon = []
        for r in cand:
            w2 = window_winner[remap[r["ws"]]]
            newwon.append(1 if w2 == r["side"] else 0)
        tot_null.append(pnl_of(cand, newwon).sum())
    tot_null = np.array(tot_null)
    obs = pnl.sum()
    print(f"  observed total = ${obs:+.2f}")
    print(f"  placebo mean   = ${tot_null.mean():+.2f}  sd = ${tot_null.std():.2f}")
    print(f"  placebo p(>= observed) = {float((tot_null >= obs).mean()):.4f}")

    # ================= 3. FULL GRID + MAX-T CORRECTION =================
    print("\n=== 3) FULL GRID (max-t multiple-testing correction) ===")
    grid = []
    for TLO in (240, 250, 260, 270):
        for PLO in (0.50, 0.70, 0.80, 0.90):
            sel = first_per_window([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                                    and r["ask"] >= PLO])
            if len(sel) < 15: continue
            p = pnl_of(sel)
            _, tt, ww = cluster_t([r["wi"] for r in sel], p)
            grid.append((tt, TLO, PLO, len(sel), ww, p.sum()))
            print(f"  T>={TLO} ask>={PLO:.2f}: N={len(sel):>3} w={ww:>3} tot=${p.sum():+8.2f} t={tt:+5.2f}")
    maxt = max(g[0] for g in grid)
    null_max = []
    for s in range(2000):
        rng = np.random.default_rng(1000 + s)
        perm = rng.permutation(ws_list); remap = dict(zip(ws_list, perm))
        best = -9
        for TLO in (240, 250, 260, 270):
            for PLO in (0.50, 0.70, 0.80, 0.90):
                sel = first_per_window([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                                        and r["ask"] >= PLO])
                if len(sel) < 15: continue
                nw2 = [1 if remap[r["ws"]] == r["side"] else 0 for r in sel]
                _, tt, _ = cluster_t([r["wi"] for r in sel], pnl_of(sel, nw2))
                best = max(best, tt)
        null_max.append(best)
    null_max = np.array(null_max)
    print(f"  observed max t over grid = {maxt:+.2f}")
    print(f"  placebo max-t mean = {null_max.mean():+.2f}  p95 = {np.percentile(null_max,95):+.2f}"
          f"  p(max>=obs) = {float((null_max >= maxt).mean()):.4f}")

    # ================= 4. PRE-REGISTERED HOLDOUT =================
    print("\n=== 4) HOLDOUT: fit-free (rule pre-registered) split by time ===")
    cut = int(nw * 0.5)
    for lbl, sel in (("first half ", [r for r in cand if r["wi"] < cut]),
                     ("second half", [r for r in cand if r["wi"] >= cut])):
        p = pnl_of(sel); _, tt, ww = cluster_t([r["wi"] for r in sel], p)
        print(f"  {lbl}: N={len(sel):>3} w={ww:>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={tt:+5.2f}")

    # ================= 5. BLOCK STABILITY + CONCENTRATION =================
    print("\n=== 5) BLOCK STABILITY (8 chronological blocks) and PnL CONCENTRATION ===")
    order = np.argsort([r["wi"] for r in cand])
    blocks = np.array_split(np.array(order), 8)
    pos = 0
    for bi, b in enumerate(blocks):
        sel = [cand[i] for i in b]
        if not sel: continue
        p = pnl_of(sel); pos += int(p.sum() > 0)
        print(f"  block {bi}: N={len(sel):>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f}")
    print(f"  positive blocks: {pos}/{len(blocks)}")
    p = pnl_of(cand); srt = np.sort(p)[::-1]
    top5 = srt[:5].sum(); tot = p.sum()
    print(f"  best 5 trades = ${top5:+.2f} of total ${tot:+.2f}  ({top5/tot*100 if tot else 0:.0f}%)")

    # ================= 6. ARTIFACT DUMP =================
    print("\n=== 6) TRADE DUMP (artifact inspection) ===")
    for r in cand[:25]:
        print(f"  ws={r['ws']} T={r['T']:.0f} {r['side']:>4} ask={r['ask']:.4f} bid={r['bid']:.4f} "
              f"mkt={r['mkt']:.4f} p_model={r['p_model']:.4f} won={r['won']}")
    losses = [r for r in cand if r["won"] == 0]
    print(f"  losses: {len(losses)}")
    for r in losses:
        print(f"    ws={r['ws']} T={r['T']:.0f} {r['side']:>4} ask={r['ask']:.4f} mkt={r['mkt']:.4f}")

    # ================= 7. P1 RE-TEST: does the gap grow with T? =================
    print("\n=== 7) MECHANISM: favourite gap (realised - mkt) by T, ask>=0.80 ===")
    for T in range(240, 300, 4):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and r["ask"] >= 0.80]
        if len(s) < 8: continue
        s1 = first_per_window(s)
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        a1 = np.array([r["ask"] for r in s1])
        gap = w1.mean() - m1.mean()
        se = np.sqrt(max(m1.mean() * (1 - m1.mean()), 1e-9) / len(s1))
        print(f"  T={T:>3} n={len(s1):>3} mkt={m1.mean():.4f} realised={w1.mean():.4f} "
              f"gap={gap:+.4f} (t={gap/se if se>0 else 0:+.2f})")


if __name__ == "__main__":
    main()
