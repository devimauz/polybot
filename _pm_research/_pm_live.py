"""Stage 32: FINAL BACKTEST - live-realistic logic + mechanism model.

Fixes:
  * p_model was NaN because the strike/eQ were computed with ffill that could
    produce a non-positive price.  Now everything is validated explicitly.
  * the backtest now matches the LIVE bot's entry rule:
        - only in the final N seconds of the window
        - the favourite's ASK must be >= ASK_MIN and <= ASK_MAX
        - the ask must be sustained for HOLD seconds (live bots need a
          persistent quote, and a one-tick quote is not fillable)
        - the OTHER side must still be tradable (ask present) - i.e. we do not
          buy a favourite in a book that has already collapsed to 0.99/0.01
          where the realisable price is not what the mid suggests
        - max $10 at the ask, hold to settlement
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


def build():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc) & (cc > 0)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])
    out = []
    for wi, d in enumerate(W):
        t, cb, n = d["t"], ffill(d["cb_px"]), len(d["t"])
        cb = np.where((cb > 0) & np.isfinite(cb), cb, np.nan)
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else np.nan
        else:
            strike = np.nan
        r = np.zeros(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.diff(np.log(cb))
        r[1:] = np.where(np.isfinite(lr), lr, 0.0)
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
            obs = cb[j240:i + 1]; obs = obs[np.isfinite(obs)]
            rem = 300.0 - T
            eQ = (obs.sum() + rem * cb[i]) / 60.0 if (len(obs) and np.isfinite(cb[i])) else np.nan
            vQ = var_mult(T) * s2
            if (np.isfinite(strike) and strike > 0 and np.isfinite(eQ) and eQ > 0
                    and np.isfinite(vQ) and vQ > 0):
                z_up = (np.log(eQ) - np.log(strike)) / np.sqrt(vQ)
                p_model = float(ndtri(max(-8.0, min(8.0, z_up))))
            else:
                p_model = float("nan")
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                out.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                bid=float(bid), asz=float(asz), mkt=float(mkt),
                                p_model=p_model, won=1 if d["_winner"] == side else 0))
    return W, out


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


def sustained(d, wi, TLO, ASK_MIN, ASK_MAX, HOLD, side_fav):
    """Live-realistic: find the first snapshot at T>=TLO where the favourite's
    ask is in [ASK_MIN, ASK_MAX] and has been sustained for HOLD seconds."""
    t = d["t"]
    n = len(t)
    side = "Up" if side_fav else "Down"
    ask = d[side + "_ask"]
    other = d["Down_ask"] if side == "Up" else d["Up_ask"]
    i0 = int(np.searchsorted(t, TLO))
    if i0 >= n: return None
    for i in range(i0, n):
        a, o = ask[i], other[i]
        if not (np.isfinite(a) and np.isfinite(o)): continue
        if not (ASK_MIN <= a <= ASK_MAX): continue
        if o <= 0.001 or o >= 0.999: continue      # other side collapsed
        # sustained?
        j = i
        while j > 0 and (t[i] - t[j - 1]) <= HOLD and ASK_MIN <= ask[j - 1] <= ASK_MAX:
            j -= 1
        if (t[i] - t[j]) >= HOLD:
            return dict(wi=wi, ws=d["ws"], T=float(t[i]), side=side, ask=float(a),
                        bid=float(d[side + "_bid"][i]), asz=float(d[side + "_asksz"][i]),
                        won=1 if d["_winner"] == side else 0)
    return None


def main():
    W, rows = build()
    print(f"windows {len(W)} rows {len(rows)}")
    nmodel = sum(1 for r in rows if np.isfinite(r["p_model"]))
    print(f"rows with finite p_model: {nmodel}/{len(rows)}")

    window_winner = {d["ws"]: d["_winner"] for d in W}
    ws_list = [d["ws"] for d in W]

    def run(name, TLO, ASK_MIN, ASK_MAX, HOLD, live_logic=True):
        trades = []
        for wi, d in enumerate(W):
            # favourite = side with the higher market mid at TLO
            i = int(np.searchsorted(d["t"], TLO))
            if i >= len(d["t"]): continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            fav_up = p_up >= 0.5
            if live_logic:
                tr = sustained(d, wi, TLO, ASK_MIN, ASK_MAX, HOLD, fav_up)
                if tr: trades.append(tr)
            else:
                cand = [r for r in rows if r["wi"] == wi and r["T"] >= TLO
                        and ((r["mkt"] >= 0.5) == fav_up) and ASK_MIN <= r["ask"] <= ASK_MAX]
                if cand:
                    c = sorted(cand, key=lambda r: r["T"])[0]
                    trades.append(dict(c, mkt=c["mkt"]))
        if not trades: return None, []
        wi = np.array([t["wi"] for t in trades])
        won = np.array([t["won"] for t in trades])
        ask = np.array([t["ask"] for t in trades])
        p = pnl_of(trades)
        m, t, w_ = cluster_t(wi, p)
        lo, hi = boot_ci(wi, p)
        k, n = int(won.sum()), len(won)
        pv = float(binom.sf(k - 1, n, ask.mean())) if n else 1.0
        print(f"  {name:<44} N={n:>3} w={w_:>3} win={won.mean():.4f} ask={ask.mean():.4f} "
              f"tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={t:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] "
              f"binom_p={pv:.4f}")
        return t, trades

    print("\n=== A) SNAPSHOT-LEVEL (no sustain requirement) ===")
    for TLO in (240, 250, 260, 270):
        for AMIN in (0.80, 0.90):
            sel = first_per_window([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                                    and AMIN <= r["ask"] < 0.99])
            p = pnl_of(sel); _, t, w_ = cluster_t([r["wi"] for r in sel], p)
            ask = np.array([r["ask"] for r in sel]); won = np.array([r["won"] for r in sel])
            lbl = f"T>={TLO} fav ask[{AMIN:.2f},0.99)"
            print(f"  {lbl:<36} N={len(sel):>3} w={w_:>3} "
                  f"win={won.mean():.4f} ask={ask.mean():.4f} tot=${p.sum():+7.2f} t={t:+5.2f}")

    print("\n=== B) LIVE-REALISTIC (sustained quote, other side alive) ===")
    best = None
    for TLO in (240, 250, 260, 270):
        for HOLD in (0.0, 2.0, 5.0):
            for AMIN, AMAX in ((0.80, 0.99), (0.85, 0.98), (0.90, 0.98), (0.90, 0.99)):
                t, tr = run(f"T>={TLO} ask[{AMIN},{AMAX}] hold={HOLD}", TLO, AMIN, AMAX, HOLD)
                if t is not None and (best is None or t > best[0]):
                    best = (t, TLO, AMIN, AMAX, HOLD, tr)
    print(f"\n  BEST live-realistic cell: t={best[0]:+.2f}  TLO={best[1]} "
          f"ask=[{best[2]},{best[3]}] hold={best[4]}")

    print("\n=== C) MAX-T CORRECTION on the live-realistic grid ===")
    grid = []
    for TLO in (240, 250, 260, 270):
        for HOLD in (0.0, 2.0, 5.0):
            for AMIN, AMAX in ((0.80, 0.99), (0.85, 0.98), (0.90, 0.98), (0.90, 0.99)):
                tr = []
                for wi, d in enumerate(W):
                    i = int(np.searchsorted(d["t"], TLO))
                    if i >= len(d["t"]): continue
                    mu, md = d["Up_mid"][i], d["Down_mid"][i]
                    if not (np.isfinite(mu) and np.isfinite(md)): continue
                    p_up = 0.5 * (mu + (1 - md))
                    r = sustained(d, wi, TLO, AMIN, AMAX, HOLD, p_up >= 0.5)
                    if r: tr.append(r)
                if len(tr) < 10: continue
                _, tt, _ = cluster_t([x["wi"] for x in tr], pnl_of(tr))
                grid.append((tt, TLO, HOLD, AMIN, AMAX, tr))
    grid.sort(reverse=True, key=lambda g: g[0])
    for tt, TLO, HOLD, AMIN, AMAX, tr in grid[:8]:
        print(f"  T>={TLO} hold={HOLD} ask[{AMIN},{AMAX}]: t={tt:+.2f} N={len(tr)}")
    obs = grid[0][0]
    null_max = []
    for s in range(400):
        rng = np.random.default_rng(9000 + s)
        perm = rng.permutation(ws_list); remap = dict(zip(ws_list, perm))
        best_t = -9.0
        for tt0, TLO, HOLD, AMIN, AMAX, tr in grid:
            nw2 = [1 if window_winner[remap[x["ws"]]] == x["side"] else 0 for x in tr]
            _, t2, _ = cluster_t([x["wi"] for x in tr], pnl_of(tr, nw2))
            best_t = max(best_t, t2)
        null_max.append(best_t)
    null_max = np.array(null_max)
    print(f"  observed max t = {obs:+.2f};  placebo max-t mean={null_max.mean():+.2f} "
          f"p95={np.percentile(null_max,95):+.2f}; p={float((null_max>=obs).mean()):.4f}")

    print("\n=== D) HALF-SPLIT on the best live cell ===")
    tt, TLO, HOLD, AMIN, AMAX, tr = grid[0]
    nw = len(W); cut = int(nw * 0.5)
    for lbl, sub in (("first half ", [x for x in tr if x["wi"] < cut]),
                     ("second half", [x for x in tr if x["wi"] >= cut])):
        p = pnl_of(sub); _, t2, w2 = cluster_t([x["wi"] for x in sub], p)
        print(f"  {lbl}: N={len(sub):>3} w={w2:>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={t2:+5.2f}")
    p = pnl_of(tr); srt = np.sort(p)[::-1]
    print(f"  best 5 trades = ${srt[:5].sum():+.2f} of ${p.sum():+.2f}")
    print(f"  total net = ${p.sum():+.2f} on ${STAKE*len(tr):.0f} staked "
          f"({p.sum()/(STAKE*len(tr))*100:+.2f}% ROI)")

    print("\n=== E) MECHANISM: market-implied vs realised, by T (all favourites) ===")
    for T in range(240, 300, 4):
        s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and r["ask"] >= 0.80]
        if len(s) < 8: continue
        s1 = first_per_window(s)
        w1 = np.array([r["won"] for r in s1]); m1 = np.array([r["mkt"] for r in s1])
        pm = np.array([r["p_model"] for r in s1]); ok = np.isfinite(pm)
        gap = w1.mean() - m1.mean()
        se = np.sqrt(max(m1.mean() * (1 - m1.mean()), 1e-9) / len(s1))
        print(f"  T={T:>3} n={len(s1):>3} mkt={m1.mean():.4f} realised={w1.mean():.4f} "
              f"gap={gap:+.4f} (t={gap/se if se>0 else 0:+.2f})  "
              f"twap_model={pm[ok].mean() if ok.any() else float('nan'):.4f}")


if __name__ == "__main__":
    main()
