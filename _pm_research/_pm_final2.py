"""Stage 34: FINAL UNIFIED EVALUATION + MECHANISM (time-weighted, corrected).

FIXES the units bug that made the mechanism residual absurd (0.36 in log
terms): the settlement is a TIME-average of price over [240,300) (60 seconds),
but I was summing ~214 price samples.  Everything here uses elapsed time.

Also fixes the live-realistic simulator: the earlier version's "sustained"
walk-back was wrong and produced a spurious N=4..12 grid.  Here the entry
rule is explicit and the trade is taken at the FIRST snapshot that satisfies
it (no look-ahead, no retrospective selection).
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


# ---------------------------------------------------------------- mechanism
def mechanism():
    W = load_all()
    by_ws = {d["ws"]: d for d in W}
    print("=== MECHANISM: time-weighted settlement & conditional model ===")
    # strike = time-average of price over the PREVIOUS window's last 60 s
    strike = {}
    for d in W:
        p = by_ws.get(d["ws"] - 300)
        if p is None: continue
        tt, cc = p["t"], ffill(p["cb_px"])
        m = (tt >= 240) & np.isfinite(cc) & (cc > 0)
        if m.sum() >= 20:
            strike[d["ws"]] = float(np.mean(cc[m]))
    agree = ok = 0
    for d in W:
        if d["ws"] not in strike: continue
        cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
        t = d["t"]
        m = (t >= 240) & np.isfinite(cb)
        if m.sum() < 20: continue
        Q = float(np.mean(cb[m])); ok += 1
        if (Q > strike[d["ws"]]) == (d["_winner"] == "Up"): agree += 1
    print(f"  settlement reconstruction: {agree}/{ok} = {agree/max(ok,1):.4f} correct")

    # conditional residual: log Q - log E[Q|F_T]  (time-weighted)
    print("\n  T    n     rms(logQ - logE[Q|F_T])   median|resid|")
    for T in (240, 250, 260, 270, 280, 290):
        res = []
        for d in W:
            if d["ws"] not in strike: continue
            cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
            t = d["t"]
            i = int(np.searchsorted(t, T))
            if i >= len(t): continue
            full = (t >= 240) & np.isfinite(cb)
            if full.sum() < 20: continue
            Q = float(np.mean(cb[full]))
            obs_m = (t >= 240) & (t <= T) & np.isfinite(cb)
            if obs_m.sum() < 5 or not np.isfinite(cb[i]): continue
            # time-average of observed part
            dt = np.diff(t[obs_m]); pv = cb[obs_m][:-1]
            obs_avg = float(np.sum(pv * dt) / max(dt.sum(), 1e-9)) if dt.sum() > 0 else float(cb[i])
            elapsed = float(t[i] - 240.0)
            eQ = (obs_avg * elapsed + (300.0 - T) * float(cb[i])) / 60.0
            if eQ > 0: res.append(np.log(Q) - np.log(eQ))
        if len(res) < 8: continue
        res = np.array(res)
        print(f"  {T:>3} {len(res):>5} {np.sqrt(np.mean(res**2)):>28.6f} {np.median(np.abs(res)):>14.6f}")

    # per-second vol from the log price
    print("\n  implied model sd (per-second vol * sqrt((300-T)^2/7200)) vs observed rms:")
    for T in (240, 250, 260, 270, 280, 290):
        rows = []
        for d in W:
            if d["ws"] not in strike: continue
            cb = ffill(d["cb_px"]); cb = np.where(cb > 0, cb, np.nan)
            t = d["t"]; i = int(np.searchsorted(t, T))
            if i >= len(t): continue
            full = (t >= 240) & np.isfinite(cb)
            if full.sum() < 20: continue
            Q = float(np.mean(cb[full]))
            obs_m = (t >= 240) & (t <= T) & np.isfinite(cb)
            if obs_m.sum() < 5 or not np.isfinite(cb[i]): continue
            dt = np.diff(t[obs_m]); pv = cb[obs_m][:-1]
            obs_avg = float(np.sum(pv * dt) / max(dt.sum(), 1e-9)) if dt.sum() > 0 else float(cb[i])
            elapsed = float(t[i] - 240.0)
            eQ = (obs_avg * elapsed + (300.0 - T) * float(cb[i])) / 60.0
            if eQ <= 0: continue
            lr = np.diff(np.log(cb[obs_m])); lr = lr[np.isfinite(lr)]
            if len(lr) < 5: continue
            dts = np.diff(t[obs_m][:-1]); dts = dts[np.isfinite(dts)]
            if len(dts) != len(lr) or dts.sum() <= 0: continue
            sig2 = float(np.sum(lr ** 2) / dts.sum())
            rows.append((np.log(Q) - np.log(eQ), sig2, 300.0 - T))
        if len(rows) < 8: continue
        a = np.array(rows)
        rms = float(np.sqrt(np.mean(a[:, 0] ** 2)))
        sbar = float(np.sqrt(np.median(a[:, 1])))
        rem = 300.0 - T
        model_sd = sbar * rem / np.sqrt(7200.0)
        print(f"  T={T:>3} rms_resid={rms:.6f}  model_sd={model_sd:.6f}  "
              f"ratio={rms/max(model_sd,1e-12):.3f}  (per-sec sd={sbar:.6f})")


# ------------------------------------------------------------------ trading
def build_rows():
    W = load_all()
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]; n = len(t)
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n: continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998): continue
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                if not (np.isfinite(o) and 0.001 < o < 0.999): continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), mkt=float(mkt),
                                 won=1 if d["_winner"] == side else 0))
    return W, rows


def pnl_of(rs, won=None):
    won = np.array([r["won"] for r in rs]) if won is None else np.asarray(won, float)
    ask = np.array([r["ask"] for r in rs])
    sh = STAKE / ask
    return sh * won - STAKE - FEE(ask, sh)


def first_per_window(rs):
    best = {}
    for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    return list(best.values())


def main():
    mechanism()
    W, rows = build_rows()
    print(f"\n=== TRADING ===  windows {len(W)}  rows {len(rows)}")
    window_winner = {d["ws"]: d["_winner"] for d in W}
    ws_list = [d["ws"] for d in W]

    def report(name, sel, won=None):
        if not sel: return None
        wi = np.array([r["wi"] for r in sel])
        w = np.array([r["won"] for r in sel]) if won is None else np.asarray(won, float)
        ask = np.array([r["ask"] for r in sel])
        p = pnl_of(sel, w)
        m, t, nwin = cluster_t(wi, p)
        lo, hi = boot_ci(wi, p)
        k, n = int(w.sum()), len(w)
        pv = float(binom.sf(k - 1, n, ask.mean())) if n else 1.0
        print(f"  {name:<40} N={n:>3} win={w.mean():.4f} ask={ask.mean():.4f} "
              f"tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={t:+5.2f} "
              f"CI=[{lo:+.2f},{hi:+.2f}] binom_p={pv:.4f}")
        return t

    print("\n--- grid (first snapshot per window, both sides live) ---")
    grid = {}
    for TLO in (240, 250, 255, 260, 265, 270):
        for AMIN in (0.70, 0.80, 0.85, 0.90):
            sel = first_per_window([r for r in rows if r["T"] >= TLO and r["mkt"] >= 0.5
                                    and AMIN <= r["ask"] < 0.99])
            if len(sel) < 15: continue
            t = report(f"T>={TLO} ask[{AMIN:.2f},0.99)", sel)
            if t is not None: grid[(TLO, AMIN)] = (t, sel)

    best_key = max(grid, key=lambda k: grid[k][0])
    print(f"\n  best cell: {best_key} t={grid[best_key][0]:+.2f}")

    print("\n--- max-t family-wise correction over the grid ---")
    null_max = []
    for s in range(3000):
        rng = np.random.default_rng(7000 + s)
        perm = rng.permutation(ws_list); remap = dict(zip(ws_list, perm))
        best = -9.0
        for key, (t0, sel) in grid.items():
            nw2 = [1 if window_winner[remap[r["ws"]]] == r["side"] else 0 for r in sel]
            _, t2, _ = cluster_t([r["wi"] for r in sel], pnl_of(sel, nw2))
            best = max(best, t2)
        null_max.append(best)
    null_max = np.array(null_max)
    print(f"  observed max t = {grid[best_key][0]:+.2f}   placebo max-t mean={null_max.mean():+.2f} "
          f"p95={np.percentile(null_max,95):+.2f}  p={float((null_max>=grid[best_key][0]).mean()):.4f}")

    sel = grid[best_key][1]
    print("\n--- stability of the best cell ---")
    nw = len(W); cut = int(nw * 0.5)
    for lbl, sub in (("first half ", [r for r in sel if r["wi"] < cut]),
                     ("second half", [r for r in sel if r["wi"] >= cut])):
        p = pnl_of(sub); _, t2, w2 = cluster_t([r["wi"] for r in sub], p)
        print(f"  {lbl}: N={len(sub):>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f} t={t2:+5.2f}")
    order = sorted(range(len(sel)), key=lambda i: sel[i]["wi"])
    for bi, blk in enumerate(np.array_split(np.array(order), 8)):
        sub = [sel[i] for i in blk]
        if not sub: continue
        p = pnl_of(sub)
        print(f"    block {bi}: N={len(sub):>3} tot=${p.sum():+7.2f} avg=${p.mean():+.3f}")
    p = pnl_of(sel); srt = np.sort(p)[::-1]
    print(f"  best 5 trades ${srt[:5].sum():+.2f} of ${p.sum():+.2f} total "
          f"({srt[:5].sum()/p.sum()*100:.0f}%)")
    print(f"  ROI on stake: {p.sum()/(STAKE*len(sel))*100:+.2f}%")
    print(f"  losses: {sum(1 for r in sel if r['won']==0)} / {len(sel)}")

    print("\n--- exit variants on the best cell (can we cut the tail?) ---")
    for label, stop in (("hold to settlement", None), ("stop 0.70", 0.70),
                        ("stop 0.80", 0.80), ("stop 0.90", 0.90), ("TP 0.99", "tp99"),
                        ("TP 0.995", "tp995")):
        pnls = []
        for r in sel:
            sh = STAKE / r["ask"]
            pnl = None
            if stop is None or stop == "tp99" or stop == "tp995":
                pnl = sh * r["won"] - STAKE - FEE(r["ask"], sh)
            else:
                pnl = -STAKE - FEE(r["ask"], sh)          # stopped out at the stop price
            pnls.append(pnl)
        pnls = np.array(pnls)
        m, t, w2 = cluster_t([r["wi"] for r in sel], pnls)
        print(f"  {label:<20} tot=${pnls.sum():+7.2f} avg=${pnls.mean():+.3f} t={t:+5.2f}")


if __name__ == "__main__":
    main()
