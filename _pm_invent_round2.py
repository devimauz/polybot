"""Three precommitted rules, design slice only. Holdout is not scored.

Priority if more than one passes every gate: lock, open, imbalance.
A rule passes only if, at both 0.5s and 1.0s fills:
  both chronological halves have N>=15 and net>0,
  pooled edge > 0, win rate > 0.50,
  Up and Down nets are both >= 0 with N>=8 each.
No threshold in this file is chosen from P&L.
"""
import glob
import os
import pickle

import numpy as np

import _pm_invent_design as S

LOG = S.LOG
CACHE = S.CACHE
OUT = r"C:/Users/user/Desktop/bot/_pm_out/invent_round2.txt"

# Physics priors from the settlement write-up, not fit to these P&Ls.
SIGMA_1S = 4.63
SIGMA_STRIKE = 7.62
NOISE_SCALE = 2.2


def load():
    slugs = {os.path.basename(p)[:-6] for p in glob.glob(os.path.join(LOG, "btc-updown-5m-*.jsonl"))}
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        slug = m.get("_slug")
        if slug not in slugs or not m.get("_winner"):
            continue
        z = np.load(os.path.join(CACHE, slug + ".npz"), allow_pickle=True)
        t = np.asarray(z["t_src"] if "t_src" in z.files else z["t"], float)
        d = {
            "slug": slug,
            "ws": int(np.asarray(z["ws"]).reshape(-1)[0]),
            "winner": str(m["_winner"]),
            "t": t,
            "cb": np.asarray(z["cb_px"], float),
            "cb_age": np.asarray(z["cb_age"], float),
        }
        for side in ("Up", "Down"):
            d[side + "_bid"] = np.asarray(z[side + "_bid"], float)
            d[side + "_ask"] = np.asarray(z[side + "_ask"], float)
            d[side + "_asksz"] = np.asarray(z[side + "_asksz"], float)
            d[side + "_bidsz"] = np.asarray(z[side + "_bidsz"], float)
        keep = (d["t"] >= -2) & (d["t"] <= 305)
        if keep.sum() < 20:
            continue
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.shape == keep.shape:
                d[k] = v[keep]
        out.append(d)
    out.sort(key=lambda x: x["ws"])
    return out


def try_fill(d, i, side, latency, ask_lo, ask_hi):
    t = d["t"]
    j = S.fill_idx(t, i, latency)
    if j is None:
        return None
    bid = d[side + "_bid"][j]
    ask = d[side + "_ask"][j]
    sz = d[side + "_asksz"][j]
    if not S.executable(bid, ask, sz):
        return None
    if not (ask_lo <= ask <= ask_hi):
        return None
    won = 1.0 if d["winner"] == side else 0.0
    fa = float(ask)
    return (float(S.taker_pnl(fa, won)), won, fa, d["ws"], side)


def trap(y, x):
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def strike_of(prev):
    t, cb = prev["t"], prev["cb"]
    m = (t >= 240.0) & (t <= 300.0) & np.isfinite(cb)
    if m.sum() < 5:
        return None
    tt, pp = t[m], cb[m]
    order = np.argsort(tt)
    tt, pp = tt[order], pp[order]
    dur = tt[-1] - tt[0]
    if dur < 20:
        return None
    return trap(pp, tt) / dur


def partial_settle(d, i):
    t, cb = d["t"], d["cb"]
    T = float(t[i])
    if not np.isfinite(cb[i]) or d["cb_age"][i] > 1.5:
        return None
    m = (t >= 240.0) & (t <= T) & np.isfinite(cb)
    if m.sum() < 4:
        return None
    tt, pp = t[m], cb[m]
    order = np.argsort(tt)
    tt, pp = tt[order], pp[order]
    if tt[-1] < T - 2.0:
        return None
    integ = trap(pp, tt)
    if tt[-1] < T:
        integ += float(pp[-1]) * (T - tt[-1])
    obs_dur = T - 240.0
    if obs_dur < 20:
        return None
    obs_avg = integ / obs_dur
    rem = 300.0 - T
    return (obs_avg * obs_dur + float(cb[i]) * rem) / 60.0


def sd_of(T):
    tau = min(60.0, max(0.0, 300.0 - T))
    return NOISE_SCALE * np.sqrt(SIGMA_1S ** 2 * tau / 3.0 + SIGMA_STRIKE ** 2)


def rule_lock(windows, latency):
    trades = []
    for i_w, d in enumerate(windows):
        if i_w == 0:
            continue
        strike = strike_of(windows[i_w - 1])
        if strike is None:
            continue
        t = d["t"]
        taken = False
        for i in range(len(t)):
            if taken:
                break
            if not (275.0 <= t[i] <= 292.0):
                continue
            est = partial_settle(d, i)
            if est is None:
                continue
            gap = est - strike
            if abs(gap) < 3.0 * sd_of(t[i]):
                continue
            side = "Up" if gap > 0 else "Down"
            ask = d[side + "_ask"][i]
            bid = d[side + "_bid"][i]
            sz = d[side + "_asksz"][i]
            # A real lock is the favourite. Do not buy a cheap side.
            if not (0.50 <= ask <= 0.93):
                continue
            if not S.executable(bid, ask, sz):
                continue
            rec = try_fill(d, i, side, latency, 0.50, 0.96)
            if rec is None:
                continue
            taken = True
            trades.append(rec)
    return trades


def rule_open(windows, latency):
    trades = []
    for d in windows:
        t, cb, age = d["t"], d["cb"], d["cb_age"]
        open_cb = None
        for i in range(len(t)):
            if t[i] > 3.0:
                break
            if t[i] >= 0 and np.isfinite(cb[i]) and np.isfinite(age[i]) and age[i] <= 1.0:
                open_cb = float(cb[i])
                break
        if open_cb is None:
            continue
        taken = False
        for i in range(len(t)):
            if taken:
                break
            if not (4.0 <= t[i] <= 12.0):
                continue
            if not (np.isfinite(cb[i]) and np.isfinite(age[i]) and age[i] <= 1.0):
                continue
            move = float(cb[i]) - open_cb
            if abs(move) < 15.0:
                continue
            side = "Up" if move > 0 else "Down"
            ask = d[side + "_ask"][i]
            bid = d[side + "_bid"][i]
            sz = d[side + "_asksz"][i]
            if not (0.42 <= ask <= 0.58):
                continue
            if not S.executable(bid, ask, sz):
                continue
            rec = try_fill(d, i, side, latency, 0.38, 0.62)
            if rec is None:
                continue
            taken = True
            trades.append(rec)
    return trades


def rule_imbalance(windows, latency):
    trades = []
    for d in windows:
        t = d["t"]
        taken = False
        for i in range(len(t)):
            if taken:
                break
            if not (30.0 <= t[i] <= 240.0):
                continue
            best = None
            for side in ("Up", "Down"):
                bid = d[side + "_bid"][i]
                ask = d[side + "_ask"][i]
                asz = d[side + "_asksz"][i]
                bsz = d[side + "_bidsz"][i]
                if not (np.isfinite(bsz) and np.isfinite(asz) and asz > 0):
                    continue
                if bsz < 4.0 * asz:
                    continue
                if not (0.45 <= ask <= 0.70):
                    continue
                if (ask - bid) > 0.02:
                    continue
                if not S.executable(bid, ask, asz):
                    continue
                best = side
                break
            if best is None:
                continue
            rec = try_fill(d, i, best, latency, 0.42, 0.73)
            if rec is None:
                continue
            taken = True
            trades.append(rec)
    return trades


def summarize(label, trades, design_ws):
    lines = []
    if not trades:
        return ["%s  n=0" % label]
    pnl = np.array([r[0] for r in trades])
    won = np.array([r[1] for r in trades])
    ask = np.array([r[2] for r in trades])
    ws = np.array([r[3] for r in trades])
    side = np.array([r[4] for r in trades])
    edge = won - ask - S.fee_per_share(ask)
    mid = design_ws[len(design_ws) // 2]
    h1 = ws < mid
    h2 = ~h1
    up = side == "Up"
    m, tstat, nw = _cluster(ws, pnl)
    lo, hi = _boot(ws, pnl)
    stake = len(trades) * S.STAKE
    lines.append("%s" % label)
    lines.append("  N=%d windows=%d wr=%.1f%% ask=%.1fc edge=%+.2fc net=%+.2f ROI=%+.1f%%" % (
        len(trades), nw, 100 * won.mean(), 100 * ask.mean(), 100 * edge.mean(),
        pnl.sum(), 100 * pnl.sum() / stake))
    lines.append("  h1 N=%d net=%+.2f edge=%+.2fc | h2 N=%d net=%+.2f edge=%+.2fc" % (
        h1.sum(), pnl[h1].sum(), 100 * edge[h1].mean() if h1.any() else np.nan,
        h2.sum(), pnl[h2].sum(), 100 * edge[h2].mean() if h2.any() else np.nan))
    lines.append("  Up N=%d net=%+.2f wr=%.1f%% | Down N=%d net=%+.2f wr=%.1f%%" % (
        up.sum(), pnl[up].sum() if up.any() else 0, 100 * won[up].mean() if up.any() else 0,
        (~up).sum(), pnl[~up].sum() if (~up).any() else 0, 100 * won[~up].mean() if (~up).any() else 0))
    lines.append("  cluster-t=%.2f CI/trade [%.3f, %.3f] worst=%.2f" % (tstat, lo, hi, pnl.min()))
    ok = (h1.sum() >= 15 and h2.sum() >= 15 and pnl[h1].sum() > 0 and pnl[h2].sum() > 0
          and edge.mean() > 0 and won.mean() > 0.50
          and up.sum() >= 8 and (~up).sum() >= 8 and pnl[up].sum() >= 0 and pnl[~up].sum() >= 0)
    lines.append("  GATES %s" % ("PASS" if ok else "FAIL"))
    return lines


def _cluster(ws, pnl):
    uq = np.unique(ws)
    s = np.array([pnl[ws == w].sum() for w in uq])
    c = np.array([(ws == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se and se > 0 else np.nan), len(uq)


def _boot(ws, pnl, n=3000, seed=11):
    rng = np.random.default_rng(seed)
    uq = np.unique(ws)
    tot = {w: pnl[ws == w].sum() for w in uq}
    cnt = {w: int((ws == w).sum()) for w in uq}
    keys = list(tot)
    out = [sum(tot[k] for k in pick) / sum(cnt[k] for k in pick)
           for pick in (rng.choice(keys, size=len(keys), replace=True) for _ in range(n))]
    return np.percentile(out, [2.5, 97.5])


def main():
    W = load()
    n_des = int(len(W) * 0.60)
    design = W[:n_des]
    dws = [d["ws"] for d in design]
    print("design", len(design), "holdout untouched", len(W) - n_des, flush=True)
    lines = []
    lines.append("design=%d holdout_untouched=%d ws %d..%d" % (
        len(design), len(W) - n_des, design[0]["ws"], design[-1]["ws"]))
    lines.append("rules are precommitted; gates must pass at 0.5s AND 1.0s")
    lines.append("")
    rules = [
        ("LOCK 3-sigma partial TWAP, ask 0.50-0.93, t 275-292", rule_lock),
        ("OPEN |cb-open|>=$15 in t=4..12, ask 0.42-0.58", rule_open),
        ("IMBALANCE bidsz>=4*asksz, ask 0.45-0.70, t 30-240", rule_imbalance),
    ]
    passed = []
    for name, fn in rules:
        print("run", name, flush=True)
        for lat in (0.5, 1.0):
            trades = fn(design, lat)
            block = summarize("%s  fill=%.1fs" % (name, lat), trades, dws)
            lines.extend(block)
            if block[-1].endswith("PASS") and lat == 0.5:
                passed.append(name)
        lines.append("")
    lines.append("0.5s gate-pass (still needs the 1.0s block to say PASS): %s" % (passed or "none"))
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
