"""Design-slice diagnostics only. Does not touch the holdout."""
import numpy as np
import _pm_invent_design as S

W = S.load_windows()
design = W[: int(len(W) * 0.60)]
dws = [d["ws"] for d in design]
mid = dws[len(dws) // 2]


def boot_ci(ws, pnl, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    ws = np.asarray(ws)
    pnl = np.asarray(pnl, float)
    uq = np.unique(ws)
    tot = {w: pnl[ws == w].sum() for w in uq}
    cnt = {w: int((ws == w).sum()) for w in uq}
    keys = list(tot)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(sum(tot[k] for k in pick) / sum(cnt[k] for k in pick))
    return np.percentile(out, [2.5, 97.5])


def cluster_t(ws, pnl):
    ws = np.asarray(ws)
    pnl = np.asarray(pnl, float)
    uq = np.unique(ws)
    s = np.array([pnl[ws == w].sum() for w in uq])
    c = np.array([(ws == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se and se > 0 else np.nan), len(uq)


def placebo(ws, won, ask, n=2000, seed=3):
    """Shuffle window winners. p = fraction of shuffles with edge >= observed."""
    rng = np.random.default_rng(seed)
    ws = np.asarray(ws)
    won = np.asarray(won, float)
    ask = np.asarray(ask, float)
    obs = (won - ask - S.fee_per_share(ask)).mean()
    # one win-bit per window (a window can contribute two sides; keep pairing)
    # simpler: permute the won flags across trades
    ge = 0
    for _ in range(n):
        sh = rng.permutation(won)
        e = (sh - ask - S.fee_per_share(ask)).mean()
        if e >= obs - 1e-15:
            ge += 1
    return obs, ge / n


def report(label, trades, maker=False):
    if not trades:
        print(label, "n=0")
        return
    pnl = np.array([r[0] for r in trades])
    won = np.array([r[1] for r in trades])
    px = np.array([r[2] for r in trades])
    ws = np.array([r[3] for r in trades])
    side = np.array([r[4] for r in trades])
    edge = won - px - (0 if maker else S.fee_per_share(px))
    m, t, nw = cluster_t(ws, pnl)
    lo, hi = boot_ci(ws, pnl)
    # losing streak
    streak = worst = 0
    for x in pnl:
        streak = streak + 1 if x < 0 else 0
        worst = max(worst, streak)
    # drawdown on cumulative
    c = np.cumsum(pnl)
    dd = (np.maximum.accumulate(c) - c).max() if len(c) else 0
    up = side == "Up"
    print("%s" % label)
    print("  N=%d windows=%d wr=%.1f%% px=%.1fc edge=%+.2fc net=%+.2f" % (
        len(trades), nw, 100 * won.mean(), 100 * px.mean(), 100 * edge.mean(), pnl.sum()))
    print("  cluster-t=%.2f  mean/trade CI [%.3f, %.3f]" % (t, lo, hi))
    print("  worst=%.2f streak=%d maxDD=%.2f" % (pnl.min(), worst, dd))
    print("  Up N=%d wr=%.1f%% net=%+.2f | Down N=%d wr=%.1f%% net=%+.2f" % (
        up.sum(), 100 * won[up].mean() if up.any() else 0, pnl[up].sum(),
        (~up).sum(), 100 * won[~up].mean() if (~up).any() else 0, pnl[~up].sum()))
    if not maker:
        _, p = placebo(ws, won, px)
        print("  outcome-placebo p(edge>=obs)=%.3f" % p)


# rebuild the 180-240 / 0.20-0.35 taker cell and its price neighbor
cells = S.surface(design, 0.5)
for key, name in [((3, 1), "taker 180-240 ask 0.20-0.35"),
                  ((3, 0), "taker 180-240 ask 0.10-0.20"),
                  ((4, 2), "taker 240-270 ask 0.35-0.50")]:
    rows = cells.get(key, [])
    trades = [(r[1], r[2], r[3], r[4], r[5]) for r in rows]
    report(name, trades)

print("--- lead follow already measured; fade control, same grid ---")


def fade(windows, move_min, ask_cap, latency):
    """Buy the side AGAINST the 2s coinbase move, when that side's ask <= cap."""
    trades = []
    for d in windows:
        t = d["t"]
        cb = d["cb"]
        age = d["cb_age"]
        taken = False
        for i in range(len(t)):
            if taken:
                break
            if not (5.0 <= t[i] <= 250.0):
                continue
            if not (np.isfinite(cb[i]) and np.isfinite(age[i]) and age[i] <= 1.0):
                continue
            j0 = int(np.searchsorted(t, t[i] - 2.0, side="right")) - 1
            if j0 < 0 or not np.isfinite(cb[j0]) or t[i] - t[j0] < 1.5:
                continue
            move = cb[i] - cb[j0]
            if abs(move) < move_min:
                continue
            side = "Down" if move > 0 else "Up"
            bid = d[side + "_bid"]
            ask = d[side + "_ask"]
            sz = d[side + "_asksz"]
            if ask[i] > ask_cap:
                continue
            if not S.executable(bid[i], ask[i], sz[i]):
                continue
            j = S.fill_idx(t, i, latency)
            if j is None or not S.executable(bid[j], ask[j], sz[j]):
                continue
            if ask[j] > ask_cap + 0.02:
                continue
            taken = True
            won = 1.0 if d["winner"] == side else 0.0
            fa = float(ask[j])
            trades.append((float(S.taker_pnl(fa, won)), won, fa, d["ws"], side))
    return trades


for mv in S.LEAD_MOVES:
    for cap in S.LEAD_CAPS:
        tr = fade(design, mv, cap, 0.5)
        if not tr:
            print("fade mv>=%.0f cap<=%.2f n=0" % (mv, cap))
            continue
        pnl = np.array([r[0] for r in tr])
        ws = np.array([r[3] for r in tr])
        h1, h2 = pnl[ws < mid].sum(), pnl[ws >= mid].sum()
        print("fade mv>=%.0f cap<=%.2f N=%d net=%+.2f h1=%+.2f h2=%+.2f" % (
            mv, cap, len(tr), pnl.sum(), h1, h2))

print("--- lead follow detail mv>=25 cap<=0.60 ---")
report("lead follow 0.5s", S.lead(design, 25, 0.60, 0.5))
report("lead follow 1.0s", S.lead(design, 25, 0.60, 1.0))
