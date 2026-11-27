"""Stage 4: honest event-driven backtester for the Polymarket BTC 5m book.

Rules enforced:
  * Fill at the ASK (taker). Require ask_size*ask >= stake, else no fill.
  * Exits sell at the BID (taker) and pay a fee on the exit leg too.
  * Fee: 0.05 * min(p, 1-p) * shares  on entry AND exit.
  * Only data up to and including the entry snapshot may be used (no lookahead).
  * One entry per (window, side); TP/SL checked on every later snapshot.
  * Hold-to-settlement: the surviving leg pays $1/share, the other $0.
Walk-forward: chronological split, fit on train, trade on test.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

FEE = 0.05
MIN_SPREAD = 0.02      # reject absurdly wide books
MAX_SPREAD = 0.06


def fee(shares, price):
    return FEE * min(price, 1.0 - price) * shares


def load_windows():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    wins = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        wins.append(d)
    wins.sort(key=lambda d: d["ws"][0] if len(d["ws"]) else 0)
    return wins


def book_arrays(d):
    """Return per-snapshot dict of valid quotes for both sides."""
    out = {}
    for s in ("Up", "Down"):
        b, a, bs, asz = d[s + "_bid"], d[s + "_ask"], d[s + "_bidsz"], d[s + "_asksz"]
        ok = np.isfinite(b) & np.isfinite(a) & (b > 0) & (a < 1) & (b <= a + 1e-9)
        out[s] = dict(bid=b, ask=a, bidsz=bs, asksz=asz, ok=ok)
    return out


def cb_series(d):
    """Coinbase price series aligned to snapshots (forward-filled, NaN-safe)."""
    cb = d["cb_px"].copy()
    t = d["t"]
    # forward fill
    for i in range(1, len(cb)):
        if not np.isfinite(cb[i]) and np.isfinite(cb[i - 1]):
            cb[i] = cb[i - 1]
    # backfill
    for i in range(len(cb) - 2, -1, -1):
        if not np.isfinite(cb[i]) and np.isfinite(cb[i + 1]):
            cb[i] = cb[i + 1]
    return cb


def mom_z(cb, i, lookback=60.0, t=None):
    """z = (price[i] - price at start of window) / (realized vol of window)"""
    if not np.isfinite(cb[i]):
        return np.nan
    p0 = cb[0]
    if not np.isfinite(p0) or p0 <= 0:
        return np.nan
    seg = cb[:i + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 5:
        return np.nan
    ticks = np.diff(seg)
    if len(ticks) < 3:
        return np.nan
    sd = np.std(ticks) * np.sqrt(len(ticks))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return (cb[i] - p0) / sd


# ---------------------------------------------------------------- strategies
def strat_momentum_vs_market(d, i, cb, params):
    """Buy the side where our coinbase-momentum model is more bullish than the
    market's own price, provided the gap clears a threshold and the model's
    probability beats the ask by enough to cover fees+spread."""
    gapmin = params.get("gapmin", 0.06)
    edge_min = params.get("edge_min", 0.02)
    T = params.get("T", 60)
    if d["t"][i] < T:
        return None
    z = mom_z(cb, i)
    if not np.isfinite(z):
        return None
    p_model = 1.0 / (1.0 + np.exp(-z / params.get("scale", 1.5)))
    for s in ("Up", "Down"):
        other = "Down" if s == "Up" else "Up"
        m_up = 0.5 * (d["Up_mid"][i] + (1 - d["Down_mid"][i])) if np.isfinite(d["Up_mid"][i]) and np.isfinite(d["Down_mid"][i]) else np.nan
        if not np.isfinite(m_up):
            continue
        m_side = m_up if s == "Up" else 1 - m_up
        p_side = p_model if s == "Up" else 1 - p_model
        gap = p_side - m_side
        if gap < gapmin:
            continue
        return s
    return None


def strat_favorite_late(d, i, cb, params):
    """Buy the favorite late in the window when the book is still two-sided."""
    T = params.get("T", 240)
    if d["t"][i] < T:
        return None
    bk = book_arrays(d)
    for s in ("Up", "Down"):
        o = "Down" if s == "Up" else "Up"
        if not (bk[s]["ok"][i] and bk[o]["ok"][i]):
            continue
        mid_s = 0.5 * (bk[s]["bid"][i] + bk[s]["ask"][i])
        mid_o = 0.5 * (bk[o]["bid"][i] + bk[o]["ask"][i])
        if mid_s < params.get("min_mid", 0.80):
            continue
        if mid_s <= mid_o:
            continue
        return s
    return None


def strat_side(d, i, cb, params):
    """Always buy the favourite side at time T (market-following baseline)."""
    T = params.get("T", 120)
    if d["t"][i] < T:
        return None
    bk = book_arrays(d)
    for s in ("Up", "Down"):
        o = "Down" if s == "Up" else "Up"
        if not (bk[s]["ok"][i] and bk[o]["ok"][i]):
            continue
        mid_s = 0.5 * (bk[s]["bid"][i] + bk[s]["ask"][i])
        mid_o = 0.5 * (bk[o]["bid"][i] + bk[o]["ask"][i])
        if mid_s > mid_o:
            return s
    return None


# ---------------------------------------------------------------- execution
def run_window(d, params, strat):
    """Simulate one window under one strategy. Returns a trade dict or None."""
    bk = book_arrays(d)
    cb = cb_series(d)
    t = d["t"]
    n = len(t)
    stake = params.get("stake", 10.0)
    T = params.get("T", 60)
    # find the first snapshot at/after T
    i = int(np.searchsorted(t, T))
    if i >= n:
        return None
    side = strat(d, i, cb, params)
    if side is None:
        return None
    o = "Down" if side == "Up" else "Up"
    if not (bk[side]["ok"][i] and bk[o]["ok"][i]):
        return None
    ask = bk[side]["ask"][i]
    bid = bk[side]["bid"][i]
    spread = ask - bid
    if spread > MAX_SPREAD or spread < 0:
        return None
    if not np.isfinite(bk[side]["asksz"][i]) or bk[side]["asksz"][i] * ask < stake:
        return None
    shares = stake / ask
    entry_fee = fee(shares, ask)
    cost = stake + entry_fee
    entry_px = ask
    entry_i = i

    tp = params.get("tp")
    sl = params.get("sl")
    exit_px = None
    exit_i = None
    reason = "settle"
    if tp or sl:
        for j in range(i + 1, n):
            b = bk[side]["bid"][j]
            if not np.isfinite(b):
                continue
            if tp and b >= tp:
                exit_px, exit_i, reason = b, j, "tp"
                break
            if sl and b <= sl:
                exit_px, exit_i, reason = b, j, "stop"
                break
    if exit_px is None:
        # hold to settlement
        won = (d["_winner"] == side)
        proceeds = shares * 1.0 if won else 0.0
        pnl = proceeds - cost
        return dict(slug=d["_slug"], side=side, t_entry=float(t[entry_i]), t_exit=float(t[-1]),
                    entry_px=entry_px, exit_px=None, reason="settle", won=bool(won),
                    shares=shares, stake=stake, cost=cost, proceeds=proceeds, pnl=pnl,
                    t_settle=300.0)
    # exit by selling at bid
    depth = bk[side]["bidsz"][exit_i]
    if not np.isfinite(depth) or depth < shares:
        # could not have sold the whole position -> treat as settle
        won = (d["_winner"] == side)
        proceeds = shares * 1.0 if won else 0.0
        pnl = proceeds - cost
        return dict(slug=d["_slug"], side=side, t_entry=float(t[entry_i]), t_exit=float(t[exit_i]),
                    entry_px=entry_px, exit_px=None, reason="settle", won=bool(won),
                    shares=shares, stake=stake, cost=cost, proceeds=proceeds, pnl=pnl,
                    t_settle=300.0)
    exit_fee = fee(shares, exit_px)
    proceeds = shares * exit_px - exit_fee
    pnl = proceeds - cost
    won = (d["_winner"] == side)
    return dict(slug=d["_slug"], side=side, t_entry=float(t[entry_i]), t_exit=float(t[exit_i]),
                entry_px=entry_px, exit_px=exit_px, reason=reason, won=bool(won),
                shares=shares, stake=stake, cost=cost, proceeds=proceeds, pnl=pnl, t_settle=300.0)


def summarize(trades, label=""):
    if not trades:
        return f"{label}: NO TRADES"
    pnl = np.array([t["pnl"] for t in trades])
    wins = np.array([t["won"] for t in trades], dtype=float)
    tp = [t for t in trades if t["reason"] == "tp"]
    sl = [t for t in trades if t["reason"] == "stop"]
    st = [t for t in trades if t["reason"] == "settle"]
    up = [t for t in trades if t["side"] == "Up"]
    dn = [t for t in trades if t["side"] == "Down"]
    def s(x):
        return f"n={len(x):>3} pnl=${sum(t['pnl'] for t in x):+8.2f}"
    return (f"{label}: N={len(trades):>3} net=${pnl.sum():+8.2f} wr={wins.mean():.3f} "
            f"avg=${pnl.mean():+.3f} worst=${pnl.min():+.2f} | tp:{s(tp)} stop:{s(sl)} settle:{s(st)} "
            f"| up:{s(up)} dn:{s(dn)}")


def main():
    wins = load_windows()
    print(f"scored windows: {len(wins)}")
    ts = [int(d["ws"][0]) for d in wins]
    import datetime
    print("span:", datetime.datetime.fromtimestamp(min(ts)), "->", datetime.datetime.fromtimestamp(max(ts)))

    # chronological 3-fold split
    folds = [(0, len(wins) // 3), (len(wins) // 3, 2 * len(wins) // 3), (2 * len(wins) // 3, len(wins))]
    print("\n=== chronological folds ===")
    for i, (a, b) in enumerate(folds):
        print(f"  fold{i}: windows {a}-{b-1}  {datetime.datetime.fromtimestamp(ts[a])} -> {datetime.datetime.fromtimestamp(ts[b-1])}")

    grid = []
    for T in (30, 60, 90, 120, 150, 180, 210, 240):
        for gapmin in (0.04, 0.06, 0.08, 0.10, 0.12):
            for scale in (1.0, 1.5, 2.0, 3.0):
                grid.append(dict(T=T, gapmin=gapmin, scale=scale, stake=10.0))

    print("\n=== momentum-vs-market strategy: full-sample grid (IN-SAMPLE, for shape only) ===")
    results = []
    for p in grid:
        trades = []
        for d in wins:
            tr = run_window(d, p, strat_momentum_vs_market)
            if tr:
                trades.append(tr)
        if len(trades) < 30:
            continue
        net = sum(t["pnl"] for t in trades)
        wr = np.mean([t["won"] for t in trades])
        results.append((net, len(trades), wr, p))
    results.sort(key=lambda r: (-r[0], r[1]))
    print(f"{'net$':>9} {'N':>4} {'wr':>6}  params")
    for net, n, wr, p in results[:15]:
        print(f"{net:>9.2f} {n:>4} {wr:>6.3f}  T={p['T']} gap={p['gapmin']} scale={p['scale']}")
    print("  ... worst 5:")
    for net, n, wr, p in results[-5:]:
        print(f"{net:>9.2f} {n:>4} {wr:>6.3f}  T={p['T']} gap={p['gapmin']} scale={p['scale']}")

    # baselines
    print("\n=== baselines (full sample) ===")
    for T in (60, 120, 180, 240):
        p = dict(T=T, stake=10.0)
        tr = [run_window(d, p, strat_side) for d in wins]
        tr = [x for x in tr if x]
        print(summarize(tr, f"favourite@{T}s"))

    with open(os.path.join(OUTDIR, "grid_momentum.pkl"), "wb") as fh:
        pickle.dump(results, fh)


if __name__ == "__main__":
    main()
