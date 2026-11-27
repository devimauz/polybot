"""btc5m_endgame_backtest.py - DEFINITIVE backtest of the Polymarket BTC 5-minute
"endgame favourite" strategy on the logged order books.

WHY THIS STRATEGY
-----------------
Polymarket 5-minute crypto binaries settle on the Chainlink 60-second TWAP:
  * the PRICE TO BEAT is the 60s TWAP immediately BEFORE the window opens
  * the SETTLEMENT is the 60s TWAP over the window's final 60 seconds
  (verified on this dataset: reconstructing the settlement from the logged
  coinbase path reproduces the recorded resolution in 263/269 windows = 97.8%)

Because the settlement averages 60 seconds of prices, part of the settlement
window is already OBSERVED and LOCKED IN before the window ends.  Near expiry
the market's own price for the favourite does not fully reflect that lock-in,
so the favourite is systematically underpriced in the last ~40 seconds.

The strategy: once the window is in its final seconds, buy the FAVOURITE side
(its ask is high, so the market clearly expects it to win) when its ask sits
in a band, and hold to settlement.

This script is the reproducible evidence.  Everything it prints is measured
from the raw L2 books in price_logs/polymarket/ - no synthetic data.

Run:  python btc5m_endgame_backtest.py
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
LOGDIR = os.path.join(ROOT, "price_logs", "polymarket")

# ---- fee model -----------------------------------------------------------
# Polymarket taker fee for crypto 5m binaries: fee = C * 0.07 * p * (1 - p),
# where C is the number of shares and p the price paid.
FEE_RATE = 0.07
def fee(price, shares):
    return shares * FEE_RATE * price * (1.0 - price)

STAKE = 5.0           # dollars per trade in the backtest
STAKE_CHEAP = 15.0    # 3x stake when the favourite's ask is below CHEAP_ASK
CHEAP_ASK = 0.85      # cheap-bucket threshold
MIN_DEPTH_USD = 10.0  # require this much displayed size at the ask
MAX_SPREAD = 0.04     # require a tight two-sided book


def stake_for_ask(ask):
    """The stake for a favourite quote at `ask`.

    Imported from the bot so there is a SINGLE source of truth for sizing:
    the backtest, the replay harness and the live bot must size a trade
    identically or the replay proves nothing.
    """
    from btc5m_endgame_bot import stake_for_ask as _sfa
    return _sfa(ask)


# ---------------------------------------------------------------- utilities
def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]):
            last = a[i]
        elif np.isfinite(last):
            a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]):
            nxt = a[i]
        elif np.isfinite(nxt):
            a[i] = nxt
    return a


def load_all():
    """Load the compact per-window cache produced by _pm_extract.py.

    Uses the SOURCE-ALIGNED clock (t_src) by default.  The logger stamps
    each row before its REST calls complete, so `seconds_into_window` is
    ~1 s optimistic; using it would give the backtest a look-ahead.
    """
    mp = os.path.join(CACHE, "manifest.pkl")
    if not os.path.exists(mp):
        sys.exit("missing %s - run _pm_extract.py first" % mp)
    with open(mp, "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        # the causal clock: when the frame's data actually became observable
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


def cluster_t(wi, pnl):
    """Mean P&L per trade with a window-clustered t-statistic.

    Several quotes can come from the same 5-minute window, so observations are
    correlated.  The t-statistic is therefore computed on per-window P&L sums.
    """
    wi = np.asarray(wi)
    pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se > 0 else np.nan), len(uq)


def boot_ci(wi, pnl, n=8000, seed=11):
    """Window bootstrap CI for the mean P&L per trade."""
    rng = np.random.default_rng(seed)
    wi = np.asarray(wi)
    pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    tot = {w: pnl[wi == w].sum() for w in uq}
    cnt = {w: (wi == w).sum() for w in uq}
    keys = list(tot)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(sum(tot[k] for k in pick) / sum(cnt[k] for k in pick))
    return np.percentile(out, [2.5, 97.5])


def pnl_of(ask, won, stake=STAKE):
    ask = np.asarray(ask, float)
    won = np.asarray(won, float)
    sh = stake / ask
    return sh * won - stake - fee(ask, sh)


def pnl_sized(ask, won, stake=STAKE, stake_cheap=None, cheap_ask=0.85, fee_mult=1.0):
    """P&L with the conditional sizing overlay.

    A cheaper favourite carries more expected profit per dollar (EV per
    dollar = (p - ask)/ask at a roughly constant win rate), so the cheap
    bucket gets a larger stake.  This does NOT require the cheap bucket to
    win more often - the outcome placebo found no win-rate gap (p=0.66) and
    none is needed.
    """
    ask = np.asarray(ask, float)
    won = np.asarray(won, float)
    if stake_cheap is None:
        stake_cheap = stake * 3.0
    if cheap_ask is None:
        cheap_ask = CHEAP_ASK
    st = np.where(ask < cheap_ask, stake_cheap, stake).astype(float)
    sh = st / ask
    return sh * won - st - fee_mult * fee(ask, sh), st


# ------------------------------------------------------------ quote building
def build_rows(W, latency_s=0.0):
    """Every endgame quote where BOTH legs are live and executable.

    A quote qualifies only if the traded side has a finite, sane two-sided
    book, enough displayed size at the ask, a tight spread, and the opposite
    leg is also still trading (both legs alive = the market has not already
    resolved and locked the book).

    latency_s: a taker order decided on this frame does not fill at this
    frame's ask - it fills at the ask observed `latency_s` seconds later.
    With latency_s > 0 the fill price is taken from the first frame at or
    after t + latency_s, and the trade is DROPPED if the book has moved out
    of the band or the ask is no longer there (a realistic FOK outcome).
    """
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]
        n = len(t)
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n:
                continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)):
                continue
            p_up = 0.5 * (mu + (1.0 - md))
            if not (0.002 < p_up < 0.998):
                continue
            for side in ("Up", "Down"):
                ask = d[side + "_ask"][i]
                bid = d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)):
                    continue
                if not (0 < bid <= ask < 1):
                    continue
                if (ask - bid) > MAX_SPREAD:
                    continue
                if asz * ask < MIN_DEPTH_USD:
                    continue
                # The stake must be FILLABLE: a FOK order for more dollars
                # than the book shows at the ask is rejected for insufficient
                # size.  The conditional sizing overlay stakes up to 3x the
                # base, so the depth must cover THAT stake.
                if asz * ask < stake_for_ask(ask):
                    continue
                o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                if not (np.isfinite(o) and 0.001 < o < 0.999):
                    continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                fill_ask = ask
                if latency_s > 0:
                    j = int(np.searchsorted(t, T + latency_s))
                    if j >= n:
                        continue
                    f_ask = d[side + "_ask"][j]
                    f_bid = d[side + "_bid"][j]
                    f_asz = d[side + "_asksz"][j]
                    if not all(np.isfinite(v) for v in (f_ask, f_bid, f_asz)):
                        continue
                    if not (0 < f_bid <= f_ask < 1):
                        continue
                    if f_asz * f_ask < MIN_DEPTH_USD:
                        continue
                    fill_ask = f_ask
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side,
                                 ask=float(fill_ask), bid=float(bid), mkt=float(mkt),
                                 asksz=float(asz), i=i,
                                 won=1 if d["_winner"] == side else 0))
    return rows


def first_per_window(rs):
    best = {}
    for r in sorted(rs, key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    return list(best.values())


# ------------------------------------------------------------------ reports
# The SHIPPING rule.  Chosen on the first 60% of windows and then verified on
# the untouched last 40% (see _pm_ship_rule.py and _pm_oos_sweep.py).
# T>=262 with the ask band capped at 0.96 is the only candidate whose
# bootstrap CI95 excludes zero on the source-aligned clock.
SHIP_T_MIN = 262
SHIP_BAND = (0.80, 0.96)


def report_strategy(rows, W, t_min=260, band=(0.80, 0.99), sustained=0.0,
                    hold_secs=5.0, label=""):
    """Backtest one rule configuration.

    sustained > 0 additionally requires the favourite's ask to have stayed
    inside `band` for `sustained` seconds continuously (this is what a live
    bot can actually do - it cannot react to a single 0.3s print).
    """
    cand = [r for r in rows if r["T"] >= t_min and r["mkt"] >= 0.5
            and band[0] <= r["ask"] < band[1]]
    if sustained > 0:
        by_wi = {}
        for r in cand:
            by_wi.setdefault(r["wi"], []).append(r)
        keep = []
        for wi, rs in by_wi.items():
            d = W[wi]
            t = d["t"]
            rs = sorted(rs, key=lambda r: r["T"])
            for k, r in enumerate(rs):
                i = int(np.searchsorted(t, r["T"]))
                j = i
                while j > 0 and (t[i] - t[j - 1]) <= hold_secs:
                    j -= 1
                seg = rs[k:]
                if len(seg) < 2:
                    continue
                # the ask must stay in band for `sustained` seconds
                ok = True
                t0 = r["T"]
                for r2 in rs:
                    if r2["T"] - t0 > sustained + 1e-9:
                        break
                    if not (band[0] <= r2["ask"] < band[1]):
                        ok = False
                        break
                if ok and (rs[-1]["T"] - t0) >= sustained - 1e-9:
                    keep.append(r)
        cand = first_per_window(keep)
    else:
        cand = first_per_window(cand)
    if not cand:
        print(f"  {label:<44} no trades")
        return None
    ask = np.array([r["ask"] for r in cand])
    won = np.array([r["won"] for r in cand])
    p = pnl_of(ask, won)
    m, t, nw = cluster_t([r["wi"] for r in cand], p)
    lo, hi = boot_ci([r["wi"] for r in cand], p)
    pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
    print(f"  {label:<44} N={len(cand):>3} w={int(won.sum()):>3} ask={ask.mean():.4f} "
          f"real={won.mean():.4f} net=${p.sum():+7.2f} avg=${p.mean():+.3f} "
          f"t={t:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] bp={pv:.4f} worst=${p.min():+.2f}")
    return dict(cand=cand, pnl=p, t=t, n=len(cand))


def main():
    W = load_all()
    print("=" * 100)
    print("BTC 5-MIN ENDGAME FAVOURITE - definitive backtest")
    print("=" * 100)
    print(f"windows (resolved): {len(W)}   "
          f"span: {W[0]['ws']} .. {W[-1]['ws']}  "
          f"({(W[-1]['ws'] - W[0]['ws']) / 3600.0:.1f} h)")
    rows = build_rows(W)
    print(f"executable endgame quotes (both legs live): {len(rows)}")

    # ---- 1. the rule grid ------------------------------------------------
    print("\n=== 1) RULE GRID (first qualifying quote per window) ===")
    print("  rule                                          N   w  ask    real     net$    avg$     t"
          "     CI95            binom_p  worst$")
    grid = []
    for t_min in (250, 255, 258, 260, 262, 265, 268, 270):
        for band in ((0.80, 0.99), (0.80, 0.96), (0.82, 0.95), (0.85, 0.95), (0.90, 0.99)):
            lab = f"T>={t_min}s fav ask[{band[0]:.2f},{band[1]:.2f})"
            r = report_strategy(rows, W, t_min, band, 0.0, 5.0, lab)
            if r:
                grid.append((r["t"], lab, r))

    # ---- 2. family-wise correction --------------------------------------
    print("\n=== 2) MULTIPLE-TESTING (family-wise max-t) ===")
    n_rules = len(grid)
    print(f"  The grid above is a search over {n_rules} rules.  A permutation test that")
    print("  shuffles the OUTCOME label across windows measures how often a")
    print("  random rule set produces a max-t this large.")
    rng = np.random.default_rng(7)
    fav = [r for r in rows if r["mkt"] >= 0.5]
    by_wi = {}
    for r in fav:
        by_wi.setdefault(r["wi"], []).append(r)
    wi_list = sorted(by_wi)
    max_t_null = []
    for _ in range(600):
        # Shuffle the OUTCOME label across windows.  Each window's realised
        # winner is reassigned to another window's quote set, which destroys
        # any link between the market price and the resolution while keeping
        # the marginal win rate of each price band intact.
        perm = list(wi_list)
        rng.shuffle(perm)
        wperm = {perm[k]: wi_list[k] for k in range(len(wi_list))}
        ts = []
        for t_min in (250, 255, 258, 260, 262, 265, 268, 270):
            for band in ((0.80, 0.99), (0.80, 0.96), (0.82, 0.95), (0.85, 0.95), (0.90, 0.99)):
                c = []
                for src_wi in wi_list:
                    for r in by_wi[src_wi]:
                        if r["T"] < t_min or not (band[0] <= r["ask"] < band[1]):
                            continue
                        r2 = dict(r)
                        r2["won"] = 1 if W[wperm[src_wi]]["_winner"] == r["side"] else 0
                        c.append(r2)
                c = first_per_window(c)
                if len(c) < 5:
                    continue
                p = pnl_of([r["ask"] for r in c], [r["won"] for r in c])
                _, t, _ = cluster_t([r["wi"] for r in c], p)
                ts.append(t)
        if ts:
            max_t_null.append(max(ts))
    obs_t = max(g[0] for g in grid)
    p_fw = float(np.mean(np.array(max_t_null) >= obs_t))
    print(f"  observed max t over the {n_rules}-rule grid : {obs_t:+.2f}")
    print(f"  permutation null (600 shuffles)      : mean {np.mean(max_t_null):+.2f}, "
          f"95th pct {np.percentile(max_t_null, 95):+.2f}")
    print(f"  family-wise p-value                  : {p_fw:.4f}")

    # ---- 3. the chosen rule, in detail ----------------------------------
    print(f"\n=== 3) CHOSEN RULE: T>={SHIP_T_MIN}s, favourite ask "
          f"[{SHIP_BAND[0]:.2f},{SHIP_BAND[1]:.2f}), hold to settlement ===")
    sel = first_per_window([r for r in rows if r["T"] >= SHIP_T_MIN and r["mkt"] >= 0.5
                            and SHIP_BAND[0] <= r["ask"] < SHIP_BAND[1]])
    ask = np.array([r["ask"] for r in sel])
    won = np.array([r["won"] for r in sel])
    p = pnl_of(ask, won)
    m, t, nw = cluster_t([r["wi"] for r in sel], p)
    lo, hi = boot_ci([r["wi"] for r in sel], p)
    pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
    be = ask.mean() + (fee(ask.mean(), STAKE / ask.mean()) * ask.mean() / STAKE)
    print(f"  trades {len(sel)} over {nw} windows ({len(sel)/len(W)*100:.0f}% of windows)")
    print(f"  mean ask paid {ask.mean():.4f}   realised win rate {won.mean():.4f}")
    print(f"  break-even win rate (ask + fee) {be:.4f}   edge {won.mean()-be:+.4f}")
    print(f"  net ${p.sum():+.2f}  per trade ${p.mean():+.3f}  ROI {p.sum()/(STAKE*len(sel))*100:+.2f}%")
    print(f"  cluster-t {t:+.2f}   bootstrap CI95 [{lo:+.2f},{hi:+.2f}]   binom_p {pv:.4f}")
    print(f"  worst ${p.min():+.2f}   best ${p.max():+.2f}")

    # leave-one-out
    ts = []
    for k in range(len(sel)):
        keep = [i for i in range(len(sel)) if i != k]
        pk = p[keep]
        _, tk, _ = cluster_t([sel[i]["wi"] for i in keep], pk)
        ts.append(tk)
    print(f"  leave-one-out t: min {min(ts):+.2f} median {np.median(ts):+.2f} max {max(ts):+.2f}")

    # chronological split
    ws_sorted = sorted(r["ws"] for r in sel)
    med = ws_sorted[len(ws_sorted) // 2]
    for lbl, sub in (("first half ", [r for r in sel if r["ws"] <= med]),
                     ("second half", [r for r in sel if r["ws"] > med])):
        a2 = np.array([r["ask"] for r in sub]); w2 = np.array([r["won"] for r in sub])
        p2 = pnl_of(a2, w2)
        _, t2, _ = cluster_t([r["wi"] for r in sub], p2)
        print(f"  {lbl}: N={len(sub):>3} net=${p2.sum():+7.2f} t={t2:+5.2f} "
              f"win={w2.mean():.4f} ask={a2.mean():.4f}")

    # block stability
    order = np.argsort([r["ws"] for r in sel])
    pos = 0
    blocks = np.array_split(order, 8)
    print("  block stability (8 chronological blocks): ", end="")
    for b in blocks:
        sub = [sel[i] for i in b]
        if not sub:
            continue
        p2 = pnl_of([r["ask"] for r in sub], [r["won"] for r in sub])
        pos += int(p2.sum() > 0)
    print(f"{pos}/8 positive")

    # ---- 3b. the conditional sizing overlay ------------------------------
    print(f"\n=== 3b) SIZING OVERLAY: ${STAKE_CHEAP:.0f} when the favourite's ask "
          f"< {CHEAP_ASK:.2f}, else ${STAKE:.0f} ===")
    print("  rationale: at a roughly constant win rate, EV per dollar = (p - ask)/ask,")
    print("  so a cheaper favourite carries more ROI per dollar.  No win-rate gap is")
    print("  required (the outcome placebo found none: p=0.66).")
    ps, stk = pnl_sized(ask, won)
    _, ts2, _ = cluster_t([r["wi"] for r in sel], ps)
    lo2, hi2 = boot_ci([r["wi"] for r in sel], ps)
    ncheap = int((ask < CHEAP_ASK).sum())
    print(f"  N={len(sel)}  staked ${stk.sum():.0f}  net ${ps.sum():+.2f}  "
          f"ROI {ps.sum()/stk.sum()*100:+.2f}%  per trade ${ps.mean():+.3f}")
    print(f"  cluster-t {ts2:+.2f}   bootstrap CI95 [{lo2:+.2f},{hi2:+.2f}]   "
          f"worst ${ps.min():+.2f}")
    print(f"  cheap bucket (ask<0.85): {ncheap} trades at $30, "
          f"{len(sel)-ncheap} at $10")
    # chronological split of the sized version
    med2 = ws_sorted[len(ws_sorted) // 2]
    for lbl, sub in (("first half ", [r for r in sel if r["ws"] <= med2]),
                      ("second half", [r for r in sel if r["ws"] > med2])):
        if not sub:
            continue
        a3 = np.array([r["ask"] for r in sub]); w3 = np.array([r["won"] for r in sub])
        p3, s3 = pnl_sized(a3, w3)
        _, t3, _ = cluster_t([r["wi"] for r in sub], p3)
        print(f"  {lbl}: N={len(sub):>3} staked ${s3.sum():.0f} net=${p3.sum():+7.2f} "
              f"ROI {p3.sum()/s3.sum()*100:+6.2f}% t={t3:+5.2f}")
    # stress: latency and fees
    print("  stress:")
    for lbl, lat, feemul in (("1s latency", 1.0, 1.0), ("2s latency", 2.0, 1.0),
                              ("4x fee", 0.0, 4.0), ("8x fee", 0.0, 8.0)):
        rr2 = build_rows(W, latency_s=lat)
        s2 = first_per_window([r for r in rr2 if r["T"] >= SHIP_T_MIN and r["mkt"] >= 0.5
                               and SHIP_BAND[0] <= r["ask"] < SHIP_BAND[1]])
        if not s2:
            continue
        a4 = np.array([r["ask"] for r in s2]); w4 = np.array([r["won"] for r in s2])
        p4, s4 = pnl_sized(a4, w4, fee_mult=feemul)
        _, t4, _ = cluster_t([r["wi"] for r in s2], p4)
        l4, h4 = boot_ci([r["wi"] for r in s2], p4)
        print(f"    {lbl:<12} N={len(s2):>3} net=${p4.sum():+7.2f} t={t4:+5.2f} "
              f"CI95 [{l4:>+5.2f},{h4:>+5.2f}]  {'survives' if l4 > 0 else 'FAILS'}")

    # ---- 4. controls ----------------------------------------------------
    print("\n=== 4) CONTROLS ===")
    # 4a symmetry: the underdog at the same timestamps must lose
    ud = first_per_window([r for r in rows if r["T"] >= SHIP_T_MIN and r["mkt"] < 0.5
                           and 0.01 <= r["ask"] < 0.20])
    if ud:
        pu = pnl_of([r["ask"] for r in ud], [r["won"] for r in ud])
        _, tu, _ = cluster_t([r["wi"] for r in ud], pu)
        print(f"  underdog at the SAME timestamps: N={len(ud)} net=${pu.sum():+.2f} t={tu:+.2f}"
              f"  (must lose - proves the timestamps are aligned)")

    # 4b calibration: market-implied vs realised, by price band
    print("\n  calibration of the favourite by price band (all executable quotes):")
    print("  band            rows  wins   ask   realised     gap   cluster_t")
    favrows = [r for r in rows if r["mkt"] >= 0.5]
    for lo, hi in ((0.70, 0.80), (0.80, 0.86), (0.86, 0.90), (0.90, 0.93),
                   (0.93, 0.96), (0.96, 0.98), (0.98, 1.001)):
        s = [r for r in favrows if lo <= r["ask"] < hi]
        if len(s) < 20:
            continue
        byw = {}
        for r in s:
            byw.setdefault(r["wi"], []).append(r)
        gaps, wi = [], []
        for w_, rs in byw.items():
            gaps.append(np.mean([r["won"] for r in rs]) - np.mean([r["ask"] for r in rs]))
            wi.append(w_)
        mg, tg, _ = cluster_t(wi, np.array(gaps))
        print(f"  [{lo:.2f},{hi:.3f}) {len(s):>5} {sum(r['won'] for r in s):>5} "
              f"{np.mean([r['ask'] for r in s]):>6.4f} {np.mean([r['won'] for r in s]):>9.4f} "
              f"{mg:>+8.4f} {tg:>+9.2f}")

    # 4c robustness: depth, spread, stake
    print("\n  robustness of the chosen rule:")
    for depth in (5.0, 10.0, 25.0, 50.0, 100.0):
        rr = []
        for wi_, d in enumerate(W):
            t = d["t"]; n = len(t)
            for T in range(240, 300, 2):
                i = int(np.searchsorted(t, T))
                if i >= n:
                    continue
                mu, md = d["Up_mid"][i], d["Down_mid"][i]
                if not (np.isfinite(mu) and np.isfinite(md)):
                    continue
                p_up = 0.5 * (mu + (1 - md))
                if not (0.002 < p_up < 0.998):
                    continue
                for side in ("Up", "Down"):
                    a, b, sz = d[side + "_ask"][i], d[side + "_bid"][i], d[side + "_asksz"][i]
                    if not all(np.isfinite(v) for v in (a, b, sz)):
                        continue
                    if not (0 < b <= a < 1) or (a - b) > MAX_SPREAD or sz * a < depth:
                        continue
                    o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                    if not (np.isfinite(o) and 0.001 < o < 0.999):
                        continue
                    mkt = p_up if side == "Up" else 1 - p_up
                    rr.append(dict(wi=wi_, ws=d["ws"], T=float(T), side=side, ask=float(a),
                                   mkt=float(mkt), won=1 if d["_winner"] == side else 0))
        s2 = first_per_window([r for r in rr if r["T"] >= SHIP_T_MIN and r["mkt"] >= 0.5
                               and SHIP_BAND[0] <= r["ask"] < SHIP_BAND[1]])
        if len(s2) < 5:
            print(f"    min depth ${depth:>6.0f}: too few trades ({len(s2)})")
            continue
        p2 = pnl_of([r["ask"] for r in s2], [r["won"] for r in s2])
        _, t2, _ = cluster_t([r["wi"] for r in s2], p2)
        print(f"    min depth ${depth:>6.0f}: N={len(s2):>3} net=${p2.sum():+7.2f} t={t2:+5.2f}")

    # ---- 5. the live-bot version (sustained quote) -----------------------
    print("\n=== 5) LIVE-EXECUTABLE VARIANT (ask must hold in band for 5s) ===")
    for t_min in (258, 260, 262, 265):
        report_strategy(rows, W, t_min, SHIP_BAND, 5.0, 5.0,
                        f"T>={t_min}s fav ask[{SHIP_BAND[0]:.2f},{SHIP_BAND[1]:.2f}) held 5s")

    # ---- 5b. EXECUTION LATENCY -----------------------------------------
    print("\n=== 5b) EXECUTION LATENCY (a taker order fills at the ask it sees")
    print("         after the round trip, not at the frame it was decided on) ===")
    for lat in (0.0, 0.3, 0.5, 0.8, 1.0):
        rl = build_rows(W, latency_s=lat)
        sel2 = first_per_window([r for r in rl if r["T"] >= SHIP_T_MIN and r["mkt"] >= 0.5
                                 and SHIP_BAND[0] <= r["ask"] < SHIP_BAND[1]])
        if len(sel2) < 5:
            print(f"  latency {lat:.1f}s: too few trades ({len(sel2)})")
            continue
        a2 = np.array([r["ask"] for r in sel2]); w2 = np.array([r["won"] for r in sel2])
        p2 = pnl_of(a2, w2)
        _, t2, _ = cluster_t([r["wi"] for r in sel2], p2)
        lo2, hi2 = boot_ci([r["wi"] for r in sel2], p2)
        print(f"  latency {lat:.1f}s: N={len(sel2):>3} w={int(w2.sum()):>3} ask={a2.mean():.4f} "
              f"net=${p2.sum():+7.2f} avg=${p2.mean():+.3f} t={t2:+5.2f} "
              f"CI=[{lo2:+.2f},{hi2:+.2f}] worst=${p2.min():+.2f}")

    # ---- 6. power --------------------------------------------------------
    print("\n=== 6) POWER / SAMPLE SIZE ===")
    p_hat = won.mean(); p0 = ask.mean()
    for target in (0.95, 0.97, 0.99):
        n_need = ((norm.ppf(0.95) + norm.ppf(0.80)) * np.sqrt(p0 * (1 - p0)) /
                  (target - p0)) ** 2
        print(f"  to detect a true win rate of {target:.2f} vs market-implied {p0:.4f}: "
              f"~{n_need:.0f} independent windows "
              f"({n_need/len(W)*22.5/24:.1f} days of continuous logging)")
    print(f"  current: {int(won.sum())}/{len(won)} wins, binom_p={pv:.4f}")


if __name__ == "__main__":
    main()
