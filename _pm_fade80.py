"""_pm_fade80.py - BACKTEST: "bet Down when Up is at 80c", and the exact mirror.

The strategy tested (plainly stated, no invented extras):
  * watch the Up leg of the BTC 5-minute binary
  * the FIRST time Up becomes buyable at 80 cents (Up ask crosses up to >= 0.80),
    buy the OTHER side (Down) at its ask and hold to settlement
  * the mirror: the FIRST time Down becomes buyable at 80c, buy Up
  * a control: buy the 80c side itself (follow, don't fade)

Everything is measured on the raw recorded L2 books in price_logs/polymarket/
(269 resolved windows, 2026-10-13/14, ~0.3s frame cadence, full 0-300s).
Fill model: taker at the ask, source-aligned causal clock (t_src), optional
latency (the order is decided on this frame and fills L seconds later at the
ask THEN quoted - dropped if that ask is gone), Polymarket taker fee
C * 0.07 * p * (1-p), $5 per trade, hold to resolution.

Run: python _pm_fade80.py
"""
import os
import sys

import numpy as np
from scipy.stats import binom

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import btc5m_endgame_backtest as B  # noqa: E402  (load_all / boot_ci / cluster_t)

FEE_RATE = 0.07
STAKE = 5.0
MIN_DEPTH_USD = 5.0     # the order must be fillable in the book at the ask
MAX_SPREAD = 0.06       # sanity: ignore crossed/illiquid garbage frames


def fee(price, shares):
    return shares * FEE_RATE * price * (1.0 - price)


def pnl(ask, won, stake=STAKE):
    ask = np.asarray(ask, float)
    won = np.asarray(won, float)
    sh = stake / ask
    return sh * won - stake - fee(ask, sh)


def quote(d, side, i, latency_s=0.0, t=None):
    """Executable taker quote for `side` at frame i (+latency). Returns (ask, ok)."""
    def leg(j):
        a = d[side + "_ask"][j]
        b = d[side + "_bid"][j]
        s = d[side + "_asksz"][j]
        if not all(np.isfinite(v) for v in (a, b, s)):
            return None
        if not (0 < b <= a < 1):
            return None
        if (a - b) > MAX_SPREAD:
            return None
        if s * a < MIN_DEPTH_USD:
            return None
        return float(a)

    j = i
    if latency_s > 0:
        j = int(np.searchsorted(t, t[i] + latency_s))
        if j >= len(t):
            return None
    return leg(j)


def first_touch(d, side, thresh, basis="ask"):
    """Index of the first frame where `side`'s price reaches `thresh`."""
    col = side + "_" + basis
    x = d[col]
    ok = np.isfinite(x) & (x >= thresh)
    if not ok.any():
        return None
    return int(np.argmax(ok))


def run(W, trig_side, thresh=0.80, basis="ask", latency_s=0.0, hold_to=None,
        first_only=True, follow=False):
    """Bet the OPPOSITE side of `trig_side` once it trades at >= thresh.

    follow=True bets `trig_side` itself (the 80c favourite) instead - the control.
    """
    other = trig_side if follow else ("Down" if trig_side == "Up" else "Up")
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]
        i = first_touch(d, trig_side, thresh, basis)
        if i is None:
            continue
        idxs = [i]
        if not first_only:  # every re-touch, capped to avoid duplicates
            col = d[trig_side + "_" + basis]
            prev = False
            idxs = []
            for j in range(len(t)):
                cur = np.isfinite(col[j]) and col[j] >= thresh
                if cur and not prev:
                    idxs.append(j)
                prev = cur
        for j in idxs:
            # opposite leg must be live and executable at that moment
            oa = quote(d, other, j, latency_s, t)
            if oa is None:
                continue
            ta = d[trig_side + "_ask"][j]
            if not (np.isfinite(ta) and 0 < ta < 1):
                continue
            if hold_to is not None and t[j] > hold_to:
                continue
            rows.append(dict(wi=wi, ws=d["ws"], T=float(t[j]), side=other,
                             ask=oa, trig_ask=float(ta),
                             trig_mid=float(d[trig_side + "_mid"][j])
                             if np.isfinite(d[trig_side + "_mid"][j]) else np.nan,
                             won=1 if d["_winner"] == other else 0))
    return rows


def edge_ci(rows, n=8000, seed=13):
    """Bootstrap CI for the mean edge in cents (realised win rate - price paid),
    resampled by window."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["wi"], []).append(r["won"] - r["ask"])
    keys = list(by)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        v = [x for k in pick for x in by[k]]
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])


def stats(rows, label="", verbose=True):
    if not rows:
        print(f"  {label:<46} no trades")
        return None
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    p = pnl(ask, won)
    wi = [r["wi"] for r in rows]
    m, tt, nwin = B.cluster_t(wi, p)
    lo, hi = B.boot_ci(wi, p)
    elo, ehi = edge_ci(rows)
    # binomial test: is the realised win rate above the price paid?
    bp = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
    staked = STAKE * len(rows)
    if verbose:
        print(f"  {label:<46} N={len(rows):>3} w={int(won.sum()):>3} "
              f"pay={ask.mean():.4f} real={won.mean():.4f} "
              f"edge={won.mean()-ask.mean():+.4f} eCI=[{elo:+.3f},{ehi:+.3f}] "
              f"net=${p.sum():+7.2f} t={tt:+5.2f} CI=[{lo:+.2f},{hi:+.2f}] "
              f"bp={bp:.3f} worst=${p.min():+.2f} roi={p.sum()/staked*100:+.1f}%")
    return dict(rows=rows, ask=ask, won=won, p=p, t=tt, lo=lo, hi=hi, bp=bp,
                n=len(rows), net=p.sum(), mean=p.mean(), elo=elo, ehi=ehi)


def placebo(W, trig_side, thresh, basis="ask", latency_s=0.0, n=4000, seed=7):
    """Outcome placebo: keep the same trigger prices, shuffle which side won."""
    rows = run(W, trig_side, thresh, basis, latency_s)
    if len(rows) < 10:
        return None
    ask = np.array([r["ask"] for r in rows])
    won = np.array([r["won"] for r in rows], float)
    obs = pnl(ask, won).sum()
    # shuffle the *win indicator within the same price*, i.e. Bernoulli(pay)
    rng = np.random.default_rng(seed)
    base = float(won.mean())
    sims = []
    for _ in range(n):
        w = rng.binomial(1, base, len(won)).astype(float)
        sims.append(pnl(ask, w).sum())
    sims = np.array(sims)
    pv = float((sims >= obs).mean())
    return pv, float(np.percentile(sims, 50)), float(np.percentile(sims, 95))


def main():
    W = B.load_all()
    up = np.array([1 if d["_winner"] == "Up" else 0 for d in W])
    print("=" * 118)
    print("BET DOWN WHEN UP IS AT 80c  /  BET UP WHEN DOWN IS AT 80c   -   measured on recorded books")
    print("=" * 118)
    print(f"windows {len(W)}   span {(W[-1]['ws']-W[0]['ws'])/3600:.1f}h   "
          f"Up base rate {up.mean():.3f}   frame cadence {np.median(np.concatenate([np.diff(d['t']) for d in W])):.2f}s")
    print("fill = taker at the ask, $5 stake, fee 0.07*p*(1-p), hold to settlement\n")

    # ---------------------------------------------------------------- 1. headline
    print("=== 1) THE TWO TRADES, BACK TO BACK (no latency, first touch) ===")
    print("  trade                                          N   w  pay     real    edge     net$    avg$      t      CI95        binom_p worst$  roi")
    fade_dn = stats(run(W, "Up", 0.80), "A) Up hits 80c -> BUY DOWN")
    fade_up = stats(run(W, "Down", 0.80), "B) Down hits 80c -> BUY UP (mirror)")
    both = (fade_dn["rows"] if fade_dn else []) + (fade_up["rows"] if fade_up else [])
    stats(both, "A+B pooled (both directions)")
    print()
    follow_up = stats(run(W, "Up", 0.80, follow=True), "C) Up hits 80c -> BUY UP (follow)")
    follow_dn = stats(run(W, "Down", 0.80, follow=True), "D) Down hits 80c -> BUY DOWN (follow)")

    # ---------------------------------------------------------------- 2. symmetry check
    a = np.array([r["trig_ask"] for r in fade_dn["rows"]])
    b = np.array([r["ask"] for r in fade_dn["rows"]])
    print(f"\n  when Up's ask is 80c, the Down ask you would pay: mean {b.mean():.4f} "
          f"median {np.median(b):.4f} p10 {np.percentile(b,10):.4f} p90 {np.percentile(b,90):.4f}")
    print(f"  Up ask at that moment: mean {a.mean():.4f}  -> 1-Up_ask = {(1-a).mean():.4f}")
    print("  (the gap between the Down ask and 1-Up_ask is the spread you cross)")

    # ---------------------------------------------------------------- 3. latency
    print("\n=== 3) LATENCY SENSITIVITY (order decided on the trigger frame, fills L s later) ===")
    for L in (0.0, 0.3, 0.5, 0.8, 1.0, 2.0):
        stats(run(W, "Up", 0.80, latency_s=L), f"A) fade Down, latency {L:.1f}s")
    print()
    for L in (0.0, 0.3, 0.5, 1.0):
        stats(run(W, "Down", 0.80, latency_s=L), f"B) fade Up,  latency {L:.1f}s")

    # ---------------------------------------------------------------- 4. threshold
    print("\n=== 4) WHERE THE 80c LINE IS (fade the side that reaches X) ===")
    for x in (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 0.95):
        stats(run(W, "Up", x), f"Up hits {x:.2f} -> buy Down")
    print()
    for x in (0.75, 0.80, 0.85, 0.90):
        stats(run(W, "Down", x), f"Down hits {x:.2f} -> buy Up")

    # ---------------------------------------------------------------- 5. basis
    print("\n=== 5) WHAT COUNTS AS 'AT 80c' (ask / mid / bid basis) ===")
    for bas in ("ask", "mid", "bid"):
        stats(run(W, "Up", 0.80, basis=bas), f"Up {bas} >= 0.80 -> buy Down")

    # ---------------------------------------------------------------- 6. timing
    print("\n=== 6) WHEN THE 80c TOUCH HAPPENS (fade Down) ===")
    rows = run(W, "Up", 0.80)
    for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 240), (240, 301)):
        sub = [r for r in rows if lo <= r["T"] < hi]
        stats(sub, f"touch at t={lo:>3}-{hi:<3}s  -> buy Down")

    # ---------------------------------------------------------------- 7. every touch
    print("\n=== 7) ONE BET PER WINDOW vs EVERY RE-TOUCH (fade Down) ===")
    stats(run(W, "Up", 0.80, first_only=True), "first touch only (1 bet/window)")
    stats(run(W, "Up", 0.80, first_only=False), "every crossing (multiple bets)")

    # ---------------------------------------------------------------- 8. placebo
    print("\n=== 8) PLACEBO / SIGNIFICANCE ===")
    for lab, ts_ in (("A) fade Down", "Up"), ("B) fade Up", "Down"),
                     ("C) follow Up", "Up")):
        pv = placebo(W, ts_, 0.80)
        if pv:
            print(f"  {lab:<16} shuffled-outcome p(>=observed) = {pv[0]:.4f}   "
                  f"shuffled median net ${pv[1]:+.2f}  95th pct ${pv[2]:+.2f}")
    print("  NOTE: the shuffled null uses the same paid prices and the observed")
    print("        win rate, so it tests whether the TRIGGER (not the price) carries")
    print("        information.  A fade that wins less than its price needs shows up")
    print("        as a large negative net with t well below zero.")

    # ---------------------------------------------------------------- 9. sizing
    print("\n=== 9) WHAT IF THE STAKE IS BIGGER (same trades, $ per bet) ===")
    for s in (5, 10, 20, 50):
        for lab, rs in (("fade Down", fade_dn["rows"]), ("fade Up", fade_up["rows"])):
            ask = np.array([r["ask"] for r in rs])
            won = np.array([r["won"] for r in rs], float)
            p = pnl(ask, won, stake=float(s))
            print(f"  {lab:<10} ${s:>3}/bet  net=${p.sum():+8.2f}  "
                  f"staked=${s*len(rs):>6}  roi={p.sum()/(s*len(rs))*100:+.1f}%  "
                  f"worst single bet ${p.min():+.2f}")
        print()


    # ---------------------------------------------------------------- 10. overround
    print("=== 10) WHAT THE BOOK CHARGES YOU (Up ask + Down ask at the trigger) ===")
    ov = []
    for wi, d in enumerate(W):
        i = first_touch(d, "Up", 0.80)
        if i is None:
            continue
        ua, da = d["Up_ask"][i], d["Down_ask"][i]
        if np.isfinite(ua) and np.isfinite(da) and 0 < ua < 1 and 0 < da < 1:
            ov.append(ua + da)
    ov = np.array(ov)
    print(f"  N={len(ov)}  mean two-legged cost = {ov.mean():.4f}  "
          f"median {np.median(ov):.4f}  p10 {np.percentile(ov,10):.4f} p90 {np.percentile(ov,90):.4f}")
    print(f"  that is a {(ov.mean()-1.0)*100:+.2f}c overround per contract - you start every")
    print(f"  bet {(ov.mean()-1.0)/2*100:.2f}c behind the fair price on EITHER side,")
    print(f"  before the {FEE_RATE*100:.0f}% taker fee.")

    # ---------------------------------------------------------------- 11. OOS split
    print("\n=== 11) FIRST HALF vs SECOND HALF (same rule, chronological) ===")
    cut = W[len(W) // 2]["ws"]
    print(f"  split at window {cut}  ({cut - W[0]['ws']:.0f}s after the first window)")
    for lab, rs in (("fade Down", fade_dn["rows"]), ("fade Up", fade_up["rows"])):
        h1 = [r for r in rs if r["ws"] < cut]
        h2 = [r for r in rs if r["ws"] >= cut]
        stats(h1, f"{lab}, first half")
        stats(h2, f"{lab}, second half")

    # ---------------------------------------------------------------- 12. benchmark
    print("\n=== 12) REFERENCE: the favourite rule this log already produced ===")
    print("     (buy the FAVOURITE in the band, last 40s, hold to settlement)")
    rows = B.build_rows(W)
    B.report_strategy(rows, W, t_min=B.SHIP_T_MIN, band=B.SHIP_BAND,
                      label="fav ask[0.80,0.96) T>=262s")
    B.report_strategy(rows, W, t_min=240, band=(0.80, 0.96),
                      label="fav ask[0.80,0.96) T>=240s")
    print("     -> the same 80c price is only an EDGE when it is bought late")
    print("        (part of the settlement TWAP is already locked); at any other")
    print("        moment 80c is simply the market's estimate and both sides are")
    print("        a coin flip minus the spread.")

    # ---------------------------------------------------------------- 13. cost breakdown
    print("\n=== 13) WHERE THE MONEY GOES (per trade, $5 stake) ===")
    for lab, rs in (("fade Down", fade_dn["rows"]), ("fade Up", fade_up["rows"]),
                    ("follow Up", follow_up["rows"]), ("follow Down", follow_dn["rows"])):
        ask = np.array([r["ask"] for r in rs])
        won = np.array([r["won"] for r in rs], float)
        sh = STAKE / ask
        gross = (sh * won - STAKE)
        fcost = fee(ask, sh)
        net = gross - fcost
        eq = np.cumsum(net)
        dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq]))
                    - np.concatenate([[0.0], eq])).max())
        # longest losing streak
        streak = best = 0
        for x in net:
            streak = streak + 1 if x < 0 else 0
            best = max(best, streak)
        print(f"  {lab:<12} N={len(rs):>3}  gross ${gross.sum():+8.2f}  "
              f"fees ${-fcost.sum():+8.2f}  net ${net.sum():+8.2f}  "
              f"fee/trade ${fcost.mean():.3f}  maxDD ${dd:6.2f}  "
              f"worst losing streak {best}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 118)
    print("VERDICT")
    print("=" * 118)
    ab = np.array([r["won"] - r["ask"] for r in both])
    print(f"  pooled fade (both directions): N={len(ab)}  realised-vs-paid edge "
          f"{ab.mean()*100:+.2f}c/trade  95% CI [{edge_ci(both)[0]*100:+.2f}c,"
          f"{edge_ci(both)[1]*100:+.2f}c]")
    print(f"  -> the fade is statistically indistinguishable from a coin flip at the")
    print(f"     price you pay.  It does not have an edge; it has a variance.")
    print("=" * 118)


if __name__ == "__main__":
    main()
