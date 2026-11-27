"""Design-slice edge search on price_logs/polymarket jsonl windows only.

Does not read the later chronological slice. Precommitted families:
  1. taker edge surface, time band x ask band, first touch, 0.5s fill
  2. maker join-bid, fee 0, fill only if the ask trades through the bid
  3. coinbase 2s lead, both directions, small precommitted grid
Plus the riskless complete-set check (re-price the second leg after 1s).
"""
import glob
import os
import pickle
import sys

import numpy as np

LOG = r"C:/Users/user/Desktop/bot/price_logs/polymarket"
CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out/invent_design.txt"
FEE_RATE = 0.07
STAKE = 5.0
LATENCY = 0.5

T_BANDS = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 270), (270, 300)]
P_BANDS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
           (0.65, 0.80), (0.80, 0.90), (0.90, 0.97)]
LEAD_MOVES = (12.0, 25.0, 40.0)
LEAD_CAPS = (0.60, 0.75, 0.90)
MAKER_HORIZONS = (8.0, 15.0, 30.0)


def fee_per_share(p):
    return FEE_RATE * p * (1.0 - p)


def taker_pnl(ask, won, stake=STAKE):
    sh = stake / ask
    return sh * won - stake - sh * fee_per_share(ask)


def executable(bid, ask, asksz, stake=STAKE):
    if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(asksz)):
        return False
    if not (0.0 < bid <= ask < 1.0):
        return False
    if (ask - bid) > 0.04:
        return False
    sh = stake / ask
    if sh < 5.0:
        return False
    if asksz + 1e-9 < sh:
        return False
    return True


def band_of(x, bands):
    for i, (a, b) in enumerate(bands):
        if a <= x < b:
            return i
    return None


def load_windows():
    slugs = set()
    for p in glob.glob(os.path.join(LOG, "btc-updown-5m-*.jsonl")):
        slugs.add(os.path.basename(p)[:-6])
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
            "cb_age": np.asarray(z["cb_age"], float) if "cb_age" in z.files else np.full(len(t), np.nan),
        }
        for side in ("Up", "Down"):
            d[side + "_bid"] = np.asarray(z[side + "_bid"], float)
            d[side + "_ask"] = np.asarray(z[side + "_ask"], float)
            d[side + "_asksz"] = np.asarray(z[side + "_asksz"], float)
        # keep frames inside the window on the causal clock
        keep = (d["t"] >= -1.0) & (d["t"] <= 305.0)
        if keep.sum() < 20:
            continue
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.shape == keep.shape:
                d[k] = v[keep]
        out.append(d)
    out.sort(key=lambda x: x["ws"])
    return out


def fill_idx(t, i, latency):
    j = int(np.searchsorted(t, t[i] + latency, side="left"))
    if j >= len(t) or j <= i:
        return None
    return j


def surface(windows, latency):
    # cell -> lists
    cells = {}
    for d in windows:
        t = d["t"]
        seen = set()
        for side in ("Up", "Down"):
            bid = d[side + "_bid"]
            ask = d[side + "_ask"]
            sz = d[side + "_asksz"]
            obid = d[("Down" if side == "Up" else "Up") + "_bid"]
            oask = d[("Down" if side == "Up" else "Up") + "_ask"]
            for i in range(len(t)):
                tb = band_of(t[i], T_BANDS)
                pb = band_of(ask[i], P_BANDS)
                if tb is None or pb is None:
                    continue
                key = (side, tb, pb)
                if key in seen:
                    continue
                if not (np.isfinite(obid[i]) and np.isfinite(oask[i]) and 0 < oask[i] < 1):
                    continue
                if not executable(bid[i], ask[i], sz[i]):
                    continue
                j = fill_idx(t, i, latency)
                if j is None:
                    seen.add(key)
                    continue
                if not executable(bid[j], ask[j], sz[j]):
                    seen.add(key)
                    continue
                if band_of(ask[j], P_BANDS) != pb:
                    seen.add(key)
                    continue
                seen.add(key)
                won = 1.0 if d["winner"] == side else 0.0
                fa = float(ask[j])
                edge = won - fa - fee_per_share(fa)
                pnl = float(taker_pnl(fa, won))
                cells.setdefault((tb, pb), []).append((edge, pnl, won, fa, d["ws"], side))
    return cells


def summarize_cells(cells, design_ws):
    lines = []
    mid = design_ws[len(design_ws) // 2]
    lines.append("time_band price_band    N   wr    ask   edge_c   net$   ROI%  half1_c/N  half2_c/N")
    candidates = []
    for tb, (t0, t1) in enumerate(T_BANDS):
        for pb, (p0, p1) in enumerate(P_BANDS):
            rows = cells.get((tb, pb), [])
            if len(rows) < 8:
                continue
            edge = np.array([r[0] for r in rows])
            pnl = np.array([r[1] for r in rows])
            won = np.array([r[2] for r in rows])
            ask = np.array([r[3] for r in rows])
            ws = np.array([r[4] for r in rows])
            h1 = edge[ws < mid]
            h2 = edge[ws >= mid]
            stake = len(rows) * STAKE
            roi = 100.0 * pnl.sum() / stake
            lines.append(
                "%3.0f-%3.0f %4.2f-%4.2f %4d %5.1f%% %5.1fc %+6.2fc %+7.1f %6.1f  %+5.2f/%-3d %+5.2f/%-3d"
                % (t0, t1, p0, p1, len(rows), 100 * won.mean(), 100 * ask.mean(),
                   100 * edge.mean(), pnl.sum(), roi,
                   100 * h1.mean() if len(h1) else np.nan, len(h1),
                   100 * h2.mean() if len(h2) else np.nan, len(h2))
            )
            if (len(h1) >= 15 and len(h2) >= 15 and h1.mean() > 0 and h2.mean() > 0
                    and edge.mean() > 0):
                candidates.append((tb, pb, edge.mean(), len(rows)))
    return lines, candidates


def adjacent_ok(candidates, cells):
    good = []
    cand = {(tb, pb) for tb, pb, _, _ in candidates}
    for tb, pb, edge, n in candidates:
        neighbors = []
        for dt, dp in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            key = (tb + dt, pb + dp)
            rows = cells.get(key, [])
            if len(rows) >= 15:
                e = np.mean([r[0] for r in rows])
                neighbors.append(e > 0)
        if neighbors and all(neighbors):
            good.append((tb, pb, edge, n, len(neighbors)))
    return good


def complete_set(windows):
    single = 0
    delayed = 0
    delayed_pnl = 0.0
    for d in windows:
        t = d["t"]
        ua, da = d["Up_ask"], d["Down_ask"]
        us, ds = d["Up_asksz"], d["Down_asksz"]
        ub, db = d["Up_bid"], d["Down_bid"]
        for i in range(len(t)):
            if not (executable(ub[i], ua[i], us[i]) and executable(db[i], da[i], ds[i])):
                continue
            # fee on each leg at $5, need both asks + fees < 1 per share-pair.
            # Compare one share of each: cost = ua+da + fee_ps(ua)+fee_ps(da)
            cost = ua[i] + da[i] + fee_per_share(ua[i]) + fee_per_share(da[i])
            if cost >= 1.0:
                continue
            # depth for 5 shares each, already in executable at $5
            if us[i] < 5 or ds[i] < 5:
                continue
            single += 1
            j = fill_idx(t, i, 1.0)
            if j is None:
                break
            if not (executable(ub[j], ua[j], us[j]) and executable(db[j], da[j], ds[j])):
                break
            cost2 = ua[j] + da[j] + fee_per_share(ua[j]) + fee_per_share(da[j])
            if cost2 < 1.0 and us[j] >= 5 and ds[j] >= 5:
                delayed += 1
                # buy 5 shares of each at the delayed asks
                delayed_pnl += 5.0 * (1.0 - cost2)
            break  # first frame only, per window
    return single, delayed, delayed_pnl


def maker(windows, horizon):
    trades = []
    posts = 0
    for d in windows:
        t = d["t"]
        used = set()
        for side in ("Up", "Down"):
            bid = d[side + "_bid"]
            ask = d[side + "_ask"]
            for i in range(len(t)):
                if not (20.0 <= t[i] <= 180.0):
                    continue
                if side in used:
                    break
                b = bid[i]
                a = ask[i]
                if not (np.isfinite(b) and np.isfinite(a)):
                    continue
                if not (0.20 <= b <= 0.80 and a > b and (a - b) >= 0.02):
                    continue
                used.add(side)
                posts += 1
                # through-fill: a later ask at or below our bid consumes the level
                deadline = t[i] + horizon
                filled = False
                for j in range(i + 1, len(t)):
                    if t[j] > deadline:
                        break
                    if np.isfinite(ask[j]) and ask[j] <= b + 1e-12:
                        filled = True
                        break
                if not filled:
                    continue
                won = 1.0 if d["winner"] == side else 0.0
                shares = 5.0
                pnl = shares * won - shares * b  # maker fee 0, no rebate
                trades.append((pnl, won, b, d["ws"], side))
    return posts, trades


def lead(windows, move_min, ask_cap, latency):
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
            if j0 < 0 or not np.isfinite(cb[j0]):
                continue
            if t[i] - t[j0] < 1.5:
                continue
            move = cb[i] - cb[j0]
            if abs(move) < move_min:
                continue
            side = "Up" if move > 0 else "Down"
            bid = d[side + "_bid"]
            ask = d[side + "_ask"]
            sz = d[side + "_asksz"]
            if ask[i] > ask_cap:
                continue
            if not executable(bid[i], ask[i], sz[i]):
                continue
            j = fill_idx(t, i, latency)
            if j is None or not executable(bid[j], ask[j], sz[j]):
                continue
            if ask[j] > ask_cap + 0.02:
                continue
            taken = True
            won = 1.0 if d["winner"] == side else 0.0
            fa = float(ask[j])
            trades.append((float(taker_pnl(fa, won)), won, fa, d["ws"], side,
                           won - fa - fee_per_share(fa)))
    return trades


def trade_line(label, trades, design_ws):
    if not trades:
        return "%-28s  n=0" % label
    pnl = np.array([r[0] for r in trades])
    won = np.array([r[1] for r in trades])
    ask = np.array([r[2] for r in trades])
    ws = np.array([r[3] for r in trades])
    edge = np.array([r[5] for r in trades]) if len(trades[0]) > 5 else (won - ask)
    mid = design_ws[len(design_ws) // 2]
    h1 = pnl[ws < mid]
    h2 = pnl[ws >= mid]
    stake = ask * 5.0 if len(trades[0]) <= 5 else np.full(len(trades), STAKE)
    # maker records have no edge slot; stake is 5*bid, and r[2] is bid
    if len(trades[0]) <= 5:
        stake = ask * 5.0
        edge = won - ask
    roi = 100.0 * pnl.sum() / stake.sum() if stake.sum() else np.nan
    return ("%-28s N=%4d wr=%5.1f%% px=%5.1fc edge=%+5.2fc net=%+8.2f ROI=%+6.1f%%  "
            "h1=%+7.2f/%d h2=%+7.2f/%d" % (
                label, len(trades), 100 * won.mean(), 100 * ask.mean(), 100 * edge.mean(),
                pnl.sum(), roi, h1.sum() if len(h1) else 0, len(h1),
                h2.sum() if len(h2) else 0, len(h2)))


def main():
    W = load_windows()
    n_des = int(len(W) * 0.60)
    design = W[:n_des]
    hold = W[n_des:]
    # touch holdout only to record the boundary, never its books
    lines = []
    lines.append("logs_only windows=%d resolved_loaded=%d design=%d holdout_untouched=%d" % (
        len(glob.glob(os.path.join(LOG, "btc-updown-5m-*.jsonl"))), len(W), len(design), len(hold)))
    lines.append("design_ws %d .. %d | holdout_starts %d" % (
        design[0]["ws"], design[-1]["ws"], hold[0]["ws"]))
    lines.append("causal clock = t_src; taker fee 0.07*p*(1-p); maker fee 0 (no rebate credited)")
    lines.append("stake taker $5 and >=5 shares; maker size 5 shares; FOK depth gate on taker")
    lines.append("")
    print("loaded", len(W), "design", len(design), flush=True)

    single, delayed, dpnl = complete_set(design)
    lines.append("complete-set design: single-frame discounts=%d  still cheap 1s later=%d  delayed_pnl=$%.2f" % (
        single, delayed, dpnl))
    lines.append("")

    print("surface", flush=True)
    cells = surface(design, LATENCY)
    slines, cands = summarize_cells(cells, [d["ws"] for d in design])
    lines.append("TAKER SURFACE first-touch per side per time-band, fill +%.1fs, drop if ask leaves the band" % LATENCY)
    lines.extend(slines)
    lines.append("cells positive in BOTH design halves with N>=15 each: %d" % len(cands))
    plateau = adjacent_ok(cands, cells)
    lines.append("of those, every adjacent populated band also positive: %d" % len(plateau))
    for tb, pb, edge, n, nn in plateau:
        lines.append("  plateau %s %s edge=%+.2fc N=%d neighbors=%d" % (
            T_BANDS[tb], P_BANDS[pb], 100 * edge, n, nn))
    lines.append("")

    dws = [d["ws"] for d in design]
    print("maker", flush=True)
    lines.append("MAKER join bid, fill only if ask trades through our bid, hold to settlement, fee 0")
    for h in MAKER_HORIZONS:
        posts, trades = maker(design, h)
        lines.append("  horizon %.0fs posts=%d fills=%d (%.1f%%)" % (
            h, posts, len(trades), 100.0 * len(trades) / posts if posts else 0))
        lines.append("  " + trade_line("through-fill", trades, dws))
    lines.append("")

    print("lead", flush=True)
    lines.append("COINBASE 2s lead, buy the side of the move, first signal per window, fill +%.1fs" % LATENCY)
    for mv in LEAD_MOVES:
        for cap in LEAD_CAPS:
            trades = lead(design, mv, cap, LATENCY)
            lines.append(trade_line("mv>=%.0f cap<=%.2f" % (mv, cap), trades, dws))
    lines.append("")
    lines.append("same grid at 1.0s fill (sign flip here kills a rule before holdout)")
    for mv in LEAD_MOVES:
        for cap in LEAD_CAPS:
            trades = lead(design, mv, cap, 1.0)
            lines.append(trade_line("1s mv>=%.0f cap<=%.2f" % (mv, cap), trades, dws))

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
