"""Backtest: Binance vs Coinbase price source for twap_bot, on recorded windows.

For every recorded sample we recompute the model probability on BOTH feeds.
The recorded `spot` is Binance (live feed at the time); the CB spot is the CB
1-min candle close at the sample time. e_final / ref / sigma_min are held
source-independent (they're the model's running estimates; only the *basis*
spot-vs-Chainlink and the *sigma* level move with the source).

We replay the calibrated gate over ALL samples in the tradeable band (enter at
the first fire per window, hold to settlement) and score the windows with a
known winner. Compares the two feeds.
"""
import json, math, os, bisect
from collections import defaultdict

ROOT = r"C:\Users\user\Desktop\bot"
SCRATCH = r"C:\Users\user\AppData\Local\hermes\cache\scratch"

# ---- calibrated gate constants (current twap_bot.py) ----
CONF_MIN_FAV = 0.85
ASK_LO, ASK_HI = 0.60, 0.88
GAP_MIN = 0.08
EDGE_FRAC = 0.02
FEE_R = 0.05
STAKE = 100.0

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def p_up_from(e_final, ref, sd, spot=None):
    if sd is None or e_final is None or ref is None:
        return None
    sd = max(sd, 1e-18)
    z = (e_final - ref) / sd
    return norm_cdf(z)

def model_p_at(sample_spot_basis):
    pass

# ---- load CB candles: asset -> sorted list of (ts, close) ----
cb = json.load(open(os.path.join(SCRATCH, "cb_candles.json")))
cb_map = {a: {int(k): v for k, v in d.items()} for a, d in cb.items()}
# pre-sort for bisection
cb_sorted = {}
for a in cb_map:
    items = sorted(cb_map[a].items())
    cb_sorted[a] = (items, [ts for ts, _ in items])

def cb_spot(asset, ts_unix):
    """CB 1-min candle close at/just before ts (step function)."""
    items, tss = cb_sorted[asset]
    i = bisect.bisect_right(tss, int(ts_unix)) - 1
    if i < 0:
        return None
    return items[i][1]

# ---- load scored winners ----
winners = {}
for f in ["resolved_windows.json", "window_outcomes.json"]:
    d = json.load(open(os.path.join(ROOT, f), encoding="utf-8"))
    if isinstance(d, list):
        for e in d:
            if e.get("winner") and e.get("slug"):
                winners[e["slug"]] = e["winner"]
    elif isinstance(d, dict):
        for k, v in d.items():
            if v.get("winner"):
                winners[k] = v["winner"]
for line in open(os.path.join(ROOT, "price_logs", "twap", "trades.jsonl"), encoding="utf-8"):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("event") == "settle" and d.get("winner") and d.get("slug"):
        winners[d["slug"]] = d["winner"]

# ---- group samples by window, keeping (ts, sample) in time order ----
windows = defaultdict(list)
total = 0
with open(os.path.join(ROOT, "price_logs", "twap", "windows.jsonl"), encoding="utf-8") as f:
    for line in f:
        total += 1
        try:
            d = json.loads(line)
        except ValueError:
            continue
        info = d.get("info") or {}
        t = info.get("t")
        if t is None:
            continue
        if not (90 <= t <= 295) or (100 <= t < 110):
            continue
        if d.get("p_up_model") is None:
            continue
        # parse sample timestamp
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(d["t"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        windows[d["window"]].append((ts, d))

print(f"total lines={total}, windows with in-band samples={len(windows)}")

# ---- replay gate for a given feed ("bn" or "cb") ----
def replay(feed):
    """Return list of trades: dict(slug, side, ask, model_p, pnl, scored, won)."""
    trades = []
    for slug, samples in windows.items():
        samples.sort(key=lambda x: x[0])
        asset = slug.split("-")[0].lower()
        entered = False
        for ts, d in samples:
            info = d.get("info") or {}
            e_final = info.get("e_final")
            ref = d.get("ref")
            sigma_min = info.get("sigma_min")
            sd = info.get("sd")
            b_spot = d.get("spot")
            if e_final is None or ref is None:
                continue
            # choose spot + sigma per feed
            if feed == "bn":
                spot = b_spot
                sig = sigma_min
            else:
                spot = cb_spot(asset, ts)
                if spot is None:
                    continue
                sig = sigma_min  # sigma level: use recorded (Binance) as proxy; see note
            # sd: recompute from the same formula the bot uses
            T = 300.0
            t = info.get("t")
            r = info.get("remaining")
            if spot is None or sig is None or t is None or r is None:
                continue
            var = spot ** 2 * (sig / 60.0) ** 2 * r ** 3 / (3 * T * T)
            sd = max(math.sqrt(max(var, 1e-18)), 0.001 * spot)
            p_up = norm_cdf((e_final - ref) / sd)
            if p_up is None:
                continue
            best = None
            for outcome in ("Up", "Down"):
                ask = d.get("ask_" + outcome)
                if not ask:
                    continue
                model_p = p_up if outcome == "Up" else 1 - p_up
                shares = STAKE / ask
                fee = FEE_R * min(ask, 1 - ask) * shares
                ev = model_p * (shares - STAKE - fee) - (1 - model_p) * (STAKE + fee)
                ok = (model_p >= CONF_MIN_FAV and ASK_LO <= ask <= ASK_HI
                      and (model_p - ask) >= GAP_MIN and ev > EDGE_FRAC * STAKE)
                if ok and (best is None or ev > best[0]):
                    best = (ev, model_p, outcome, ask, shares, fee)
            if best:
                ev, model_p, outcome, ask, shares, fee = best
                won_side = winners.get(slug)
                won = (won_side == outcome) if won_side else None
                pnl = (shares if won else 0.0) - (STAKE + fee) if won_side else None
                trades.append({"slug": slug, "side": outcome, "ask": ask,
                               "model_p": model_p, "scored": won_side is not None,
                               "won": won, "pnl": pnl})
                entered = True
                break
    return trades

for feed in ("bn", "cb"):
    tr = replay(feed)
    scored = [t for t in tr if t["scored"]]
    wins = [t for t in scored if t["won"]]
    pnl = sum(t["pnl"] for t in scored if t["pnl"] is not None)
    worst = min((t["pnl"] for t in scored if t["pnl"] is not None), default=None)
    up = sum(t["pnl"] for t in scored if t["side"] == "Up" and t["pnl"] is not None)
    dn = sum(t["pnl"] for t in scored if t["side"] == "Down" and t["pnl"] is not None)
    print(f"\n=== FEED={feed} ===")
    print(f"  gate-fired windows: {len(tr)} (scored: {len(scored)})")
    print(f"  win rate: {len(wins)}/{len(scored)} = {100*len(wins)/max(len(scored),1):.1f}%")
    print(f"  net P&L (scored): ${pnl:.2f}   worst loss: ${worst if worst is None else round(worst,2)}")
    print(f"  Up P&L: ${up:.2f}   Down P&L: ${dn:.2f}")
    # distribution of model_p among fired
    if scored:
        import statistics
        mp = [t["model_p"] for t in scored]
        print(f"  model_p mean={statistics.mean(mp):.3f} min={min(mp):.3f} max={max(mp):.3f}")
    for t in sorted(scored, key=lambda x: x["slug"]):
        w = t["won"]
        print(f"    {t['slug']} {t['side']:4} ask={t['ask']:.2f} mp={t['model_p']:.3f} "
              f"{'WIN' if w else 'LOSS'} pnl={t['pnl']:+.2f}" if w is not None else
              f"    {t['slug']} {t['side']} (unscored)")
