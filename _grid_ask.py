"""Grid search ASK_LO x ASK_HI x GAP_MIN (conf fixed at 0.76) on the CB feed.

Precomputes per-sample per-side (mp, ask, shares, fee, ev); only the three
bounds vary per combination. Reports the top combos by net P&L and flags
plateaus (neighboring combos with similar profit).
"""
import json, math, os, bisect, itertools
from collections import defaultdict
from datetime import datetime

ROOT = r"C:\Users\user\Desktop\bot"
SCRATCH = r"C:\Users\user\AppData\Local\hermes\cache\scratch"
CONF = 0.76
EDGE_FRAC = 0.02
FEE_R = 0.05
STAKE = 100.0

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

cb = json.load(open(os.path.join(SCRATCH, "cb_candles.json")))
cb_map = {a: {int(k): v for k, v in d.items()} for a, d in cb.items()}
cb_sorted = {a: sorted(cb_map[a].items()) for a in cb_map}
cb_tss = {a: [ts for ts, _ in cb_sorted[a]] for a in cb_map}
def cb_spot(a, ts):
    tss = cb_tss.get(a, [])
    i = bisect.bisect_right(tss, int(ts)) - 1
    return cb_sorted[a][i][1] if i >= 0 and tss else None

winners = {}
for f in ["resolved_windows.json", "window_outcomes.json"]:
    d = json.load(open(os.path.join(ROOT, f), encoding="utf-8"))
    if isinstance(d, list):
        for e in d:
            if e.get("winner") and e.get("slug"):
                winners[e["slug"]] = e["winner"]
    else:
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

# Precompute per-window: list of {outcome: (mp, ask, shares, fee, ev)} in time order
windows = defaultdict(list)
with open(os.path.join(ROOT, "price_logs", "twap", "windows.jsonl"), encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        info = d.get("info") or {}
        t = info.get("t")
        if t is None or not (90 <= t <= 295) or (100 <= t < 110):
            continue
        if d.get("p_up_model") is None:
            continue
        try:
            ts = datetime.fromisoformat(d["t"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        windows[d["window"]].append((ts, d))

precomp = []
for slug, samples in windows.items():
    samples.sort(key=lambda x: x[0])
    asset = slug.split("-")[0].lower()
    w = []
    for ts, d in samples:
        info = d.get("info") or {}
        sig = info.get("sigma_min")
        spot = cb_spot(asset, ts)
        if spot is None or sig is None:
            continue
        if None in (info.get("e_final"), d.get("ref"), info.get("t"), info.get("remaining")):
            continue
        T = 300.0
        var = spot ** 2 * (sig / 60.0) ** 2 * info["remaining"] ** 3 / (3 * T * T)
        sd = max(math.sqrt(max(var, 1e-18)), 0.001 * spot)
        p_up = norm_cdf((info["e_final"] - d["ref"]) / sd)
        side = {}
        for outcome in ("Up", "Down"):
            ask = d.get("ask_" + outcome)
            if not ask:
                continue
            mp = p_up if outcome == "Up" else 1 - p_up
            shares = STAKE / ask
            fee = FEE_R * min(ask, 1 - ask) * shares
            ev = mp * (shares - STAKE - fee) - (1 - mp) * (STAKE + fee)
            side[outcome] = (mp, ask, shares, fee, ev)
        w.append(side)
    precomp.append((slug, winners.get(slug), w))

def replay(ask_lo, ask_hi, gap_min):
    fired = 0
    scored = []
    for slug, win, w in precomp:
        entry = None
        for si in w:
            best = None
            for outcome, (mp, ask, shares, fee, ev) in si.items():
                if (mp >= CONF and ask_lo <= ask <= ask_hi
                        and (mp - ask) >= gap_min and ev > EDGE_FRAC * STAKE):
                    if best is None or ev > best[0]:
                        best = (ev, outcome, mp, ask, shares, fee)
            if best:
                entry = best
                break
        if entry is None:
            continue
        fired += 1
        ev, outcome, mp, ask, shares, fee = entry
        if win is not None:
            won = (win == outcome)
            pnl = (shares if won else 0.0) - (STAKE + fee)
            scored.append((outcome, won, pnl))
    n = len(scored)
    wins = sum(1 for _, w, _ in scored if w)
    pnl = sum(p for _, _, p in scored)
    up = sum(p for o, _, p in scored if o == "Up")
    dn = sum(p for o, _, p in scored if o == "Down")
    worst = min((p for _, _, p in scored), default=None)
    return fired, n, wins, pnl, up, dn, worst

ask_los = [0.50, 0.55, 0.60, 0.65]
ask_his = [0.85, 0.88, 0.90, 0.93]
gap_mins = [0.04, 0.06, 0.08, 0.10, 0.12]

results = {}
print(f"conf={CONF}  grid: ASK_LO={ask_los} ASK_HI={ask_his} GAP_MIN={gap_mins}")
for alo, ahi, gap in itertools.product(ask_los, ask_his, gap_mins):
    fired, n, wins, pnl, up, dn, worst = replay(alo, ahi, gap)
    results[(alo, ahi, gap)] = (fired, n, wins, pnl, up, dn, worst)
    if n < 10:
        continue
    wr = 100 * wins / n
    wstr = f"{worst:+.0f}" if worst is not None else "-"
    print(f"AL={alo:.2f} AH={ahi:.2f} GAP={gap:.2f} | fired={fired:>4} scored={n:>3} "
          f"win%={wr:4.1f} net=${pnl:+8.2f} worst=${wstr} Up=${up:+.0f} Down=${dn:+.0f}")

valid = {k: v for k, v in results.items() if v[1] >= 10}
best_k = max(valid, key=lambda k: valid[k][3])
bf, bn, bw, bp, bu, bd, bwo = valid[best_k]
print(f"\n*** BEST: ASK_LO={best_k[0]} ASK_HI={best_k[1]} GAP_MIN={best_k[2]} "
      f"-> net=${bp:+.2f} fired={bf} scored={bn} win%={100*bw/bn:.1f} worst=${bwo:+.2f} "
      f"Up=${bu:+.2f} Down=${bd:+.2f}")
print(f"    baseline (0.60,0.88,0.08): net=${results[(0.60,0.88,0.08)][3]:+.2f}")

# plateau: neighbors (within grid step) that keep net >= best*0.8
print("\nNEAR-BEST combos (net >= 80% of best, descending):")
for k, v in sorted(valid.items(), key=lambda kv: -kv[1][3]):
    if v[3] >= 0.8 * bp and v[3] > 0:
        print(f"  AL={k[0]:.2f} AH={k[1]:.2f} GAP={k[2]:.2f} -> net=${v[3]:+.2f} "
              f"scored={v[1]} win%={100*v[2]/v[1]:.1f} worst=${v[6]:+.0f}")
