"""Sweep CONF_MIN_FAV from 0.50 to 0.85 on the CB feed, report max-profit gate.

Replays the calibrated gate (GAP_MIN, ASK_LO/HI, EV all fixed) over every
recorded in-band sample, varying only CONF_MIN_FAV. Report per conf:
N fired, N scored, win rate, net P&L, worst loss, Up/Down split.
"""
import json, math, os, bisect
from collections import defaultdict
from datetime import datetime

ROOT = r"C:\Users\user\Desktop\bot"
SCRATCH = r"C:\Users\user\AppData\Local\hermes\cache\scratch"
ASK_LO, ASK_HI = 0.60, 0.88
GAP_MIN = 0.08
EDGE_FRAC = 0.02
FEE_R = 0.05
STAKE = 100.0
FEED = "cb"

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

# Precompute per-sample: (model_p_up, model_p_down, ask_Up, ask_Down, shares, fee, ev...)
# but model_p depends only on conf-independent inputs (spot, sig, e_final, ref),
# so precompute p_up once per sample; conf only gates.
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

# For each window, find the FIRST sample where gate fires for each conf -> entry.
# Precompute per-sample p_up and per-side (ask, shares, fee, ev) so the conf
# sweep only re-checks the conf/gap conditions.
samples_by_window = []
for slug, samples in windows.items():
    samples.sort(key=lambda x: x[0])
    asset = slug.split("-")[0].lower()
    w = []
    for ts, d in samples:
        info = d.get("info") or {}
        sig = info.get("sigma_min")
        spot = cb_spot(asset, ts) if FEED == "cb" else d.get("spot")
        if spot is None or sig is None:
            continue
        if None in (info.get("e_final"), d.get("ref"), info.get("t"), info.get("remaining")):
            continue
        T = 300.0
        var = spot ** 2 * (sig / 60.0) ** 2 * info["remaining"] ** 3 / (3 * T * T)
        sd = max(math.sqrt(max(var, 1e-18)), 0.001 * spot)
        p_up = norm_cdf((info["e_final"] - d["ref"]) / sd)
        side_info = {}
        for outcome in ("Up", "Down"):
            ask = d.get("ask_" + outcome)
            if not ask:
                continue
            mp = p_up if outcome == "Up" else 1 - p_up
            shares = STAKE / ask
            fee = FEE_R * min(ask, 1 - ask) * shares
            ev = mp * (shares - STAKE - fee) - (1 - mp) * (STAKE + fee)
            side_info[outcome] = (mp, ask, shares, fee, ev)
        w.append(side_info)
    samples_by_window.append((slug, winners.get(slug), w))

def replay(conf):
    scored = []
    fired = 0
    for slug, win, w in samples_by_window:
        entry = None
        for si in w:
            best = None
            for outcome, (mp, ask, shares, fee, ev) in si.items():
                if (mp >= conf and ASK_LO <= ask <= ASK_HI
                        and (mp - ask) >= GAP_MIN and ev > EDGE_FRAC * STAKE):
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

confs = [round(0.50 + 0.01 * i, 2) for i in range(36)]  # 0.50..0.85
print(f"{'conf':>5} {'fired':>6} {'scored':>7} {'wins':>5} {'win%':>6} {'net$':>9} {'worst$':>8} {'Up$':>8} {'Down$':>8}")
best = None
for conf in confs:
    fired, n, wins, pnl, up, dn, worst = replay(conf)
    wr = 100 * wins / n if n else 0
    wstr = f"{worst:+.2f}" if worst is not None else "-"
    print(f"{conf:>5.2f} {fired:>6} {n:>7} {wins:>5} {wr:>6.1f} {pnl:>+9.2f} {wstr:>8} {up:>+8.2f} {dn:>+8.2f}")
    if n >= 10 and (best is None or pnl > best[1]):
        best = (conf, pnl, fired, n, wins, worst)
print(f"\nBEST net P&L (scored>=10): conf={best[0]:.2f} net=${best[1]:+.2f} "
      f"fired={best[2]} scored={best[3]} wins={best[4]}/{best[3]} worst=${best[5]:+.2f}")
