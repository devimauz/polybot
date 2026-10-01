"""Sweep EDGE_FRAC at best config (ASK_LO=0.65, ASK_HI=0.85, GAP_MIN=0.08, conf=0.76).

Proper replay: per window, enter at the FIRST sample where some side passes ALL
gates (conf, ask band, gap, EV=edge_frac*stake). Matches live code exactly.
"""
import json, math, os, bisect
from collections import defaultdict
from datetime import datetime

ROOT = r"C:\Users\user\Desktop\bot"
SCRATCH = r"C:\Users\user\AppData\Local\hermes\cache\scratch"
CONF = 0.76
ASK_LO, ASK_HI, GAP_MIN = 0.65, 0.85, 0.08
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

def replay(ef):
    ev_min = ef * STAKE
    fired = 0
    scored = []
    for slug, win, w in precomp:
        entry = None
        for si in w:
            best = None
            for outcome, (mp, ask, shares, fee, ev) in si.items():
                if (mp >= CONF and ASK_LO <= ask <= ASK_HI
                        and (mp - ask) >= GAP_MIN and ev > ev_min):
                    if best is None or ev > best[0]:
                        best = (ev, outcome, shares, fee)
            if best:
                entry = best
                break
        if entry is None:
            continue
        fired += 1
        ev, outcome, shares, fee = entry
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

edge_fracs = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
print(f"conf={CONF} ASK_LO={ASK_LO} ASK_HI={ASK_HI} GAP_MIN={GAP_MIN}")
print(f"{'EDGE':>6} {'ev$':>6} {'fired':>6} {'scored':>7} {'wins':>5} {'win%':>6} "
      f"{'net$':>9} {'worst$':>7} {'Up$':>7} {'Down$':>7}")
rows = []
for ef in edge_fracs:
    fired, n, wins, pnl, up, dn, worst = replay(ef)
    wr = 100 * wins / n if n else 0
    wstr = f"{worst:+.0f}" if worst is not None else "-"
    rows.append((ef, fired, n, wins, pnl, up, dn, worst, wr))
    print(f"{ef:>6.2f} {ef*STAKE:>6.1f} {fired:>6} {n:>7} {wins:>5} {wr:>6.1f} "
          f"{pnl:>+9.2f} {wstr:>7} {up:>+7.2f} {dn:>+7.2f}")

valid = [r for r in rows if r[2] >= 10]
best = max(valid, key=lambda r: r[4])
print(f"\n*** BEST EDGE_FRAC: {best[0]:.2f} (ev>{best[0]*STAKE:.1f}) -> net=${best[4]:+.2f} "
      f"fired={best[1]} scored={best[2]} win%={best[8]:.1f} worst=${best[7]:+.0f} "
      f"Up=${best[5]:+.2f} Down=${best[6]:+.2f}")
print(f"    current (0.02): net=${[r for r in rows if r[0]==0.02][0][4]:+.2f}")
