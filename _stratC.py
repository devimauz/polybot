"""Final: find robust +EV cells (win rate >> ask + fee), combine them into one
live strategy, validate hourly P&L + time-split stability.
Rule: buy favored side (model_p > 0.5) if (t, mp, ask) inside a +EV cell.
One fill per side per window (first qualifying sample); optional max 2."""
import json
from collections import defaultdict
from statistics import mean

ROOT = "C:/Users/user/Desktop/bot"
PATH = ROOT + "/price_logs/twap/windows.jsonl"
CLOB = ROOT + "/price_logs/twap/clob_winners.json"
FEE_R = 0.05
STAKE = 10.0

wins = defaultdict(list)
with open(PATH, encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("asset") == "btc":
            wins[d["window"]].append(d)
clob = json.load(open(CLOB, encoding="utf-8"))
W = {}
for slug, rows in wins.items():
    cw = clob.get(slug)
    if not cw or not cw.get("winner"):
        continue
    W[slug] = {"rows": rows, "winner": cw["winner"], "start": int(slug.rsplit("-", 1)[1])}
starts = sorted(x["start"] for x in W.values())
HOURS = (starts[-1] - starts[0]) / 3600 + 1
MID = (starts[-1] + starts[0]) / 2

CAND = []
for slug, win in W.items():
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295) or (100 <= t < 110): continue
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0.20 <= ask <= 0.99): continue
            mp = p_up if side == "Up" else 1 - p_up
            if mp <= 0.5: continue
            fee = FEE_R * min(ask, 1 - ask)
            won = win["winner"] == side
            pnl = (STAKE/ask - STAKE - fee*STAKE/ask) if won else -(STAKE + fee*STAKE/ask)
            CAND.append({"t": t, "mp": mp, "ask": ask, "slug": slug, "side": side,
                         "won": won, "pnl": pnl, "start": win["start"]})

# 1) Cell mining: require n>=60, realized +EV
print("== Robust +EV cells ==")
cells = []
for tlo, thi in [(90, 150), (150, 200), (200, 240), (240, 296), (90, 296)]:
    for mlo, mhi in [(0.52, 0.62), (0.55, 0.65), (0.55, 0.68), (0.58, 0.68), (0.60, 0.70), (0.62, 0.72)]:
        for alo, ahi in [(0.40, 0.80), (0.50, 0.80), (0.55, 0.80), (0.55, 0.85), (0.60, 0.85), (0.65, 0.85), (0.70, 0.85)]:
            sub = [c for c in CAND if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
            n = len(sub)
            if n < 60: continue
            wr = sum(1 for c in sub if c["won"]) / n
            aq = mean(c["ask"] for c in sub)
            pnl = sum(c["pnl"] for c in sub)
            if wr < 0.55 or aq > 0.85 or pnl <= 0: continue
            h1 = [c for c in sub if c["start"] < MID]; h2 = [c for c in sub if c["start"] >= MID]
            p1 = sum(c["pnl"] for c in h1) if h1 else 0
            p2 = sum(c["pnl"] for c in h2) if h2 else 0
            cells.append((pnl, tlo, thi, mlo, mhi, alo, ahi, n, wr, aq, p1, p2))
            print(f"  t[{tlo},{thi}) mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}): n={n:4d} wr={wr:.3f} ask={aq:.3f} pnl=${pnl:8.2f} (h1 ${p1:+.1f} / h2 ${p2:+.1f})")

# 2) Union of all cells, simulate
def in_cell(c):
    for _, tlo, thi, mlo, mhi, alo, ahi, *_ in cells:
        if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi:
            return True
    return False

print("\n== Strategy: buy favored side when (t,mp,ask) in any robust +EV cell ==")
for max_side in (1, 2):
    used = defaultdict(int)
    sel = []
    for c in sorted(CAND, key=lambda c: (c["start"], c["t"])):
        if not in_cell(c): continue
        key = (c["slug"], c["side"])
        if used[key] >= max_side: continue
        used[key] += 1
        sel.append(c)
    n = len(sel)
    if not n: continue
    w = sum(1 for c in sel if c["won"])
    tot = sum(c["pnl"] for c in sel)
    per_hr = tot / HOURS
    h1 = sum(c["pnl"] for c in sel if c["start"] < MID)
    h2 = sum(c["pnl"] for c in sel if c["start"] >= MID)
    worst = cur = 0.0
    for c in sel:
        cur += c["pnl"]
        worst = min(worst, cur)
    hrs = defaultdict(float)
    for c in sel:
        hrs[c["start"] // 3600 * 3600] += c["pnl"]
    neg_hours = sum(1 for v in hrs.values() if v < 0)
    print(f"  max_side={max_side}: n={n} ({n/HOURS:.2f}/hr) win%={100*w/n:.1f} net=${tot:.2f} (${per_hr:.2f}/hr)")
    print(f"     half1=${h1:+.2f} half2=${h2:+.2f} min_eq=${worst:.2f} hours_with_trades={len(hrs)} losing_hours={neg_hours}")

# 3) Simplified live-friendly single-band rules
print("\n== Simplified single-band rules (live-friendly, one check per poll) ==")
for tlo, thi in [(90, 296), (90, 200), (150, 296)]:
    for mlo, mhi in [(0.55, 0.70), (0.55, 0.68), (0.55, 0.65), (0.58, 0.68), (0.55, 0.72)]:
        for alo, ahi in [(0.40, 0.85), (0.45, 0.85), (0.50, 0.85), (0.50, 0.82), (0.55, 0.85)]:
            sel_all = [c for c in CAND if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
            if len(sel_all) < 100: continue
            n0 = len(sel_all); wr0 = sum(1 for c in sel_all if c["won"])/n0
            pnl0 = sum(c["pnl"] for c in sel_all)
            if wr0 < 0.55 or pnl0 <= 0: continue
            used = defaultdict(int); sel = []
            for c in sorted(sel_all, key=lambda c: (c["start"], c["t"])):
                key = (c["slug"], c["side"])
                if used[key] >= 1: continue
                used[key] += 1
                sel.append(c)
            n = len(sel); tot = sum(c["pnl"] for c in sel)
            per_hr = tot / HOURS
            if per_hr <= 0.3: continue
            h1 = sum(c["pnl"] for c in sel if c["start"] < MID)
            h2 = sum(c["pnl"] for c in sel if c["start"] >= MID)
            w = sum(1 for c in sel if c["won"])
            print(f"  t[{tlo},{thi}) mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}): cell-n={n0} cell-wr={wr0:.3f} | 1/side: n={n} ({n/HOURS:.2f}/hr) win%={100*w/n:.1f} net=${tot:.2f} (${per_hr:.2f}/hr) h1=${h1:+.1f} h2=${h2:+.1f}")
