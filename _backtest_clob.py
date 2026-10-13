"""Backtest using AUTHORITATIVE CLOB winners (no e_final circularity)."""
import json
from collections import defaultdict

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
    rows.sort(key=lambda r: (r.get("info") or {}).get("t") if r.get("info") else -1)
    cw = clob.get(slug)
    if not cw or not cw.get("winner"):
        continue  # only windows with authoritative outcomes
    W[slug] = {"rows": rows, "winner": cw["winner"],
               "start": int(slug.rsplit("-", 1)[1])}

print("windows with CLOB winner:", len(W))
starts = sorted(x["start"] for x in W.values())
HOURS = (starts[-1] - starts[0]) / 3600 + 1
print("span hours:", round(HOURS, 1))
from collections import Counter
print("outcome split:", Counter(x["winner"] for x in W.values()))

def sim(t_lo, t_hi, mp_min, gap_min, ask_lo, ask_hi, max_side=1):
    total = 0.0; n = 0; w = 0
    for slug, win in W.items():
        used = {"Up": 0, "Down": 0}
        for r in win["rows"]:
            inf = r.get("info"); p_up = r.get("p_up_model")
            if inf is None or p_up is None: continue
            t = inf["t"]
            if not (t_lo <= t <= t_hi): continue
            if 100 <= t < 110: continue
            for side in ("Up", "Down"):
                if used[side] >= max_side: continue
                ask = r.get("ask_" + side)
                if not ask or not (0 < ask < 1): continue
                mp = p_up if side == "Up" else 1 - p_up
                if mp < mp_min: continue
                if (mp - ask) < gap_min: continue
                if not (ask_lo <= ask <= ask_hi): continue
                shares = STAKE / ask
                fee = FEE_R * min(ask, 1 - ask) * shares
                won = win["winner"] == side
                total += (shares - STAKE - fee) if won else -(STAKE + fee)
                n += 1; used[side] += 1
                if won: w += 1
    wr = 100.0 * w / n if n else 0
    return n, wr, total, n / HOURS

print(f"\n{'mp_min':>6} {'gap':>5} | {'tr':>4} {'tr/hr':>6} {'win%':>6} {'net$':>9} {'$/tr':>7}")
for mp_min in (0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78):
    for gap_min in (0.00, 0.02, 0.05):
        n, wr, total, per_hr = sim(90, 295, mp_min, gap_min, 0.30, 0.97)
        per = total / n if n else 0
        flag = "  *" if total > 0 and per_hr >= 2 else ""
        print(f"{mp_min:>6.2f} {gap_min:>5.2f} | {n:>4} {per_hr:>6.2f} {wr:>6.1f} {total:>9.2f} {per:>7.2f}{flag}")
