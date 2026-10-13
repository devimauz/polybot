import json
from collections import defaultdict
ROOT = "C:/Users/user/Desktop/bot"
wins = defaultdict(list)
with open(ROOT + "/price_logs/twap/windows.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("asset") != "btc": continue
        wins[d["window"]].append(d)
for v in wins.values():
    v.sort(key=lambda d: (d.get("info") or {}).get("t") or 0)
clob = json.load(open(ROOT + "/price_logs/twap/clob_winners.json", encoding="utf-8"))
FEE_R = 0.05
STAKE = 10.0
HR = 154.25

def fee_ps(p): return FEE_R * min(p, 1.0 - p)
def up_ask(d):
    a = d.get("ask_Up")
    if a is not None: return a
    b = d.get("bid_Down")
    return (1.0 - b) if b is not None else None
def down_ask(d):
    a = d.get("ask_Down")
    if a is not None: return a
    b = d.get("bid_Up")
    return (1.0 - b) if b is not None else None

def ev_pnl(price, won):
    shares = STAKE / price
    return (shares if won else 0.0) - STAKE - fee_ps(price) * shares

# For each window, take the FIRST sample with model available at a given t.
# Then bin by (fav_p bucket, fav_ask bucket) over windows where we'd BUY the fav side.
# We only get ONE trade per window, so per window pick the sample closest to dec_t.
def cells_at(dec_t):
    """For each window: sample at ~dec_t (first sample with t>=dec_t and model).
    Bin by (fav_p, fav_ask). Return dict[(pb,ab)] -> list of won booleans."""
    cells = defaultdict(list)
    n = 0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok": continue
        winner = c["winner"]
        # first sample with t >= dec_t and model
        d = None
        for s in samples:
            info = s.get("info"); t = info.get("t") if info else None
            if t is not None and t >= dec_t and s.get("p_up_model") is not None:
                d = s; break
        if d is None: continue
        n += 1
        mp = d["p_up_model"]
        fav_p = mp if mp >= 0.5 else 1.0 - mp
        side = "Up" if mp >= 0.5 else "Down"
        ask = up_ask(d) if side == "Up" else down_ask(d)
        if ask is None or not (0.02 <= ask <= 0.98): continue
        won = (side == winner)
        pb = round(fav_p * 5) / 5
        ab = round(ask * 5) / 5
        if ab < 0.04: continue
        won = (side == winner)
        cells[(pb, ab)].append(won)
    return cells, n

for dec_t in [100, 130, 160]:
    cells, n = cells_at(dec_t)
    print(f"\n=== dec_t={dec_t}  windows={n} ===")
    print(f"{'fav_p':>5} {'fav_ask':>8} {'n':>5} {'win%':>6} {'edge':>7} {'$/trade':>9}")
    rows = []
    for (pb, ab), arr in cells.items():
        m = len(arr)
        if m < 8: continue
        wr = sum(arr) / m
        # per-trade pnl using avg ask ~ ab
        avg_pnl = sum(ev_pnl(ab, w) for w in arr) / m
        edge = wr - ab
        rows.append((edge, pb, ab, m, wr, avg_pnl))
    for edge, pb, ab, m, wr, avg_pnl in sorted(rows, reverse=True):
        flag = " <-- +EV" if (wr > ab + 0.02) else ""
        print(f"{pb:5.2f} {ab:8.2f} {m:5d} {wr*100:5.1f}% {edge:+7.3f} {avg_pnl:+9.3f}{flag}")
