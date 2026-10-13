"""Find the +EV region: for each (mp band, ask band), realized P&L per trade
using authoritative CLOB winners. Then grid a single contiguous gate and
report per-hour P&L with max_side cap."""
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
    cw = clob.get(slug)
    if not cw or not cw.get("winner"):
        continue
    W[slug] = {"rows": rows, "winner": cw["winner"],
               "start": int(slug.rsplit("-", 1)[1])}
starts = sorted(x["start"] for x in W.values())
HOURS = (starts[-1] - starts[0]) / 3600 + 1
print("windows:", len(W), "span hours:", round(HOURS, 1))

# Collect all (sample) candidates with their real win flag
CAND = []
for slug, win in W.items():
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295): continue
        if 100 <= t < 110: continue
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0 < ask < 1): continue
            mp = p_up if side == "Up" else 1 - p_up
            won = win["winner"] == side
            shares = STAKE / ask
            fee = FEE_R * min(ask, 1 - ask) * shares
            pnl = (shares - STAKE - fee) if won else -(STAKE + fee)
            CAND.append({"slug": slug, "side": side, "t": t, "mp": mp,
                         "ask": ask, "won": won, "pnl": pnl,
                         "start": win["start"]})

print("total candidate samples (both sides):", len(CAND))

# Region scan: mp band x ask band, realized $/trade and win rate
print("\n== Realized P&L per trade by (mp band, ask band) ==")
print(f"{'mp':>7} | " + " ".join(f"a{lo:.1f}-{hi:.1f}".rjust(9) for lo, hi in [(0.3,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,0.95)]))
for mlo, mhi in [(0.5,0.55),(0.55,0.6),(0.6,0.65),(0.65,0.7),(0.7,0.75),(0.75,0.8),(0.8,0.85),(0.85,0.9),(0.9,0.95)]:
    row = []
    for alo, ahi in [(0.3,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,0.95)]:
        sub = [c for c in CAND if mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
        if not sub:
            row.append("    -  ")
            continue
        per = sum(c["pnl"] for c in sub)/len(sub)
        wr = 100*sum(1 for c in sub if c["won"])/len(sub)
        mark = "+" if per > 0 else " "
        row.append(f"{mark}{per:5.2f}({len(sub)})")
    print(f"{mlo:.2f}-{mhi:.2f} | " + "  ".join(row))

# Now: gate = mp in [mp_lo,mp_hi], ask in [ask_lo,ask_hi], one trade/side/window
def gate_sim(mp_lo, mp_hi, ask_lo, ask_hi, t_lo=90, t_hi=295, max_side=1):
    used = defaultdict(int)
    sel = []
    # iterate in time order so first-qualifying wins the slot
    for c in sorted(CAND, key=lambda c: (c["start"], c["t"])):
        if c["mp"] < mp_lo or c["mp"] >= mp_hi: continue
        if c["ask"] < ask_lo or c["ask"] >= ask_hi: continue
        if not (t_lo <= c["t"] <= t_hi): continue
        if 100 <= c["t"] < 110: continue
        key = (c["slug"], c["side"])
        if used[key] >= max_side: continue
        used[key] += 1
        sel.append(c)
    n = len(sel)
    if not n:
        return 0, 0, 0, 0
    w = sum(1 for c in sel if c["won"])
    tot = sum(c["pnl"] for c in sel)
    return n, 100*w/n, tot, n/HOURS

print("\n== Gate grid (mp range x ask range), one trade/side/window ==")
print(f"{'mp_lo':>6} {'mp_hi':>6} {'ask_lo':>7} {'ask_hi':>7} | {'tr':>5} {'tr/hr':>6} {'win%':>6} {'net$':>9} {'$/hr':>7}")
best = []
for (mp_lo, mp_hi) in [(0.65,0.72),(0.65,0.75),(0.68,0.75),(0.70,0.78),(0.65,0.80),(0.70,0.85),(0.75,0.85),(0.75,0.90),(0.80,0.90),(0.80,0.95),(0.68,0.85)]:
    for (ask_lo, ask_hi) in [(0.45,0.80),(0.50,0.80),(0.55,0.80),(0.55,0.85),(0.60,0.85),(0.60,0.90),(0.55,0.90),(0.60,0.92),(0.50,0.90),(0.65,0.90)]:
        n, wr, tot, per_hr = gate_sim(mp_lo, mp_hi, ask_lo, ask_hi)
        if not n: continue
        per_hr_net = tot/HOURS
        best.append((per_hr_net, mp_lo, mp_hi, ask_lo, ask_hi, n, wr, tot, per_hr))
        mark = "  *" if per_hr_net > 0 else ""
        print(f"{mp_lo:>6.2f} {mp_hi:>6.2f} {ask_lo:>7.2f} {ask_hi:>7.2f} | {n:>5} {per_hr:>6.2f} {wr:>6.1f} {tot:>9.2f} {per_hr_net:>7.2f}{mark}")

print("\nTop 8 gates by $/hr:")
for b in sorted(best, reverse=True)[:8]:
    per_hr_net, mp_lo, mp_hi, ask_lo, ask_hi, n, wr, tot, per_hr = b
    print(f"  mp[{mp_lo},{mp_hi}) ask[{ask_lo},{ask_hi}) -> {n} trades ({per_hr:.2f}/hr) win%={wr:.1f} net=${tot:.2f} ${per_hr_net:.2f}/hr")
