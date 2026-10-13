"""Re-mine +EV cells with the TRUE tradable sample (skip 100-110s everywhere)
and real 1-per-side-per-window execution. This is what a live bot achieves."""
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
        try: d = json.loads(line)
        except ValueError: continue
        if d.get("asset") == "btc":
            wins[d["window"]].append(d)
clob = json.load(open(CLOB, encoding="utf-8"))
W = {}
for slug, rows in wins.items():
    cw = clob.get(slug)
    if not cw or not cw.get("winner"): continue
    W[slug] = {"rows": rows, "winner": cw["winner"], "start": int(slug.rsplit("-", 1)[1])}
starts = sorted(x["start"] for x in W.values())
HOURS = (starts[-1] - starts[0]) / 3600 + 1
MID = (starts[-1] + starts[0]) / 2

def tradable(r):
    inf = r.get("info")
    if inf is None: return False
    t = inf["t"]
    return (90 <= t <= 295) and not (100 <= t < 110)

CAND = []
for slug, win in W.items():
    for r in win["rows"]:
        if not tradable(r): continue
        p_up = r.get("p_up_model")
        if p_up is None: continue
        t = r["info"]["t"]
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

print("tradable candidates:", len(CAND))

def evaluate(tlo, thi, mlo, mhi, alo, ahi, max_side=1):
    qual = [c for c in CAND if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
    used = defaultdict(int); sel = []
    for c in sorted(qual, key=lambda c: (c["start"], c["t"])):
        key = (c["slug"], c["side"])
        if used[key] >= max_side: continue
        used[key] += 1
        sel.append(c)
    n = len(sel)
    if n < 20: return None
    w = sum(1 for c in sel if c["won"])
    tot = sum(c["pnl"] for c in sel)
    h1 = sum(c["pnl"] for c in sel if c["start"] < MID)
    h2 = sum(c["pnl"] for c in sel if c["start"] >= MID)
    worst = cur = 0.0
    for c in sel:
        cur += c["pnl"]; worst = min(worst, cur)
    return dict(n=n, tr_hr=n/HOURS, wr=100*w/n, net=tot, per_hr=tot/HOURS,
                h1=h1, h2=h2, worst=worst, sel=sel)

# Coarse map: which (t,mp,ask) regions are +EV after 1/side execution
print("\n== +EV regions (1/side/window), coarse map ==")
print(f"{'t':>12}{'mp':>12}{'ask':>12}|{'tr':>5}{'tr/hr':>7}{'win%':>6}{'net$':>9}{'$/hr':>7}{'h1':>8}{'h2':>8}{'worst':>8}")
good = []
for (tlo, thi) in [(90,150),(150,200),(200,240),(240,296),(90,200),(200,296)]:
    for (mlo,mhi) in [(0.52,0.62),(0.55,0.68),(0.58,0.72),(0.60,0.75),(0.62,0.78),(0.55,0.75)]:
        for (alo,ahi) in [(0.30,0.65),(0.40,0.70),(0.50,0.75),(0.55,0.80),(0.55,0.85),(0.60,0.85),(0.60,0.90),(0.65,0.90)]:
            r = evaluate(tlo, thi, mlo, mhi, alo, ahi, 1)
            if not r: continue
            if r["per_hr"] > 0:
                good.append((r["per_hr"], tlo, thi, mlo, mhi, alo, ahi, r))
                print(f"{tlo:>4}-{thi:<4}{mlo:>5.2f}-{mhi:.2f}  {alo:>4.2f}-{ahi:.2f}|{r['n']:>5}{r['tr_hr']:>7.2f}{r['wr']:>6.1f}{r['net']:>9.2f}{r['per_hr']:>7.2f}{r['h1']:>8.1f}{r['h2']:>8.1f}{r['worst']:>8.1f}")

print(f"\n+EV regions: {len(good)}")
# The best, and check if any has both halves positive
print("\nBest by $/hr:")
for per_hr, tlo, thi, mlo, mhi, alo, ahi, r in sorted(good, reverse=True)[:12]:
    both = "BOTH+" if r["h1"] > 0 and r["h2"] > 0 else ("h1-only" if r["h1"]>0 else "h2-only")
    print(f"  t[{tlo},{thi}) mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}): n={r['n']} ({r['tr_hr']:.2f}/hr) win%={r['wr']:.1f} ${r['per_hr']:.2f}/hr h1=${r['h1']:+.1f} h2=${r['h2']:+.1f} {both}")
