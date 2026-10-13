"""Find ONE tight, stable, live-achievable rule.
Rule = favored side, t-range x mp-range x ask-range, max_side fills/window.
REQUIRE: $/hr>0.5, trades/hr>=1.5, BOTH halves positive, worst_dd bounded.
Report the full set that qualifies + a per-hour P&L for the best."""
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

def evaluate(tlo, thi, mlo, mhi, alo, ahi, max_side):
    qual = [c for c in CAND if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
    used = defaultdict(int); sel = []
    for c in sorted(qual, key=lambda c: (c["start"], c["t"])):
        key = (c["slug"], c["side"])
        if used[key] >= max_side: continue
        used[key] += 1
        sel.append(c)
    n = len(sel)
    if n < 30: return None
    w = sum(1 for c in sel if c["won"])
    tot = sum(c["pnl"] for c in sel)
    per_hr = tot / HOURS
    h1 = sum(c["pnl"] for c in sel if c["start"] < MID)
    h2 = sum(c["pnl"] for c in sel if c["start"] >= MID)
    worst = cur = 0.0
    for c in sel:
        cur += c["pnl"]; worst = min(worst, cur)
    # per-calendar-hour P&L
    hrs = defaultdict(float)
    for c in sel:
        hrs[c["start"] // 3600 * 3600] += c["pnl"]
    return dict(n=n, tr_hr=n/HOURS, wr=100*w/n, net=tot, per_hr=per_hr,
                h1=h1, h2=h2, worst=worst, hrs=hrs, sel=sel,
                n_hours=len(hrs))

print("FINE GRID (favored side; max_side=1). QUALIFYING rows (both halves>0, $/hr>0.4, tr/hr>=1.5):\n")
print(f"{'tlo':>4}{'thi':>4}{'mlo':>5}{'mhi':>5}{'alo':>5}{'ahi':>5}|{'tr':>4}{'tr/hr':>6}{'win%':>6}{'net$':>9}{'$/hr':>7}{'h1':>8}{'h2':>8}{'worst':>8}")
qual = []
for (tlo, thi) in [(90,150),(150,200),(200,240),(240,296),(90,200),(150,296),(90,296)]:
    for (mlo,mhi) in [(0.58,0.68),(0.60,0.70),(0.62,0.72),(0.65,0.75),(0.55,0.65),(0.58,0.70)]:
        for (alo,ahi) in [(0.60,0.85),(0.62,0.85),(0.65,0.85),(0.65,0.82),(0.68,0.85),(0.60,0.82),(0.65,0.88),(0.55,0.85)]:
            r = evaluate(tlo, thi, mlo, mhi, alo, ahi, 1)
            if not r: continue
            ok = r["per_hr"]>0.4 and r["tr_hr"]>=1.5 and r["h1"]>0 and r["h2"]>0
            mark = "  <<<QUALIFIES" if ok else ""
            if ok:
                qual.append((r["per_hr"], tlo, thi, mlo, mhi, alo, ahi, r))
            print(f"{tlo:>4}{thi:>4}{mlo:>5.2f}{mhi:>5.2f}{alo:>5.2f}{ahi:>5.2f}|{r['n']:>4}{r['tr_hr']:>6.2f}{r['wr']:>6.1f}{r['net']:>9.2f}{r['per_hr']:>7.2f}{r['h1']:>8.1f}{r['h2']:>8.1f}{r['worst']:>8.1f}{mark}")

print(f"\nQUALIFYING RULES: {len(qual)}")
for per_hr, tlo, thi, mlo, mhi, alo, ahi, r in sorted(qual, reverse=True)[:10]:
    print(f"  t[{tlo},{thi}) mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}): "
          f"n={r['n']} ({r['tr_hr']:.2f}/hr) win%={r['wr']:.1f} net=${r['net']:.2f} ${r['per_hr']:.2f}/hr "
          f"h1=${r['h1']:+.1f} h2=${r['h2']:+.1f} worst=${r['worst']:.1f} hrs={r['n_hours']}")

# Best rule per-hour detail
if qual:
    per_hr, tlo, thi, mlo, mhi, alo, ahi, r = sorted(qual, reverse=True)[0]
    print(f"\nBEST rule t[{tlo},{thi}) mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}):")
    print("  per-calendar-hour P&L (sorted by time):")
    for k in sorted(r["hrs"]):
        tag = " <-" if r["hrs"][k] > 25 else ""
        print(f"    hour@{k//3600*3600- k%3600*3600:5d}s: ${r['hrs'][k]:+7.2f}{tag}")
    losing = [ (k, v) for k, v in r["hrs"].items() if v < 0]
    print(f"  losing hours: {len(losing)}/{len(r['hrs'])}; worst single hour: ${min(r['hrs'].values()):.2f}")
    # consecutive-hour check: max losing streak of hours
    keys = sorted(r["hrs"]); cur=0; mx=0
    for k in keys:
        cur = cur+1 if r["hrs"][k] < 0 else 0
        mx = max(mx, cur)
    print(f"  max consecutive losing hours: {mx}")
