"""Final gate validation for the live bot.
GATE (endgame mispricing, favored side):
  t >= 235, model_p >= MP_MIN, MP_MIN <= model_p <= 0.92,
  ASK_LO <= ask <= ASK_HI, (model_p - ask) >= GAP_MIN,
  max 1 fill per (window, side).
Report trades/hr, win%, $/hr, both-halves, worst drawdown, per-calendar-hour.
Also sweep a small neighborhood to confirm stability."""
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
        info = r.get("info")
        p_up = r.get("p_up_model")
        if info is None or p_up is None: continue
        t = info["t"]
        if not (90 <= t <= 295) or (100 <= t < 110): continue
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0.30 <= ask <= 0.97): continue
            mp = p_up if side == "Up" else 1 - p_up
            if mp <= 0.5: continue
            fee = FEE_R * min(ask, 1 - ask)
            won = win["winner"] == side
            pnl = (STAKE/ask - STAKE - fee*STAKE/ask) if won else -(STAKE + fee*STAKE/ask)
            CAND.append({"t": t, "mp": mp, "ask": ask, "slug": slug, "side": side,
                         "won": won, "pnl": pnl, "start": win["start"]})

def evaluate(TLO, MP_MIN, MP_MAX, ALO, AHI, GAP, max_side=1):
    qual = [c for c in CAND if c["t"] >= TLO and MP_MIN <= c["mp"] <= MP_MAX
            and ALO <= c["ask"] <= AHI and (c["mp"] - c["ask"]) >= GAP]
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
    hrs = defaultdict(float)
    for c in sel:
        hrs[c["start"] // 3600 * 3600] += c["pnl"]
    return dict(n=n, tr_hr=n/HOURS, wr=100*w/n, net=tot, per_hr=tot/HOURS,
                h1=h1, h2=h2, worst=worst, hrs=hrs, sel=sel)

print("== neighborhood sweep (favored side, max_side=1) ==")
print(f"{'TLO':>4}{'MPmin':>6}{'MPmax':>6}{'ALo':>5}{'AHi':>5}{'GAP':>5}|{'tr':>4}{'tr/hr':>7}{'win%':>6}{'net$':>9}{'$/hr':>7}{'h1':>8}{'h2':>8}{'worst':>8}")
for TLO in (230, 235, 240):
    for MP_MIN in (0.55, 0.58, 0.60):
        for ALO, AHI in ((0.40, 0.92), (0.45, 0.90), (0.50, 0.90)):
            for GAP in (0.08, 0.10, 0.12):
                r = evaluate(TLO, MP_MIN, 0.92, ALO, AHI, GAP, 1)
                if not r: continue
                both = r["h1"] > 0 and r["h2"] > 0
                if r["per_hr"] > 0:
                    print(f"{TLO:>4}{MP_MIN:>6.2f}{0.92:>6.2f}{ALO:>5.2f}{AHI:>5.2f}{GAP:>5.2f}|"
                          f"{r['n']:>4}{r['tr_hr']:>7.2f}{r['wr']:>6.1f}{r['net']:>9.2f}"
                          f"{r['per_hr']:>7.2f}{r['h1']:>8.1f}{r['h2']:>8.1f}{r['worst']:>8.1f}"
                          f"{'  BOTH+' if both else ''}")

# Chosen rule detail
print("\n== CHOSEN RULE DETAIL ==")
r = evaluate(235, 0.58, 0.92, 0.45, 0.90, 0.10, 1)
if r:
    print(f"t>=235 mp>=0.58 ask[0.45,0.90] gap>=0.10: n={r['n']} ({r['tr_hr']:.2f}/hr) "
          f"win%={r['wr']:.1f} net=${r['net']:.2f} ${r['per_hr']:.2f}/hr "
          f"h1=${r['h1']:+.1f} h2=${r['h2']:+.1f} worst=${r['worst']:.1f}")
    print("  per-calendar-hour P&L:")
    for k in sorted(r["hrs"]):
        print(f"    hour@{k//3600*3600:5d}s: ${r['hrs'][k]:+7.2f}")
    lose = sum(1 for c in r["sel"] if not c["won"])
    print(f"  wins={r['n']-lose} losses={lose}; avg ask={sum(c['ask'] for c in r['sel'])/r['n']:.3f}")
    # a typical winning hour: how many trades & $ ?
    wins_h = [v for v in r["hrs"].values() if v > 0]
    print(f"  winning hours: {len(wins_h)}/{len(r['hrs'])}; best hour ${max(r['hrs'].values()):+.2f}; worst ${min(r['hrs'].values()):+.2f}")
