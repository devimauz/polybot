"""Strategy B: buy favored side when calibrated(mp) > ask + fee buffer.
Calibration table from real CLOB outcomes: mp -> real P(side wins).
Sweep the buffer + freq caps; report $/hr, trades/hr, robustness."""
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

# Build calibration: for each mp, real P(the favored side wins)
cal = defaultdict(lambda: [0, 0])
for slug, win in W.items():
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295) or (100 <= t < 110): continue
        # favored side = the one with higher model prob
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0 < ask < 1): continue
            mp = p_up if side == "Up" else 1 - p_up
            if mp <= 0.5: continue  # no favored side
            b = round(mp, 2)
            cal[b][0] += 1
            cal[b][1] += 1 if win["winner"] == side else 0

# Smooth calibration
def calibrated(mp):
    # use mp rounded to 0.05 buckets for stability
    b = round(round(mp * 20) / 20, 2)
    keys = [b + d for d in (-0.05, 0, 0.05)]
    n = w = 0
    for k in keys:
        if k in cal:
            n += cal[k][0]; w += cal[k][1]
    return w / n if n else None

# Build candidates
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
            calmp = calibrated(mp)
            if calmp is None: continue
            fee = FEE_R * min(ask, 1 - ask)
            won = win["winner"] == side
            pnl = (STAKE/ask - STAKE - fee*STAKE/ask) if won else -(STAKE + fee*STAKE/ask)
            # breakeven cost: p_win > ask*(1+fee)/(1-ask*fee) ... simpler: EV per trade
            ev = calmp * (STAKE/ask - STAKE - fee*STAKE/ask) - (1-calmp) * (STAKE + fee*STAKE/ask)
            CAND.append({"t": t, "mp": mp, "ask": ask, "calmp": calmp,
                         "slug": slug, "side": side, "won": won, "pnl": pnl,
                         "start": win["start"], "ev": ev})

print("candidates:", len(CAND))
print("\n== calibrated win rate by mp (favored side) ==")
buckets = sorted(set(c["mp"] for c in CAND))
mp_sorted = []
seen = set()
for c in sorted(CAND, key=lambda x: x["mp"]):
    b = round(c["mp"], 2)
    if b in seen: continue
    seen.add(b)
    mp_sorted.append(b)
for b in mp_sorted[::5]:
    sub = [c for c in CAND if abs(c["mp"] - b) < 0.03]
    if sub:
        n = len(sub); wr = sum(1 for c in sub if c["won"])/n
        aq = mean(c["ask"] for c in sub)
        print(f"  mp={b:.2f} cal={calibrated(b):.3f} real_wr={wr:.3f} avg_ask={aq:.3f}")

def sim(buf, t_lo, t_hi, max_per_side, ask_lo, ask_hi, mp_lo):
    used = defaultdict(int)
    sel = []
    for c in sorted(CAND, key=lambda c: (c["start"], c["t"])):
        if not (t_lo <= c["t"] <= t_hi): continue
        if not (ask_lo <= c["ask"] <= ask_hi): continue
        if c["mp"] < mp_lo: continue
        if c["calmp"] - c["ask"] < buf: continue  # cal edge after ask
        if c["ev"] <= 0: continue
        key = (c["slug"], c["side"])
        if used[key] >= max_per_side: continue
        used[key] += 1
        sel.append(c)
    n = len(sel)
    if not n:
        return dict(n=0)
    w = sum(1 for c in sel if c["won"])
    tot = sum(c["pnl"] for c in sel)
    # worst streak
    worst = cur = 0
    for c in sel:
        cur = cur - c["pnl"] if not c["won"] else cur
        worst = max(worst, cur)
    # hourly max: bucket trades by start hour
    hrs = defaultdict(float)
    for c in sel:
        hrs[c["start"] // 3600 * 3600] += c["pnl"]
    best_hour = max(hrs.values()) if hrs else 0
    n_hours = len(set(c["start"] // 3600 for c in sel))
    return dict(n=n, wr=100*w/n, net=tot, per_hr=tot/HOURS,
                worst=-worst, best_hour=best_hour, hours_with=n_hours)

print("\n== Sweep: cal edge buffer (require calibrated_p - ask >= buf) ==")
print(f"{'buf':>5} {'t_lo':>5} {'t_hi':>5} {'mps':>4} {'a_lo':>5} {'a_hi':>5} | {'tr':>4} {'tr/hr':>6} {'win%':>6} {'net$':>9} {'$/hr':>7} {'worst':>7} {'besthr':>7}")
results = []
for buf in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
    for (t_lo, t_hi) in [(90, 295), (90, 200), (120, 280)]:
        for mps in [1, 2]:
            for (a_lo, a_hi) in [(0.20, 0.99), (0.20, 0.95), (0.25, 0.95), (0.30, 0.95), (0.30, 0.90)]:
                for mp_lo in [0.52, 0.55, 0.60]:
                    r = sim(buf, t_lo, t_hi, mps, a_lo, a_hi, mp_lo)
                    if r.get("n", 0) < 5: continue
                    results.append((r["per_hr"], buf, t_lo, t_hi, mps, a_lo, a_hi, mp_lo, r))
                    mark = " *" if r["per_hr"] > 0.5 else ("  " if r["per_hr"] > 0 else " ")
                    print(f"{buf:>5.2f} {t_lo:>5} {t_hi:>5} {mps:>4} {a_lo:>5.2f} {a_hi:>5.2f} | {r['n']:>4} {r['n']/HOURS:>6.2f} {r['wr']:>6.1f} {r['net']:>9.2f} {r['per_hr']:>7.2f} {-r['worst']:>7.1f} {r['best_hour']:>7.1f}{mark}")

print("\nTop 15 gates by $/hr (n>=15):")
for per_hr, buf, t_lo, t_hi, mps, a_lo, a_hi, mp_lo, r in sorted(results, key=lambda x: -x[0])[:15]:
    if r["n"] < 15: continue
    print(f"  buf={buf:.2f} t[{t_lo},{t_hi}] mps={mps} ask[{a_lo},{a_hi}] mp>={mp_lo} -> "
          f"n={r['n']} ({r['n']/HOURS:.2f}/hr) win%={r['wr']:.1f} net=${r['net']:.2f} ${r['per_hr']:.2f}/hr worst=${-r['worst']:.1f} besthr=${r['best_hour']:.2f} hours={r['hours_with']}")
