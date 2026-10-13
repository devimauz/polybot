"""Where is the edge? Find (t, mp, ask) cells where real win rate - ask - fee > 0
with decent n. Edge = buying that side and holding to settle.
Scan fine grid on t, model_p, ask."""
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

CAND = []
for slug, win in W.items():
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295) or (100 <= t < 110): continue
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0.2 <= ask <= 0.99): continue
            mp = p_up if side == "Up" else 1 - p_up
            won = win["winner"] == side
            # EV per $1 stake: p_win*(1/ask - 1 - fee_frac) - (1-p_win)*(1 + fee_frac)
            # where fee_frac = FEE_R*min(ask,1-ask)
            fee_frac = FEE_R * min(ask, 1 - ask)
            ev_per_dollar = (won and 1 or 0) / ask - 1 - fee_frac if won else -(1 + fee_frac)
            CAND.append({"t": t, "mp": mp, "ask": ask, "won": won, "slug": slug,
                         "side": side, "start": win["start"], "ev": ev_per_dollar,
                         "pnl": (STAKE/ask - STAKE - FEE_R*min(ask,1-ask)*STAKE/ask) if won else -(STAKE + FEE_R*min(ask,1-ask)*STAKE/ask)})

def cell_stats(sub, n_min=30):
    if not sub: return None
    n = len(sub)
    wr = sum(1 for c in sub if c["won"]) / n
    avg_ask = mean(c["ask"] for c in sub)
    avg_mp = mean(c["mp"] for c in sub)
    avg_t = mean(c["t"] for c in sub)
    avg_ev = mean(c["ev"] for c in sub)
    avg_pnl = mean(c["pnl"] for c in sub)
    # EV assuming real wr: wr*(1/avg_ask - 1 - fee) - (1-wr)*(1+fee)
    fee = FEE_R * min(avg_ask, 1 - avg_ask)
    ev_assumed = wr * (1/avg_ask - 1 - fee) - (1 - wr) * (1 + fee)
    return n, wr, avg_ask, avg_mp, avg_t, ev_assumed, avg_pnl

print("== t-band x mp-band: real win rate vs ask (need wr > ask + fee for +EV) ==")
t_bands = [(90,150),(150,200),(200,240),(240,270),(270,296)]
mp_bands = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,0.90)]
ask_bands = [(0.20,0.40),(0.40,0.55),(0.55,0.70),(0.70,0.80),(0.80,0.88),(0.88,0.99)]

for tlo, thi in t_bands:
    print(f"\n-- t[{tlo},{thi}) --")
    for mlo, mhi in mp_bands:
        for alo, ahi in ask_bands:
            sub = [c for c in CAND if tlo <= c["t"] < thi and mlo <= c["mp"] < mhi and alo <= c["ask"] < ahi]
            s = cell_stats(sub)
            if not s or s[0] < 30: continue
            n, wr, aq, am, at, ev, pnl = s
            flag = "  *+EV" if ev > 0 else ""
            print(f"  mp[{mlo:.2f},{mhi:.2f}) ask[{alo:.2f},{ahi:.2f}): n={n:4d} wr={wr:.3f} ask={aq:.3f} ev=${ev*STAKE:+.2f} pnl=${pnl:+.2f}{flag}")
