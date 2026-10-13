"""Calibration: model p_up vs REAL CLOB outcomes, and ask vs real outcomes.
Find if ANY (model_p, ask) region has real win rate beating the ask-implied
cost after fees. Also test the complement: maybe the model is anti-predictive."""
import json
from collections import defaultdict, Counter

ROOT = "C:/Users/user/Desktop/bot"
PATH = ROOT + "/price_logs/twap/windows.jsonl"
CLOB = ROOT + "/price_logs/twap/clob_winners.json"

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

# Calibration of p_up on all valid samples (each sample: does Up win?)
bands = defaultdict(lambda: [0, 0])  # band -> [n, wins_up]
ask_real = defaultdict(lambda: [0, 0])  # (mp_band, ask_band) -> [n, wins_side]
t_bands = defaultdict(lambda: [0, 0])
for slug, win in W.items():
    w_up = 1 if win["winner"] == "Up" else 0
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295): continue
        if 100 <= t < 110: continue
        # model calibration
        b = round(p_up, 1)
        bands[b][0] += 1
        bands[b][1] += w_up
        # per side: model says side prob = mp (up) / 1-mp (down); ask = price
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0 < ask < 1): continue
            mp = p_up if side == "Up" else 1 - p_up
            won = (win["winner"] == side)
            mkb = f"{mp:.1f}"
            akb = f"{ask:.1f}"
            ask_real[(mkb, akb)][0] += 1
            ask_real[(mkb, akb)][1] += 1 if won else 0
        tb = "90-150" if t < 150 else ("150-220" if t < 220 else "220-295")
        t_bands[tb][0] += 1
        t_bands[tb][1] += w_up

print("== Model calibration: p_up band -> real Up win rate ==")
print(f"{'p_up':>5} | {'n':>6} {'real P(Up)':>11} {'bias':>7}")
for b in sorted(bands, key=lambda x: float(x)):
    n, wu = bands[b]
    if n < 20: continue
    real = wu / n
    print(f"{b:>5} | {n:>6} {real:>11.3f} {real - float(b):>7.3f}")

print("\n== Real win rate of BUYING side vs model_p band (mp>=0.55) ==")
# aggregate over ask
mp_real = defaultdict(lambda: [0, 0])
for (mkb, akb), (n, w) in ask_real.items():
    mp_real[mkb][0] += n
    mp_real[mkb][1] += w
print(f"{'mp':>5} | {'n':>6} {'real P(win)':>12} {'avg cost proxy':>14}")
for mkb in sorted(mp_real, key=lambda x: float(x)):
    if float(mkb) < 0.5: continue
    n, w = mp_real[mkb]
    if n < 30: continue
    print(f"{mkb:>5} | {n:>6} {w/n:>12.3f}")

print("\n== Grid: real win rate where model says mp in band, ask in band ==")
# for each (mp band, ask band): real win rate of buying
print(f"{'mp':>5} | " + " ".join(f"{a:>7}" for a in ["0.3-4","0.4-5","0.5-6","0.6-7","0.7-8","0.8-9"]))
mp_bands = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.001]
ask_bands = [(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9)]
grid = defaultdict(lambda: [0, 0])
for slug, win in W.items():
    for r in win["rows"]:
        inf = r.get("info"); p_up = r.get("p_up_model")
        if inf is None or p_up is None: continue
        t = inf["t"]
        if not (90 <= t <= 295) or (100 <= t < 110): continue
        for side in ("Up", "Down"):
            ask = r.get("ask_" + side)
            if not ask or not (0 < ask < 1): continue
            mp = p_up if side == "Up" else 1 - p_up
            if mp < 0.5: continue
            def band(lo, hi):
                for (mlo2, mhi2) in [(mp_bands[i], mp_bands[i+1]) for i in range(len(mp_bands)-1)]:
                    if mlo2 <= mp < mhi2:
                        return mlo2
                return None
            mlo = band(0.5, 1.001)
            if mlo is None: continue
            for (alo, ahi) in ask_bands:
                if alo <= ask < ahi:
                    grid[(mlo, alo)][0] += 1
                    grid[(mlo, alo)][1] += 1 if win["winner"] == side else 0
                    break
for (mlo, alo) in sorted(grid):
    n, w = grid[(mlo, alo)]
    if n < 15: 
        cell = f"{n:>4}-"
    else:
        cell = f"{w/n*100:5.0f}({n})"
    print(f"{mlo:.2f}  | {cell}")
