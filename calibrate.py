"""Calibrate + choose a +EV config for twap_bot from calib.jsonl.

Each sample stores e_final, ref, sd (floored), spot, sigma, t, and the live
ask/bid for Up & Down at that instant. sd can only be raised (a higher
uncertainty floor), so for a floor F:
    sd_F = max(sd_stored, F*spot);  z_F=(e_final-ref)/sd_F;  p_up=Phi(z_F)
Join each sample to its window's true winner, then replay the bot's exact
trade rule for a sweep of (floor, conf band, edge) and report realized PnL.
"""
import json, math, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
p = ROOT / "price_logs/twap/calib.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
samples = [r for r in rows if r.get("kind") == "sample"]
outcomes = {r["window"]: r for r in rows if r.get("kind") == "outcome"}

def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# Build replayable tradeable moments
moments = []
for s in samples:
    if s.get("p_up") is None:
        continue
    win = s.get("window")
    o = outcomes.get(win)
    if not o or not o.get("winner"):
        continue
    info = s.get("info") or {}
    e = info.get("e_final"); ref = s.get("ref"); sd = info.get("sd"); spot = s.get("spot")
    if not (e and ref and sd and spot):
        continue
    t = s.get("t")
    if not (60 <= t <= 240):          # bot's tradeable band
        continue
    moments.append({
        "window": win, "t": t, "winner": o["winner"],
        "e_final": e, "ref": ref, "sd": sd, "spot": spot,
        "ask_Up": s.get("ask_Up"), "bid_Up": s.get("bid_Up"),
        "ask_Down": s.get("ask_Down"), "bid_Down": s.get("bid_Down"),
    })

print(f"resolved windows with tradeable moments: {len({m['window'] for m in moments})}, moments={len(moments)}")

FEE_R = 0.05
def simulate(floor_frac, lo, hi, max_ask, min_ask, min_edge, stake=10.0, t_lo=60, t_hi=240):
    trades = []
    for m in moments:
        if not (t_lo <= m["t"] <= t_hi):
            continue
        sd_f = max(m["sd"], floor_frac * m["spot"])
        z = (m["e_final"] - m["ref"]) / sd_f
        p_up = phi(z)
        best = None
        for outcome in ("Up", "Down"):
            ask = m.get(f"ask_{outcome}")
            model_p = p_up if outcome == "Up" else 1 - p_up
            if not (lo <= model_p <= hi):
                continue
            if not ask or not (min_ask <= ask <= max_ask):
                continue
            shares = stake / ask
            fee = FEE_R * min(ask, 1 - ask) * shares
            ev = (model_p * (shares - stake - fee) - (1 - model_p) * (stake + fee))
            if ev <= min_edge * stake:
                continue
            cand = (ev, model_p, outcome, ask)
            if best is None or cand[0] > best[0]:
                best = cand
        if best:
            ev, model_p, outcome, ask = best
            won = (outcome == m["winner"])
            shares = stake / ask
            fee = FEE_R * min(ask, 1 - ask) * shares
            pnl = (shares if won else 0.0) - stake - fee
            trades.append({"outcome": outcome, "won": won, "model_p": model_p,
                           "ask": ask, "pnl": pnl, "ev": ev, "window": m["window"]})
    return trades

def report(trades, label):
    n = len(trades)
    if n == 0:
        print(f"{label:42s}  no trades")
        return
    pnl = sum(t["pnl"] for t in trades)
    ev = sum(t["ev"] for t in trades)
    wr = sum(1 for t in trades if t["won"]) / n
    avg_mp = statistics.mean(t["model_p"] for t in trades)
    avg_ask = statistics.mean(t["ask"] for t in trades)
    avg_pnl = pnl / n
    print(f"{label:42s}  n={n:3d} wr={wr*100:5.1f}%  avgP={avg_mp:.3f} avgAsk={avg_ask:.3f}  EV=${ev:7.2f} PnL=${pnl:8.2f} avg=${avg_pnl:+.3f}")
    return n, wr, pnl

print("\n=== Sweep uncertainty floor (conf [0.35,0.65], edge 3%, ask [0.15,0.85]) ===")
for F in (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015, 0.02, 0.03):
    report(simulate(F, 0.35, 0.65, 0.85, 0.15, 0.03), f"floor={F}")

print("\n=== Sweep conf band (floor 0.006, edge 3%) ===")
for lo, hi in ((0.40, 0.60), (0.35, 0.65), (0.45, 0.55), (0.30, 0.70), (0.40, 0.65), (0.50, 0.50)):
    report(simulate(0.006, lo, hi, 0.85, 0.15, 0.03), f"band=[{lo},{hi}]")

print("\n=== Sweep edge threshold (floor 0.006, band [0.35,0.65]) ===")
for e in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
    report(simulate(0.006, 0.35, 0.65, 0.85, 0.15, e), f"edge={e}")

print("\n=== Calibration: model p_up vs actual Up rate (raw, t in 60-240) ===")
buckets = defaultdict(list)
for m in moments:
    z = (m["e_final"] - m["ref"]) / m["sd"]
    p_up = phi(z)
    buckets[min(0.9, int(p_up * 10) / 10)].append(m["winner"] == "Up")
for k in sorted(buckets):
    v = buckets[k]
    print(f"  model in [{k:.1f},{min(k+0.1,1.0):.1f}): n={len(v):3d} actualUp={sum(v)/len(v)*100:5.1f}%")
