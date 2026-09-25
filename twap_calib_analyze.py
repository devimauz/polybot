"""Analyze calibration data: does the TWAP model beat the market?

Reads price_logs/twap/calib.jsonl. For each (asset, window, t*) sample we have:
  p_up (model), ask/bid for both outcomes, market mid, and (from outcome rows)
  the true winner.

Outputs:
  1. Model Brier vs market-mid Brier (lower is better).
  2. Calibration of the model (predicted prob vs realized frequency).
  3. Taker EV: for the best side, EV = model_p*(shares - stake - fee) - (1-model_p)*(stake+fee)
     using the ask, stake=$10, fee = 0.05*min(p,1-p)*shares. Mean/median EV, and
     EV of the "trade only when EV > threshold" strategy.
  4. Edge over market: model_p - ask (for the side we'd buy), per bin.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEY = 0.05
STAKE = 10.0

def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def load(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def main():
    rows = load(ROOT / "price_logs" / "twap" / "calib.jsonl")
    samples = [r for r in rows if r.get("kind") == "sample" and r.get("p_up") is not None]
    outcomes = {r["window"]: r["winner"] for r in rows if r.get("kind") == "outcome"}
    print(f"total rows: {len(rows)}, decision samples: {len(samples)}, outcomes: {len(outcomes)}")
    if len(samples) < 10:
        print("not enough data yet"); return

    # attach winner to samples
    matched = [s for s in samples if s["window"] in outcomes]
    print(f"samples with known outcome: {len(matched)}")
    if len(matched) < 10:
        print("waiting for more resolved windows"); return

    # ---- Brier: model vs market mid ----
    def brier(p, y): return (p - y) ** 2
    mb, mkb = 0.0, 0.0
    n = 0
    for s in matched:
        y = 1.0 if outcomes[s["window"]] == "Up" else 0.0
        mb += brier(s["p_up"], y)
        mid = None
        if s.get("bid_Up") is not None and s.get("ask_Up") is not None:
            mid = (s["bid_Up"] + s["ask_Up"]) / 2
        if mid is not None:
            mkb += brier(mid, y); n += 1
    print(f"\n== Brier (lower=better) over {len(matched)} windows ==")
    print(f"  model p_up : {mb/len(matched):.4f}")
    if n: print(f"  market mid : {mkb/n:.4f}   (n={n})")

    # ---- calibration of model ----
    print("\n== Model calibration (predicted p_up vs realized Up rate) ==")
    bins = defaultdict(list)
    for s in matched:
        y = 1.0 if outcomes[s["window"]] == "Up" else 0.0
        b = min(int(s["p_up"] * 5), 4)
        bins[b].append(y)
    for b in range(5):
        v = bins.get(b, [])
        if not v: continue
        print(f"  p_up in [{b*0.2:.1f},{(b+1)*0.2:.1f}): n={len(v):3d}  realized={sum(v)/len(v):.3f}")

    # ---- Taker EV per side ----
    print("\n== Taker EV (stake=$%.0f, fee=%.0f%%) ==" % (STAKE, KEY*100))
    evs = []
    for s in matched:
        for outcome in ("Up", "Down"):
            ask = s.get(f"ask_{outcome}")
            mp = s.get(f"model_{outcome}")
            if ask is None or mp is None or not (0.02 <= ask <= 0.98):
                continue
            shares = STAKE / ask
            fee = KEY * min(ask, 1 - ask) * shares
            ev = mp * (shares - STAKE - fee) - (1 - mp) * (STAKE + fee)
            evs.append((ev, outcome, ask, mp, s["window"], s.get("t")))
    evs.sort(reverse=True)
    if evs:
        import statistics
        vals = [e[0] for e in evs]
        print(f"  n side-opportunities: {len(evs)}")
        print(f"  mean EV: {statistics.mean(vals):+.4f}   median: {statistics.median(vals):+.4f}")
        print(f"  best: EV={evs[0][0]:+.3f} {evs[0][1]} ask={evs[0][2]:.3f} model={evs[0][3]:.3f}")
        print(f"  worst: EV={evs[-1][0]:+.3f} {evs[-1][1]} ask={evs[-1][2]:.3f} model={evs[-1][3]:.3f}")
        for thr in (0.0, 0.5, 1.0, 2.0):
            sel = [e for e in evs if e[0] > thr]
            if sel:
                print(f"  EV>{thr:+.1f}: n={len(sel):3d}  total=${sum(e[0] for e in sel):+.2f}  avg=${statistics.mean([e[0] for e in sel]):+.3f}")

if __name__ == "__main__":
    main()
