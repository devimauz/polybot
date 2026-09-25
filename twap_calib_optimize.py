"""Offline param optimizer for the TWAP model.

Loads calib.jsonl. Each 'sample' row has the raw model ingredients
(e_final, ref, spot, sd_raw, sigma, t) plus market ask/bid/mid for both
outcomes; each 'outcome' row gives the true winner per window.

We recompute p_up under different sd floors and vol scales, then score:
  1. Model Brier (lower = better) vs market-mid Brier
  2. Calibration (confidence buckets)
  3. Taker EV: buy the side where model_p - ask > edge_thresh, size=STAKE,
     fee=KEY*min(ask,1-ask)*shares, only if CONF_LO<=model_p<=CONF_HI.
     Report n, win-rate, total EV, and PnL after a realistic fill.

The goal: find (floor, vol_scale, edge_thresh, conf_band) that maximizes
realistic per-trade EV with enough trades.
"""
import json, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
def norm_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def main(path=ROOT/"price_logs/twap/calib.jsonl", STAKE=10.0, KEY=0.05):
    rows = load(path)
    outc = {r["window"]: r["winner"] for r in rows if r.get("kind")=="outcome"}
    samples = [r for r in rows if r.get("kind")=="sample" and r.get("info")]
    samples = [s for s in samples if s["window"] in outc]
    print(f"samples with outcome: {len(samples)}")
    if len(samples) < 20:
        print("not enough data"); return

    # Recompute p_up under (floor, vol_scale):
    #   sd_use = max(sd_raw*vol_scale, floor*spot)
    #   z = (e_final - ref)/sd_use
    def p_up_for(s, floor, vol_scale):
        inf = s["info"]
        e_final = inf["e_final"]; ref = s["ref"]; spot = s["spot"]
        sd_raw = inf.get("sd_raw", inf["sd"])
        sd = max(sd_raw*vol_scale, floor*spot)
        z = (e_final - ref)/sd if sd>0 else 0.0
        return norm_cdf(z)

    def evaluate(floor, vol_scale, edge_thresh, conf_lo, conf_hi):
        # per-sample model prob
        brier = 0.0
        trades = []   # (ev, won, side, ask, mp)
        for s in samples:
            y = (outc[s["window"]] == "Up")
            p = p_up_for(s, floor, vol_scale)
            brier += (p - (1.0 if y else 0.0))**2
            for outcome in ("Up","Down"):
                ask = s.get(f"ask_{outcome}")
                if ask is None: continue
                mp = p if outcome=="Up" else 1-p
                if mp - ask <= edge_thresh: continue
                if not (conf_lo <= mp <= conf_hi): continue
                shares = STAKE/ask
                fee = KEY*min(ask,1-ask)*shares
                won = (outcome=="Up" and y) or (outcome=="Down" and not y)
                ev = mp*(shares-STAKE-fee) - (1-mp)*(STAKE+fee)
                trades.append((ev, won, outcome, ask, mp))
            break  # only evaluate each window once at its t
        # market-mid Brier for comparison
        mb = 0.0; nmid=0
        for s in samples:
            y = (outc[s["window"]]=="Up")
            if s.get("mid_Up") is not None:
                mb += (s["mid_Up"] - (1.0 if y else 0.0))**2; nmid+=1
        brier /= len(samples)
        mb = mb/nmid if nmid else float("nan")
        if not trades:
            return {"brier": brier, "mkt_brier": mb, "n":0}
        tot = sum(t[0] for t in trades)
        wins = sum(1 for t in trades if t[1])
        return {"brier": brier, "mkt_brier": mb, "n": len(trades),
                "win": wins/len(trades), "tot_ev": tot,
                "avg_ev": tot/len(trades)}

    print(f"\n{'floor':>7} {'volscl':>6} {'edge':>5} {'conf':>9}  {'n':>3} {'win%':>5} {'Brier':>6} {'mktB':>6} {'totEV':>8} {'avgEV':>7}")
    best = None
    for floor in (0.0, 0.0005, 0.001, 0.0012, 0.0015, 0.002, 0.003, 0.005):
        for vol_scale in (1.0, 1.5, 2.0, 3.0):
            for edge_thresh in (0.03, 0.05, 0.08, 0.10):
                for conf_lo, conf_hi in ((0.15,0.85),(0.20,0.80),(0.10,0.90)):
                    r = evaluate(floor, vol_scale, edge_thresh, conf_lo, conf_hi)
                    if r["n"] < 5: continue
                    print(f"{floor:7.4f} {vol_scale:6.2f} {edge_thresh:5.2f} {conf_lo}-{conf_hi}  {r['n']:3d} {r['win']:5.1%} {r['brier']:6.4f} {r['mkt_brier']:6.4f} {r['tot_ev']:8.2f} {r['avg_ev']:7.3f}")
                    # pick best by total EV with n>=10 and avg_ev>0
                    if best is None or (r["n"]>=10 and r["avg_ev"]>0 and r["tot_ev"]>best[1]["tot_ev"]):
                        best = ((floor,vol_scale,edge_thresh,conf_lo,conf_hi), r)
    if best:
        print(f"\nBEST (max total EV, n>=10, avgEV>0): floor={best[0][0]} vol_scale={best[0][1]} edge={best[0][2]} conf=[{best[0][3]},{best[0][4]}]")
        print(f"   n={best[1]['n']} win={best[1]['win']:.1%} Brier={best[1]['brier']:.4f} totEV=${best[1]['tot_ev']:+.2f} avgEV=${best[1]['avg_ev']:+.3f}")
    else:
        print("\nNo config with n>=10 and positive avgEV.")

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv)>1 else None
    main(p)
