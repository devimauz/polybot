"""Pair-arb test: buy BOTH legs when ask sum <= TH; latency + FOK depth on both legs."""
from backtest import items, N
from loader import TAKER_FEE
import statistics, itertools

train = set(range(0, N//2)); test = set(range(N//2, N))

def run_pair(TH, latency=0.0, stake=10.0, subset=None):
    trades = []
    for idx, (slug, w) in enumerate(items):
        if subset is not None and idx not in subset: continue
        frames = w["frames"]; done = False
        for i, fr in enumerate(frames):
            if done: break
            t = fr["t"]
            if not (5 <= t < 290): continue
            if fr["up"]["ask"] + fr["dn"]["ask"] > TH: continue
            fill_fr = fr
            if latency > 0:
                j = next((k for k in range(i+1, len(frames)) if frames[k]["t"] >= t + latency), None)
                if j is None: break
                fill_fr = frames[j]
            ok = True; legs = {}
            for side in ("up", "dn"):
                d2 = fill_fr[side]
                if d2["ask"] >= 0.99: ok = False; break
                rem = stake; sh = 0.0
                touch = d2["asks"][0] if d2["asks"] else None
                if touch is None or touch[0]*touch[1] < min(stake, 5.0*touch[0]): ok = False; break
                for p, sz in d2["asks"]:
                    take = min(rem/p, sz); sh += take; rem -= take*p
                    if rem <= 1e-9: break
                if rem > 1e-6: ok = False; break
                legs[side] = (sh, TAKER_FEE*min(d2["ask"], 1-d2["ask"])*sh)
            if not ok: done = True; continue  # pair missed; don't retry this window (one pair per window)
            cost = 2*stake + legs["up"][1] + legs["dn"][1]
            win_side = "up" if w["winner"] == "Up" else "dn"
            pnl = legs[win_side][0] - cost
            trades.append(dict(slug=slug, pnl=pnl, win=True, cost=cost))
            done = True
    return trades

def stats(tr):
    if not tr: return dict(n=0, net=0.0, t=0.0)
    pnls=[x["pnl"] for x in tr]; net=sum(pnls)
    sd=statistics.pstdev(pnls) or 1e-9
    return dict(n=len(pnls), net=net, t=(net/len(pnls))/(sd/statistics.sqrt if False else __import__('math').sqrt(len(pnls))) if len(pnls)>1 else 0.0)

if __name__ == "__main__":
    for TH in (0.94, 0.96, 0.97):
        for L in (0.0, 0.5, 1.0):
            a = stats(run_pair(TH, L, subset=train)); b = stats(run_pair(TH, L, subset=test))
            print(f"TH={TH} L={L}: train n={a['n']:>3} net={a['net']:+8.2f} | test n={b['n']:>3} net={b['net']:+8.2f}")