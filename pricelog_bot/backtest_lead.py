"""Leader-side grid backtest + validation: buy the FAVORITE (higher ask) in [LO,HI) during [T0,T1)."""
from backtest import run, stats, items, N
import itertools

train = set(range(0, N//2)); test = set(range(N//2, N))

def run_lead(T0, T1, LO, HI, stake=10.0, latency=0.0, subset=None, tp=None):
    """Copy of backtest.run but buying the LEADER (higher-ask side)."""
    from loader import TAKER_FEE
    trades = []
    for idx, (slug, w) in enumerate(items):
        if subset is not None and idx not in subset: continue
        frames = w["frames"]; pos = None
        for i, fr in enumerate(frames):
            t = fr["t"]
            if t < T0: continue
            if t >= T1: break
            if pos is not None: continue
            lead = "dn" if fr["up"]["ask"] <= fr["dn"]["ask"] else "up"  # leader = higher ask
            d = fr[lead]; a = d["ask"]
            if LO <= a < HI:
                fill_fr = fr
                if latency > 0:
                    j = next((k for k in range(i+1, len(frames)) if frames[k]["t"] >= t + latency), None)
                    if j is None: break
                    fill_fr = frames[j]
                d2 = fill_fr[lead]
                if not (LO <= d2["ask"] < HI): pos = "REJ"; continue
                sh = None; rem = stake
                if d2["asks"]:
                    touch = d2["asks"][0]
                    if touch[0]*touch[1] < min(stake, 5.0*touch[0]): pos = "REJ"; continue
                    sh = 0.0
                    for p, sz in d2["asks"]:
                        take = min(rem/p, sz); sh += take; rem -= take*p
                        if rem <= 1e-9: break
                if sh is None or rem > 1e-6: pos = "REJ"; continue
                fee = TAKER_FEE * min(d2["ask"], 1-d2["ask"]) * sh
                cost = stake + fee
                win = (lead == "up") == (w["winner"] == "Up")
                trades.append(dict(slug=slug, side=lead, t=t, ask=d2["ask"], shares=sh,
                                  pnl=(sh if win else 0.0) - cost, win=win, cost=cost))
                pos = "IN"
    return trades

if __name__ == "__main__":
    grid_T0 = [90, 120, 150, 180]
    grid_T1 = [210, 240, 270]
    grid_LO = [0.60, 0.70, 0.80, 0.85]
    grid_HI = [0.90, 0.93, 0.95]
    res = []
    for T0, T1, LO, HI in itertools.product(grid_T0, grid_T1, grid_LO, grid_HI):
        if T0 >= T1: continue
        a = stats(run_lead(T0, T1, LO, HI, subset=train))
        b = stats(run_lead(T0, T1, LO, HI, subset=test))
        res.append((a["net"] + b["net"], T0, T1, LO, HI, a, b))
    res.sort(reverse=True)
    for tot, T0, T1, LO, HI, a, b in res[:10]:
        print(f"T[{T0},{T1}) fav[{LO},{HI}) | train n={a['n']:>3} net={a['net']:+8.2f} t={a['t']:+5.2f} wr={a['wr']:.3f} | test n={b['n']:>3} net={b['net']:+8.2f} t={b['t']:+5.2f} wr={b['wr']:.3f}")
    pos_test = sum(1 for r in res if r[6]["net"] > 0)
    print(f"\nfamily sign: {pos_test}/{len(res)} positive out of sample")