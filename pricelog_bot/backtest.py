"""Backtest: late-window dog value strategy on the full-book 5m logs.

Strategy: at each frame (causal clock), if not yet entered this window, and
t in [T0,T1], dog ask in [LO,HI] (dog = side with lower mid), buy the dog with
stake S, walk-book fill (FOK: require displayed depth at the ask to cover S),
fee on entry, hold to settlement (winner known from resolution line).

Validation: train/test halves, latency stress (fill at T+L ask, drop if band left),
outcome permutation placebo, concentration, plateau over the grid.
"""
from loader import load, TAKER_FEE
import itertools, math, statistics, sys

ws = load()
items = sorted(ws.items())  # chronological by slug start
N = len(items)

def dog_leader(fr):
    # dog = LOWER-ask side, leader = higher-ask side
    if fr["up"]["ask"] <= fr["dn"]["ask"]:
        return "up", "dn"
    return "dn", "up"

def fill_levels(levels, stake):
    """Walk the ask ladder, fill stake dollars FOK. Returns (shares, fee) or None if depth insufficient at touch for full stake."""
    # FOK strict: full stake must fill; require touch displayed size*price >= stake portion at touch,
    # allow walking levels (market order).
    rem = stake
    shares = 0.0
    if not levels: return None
    touch_val = levels[0][0] * levels[0][1]
    if touch_val < min(stake, 5.0 * levels[0][0]):
        # displayed depth at touch can't even cover the min order -> reject (FOK insufficient size)
        return None
    for p, sz in levels:
        take = min(rem / p, sz)
        shares += take
        rem -= take * p
        if rem <= 1e-9: break
    if rem > 1e-6:
        return None
    return shares

def run(T0, T1, LO, HI, stake=10.0, latency=0.0, subset=None):
    trades = []
    for idx, (slug, w) in enumerate(items):
        if subset is not None and idx not in subset: continue
        frames = w["frames"]
        pos = None
        for i, fr in enumerate(frames):
            t = fr["t"]
            if t < T0: continue
            if t >= T1: break
            if pos is not None: continue  # one entry per window (no lookahead state needed)
            dog, lead = dog_leader(fr)
            d = fr[dog]
            a = d["ask"]
            if LO <= a < HI:
                # decide now; fill at t+latency book
                fill_fr = fr
                if latency > 0:
                    j = next((k for k in range(i+1, len(frames)) if frames[k]["t"] >= t + latency), None)
                    if j is None: break
                    fill_fr = frames[j]
                d2 = fill_fr[dog]
                if not (LO <= d2["ask"] < HI):  # band left under latency -> FOK reject
                    pos = "REJ"
                    continue
                sh = fill_levels(d2["asks"], stake)
                if sh is None:
                    pos = "REJ"
                    continue
                fee = TAKER_FEE * min(d2["ask"], 1 - d2["ask"]) * sh
                cost = stake + fee
                win = (dog == "up") == (w["winner"] == "Up")
                pnl = (sh * 1.0 if win else 0.0) - cost
                trades.append(dict(slug=slug, side=dog, t=t, ask=d2["ask"], shares=sh,
                                  pnl=pnl, win=win, cost=cost))
                pos = "IN"
        # REJ entries mean no trade this window
    return [tr for tr in trades]

def stats(trades):
    if not trades: return dict(n=0, net=0.0, t=0.0, wr=0.0, worst=0.0, roi=0.0, conc5=0.0)
    pnls = [tr["pnl"] for tr in trades]
    net = sum(pnls)
    sd = statistics.pstdev(pnls) or 1e-9
    tstat = (net / len(pnls)) / (sd / math.sqrt(len(pnls))) if len(pnls) > 1 else 0.0
    top5 = sum(sorted(pnls)[-5:])
    spend = sum(tr["cost"] for tr in trades)
    return dict(n=len(pnls), net=net, t=tstat, wr=sum(tr["win"] for tr in trades)/len(pnls),
                worst=min(pnls), roi=net/spend*100, conc5=(top5/net if abs(net)>1e-9 else float('nan')),
                up=sum(1 for tr in trades if tr['side']=='up'), dn=len(pnls)-sum(1 for tr in trades if tr['side']=='up'))

if __name__ == "__main__":
    grid_T0 = [20, 60, 120, 180]
    grid_T1 = [240, 270, 285]
    grid_LO = [0.05, 0.10, 0.15, 0.20]
    grid_HI = [0.30, 0.40, 0.50]
    train = set(range(0, N//2)); test = set(range(N//2, N))
    res = []
    for T0, T1, LO, HI in itertools.product(grid_T0, grid_T1, grid_LO, grid_HI):
        if T0 >= T1: continue
        tr_tr = stats(run(T0, T1, LO, HI, subset=train))
        te_tr = stats(run(T0, T1, LO, HI, subset=test))
        if tr_tr["n"] >= 25:
            res.append((te_tr["net"] + tr_tr["net"], T0, T1, LO, HI, tr_tr, te_tr))
    res.sort(reverse=True)
    print(f"windows train={len(train)} test={len(test)}; grid cells with >=25 train trades: {len(res)}")
    pos_test = sum(1 for r in res if r[6]["net"] > 0)
    print(f"family sign test: {pos_test}/{len(res)} rules positive OUT of sample (test half)")
    print("\ntop 10 by full-sample net:")
    for tot, T0, T1, LO, HI, a, b in res[:10]:
        print(f"T[{T0},{T1}) ask[{LO},{HI}) | train n={a['n']:>3} net={a['net']:+8.2f} t={a['t']:+5.2f} wr={a['wr']:.2f} | "
              f"test n={b['n']:>3} net={b['net']:+8.2f} t={b['t']:+5.2f} wr={b['wr']:.2f}")
    # plateau check on neighbors of the winner
    print("\nPLATEAU around winner:")
    _, T0, T1, LO, HI, _, _ = res[0]
    for dT0 in (-15, 0, 15):
        for dLO in (-0.05, 0, 0.05):
            a = stats(run(T0+dT0, T1, round(LO+dLO,2), HI, subset=train))
            b = stats(run(T0+dT0, T1, round(LO+dLO,2), HI, subset=test))
            print(f"T0={T0+dT0} LO={round(LO+dLO,2)}: train n={a['n']} net={a['net']:+.2f} | test net={b['net']:+.2f} (n={b['n']})")
    # latency stress on winner
    print("\nLATENCY stress on winner (full sample):")
    for L in (0.0, 0.3, 0.5, 0.8, 1.0):
        s = stats(run(T0, T1, LO, HI, latency=L))
        print(f"L={L}: n={s['n']} net={s['net']:+.2f} t={s['t']:+.2f} wr={s['wr']:.3f} worst={s['worst']:+.2f}")
    # placebo: permute winners across windows
    import random
    print("\nPLACEBO (outcome permutation, 200 draws):")
    winners = [w["winner"] for _, w in items]
    obs = stats(run(T0, T1, LO, HI))
    obs_net = obs["net"]
    cnt = 0
    random.seed(7)
    for _ in range(200):
        perm = winners[:]
        random.shuffle(perm)
        # rebuild winner map
        for (slug, w), pw in zip(items, perm):
            w["winner"] = pw
        pl = stats(run(T0, T1, LO, HI))
        if pl["net"] >= obs_net: cnt += 1
    print(f"observed net={obs_net:+.2f}; placebo p(perm >= obs) = {cnt/200:.3f}")
    for (slug, w), pw in zip(items, winners):
        w["winner"] = pw
    print("\nfinal winner rule stats (full sample):", {k: (round(v,3) if isinstance(v,float) else v) for k,v in obs.items()})