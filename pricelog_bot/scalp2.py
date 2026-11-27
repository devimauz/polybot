"""Scalp variant: enter on p_model - ask >= EDGE_IN, exit at bid when p_model - bid <= EDGE_OUT.
Multiple round trips per window, one position at a time. Fees both legs. Settle if still held at end."""
from loader import load, TAKER_FEE
from model import cdf, SIGMA_1S, model_frames, window_open
import math, statistics, itertools

ws = load(); items = sorted(ws.items())
N = len(items)

_p_cache = {}
def p_side(w, side, t):
    key = id(w)
    if key not in _p_cache:
        op = window_open(w) or 1.0
        _p_cache[key] = {round(ft, 1): p for ft, p, _ in model_frames(w, op)}
    p = _p_cache[key].get(round(t, 1), 0.5)
    return p if side == "up" else 1 - p

def run(T0, T1, EDGE_IN, EDGE_OUT, stake=10.0, latency=0.0, subset=None, t_exit=292):
    trades = []
    for idx, (slug, w) in enumerate(items):
        if subset is not None and idx not in subset: continue
        open_px = window_open(w)
        if not open_px: continue
        mfs = model_frames(w, open_px)
        pos = None
        for i, (t, p_up, fr) in enumerate(mfs):
            if t >= T1: break
            if pos is not None:
                p_here = p_side(w, pos["side"], t)
                bid = fr[pos["side"]]["bid"]
                if (p_here - bid) <= EDGE_OUT or t >= t_exit:
                    fills = fr[pos["side"]]["bids"]
                    rem_sh = pos["shares"]; proceeds = 0.0
                    for p, sz in fills:
                        take = min(rem_sh, sz); proceeds += take * p; rem_sh -= take
                        if rem_sh <= 1e-9: break
                    if rem_sh > 1e-6:
                        if t >= t_exit:  # must settle
                            win = (pos["side"] == "up") == (w["winner"] == "Up")
                            trades.append(dict(slug=slug, side=pos["side"], t=pos["t_entry"],
                                               pnl=(pos["shares"] if win else 0.0) - pos["cost"],
                                               win=win, cost=pos["cost"], ex="settle"))
                            pos = None
                        continue  # can't fully exit yet
                    fee_ex = TAKER_FEE * min(bid, 1 - bid) * pos["shares"]
                    pnl = proceeds - fee_ex - pos["cost"]
                    trades.append(dict(slug=slug, side=pos["side"], t=pos["t_entry"], pnl=pnl,
                                       win=pnl > 0, cost=pos["cost"], ex="scalp"))
                    pos = None
                continue
            if t < T0: continue
            gu = p_up - fr["up"]["ask"]
            gd = (1 - p_up) - fr["dn"]["ask"]
            side, pm = (None, None)
            if gu >= EDGE_IN and gu >= gd: side, pm = "up", p_up
            elif gd >= EDGE_IN: side, pm = "dn", 1 - p_up
            if side is None: continue
            fill_fr = fr
            if latency > 0:
                j = next((k for k in range(i+1, len(mfs)) if mfs[k][0] >= t + latency), None)
                if j is None: break
                fill_fr = mfs[j][2]
            d2 = fill_fr[side]; a2 = d2["ask"]
            if a2 >= pm: continue  # no longer cheap; keep scanning (no latch)
            rem = stake; sh = 0.0
            if not d2["asks"]: continue
            touch = d2["asks"][0]
            if touch[0] * touch[1] < min(stake, 5.0 * touch[0]): continue
            for p, sz in d2["asks"]:
                take = min(rem / p, sz); sh += take; rem -= take * p
                if rem <= 1e-9: break
            if rem > 1e-6 or sh < 5: continue
            fee = TAKER_FEE * min(a2, 1 - a2) * sh
            pos = dict(side=side, shares=sh, cost=stake + fee, t_entry=t)
        if pos is not None:
            win = (pos["side"] == "up") == (w["winner"] == "Up")
            trades.append(dict(slug=slug, side=pos["side"], t=pos["t_entry"],
                               pnl=(pos["shares"] if win else 0.0) - pos["cost"],
                               win=win, cost=pos["cost"], ex="settle"))
    return trades

def stats(trades):
    if not trades: return dict(n=0, net=0.0, t=0.0, wr=0.0, worst=0.0, roi=0.0, up=0, dn=0, scalp=0, settle=0)
    pnls = [tr["pnl"] for tr in trades]
    net = sum(pnls); sd = statistics.pstdev(pnls) or 1e-9
    tstat = (net/len(pnls))/(sd/math.sqrt(len(pnls))) if len(pnls) > 1 else 0.0
    return dict(n=len(pnls), net=net, t=tstat, wr=sum(tr["win"] for tr in trades)/len(pnls),
                worst=min(pnls), roi=net/sum(tr["cost"] for tr in trades)*100,
                up=sum(1 for tr in trades if tr["side"]=="up"), dn=len(pnls)-sum(1 for tr in trades if tr["side"]=="up"),
                scalp=sum(1 for tr in trades if tr["ex"]=="scalp"), settle=sum(1 for tr in trades if tr["ex"]=="settle"))

if __name__ == "__main__":
    train = set(range(0, N//2)); test = set(range(N//2, N))
    res = []
    for T0, T1, EI, EO in itertools.product([30, 60], [285, 295], [0.04, 0.06, 0.08], [0.00, 0.01, 0.02]):
        a = stats(run(T0, T1, EI, EO, subset=train)); b = stats(run(T0, T1, EI, EO, subset=test))
        res.append((a["net"] + b["net"], T0, T1, EI, EO, a, b))
    res.sort(reverse=True)
    for tot, T0, T1, EI, EO, a, b in res[:10]:
        print(f"T[{T0},{T1}) EI={EI} EO={EO} | train n={a['n']:>4} net={a['net']:+8.2f} t={a['t']:+5.2f} scalp={a['scalp']} | "
              f"test n={b['n']:>4} net={b['net']:+8.2f} t={b['t']:+5.2f} scalp={b['scalp']}")
    print("family:", sum(1 for r in res if r[6]["net"] > 0), "/", len(res), "positive OOS")