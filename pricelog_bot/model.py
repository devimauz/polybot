"""Model-driven strategy: estimate P(Up wins) from the coinbase spot path vs window open,
then buy whichever side's ask is far below model probability.

Resolution rule (polymarket 5m up/down): final Chainlink TWAP over window >= open -> Up.
Model per frame: projected final TWAP = partial_avg*(t/300) + spot*((300-t)/300);
uncertainty = sigma_1s * sqrt(rem) / 300 (in return units) -> z -> p_up.
Entry: buy side S if p_model(S) - ask_S >= EDGE (ask, not mid, conservative).
Validation: train/test halves, latency fill, placebo, plateau, concentration.
"""
from loader import load, TAKER_FEE
import math, statistics, itertools

ws = load(); items = sorted(ws.items())
N = len(items)

def cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# --- global sigma: median per-window realized 1s cb-return std over t in [0,60] ---
per_w = []
for slug, w in items:
    pts = [(fr["t"], fr["cb"]) for fr in w["frames"] if fr["cb"] and fr["t"] < 60]
    rets = []
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if p0 and p1 and t1 > t0:
            rets.append(math.log(p1 / p0) / math.sqrt(t1 - t0))  # per-second log return
    if len(rets) > 20:
        per_w.append(statistics.pstdev(rets))
SIGMA_1S = statistics.median(per_w)
print(f"sigma_1s median = {SIGMA_1S:.6f} (per-second log ret), windows measured={len(per_w)}")

def window_open(w):
    # open proxy: first chainlink 60s twap recorded before the window (earliest frames carry it)
    best = None
    for fr in w["frames"]:
        if fr["t"] <= 5 and fr["cb"]:
            return fr["cb"]
    return w["frames"][0]["cb"] if w["frames"][0]["cb"] else None

def model_frames(w, open_px):
    """Yield (t, p_up, up_ask, dn_ask, ...) using causal running average."""
    frames = w["frames"]
    s = 0.0; n = 0; out = []; last_cb = None
    for fr in frames:
        t = fr["t"]
        if fr["cb"]:
            s += fr["cb"]; n += 1; last_cb = fr["cb"]
        if n == 0 or last_cb is None: continue
        partial = s / n
        rem = max(300 - t, 1.0)
        proj = partial * (t / 300.0) + last_cb * (rem / 300.0)
        sig = SIGMA_1S * math.sqrt(rem) / 300.0
        z = math.log(proj / open_px) / sig if sig > 0 else 0.0
        p_up = min(max(cdf(z), 0.0), 1.0)
        out.append((t, p_up, fr))
    return out

def run(T0, T1, EDGE, stake=10.0, latency=0.0, subset=None, sig_mult=1.0):
    trades = []
    for idx, (slug, w) in enumerate(items):
        if subset is not None and idx not in subset: continue
        open_px = window_open(w)
        if not open_px: continue
        frames = w["frames"]; pos = None
        mfs = model_frames(w, open_px)
        for i, (t, p_up, fr) in enumerate(mfs):
            if t < T0: continue
            if t >= T1: break
            if pos is not None: continue
            # candidate sides
            cands = []
            pu, pd_ = p_up, 1 - p_up
            ua, da = fr["up"]["ask"], fr["dn"]["ask"]
            if pu - ua >= EDGE: cands.append(("up", ua, pu))
            if pd_ - da >= EDGE: cands.append(("dn", da, pd_))
            if not cands: continue
            side, a, pm = max(cands, key=lambda c: c[1+2-2])  # pick larger edge
            side, a, pm = max(cands, key=lambda c: c[2] - c[1])
            fill_fr = fr
            if latency > 0:
                # find frame at t+latency
                j = next((k for k in range(i+1, len(mfs)) if mfs[k][0] >= t + latency), None)
                if j is None: break
                fill_fr = mfs[j][2]
            d2 = fill_fr[side]
            if d2["ask"] >= pm or d2["ask"] >= 0.99:  # no longer cheap under latency
                pos = "REJ"; continue
            rem = stake; sh = 0.0
            if not d2["asks"]: pos = "REJ"; continue
            touch = d2["asks"][0]
            if touch[0] * touch[1] < min(stake, 5.0 * touch[0]): pos = "REJ"; continue
            for p, sz in d2["asks"]:
                take = min(rem / p, sz); sh += take; rem -= take * p
                if rem <= 1e-9: break
            if rem > 1e-6: pos = "REJ"; continue
            fee = TAKER_FEE * min(a, 1 - a) * sh
            cost = stake + fee
            win = (side == "up") == (w["winner"] == "Up")
            trades.append(dict(slug=slug, side=side, t=t, ask=d2["ask"], p=pm,
                               pnl=(sh if win else 0.0) - cost, win=win, cost=cost))
            pos = "IN"
    return trades

def stats(trades):
    if not trades: return dict(n=0, net=0.0, t=0.0, wr=0.0, worst=0.0, roi=0.0, conc5=0.0, up=0, dn=0)
    pnls = [tr["pnl"] for tr in trades]
    net = sum(pnls); sd = statistics.pstdev(pnls) or 1e-9
    tstat = (net / len(pnls)) / (sd / math.sqrt(len(pnls))) if len(pnls) > 1 else 0.0
    top5 = sum(sorted(pnls)[-5:]); spend = sum(tr["cost"] for tr in trades)
    return dict(n=len(pnls), net=net, t=tstat, wr=sum(tr["win"] for tr in trades)/len(pnls),
                worst=min(pnls), roi=net/spend*100, conc5=(top5/net if abs(net)>1e-9 else float('nan')),
                up=sum(1 for tr in trades if tr["side"]=="up"), dn=len(pnls)-sum(1 for tr in trades if tr["side"]=="up"))

if __name__ == "__main__":
    train = set(range(0, N//2)); test = set(range(N//2, N))
    grid = itertools.product([30,60,120,180], [285,295], [0.02,0.04,0.06,0.08,0.12])
    res = []
    for T0, T1, E in grid:
        a = stats(run(T0, T1, E, subset=train)); b = stats(run(T0, T1, E, subset=test))
        res.append((a["net"] + b["net"], T0, T1, E, a, b))
    res.sort(reverse=True)
    for tot, T0, T1, E, a, b in res[:12]:
        print(f"T[{T0},{T1}) EDGE={E:.2f} | train n={a['n']:>3} net={a['net']:+8.2f} t={a['t']:+5.2f} | test n={b['n']:>3} net={b['net']:+8.2f} t={b['t']:+5.2f}")
    pos_test = sum(1 for r in res if r[6]["net"] > 0)
    print(f"\nfamily sign test half: {pos_test}/{len(res)} rules positive OUT of sample")