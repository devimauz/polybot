import json
from collections import defaultdict
ROOT = "C:/Users/user/Desktop/bot"
wins = defaultdict(list)
with open(ROOT + "/price_logs/twap/windows.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("asset") != "btc":
            continue
        wins[d["window"]].append(d)
clob = json.load(open(ROOT + "/price_logs/twap/clob_winners.json", encoding="utf-8"))
FEE_R = 0.05
STAKE = 10.0
HR = 154.25

def up_ask(d):
    a = d.get("ask_Up")
    if a is not None: return a
    b = d.get("bid_Down")
    return (1.0 - b) if b is not None else None
def down_ask(d):
    a = d.get("ask_Down")
    if a is not None: return a
    b = d.get("bid_Up")
    return (1.0 - b) if b is not None else None
def fee_ps(price): return FEE_R * min(price, 1.0 - price)

# Check ask availability by time-of-window
ask_counts = defaultdict(int)
sample_counts = defaultdict(int)
both_ask_counts = defaultdict(int)
for slug, samples in wins.items():
    if slug not in clob: continue
    for d in samples:
        info = d.get("info")
        t = info.get("t") if info else None
        if t is None: continue
        bucket = min(int(t // 30) * 30, 270)
        sample_counts[bucket] += 1
        pU, pD = up_ask(d), down_ask(d)
        has_ask = pU is not None and 0.02 <= pU <= 0.98
        has_ask_d = pD is not None and 0.02 <= pD <= 0.98
        if has_ask: ask_counts[bucket] += 1
        if has_ask and has_ask_d: both_ask_counts[bucket] += 1

print("t_range   samples  ask_U  ask_D  both_in_range")
for bucket in sorted(sample_counts):
    s = sample_counts[bucket]
    au = ask_counts[bucket]
    bd = both_ask_counts[bucket]
    print(f"{bucket:3d}-{bucket+30:3d}   {s:5d}   {au:5d}   {au:5d}   {bd:5d}  ({bd/s*100:.0f}%)")

# Now test: for each window, scan ALL samples (full window 0-295), pick the best trade.
# "best" = max EV for buying the cheaper side.
# If model active (mp available), use mp as fair prob. Else use spot-vs-ref z.
def z_from_spot(d):
    """Compute z from spot and ref (available even before model activates)."""
    spot = d.get("spot")
    ref = d.get("ref")
    if spot is None or ref is None: return None
    # approximate sigma from data: use 0.0002 (typical BTC 1-min vol)
    info = d.get("info")
    if info and info.get("sigma_min"):
        sigma = info["sigma_min"]
        t = info.get("t")
        remaining = info.get("remaining")
    else:
        sigma = 0.0002
        remaining = 300.0  # assume full window
    if t is None:
        remaining = 300.0
    if remaining <= 0: return None
    z = (spot - ref) / (sigma * (remaining ** 0.5) * 0.01)  # approximate
    # convert z to probability (standard normal CDF approximation)
    # P = 0.5 * (1 + erf(z / sqrt(2)))
    import math
    p_up = 0.5 * (1.0 + math.erf(z / (2 ** 0.5)))
    return p_up

def evaluate(name, b_lo, b_hi, use_model, use_spot_fallback):
    nwin = 0; ntrade = 0; nw = 0; tot = 0.0; nskip = 0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok": continue
        nwin += 1
        # Find best sample in band
        best_ev = -999; best_price = None; best_side = None; best_mp = None
        for d in samples:
            info = d.get("info")
            t = info.get("t") if info else None
            if t is None: continue
            if not (b_lo <= t <= b_hi): continue
            mp = d.get("p_up_model")
            if mp is None and use_spot_fallback:
                mp = z_from_spot(d)
            if mp is None: continue
            pU, pD = up_ask(d), down_ask(d)
            # Buy the cheaper side
            cands = []
            if pU is not None and 0.02 <= pU <= 0.98:
                ev = (mp / pU - 1.0 - fee_ps(pU) / pU) * STAKE
                cands.append((ev, pU, "Up"))
            if pD is not None and 0.02 <= pD <= 0.98:
                ev = ((1-mp) / pD - 1.0 - fee_ps(pD) / pD) * STAKE
                cands.append((ev, pD, "Down"))
            if not cands: continue
            # Pick the cheaper side (min price), or max EV
            ev_best, price, side = max(cands, key=lambda x: x[0])
            if ev_best > best_ev:
                best_ev = ev_best; best_price = price; best_side = side
        if best_price is None:
            nskip += 1
            continue
        ntrade += 1
        winner = c["winner"]
        won = (best_side == winner)
        shares = STAKE / best_price
        gross = shares if won else 0.0
        fee = fee_ps(best_price) * shares
        tot += (gross - STAKE) - fee
        if won: nw += 1
    per_hr = ntrade / HR
    wr = (nw / ntrade * 100.0) if ntrade else 0
    print(f"{name:40}  win={nwin:4d}  trades={ntrade:4d} ({per_hr:4.1f}/hr)  skip={nskip:4d} ({nskip/nwin*100:.0f}%)  winrate={wr:5.1f}%  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}")

print("\n=== cheap-side max-EV, full window, model only ===")
evaluate("cheap_maxEV_model_0_295", 0, 295, True, False)
evaluate("cheap_maxEV_model_90_295", 90, 295, True, False)
evaluate("cheap_maxEV_model_150_295", 150, 295, True, False)
print("\n=== cheap-side max-EV, full window, model + spot fallback ===")
evaluate("cheap_maxEV_model+spot_0_295", 0, 295, True, True)
evaluate("cheap_maxEV_model+spot_90_295", 90, 295, True, True)
evaluate("cheap_maxEV_model+spot_150_295", 150, 295, True, True)
