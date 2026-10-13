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

# For each sample in late band, check if both-side (equal-shares) arbitrage is possible.
def up_ask(d):
    a = d.get("ask_Up")
    if a is not None:
        return a
    b = d.get("bid_Down")
    return (1.0 - b) if b is not None else None


def down_ask(d):
    a = d.get("ask_Down")
    if a is not None:
        return a
    b = d.get("bid_Up")
    return (1.0 - b) if b is not None else None


def fee_per_share(price):
    # fee_for(price, shares=1)
    return FEE_R * min(price, 1.0 - price)


def scan(b_lo, b_hi):
    # returns per-window: best both-side opportunity (min combined cost) in band
    per_window = {}
    for slug, samples in wins.items():
        best = None
        for d in samples:
            info = d.get("info")
            if not info:
                continue
            t = info.get("t")
            if t is None or not (b_lo <= t <= b_hi):
                continue
            pU, pD = up_ask(d), down_ask(d)
            if pU is None or pD is None:
                continue
            if not (0.02 <= pU <= 0.98 and 0.02 <= pD <= 0.98):
                continue
            # equal shares of both; per $1 of shares:
            combined = pU + pD + fee_per_share(pU) + fee_per_share(pD)
            if best is None or combined < best:
                best = combined
        if best is not None:
            per_window[slug] = best
    return per_window


def report(b_lo, b_hi):
    best_per_window = {}
    for slug, samples in wins.items():
        best = None
        for d in samples:
            info = d.get("info")
            if not info:
                continue
            t = info.get("t")
            if t is None or not (b_lo <= t <= b_hi):
                continue
            pU, pD = up_ask(d), down_ask(d)
            if pU is None or pD is None:
                continue
            if not (0.02 <= pU <= 0.98 and 0.02 <= pD <= 0.98):
                continue
            c = pU + pD + fee_per_share(pU) + fee_per_share(pD)
            if best is None or c < best[0]:
                best = (c, pU, pD)
        if best is not None:
            best_per_window[slug] = best
    pw = {s: v[0] for s, v in best_per_window.items()}
    tot = 0.0
    n_arb = 0
    tot_profit = 0.0
    for slug, (c, pU, pD) in best_per_window.items():
        if c < 1.0:
            S = 10.0 / (pU + pD)
            payout = S
            fees = fee_per_share(pU) * S + fee_per_share(pD) * S
            profit = payout - (pU * S + pD * S) - fees
            tot_profit += profit
            n_arb += 1
    hr = 154.25
    print(f"band {b_lo}-{b_hi}: windows_with_both_ask={len(pw)}/895  arbs(combined<1.0)={n_arb}  ({n_arb/hr:4.2f}/hr)  total_profit=${tot_profit:+8.2f}  $/hr(if only arbs)={tot_profit/hr:+6.3f}")
    cs = sorted(pw.values())
    print(f"   combined cost: min={cs[0]:.4f} p25={cs[len(cs)//4]:.4f} med={cs[len(cs)//2]:.4f} p75={cs[3*len(cs)//4]:.4f} max={cs[-1]:.4f}")
    print(f"   frac combined<1.0={sum(1 for c in cs if c<1.0)/len(cs)*100:.1f}%  <0.99={sum(1 for c in cs if c<0.99)/len(cs)*100:.1f}%  <0.97={sum(1 for c in cs if c<0.97)/len(cs)*100:.1f}%")


for b_lo, b_hi in [(150, 295), (180, 295), (210, 295), (240, 295)]:
    report(b_lo, b_hi)
