import json
from collections import defaultdict
ROOT = "C:/Users/user/Desktop/bot"
wins = defaultdict(list)
with open(ROOT + "/price_logs/twap/windows.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("asset") != "btc": continue
        wins[d["window"]].append(d)
for v in wins.values():
    v.sort(key=lambda d: (d.get("info") or {}).get("t") or 0)
clob = json.load(open(ROOT + "/price_logs/twap/clob_winners.json", encoding="utf-8"))
FEE_R = 0.05
STAKE = 10.0
HR = 154.25

def fee_ps(p): return FEE_R * min(p, 1.0 - p)

def side_ask_bid(d, side):
    """Return (ask, bid) for 'Up' or 'Down'."""
    if side == "Up":
        a = d.get("ask_Up")
        b = d.get("bid_Up")
    else:
        a = d.get("ask_Down")
        b = d.get("bid_Down")
    if a is None and side == "Up" and d.get("bid_Down") is not None:
        a = 1.0 - d["bid_Down"]
    if a is None and side == "Down" and d.get("bid_Up") is not None:
        a = 1.0 - d["bid_Up"]
    if b is None and side == "Up" and d.get("ask_Down") is not None:
        b = 1.0 - d["ask_Down"]
    if b is None and side == "Down" and d.get("ask_Up") is not None:
        b = 1.0 - d["ask_Up"]
    return a, b

# For each window: find the BEST scalp (buy_ask, sell_bid) where:
# - same model-favoured side at both times
# - t_buy in [buy_lo, buy_hi]
# - t_sell in [sell_lo, sell_hi]
# - t_buy < t_sell
# - profit = (sell_bid - buy_ask) * shares - fees_buy - fees_sell
# where shares = STAKE / buy_ask
# We also need sell_bid > buy_ask (positive spread) to have any chance.

def scalp_sim(buy_lo, buy_hi, sell_lo, sell_hi, min_spread=0.03):
    nwin = 0
    nscalp = 0
    nskip = 0
    total_pnl = 0.0
    spreads = []
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        # Build timeline: list of (t, side, ask, bid)
        timeline = []
        for d in samples:
            info = d.get("info")
            t = info.get("t") if info else None
            mp = d.get("p_up_model")
            if t is None or mp is None:
                continue
            side = "Up" if mp >= 0.5 else "Down"
            ask, bid = side_ask_bid(d, side)
            if ask is None or bid is None:
                continue
            if not (0.05 <= ask <= 0.95 and 0.05 <= bid <= 0.95):
                continue
            timeline.append((t, side, ask, bid))
        if not timeline:
            nskip += 1
            continue
        # Find best scalp: buy in [buy_lo,buy_hi], sell in [sell_lo,sell_hi], same side
        best_pnl = None
        best_spread = 0
        for (t1, s1, a1, b1) in timeline:
            if not (buy_lo <= t1 <= buy_hi):
                continue
            for (t2, s2, a2, b2) in timeline:
                if not (sell_lo <= t2 <= sell_hi):
                    continue
                if t2 <= t1:
                    continue
                if s1 != s2:
                    continue
                if b2 <= a1:
                    continue
                spread = b2 - a1
                if spread < min_spread:
                    continue
                shares = STAKE / a1
                fees = fee_ps(a1) * shares + fee_ps(b2) * shares
                pnl = (b2 - a1) * shares - fees
                if best_pnl is None or pnl > best_pnl:
                    best_pnl = pnl
                    best_spread = spread
        if best_pnl is not None:
            nscalp += 1
            total_pnl += best_pnl
            spreads.append(best_spread)
        else:
            nskip += 1
    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    avg_pnl = total_pnl / nscalp if nscalp else 0
    print(f"buy[{buy_lo:3d}-{buy_hi:3d}] sell[{sell_lo:3d}-{sell_hi:3d}] min_sp={min_spread:.2f}  "
          f"windows={nwin} scalps={nscalp} skip={nskip}  avg_spread={avg_spread:.3f}  "
          f"total=${total_pnl:+9.1f}  $/scalp=${avg_pnl:+5.3f}  $/window=${total_pnl/nwin:+5.3f}  "
          f"$/hr=${total_pnl/HR:+6.2f}  cover={nscalp/nwin*100:3.0f}%")

print("=== MOMENTUM SCALP (buy fav side early, sell same side later) ===")
print()
# Wide ranges first
scalp_sim(80, 120, 150, 220, 0.03)
scalp_sim(80, 120, 150, 220, 0.05)
scalp_sim(80, 120, 150, 220, 0.08)
scalp_sim(90, 130, 160, 230, 0.03)
scalp_sim(90, 130, 160, 230, 0.05)
scalp_sim(100, 140, 170, 240, 0.03)
scalp_sim(100, 140, 170, 240, 0.05)
scalp_sim(100, 140, 170, 240, 0.08)
scalp_sim(100, 140, 180, 250, 0.03)
scalp_sim(100, 140, 180, 250, 0.05)
scalp_sim(100, 140, 180, 250, 0.08)
# Wider buy
scalp_sim(60, 140, 150, 240, 0.03)
scalp_sim(60, 140, 150, 240, 0.05)
# Wide sell
scalp_sim(90, 140, 160, 290, 0.03)
scalp_sim(90, 140, 160, 290, 0.05)
scalp_sim(90, 140, 160, 290, 0.08)
# Tight around mid
scalp_sim(110, 130, 180, 210, 0.03)
scalp_sim(110, 130, 180, 210, 0.05)
# No min spread (to see raw data)
scalp_sim(90, 140, 160, 240, 0.01)
scalp_sim(90, 140, 160, 240, 0.0)
print()
print("=== WIDER BUY WINDOW (early entries) ===")
scalp_sim(50, 150, 160, 250, 0.03)
scalp_sim(50, 150, 160, 250, 0.05)
scalp_sim(40, 150, 150, 260, 0.03)
scalp_sim(40, 150, 150, 260, 0.05)
