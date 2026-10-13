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
for v in wins.values():
    v.sort(key=lambda d: (d.get("info") or {}).get("t") or 0)
clob = json.load(open(ROOT + "/price_logs/twap/clob_winners.json", encoding="utf-8"))
FEE_R = 0.05
STAKE = 10.0
HR = 154.25

def fee_ps(price):
    return FEE_R * min(price, 1.0 - price)

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

def up_bid(d):
    a = d.get("bid_Up")
    if a is not None: return a
    b = d.get("ask_Down")
    return (1.0 - b) if b is not None else None

def down_bid(d):
    a = d.get("bid_Down")
    if a is not None: return a
    b = d.get("ask_Up")
    return (1.0 - b) if b is not None else None

def scalp_sim(buy_t, sell_t, min_conf, target_gain):
    """At buy_t: buy fav side at ask. Poll until sell_t or target_gain reached: sell at bid."""
    nwin = n_buy = n_sell = n_skip = 0
    tot = 0.0
    hold_to_end = 0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
        bought = False
        buy_price = 0.0
        side = None
        shares = 0.0
        buy_fee = 0.0
        for i, d in enumerate(samples):
            info = d.get("info")
            t = info.get("t") if info else None
            if t is None: continue
            mp = d.get("p_up_model")
            if mp is None: continue
            conf = max(mp, 1.0 - mp)
            if conf < min_conf: continue
            if not bought and t >= buy_t:
                side = "Up" if mp >= 0.5 else "Down"
                buy_price = up_ask(d) if side == "Up" else down_ask(d)
                if buy_price is None or not (0.02 <= buy_price <= 0.98): continue
                shares = STAKE / buy_price
                buy_fee = fee_ps(buy_price) * shares
                bought = True
                n_buy += 1
            if bought and t >= sell_t:
                # sell at bid
                sell_price = up_bid(d) if side == "Up" else down_bid(d)
                if sell_price is None: sell_price = buy_price
                sell_fee = fee_ps(sell_price) * shares
                tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                n_sell += 1
                break
            if bought and (sell_price := (up_bid(d) if side == "Up" else down_bid(d))) is not None:
                gain = (sell_price - buy_price) * shares
                if gain >= target_gain * shares:
                    sell_fee = fee_ps(sell_price) * shares
                    tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                    n_sell += 1
                    break
        if bought and not any(True for _ in [0]): pass
    # simpler: redo with cleaner logic
    nwin = n_buy = n_sell = n_skip = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
        bought = False
        buy_price = 0.0
        side = None
        shares = 0.0
        buy_fee = 0.0
        sold = False
        for d in samples:
            info = d.get("info")
            t = info.get("t") if info else None
            if t is None: continue
            mp = d.get("p_up_model")
            if mp is None: continue
            conf = max(mp, 1.0 - mp)
            if conf < min_conf: continue
            if not bought and t >= buy_t:
                side = "Up" if mp >= 0.5 else "Down"
                buy_price = up_ask(d) if side == "Up" else down_ask(d)
                if buy_price is None or not (0.02 <= buy_price <= 0.98): continue
                shares = STAKE / buy_price
                buy_fee = fee_ps(buy_price) * shares
                bought = True
                n_buy += 1
                # check immediate sell
                sell_price = up_bid(d) if side == "Up" else down_bid(d)
                if sell_price is not None and t >= sell_t:
                    sell_fee = fee_ps(sell_price) * shares
                    tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                    n_sell += 1
                    sold = True
                elif sell_price is not None:
                    gain = (sell_price - buy_price) * shares
                    if gain >= target_gain * shares and t >= buy_t + 5:
                        sell_fee = fee_ps(sell_price) * shares
                        tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                        n_sell += 1
                        sold = True
            elif bought and not sold:
                sell_price = up_bid(d) if side == "Up" else down_bid(d)
                if sell_price is not None:
                    if t >= sell_t:
                        sell_fee = fee_ps(sell_price) * shares
                        tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                        n_sell += 1
                        sold = True
                    else:
                        gain = (sell_price - buy_price) * shares
                        if gain >= target_gain * shares and t >= buy_t + 5:
                            sell_fee = fee_ps(sell_price) * shares
                            tot += sell_price * shares - buy_price * shares - buy_fee - sell_fee
                            n_sell += 1
                            sold = True
        if bought and not sold:
            n_skip += 1  # held to end, counted as skip
    print(f"buy_t={buy_t:3d} sell_t={sell_t:3d} conf>={min_conf:.2f} target=${target_gain:.2f}  bought={n_buy:4d} sold={n_sell:4d} held_end={n_skip:4d}  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}  $/trade=${tot/n_buy if n_buy else 0:+5.3f}")

print("=== SCALP: buy early, sell when target gained or at sell_t ===")
for buy_t in [90, 100, 110, 120]:
    for sell_t in [200, 240, 280]:
        for mc in [0.50, 0.55, 0.60]:
            scalp_sim(buy_t, sell_t, mc, target_gain=1.0)
    print()
