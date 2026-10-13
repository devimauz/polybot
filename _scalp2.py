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


def scalp(buy_t, sell_t, min_conf, target_gain, settle_on_end):
    """Honest scalp sim.
    - At buy_t (first sample with model & conf>=min_conf): buy fav side at ask.
    - After buy, each subsequent sample: sell at bid if (t >= sell_t OR gain >= target_gain AND t >= buy_t+5).
    - If still held at window end: if settle_on_end, settle at CLOB winner (1 or 0 per share); else sell at last known bid."""
    nwin = n_buy = n_sell = n_settle = n_skip = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
        bought = False
        sell_price = 0.0
        buy_price = 0.0
        side = None
        shares = 0.0
        buy_fee = 0.0
        sold = False
        for d in samples:
            info = d.get("info")
            t = info.get("t") if info else None
            if t is None:
                continue
            mp = d.get("p_up_model")
            if mp is None:
                continue
            conf = max(mp, 1.0 - mp)
            if not bought:
                if t >= buy_t and conf >= min_conf:
                    side = "Up" if mp >= 0.5 else "Down"
                    buy_price = up_ask(d) if side == "Up" else down_ask(d)
                    if buy_price is None or not (0.02 <= buy_price <= 0.98):
                        continue
                    shares = STAKE / buy_price
                    buy_fee = fee_ps(buy_price) * shares
                    bought = True
                    n_buy += 1
            if bought and not sold:
                bp = up_bid(d) if side == "Up" else down_bid(d)
                if bp is not None and 0.02 <= bp <= 0.98:
                    gain = (bp - buy_price) * shares
                    if (t >= sell_t) or (gain >= target_gain * shares and t >= buy_t + 5):
                        sell_fee = fee_ps(bp) * shares
                        tot += bp * shares - buy_price * shares - buy_fee - sell_fee
                        n_sell += 1
                        sold = True
                        break
        if bought and not sold:
            # end-of-window: use CLOB winner if settle_on_end
            if settle_on_end:
                win_val = 1.0 if side == winner else 0.0
                tot += win_val * shares - buy_price * shares - buy_fee
                n_settle += 1
            else:
                # sell at last bid (approx: use winner price)
                win_val = 1.0 if side == winner else 0.0
                tot += win_val * shares - buy_price * shares - buy_fee
                n_settle += 1
    n_trade = n_buy
    wr = (n_sell + n_settle) / n_buy * 100.0 if n_buy else 0
    print(f"buy={buy_t:3d} sell={sell_t:3d} conf>={min_conf:.2f} tgt=${target_gain:.2f}  bought={n_buy:4d} sold={n_sell:4d} settled={n_settle:4d}  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}  $/trade=${tot/n_buy if n_buy else 0:+5.3f}")


print("=== HONEST SCALP SIM (all positions closed at end if not sold) ===")
print("--- target $1/trade, sell_t=280, settle on end ---")
for buy_t in [90, 100, 110, 120]:
    for mc in [0.50, 0.55, 0.60, 0.65]:
        scalp(buy_t, 280, mc, 1.0, True)
    print()
print("--- target $2, sell_t=280 ---")
for buy_t in [90, 100]:
    for mc in [0.55, 0.60]:
        scalp(buy_t, 280, mc, 2.0, True)
    print()
print("--- target $0.5, sell_t=250 ---")
for buy_t in [90, 100, 110]:
    for mc in [0.55, 0.60]:
        scalp(buy_t, 250, mc, 0.5, True)
    print()
