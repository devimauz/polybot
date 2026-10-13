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


def pnl(price, won):
    shares = STAKE / price
    gross = (shares * 1.0) if won else 0.0
    fee = FEE_R * min(price, 1.0 - price) * shares
    return (gross - STAKE) - fee


def band(d, tlo, thi):
    info = d.get("info")
    if not info:
        return None
    t = info.get("t")
    mp = d.get("p_up_model")
    if t is None or mp is None:
        return None
    if tlo <= t <= thi:
        return (t, mp, d)
    return None


def evaluate(name, b_lo, b_hi, kind):
    nwin = 0
    ntrade = 0
    nw = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        rows = []
        for d in samples:
            b = band(d, b_lo, b_hi)
            if b:
                rows.append(b)
        if not rows:
            continue
        if kind == "fav":
            t, mp, d = max(rows, key=lambda x: max(x[1], 1.0 - x[1]))
            if mp >= 0.5:
                side, wmp = "Up", mp
            else:
                side, wmp = "Down", 1.0 - mp
            price = up_ask(d) if side == "Up" else down_ask(d)
        elif kind == "cheap":
            t, mp, d = max(rows, key=lambda x: abs(x[1] - 0.5))
            aU, aD = up_ask(d), down_ask(d)
            cands = []
            if aU is not None and 0.30 <= aU <= 0.70:
                cands.append((aU, mp, "Up"))
            if aD is not None and 0.30 <= aD <= 0.70:
                cands.append((aD, 1.0 - mp, "Down"))
            if not cands:
                continue
            price, wmp, side = min(cands, key=lambda c: c[0])
        else:
            continue
        if price is None or not (0.02 <= price <= 0.98):
            continue
        ntrade += 1
        winner = c["winner"]
        won = (side == winner)
        tot += pnl(price, won)
        if won:
            nw += 1
    hr = 155.1
    per_hour = ntrade / hr
    wr = (nw / ntrade * 100.0) if ntrade else 0
    print(f"{name:26} n_win={nwin:4d}  n_trade={ntrade:4d}  ({per_hour:5.2f}/hr)  win={wr:5.1f}%  total={tot:+8.2f}  $/hr={tot/hr:+6.3f}")


print("=== every-window strategies (1 fill/window, band = late window, pick best in band) ===")
for b_lo, b_hi in [(240, 295), (220, 295), (180, 295), (150, 295), (120, 295), (90, 295)]:
    evaluate(f"fav_{b_lo}_{b_hi}", b_lo, b_hi, "fav")
for b_lo, b_hi in [(240, 295), (220, 295), (180, 295), (150, 295), (120, 295), (90, 295)]:
    evaluate(f"cheap_{b_lo}_{b_hi}", b_lo, b_hi, "cheap")
