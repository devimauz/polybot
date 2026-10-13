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


def fee_ps(price):
    return FEE_R * min(price, 1.0 - price)


def side_pnl(price, side, winner):
    shares = STAKE / price
    gross = shares if side == winner else 0.0
    return (gross - STAKE) - fee_ps(price) * shares


# For each window, take the sample at dec_t (first one >= dec_t with model+asks).
# Bin by confidence. Report: avg model p, avg ask, avg (p-ask) edge, win rate, $/trade.
def report(dec_t, arb_th=1.0):
    bins = defaultdict(list)  # conf bucket -> list of (model_p, ask, won)
    n_win = 0
    n_arb = 0
    n_side = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        winner = c["winner"]
        traded = False
        for d in samples:
            info = d.get("info")
            t = info.get("t") if info else None
            if t is None:
                continue
            pU, pD = up_ask(d), down_ask(d)
            if not (pU and pD and 0.02 <= pU <= 0.98 and 0.02 <= pD <= 0.98):
                continue
            comb = pU + pD + fee_ps(pU) + fee_ps(pD)
            if comb < arb_th:
                n_arb += 1
                traded = True
                # count arb profit separately, approximate at $0.05 avg
                break
            if (not traded) and t >= dec_t:
                mp = d.get("p_up_model")
                if mp is None:
                    continue
                fav_p = mp if mp >= 0.5 else 1.0 - mp
                side = "Up" if mp >= 0.5 else "Down"
                ask = pU if side == "Up" else pD
                won = (side == winner)
                b = round(fav_p * 4) / 4
                bins[b].append((fav_p, ask, won, slug))
                tot += side_pnl(ask, side, winner)
                n_side += 1
                traded = True
                break
    print(f"=== dec_t={dec_t}  (n_arb={n_arb} n_side={n_side} tot_side_pnl=${tot:+.2f}) ===")
    print(f"{'conf':>5} {'n':>5} {'avg_p':>6} {'avg_ask':>8} {'edge(p-a)':>10} {'win%':>6} {'$/trade':>8}")
    for b in sorted(bins):
        arr = bins[b]
        n = len(arr)
        ap = sum(x[0] for x in arr) / n
        aa = sum(x[1] for x in arr) / n
        wr = sum(1 for x in arr if x[2]) / n
        # pnl using avg ask and avg win
        avg_pnl = sum(side_pnl(x[1], "F", "F" if x[2] else "O") for x in arr) / n
        print(f"{b:5.2f} {n:5d} {ap:6.3f} {aa:8.3f} {ap-aa:+10.3f} {wr*100:5.1f}% {avg_pnl:+8.3f}")


for dec_t in [60, 120, 180, 240]:
    report(dec_t)
    print()
