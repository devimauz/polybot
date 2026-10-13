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


def arb_profit(pU, pD, stake):
    S = stake / (pU + pD)
    fees = fee_ps(pU) * S + fee_ps(pD) * S
    return S - stake - fees


def simulate(dec_t, min_conf, arb_th, use_fallback_spot):
    """At dec_t: buy fav side. If conf < min_conf, use spot-vs-ref direction as fallback."""
    nwin = n_arb = n_side = n_skip = 0
    tot = 0.0
    price_stats = []
    win_count = 0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
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
                tot += arb_profit(pU, pD, STAKE)
                n_arb += 1
                traded = True
                break
            if (not traded) and t >= dec_t:
                mp = d.get("p_up_model")
                if mp is None and use_fallback_spot:
                    # use spot vs ref direction
                    spot = d.get("spot")
                    ref = d.get("ref")
                    if spot is None or ref is None:
                        continue
                    mp = 0.5 + 0.5 * (1 if spot > ref else -1)  # 0.5 or 1.0
                    if spot > ref:
                        mp = 0.55  # weak up bias
                    else:
                        mp = 0.45
                if mp is None:
                    continue
                conf = max(mp, 1.0 - mp)
                if conf < min_conf:
                    continue
                side = "Up" if mp >= 0.5 else "Down"
                price = pU if side == "Up" else pD
                tot += side_pnl(price, side, winner)
                price_stats.append(price)
                n_side += 1
                won = (side == winner)
                if won:
                    win_count += 1
                traded = True
                break
        if not traded:
            n_skip += 1
    n_trade = n_arb + n_side
    per_hr = n_trade / HR
    wr = (win_count / n_side * 100.0) if n_side else 0
    avg_price = sum(price_stats) / len(price_stats) if price_stats else 0
    print(f"dec_t={dec_t:3d} min_conf={min_conf:.2f} arb_th={arb_th:.3f} spot_fb={int(use_fallback_spot)}  trades={n_trade:4d} ({per_hr:4.2f}/hr)  arb={n_arb:3d} side={n_side:4d} skip={n_skip:4d}  side_win={wr:5.1f}%  avg_price={avg_price:.3f}  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}  $/trade=${tot/n_trade if n_trade else 0:+5.3f}")


print("=== sweep dec_t x min_conf (always use spot fallback if model unavailable) ===")
for dec_t in [60, 90, 120, 150]:
    for min_conf in [0.50, 0.55, 0.60, 0.65]:
        simulate(dec_t, min_conf, 1.0, True)
    print()
