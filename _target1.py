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


def simulate(arb_th, dec_t, side_rule, min_conf=0.0):
    """Honest: poll in time order. At dec_t, if model available and conf >= min_conf,
    take arb if available, else buy one side per side_rule."""
    nwin = 0
    n_arb = 0
    n_side = 0
    n_skip = 0
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
                if mp is None:
                    continue
                conf = max(mp, 1.0 - mp)
                if conf < min_conf:
                    continue
                if side_rule == "fav":
                    side = "Up" if mp >= 0.5 else "Down"
                    price = pU if side == "Up" else pD
                elif side_rule == "cheap":
                    side = "Up" if pU <= pD else "Down"
                    price = min(pU, pD)
                elif side_rule == "maxev":
                    evU = (mp / pU - 1.0 - fee_ps(pU) / pU) * STAKE
                    evD = ((1 - mp) / pD - 1.0 - fee_ps(pD) / pD) * STAKE
                    if evU >= evD:
                        side, price = "Up", pU
                    else:
                        side, price = "Down", pD
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
    print(f"dec_t={dec_t:3d} rule={side_rule:6} min_conf={min_conf:.2f} arb_th={arb_th:.3f}  trades={n_trade:4d} ({per_hr:4.2f}/hr)  arb={n_arb:3d} side={n_side:4d} skip={n_skip:4d}  side_win={wr:5.1f}%  avg_price={avg_price:.3f}  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}  $/trade=${tot/n_trade if n_trade else 0:+5.3f}")


print("=== TARGET: $1/trade ($12/hr), trade EVERY window ===")
print("--- fav rule at various dec_t, no min conf (always trade) ---")
for dec_t in [90, 100, 110, 120, 130, 140, 150, 160, 170, 180]:
    simulate(1.0, dec_t, "fav", min_conf=0.0)

print()
print("--- fav rule with min_conf (skip if too uncertain) ---")
for dec_t in [120, 140, 150]:
    for mc in [0.50, 0.52, 0.55, 0.60]:
        simulate(1.0, dec_t, "fav", min_conf=mc)

print()
print("--- cheap rule at various dec_t ---")
for dec_t in [90, 100, 120, 150, 180]:
    simulate(1.0, dec_t, "cheap", min_conf=0.0)

print()
print("--- maxev rule at various dec_t ---")
for dec_t in [90, 100, 120, 150, 180]:
    simulate(1.0, dec_t, "maxev", min_conf=0.0)

print()
print("--- arb threshold sweep (dec_t=90, fav) ---")
for arb_th in [0.99, 0.995, 1.0, 1.005, 1.01]:
    simulate(arb_th, 90, "fav", min_conf=0.0)
