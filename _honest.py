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


def arb_profit(pU, pD, stake):
    S = stake / (pU + pD)
    fees = fee_ps(pU) * S + fee_ps(pD) * S
    return S - stake - fees


def side_pnl(price, side, winner):
    shares = STAKE / price
    gross = shares if side == winner else 0.0
    return (gross - STAKE) - fee_ps(price) * shares


def simulate(arb_th, dec_t, side_rule, arb_stake=10.0):
    """Honest live simulation: poll samples in time order, act on current sample.
    arb: buy both sides when combined+fees < arb_th.
    fallback: at dec_t, if not arbed, buy one side per side_rule."""
    nwin = 0
    n_arb = 0
    n_side = 0
    n_skip = 0
    tot = 0.0
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
                tot += arb_profit(pU, pD, arb_stake)
                n_arb += 1
                traded = True
                break  # took arb, done with window
            if (not traded) and t >= dec_t:
                mp = d.get("p_up_model")
                if mp is None:
                    continue
                # choose side
                if side_rule == "maxev":
                    evU = (mp / pU - 1.0 - fee_ps(pU) / pU) * STAKE
                    evD = ((1 - mp) / pD - 1.0 - fee_ps(pD) / pD) * STAKE
                    side = "Up" if evU >= evD else "Down"
                elif side_rule == "fav":
                    side = "Up" if mp >= 0.5 else "Down"
                elif side_rule == "cheap":
                    side = "Up" if pU <= pD else "Down"
                price = pU if side == "Up" else pD
                tot += side_pnl(price, side, winner)
                n_side += 1
                traded = True
                break
        if not traded:
            n_skip += 1
    per_hr = (n_arb + n_side) / HR
    print(f"arb_th={arb_th:.3f} dec_t={dec_t:3d} rule={side_rule:8}  trades={n_arb+n_side:4d} ({per_hr:4.2f}/hr)  arb={n_arb:4d} side={n_side:4d} skip={n_skip:4d} ({n_skip/nwin*100:.0f}%skipped)  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}")


print("=== honest live simulation (poll in time order, act on current sample) ===")
print("--- take arb when available, else single-side at dec_t ---")
for arb_th in [1.0, 1.005, 1.01]:
    for dec_t in [150, 180, 210, 240]:
        for rule in ["maxev", "fav", "cheap"]:
            simulate(arb_th, dec_t, rule)
print()
print("--- arb only (no fallback side) ---")
for arb_th in [1.0, 1.005, 1.01]:
    simulate(arb_th, 999, "maxev")  # dec_t=999 means no side taken
