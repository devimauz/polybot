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
    return FEE_R * min(price, 1.0 - price)


def pick_moment(samples, b_lo, b_hi):
    # sample with max |mp-0.5| in band (most decisive)
    best = None
    for d in samples:
        info = d.get("info")
        if not info:
            continue
        t = info.get("t")
        mp = d.get("p_up_model")
        if t is None or mp is None:
            continue
        if not (b_lo <= t <= b_hi):
            continue
        if best is None or abs(mp - 0.5) > abs(best[0] - 0.5):
            best = (mp, d)
    return best


def strategy(samples, b_lo, b_hi, mode):
    """Return (action, price, side, stake_total).
    action in {'arb','fav','cheap','skip'}"""
    m = pick_moment(samples, b_lo, b_hi)
    if not m:
        return None
    mp, d = m
    pU, pD = up_ask(d), down_ask(d)
    if pU is None or pD is None:
        return None
    if not (0.02 <= pU <= 0.98 and 0.02 <= pD <= 0.98):
        return None
    c = pU + pD + fee_per_share(pU) + fee_per_share(pD)
    F_up, F_down = mp, 1.0 - mp
    # EV per $1 of stake, for single-side buy at ask p, when we believe prob F:
    # shares = 1/p, win pays 1, loss 0. EV = F/p - 1 - fee/p  (per $1 spent)
    # Multiply by STAKE for $-EV.
    ev_up = (F_up / pU - 1.0 - fee_per_share(pU) / pU) * STAKE if pU > 0 else -STAKE
    ev_down = (F_down / pD - 1.0 - fee_per_share(pD) / pD) * STAKE if pD > 0 else -STAKE
    if mode == "arb_plus_fav":
        if c < 1.0:
            return ("arb", None, None, STAKE)
        if ev_up >= ev_down:
            return ("fav", pU, "Up", STAKE)
        return ("fav", pD, "Down", STAKE)
    elif mode == "arb_plus_maxev":
        if c < 1.0:
            return ("arb", None, None, STAKE)
        if ev_up >= ev_down:
            return ("fav", pU, "Up", STAKE)
        return ("fav", pD, "Down", STAKE)
    elif mode == "arb_plus_cheap":
        if c < 1.0:
            return ("arb", None, None, STAKE)
        # buy the cheaper side
        if pU <= pD:
            return ("cheap", pU, "Up", STAKE)
        return ("cheap", pD, "Down", STAKE)
    elif mode == "single_side_only":
        if ev_up >= ev_down:
            return ("fav", pU, "Up", STAKE)
        return ("fav", pD, "Down", STAKE)
    elif mode == "cheap_only":
        if pU <= pD:
            return ("cheap", pU, "Up", STAKE)
        return ("cheap", pD, "Down", STAKE)
    elif mode == "arb_plus_better":
        # arb if c<1; else buy the side with EV>0; else buy cheap side
        if c < 1.0:
            return ("arb", None, None, STAKE)
        if ev_up > 0 and ev_up >= ev_down:
            return ("fav", pU, "Up", STAKE)
        if ev_down > 0:
            return ("fav", pD, "Down", STAKE)
        if pU <= pD:
            return ("cheap", pU, "Up", STAKE)
        return ("cheap", pD, "Down", STAKE)
    return None


def pnl_action(action, price, side, stake, winner):
    if action == "arb":
        # S = stake / (pU+pD); payout = S; fees as before. We need pU, pD here — caller handles.
        return None
    shares = stake / price
    gross = shares if (side == winner) else 0.0
    fee = fee_per_share(price) * shares
    return (gross - stake) - fee


def arb_profit(pU, pD, stake):
    S = stake / (pU + pD)
    payout = S
    fees = fee_per_share(pU) * S + fee_per_share(pD) * S
    return payout - stake - fees


def evaluate(name, b_lo, b_hi, mode):
    nwin = 0
    n_arb = 0
    n_fav = 0
    n_cheap = 0
    n_skip = 0
    wins_ = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        m = pick_moment(samples, b_lo, b_hi)
        if not m:
            n_skip += 1
            continue
        mp, d = m
        pU, pD = up_ask(d), down_ask(d)
        if pU is None or pD is None or not (0.02 <= pU <= 0.98 and 0.02 <= pD <= 0.98):
            n_skip += 1
            continue
        act = strategy(samples, b_lo, b_hi, mode)
        if act is None:
            n_skip += 1
            continue
        action, price, side, stake = act
        winner = c["winner"]
        if action == "arb":
            prof = arb_profit(pU, pD, stake)
            n_arb += 1
            # count as win (risk-free)
            wins_ += 1
        else:
            prof = pnl_action(action, price, side, stake, winner)
            if action == "fav":
                n_fav += 1
            else:
                n_cheap += 1
            won = (side == winner)
            if won:
                wins_ += 1
        tot += prof
    n_trade = n_arb + n_fav + n_cheap
    per_hour = n_trade / HR
    wr = (wins_ / n_trade * 100.0) if n_trade else 0
    print(f"{name:36} win={nwin:4d}  trades={n_trade:4d} ({per_hour:4.1f}/hr)  arb={n_arb:4d} fav={n_fav:4d} cheap={n_cheap:4d} skip={n_skip:4d}  winrate={wr:5.1f}%  total=${tot:+8.2f}  $/hr=${tot/HR:+6.3f}")


print("=== every-window strategies (each trades <= 1 window's worth, band 150-295, pick most-decisive moment) ===")
for mode in ["arb_plus_fav", "arb_plus_maxev", "arb_plus_cheap", "arb_plus_better", "single_side_only", "cheap_only"]:
    evaluate(mode, 150, 295, mode)
print()
print("=== arb_plus_better, varying band ===")
for b_lo, b_hi in [(120, 295), (150, 295), (180, 295), (210, 295), (240, 295)]:
    evaluate(f"arb_plus_better_{b_lo}_{b_hi}", b_lo, b_hi, "arb_plus_better")
