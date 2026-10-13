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


def fee_ps(p):
    return FEE_R * min(p, 1.0 - p)


def side_prices(d, side):
    if side == "Up":
        a = d.get("ask_Up"); b = d.get("bid_Up")
        if a is None and d.get("bid_Down") is not None: a = 1.0 - d["bid_Down"]
        if b is None and d.get("ask_Down") is not None: b = 1.0 - d["ask_Down"]
    else:
        a = d.get("ask_Down"); b = d.get("bid_Down")
        if a is None and d.get("bid_Up") is not None: a = 1.0 - d["bid_Up"]
        if b is None and d.get("ask_Up") is not None: b = 1.0 - d["ask_Up"]
    return a, b


def sim(buy_lo, buy_hi, min_mp, max_ask, exit_spread, sell_lo, sell_hi):
    """Honest: enter at FIRST qualifying sample; then FIRST qualifying exit; else hold to settlement."""
    nwin = nscalp = nholdwin = nholdlose = nskip = 0
    total = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
        # --- entry: first sample with t in [buy_lo,buy_hi], mp>=min_mp, fav ask <= max_ask ---
        entry = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            mp = d.get("p_up_model")
            if t is None or mp is None: continue
            if not (buy_lo <= t <= buy_hi): continue
            fav_mp = mp if mp >= 0.5 else 1.0 - mp
            if fav_mp < min_mp: continue
            side = "Up" if mp >= 0.5 else "Down"
            ask, bid = side_prices(d, side)
            if ask is None or ask < 0.20 or ask > max_ask: continue
            entry = (t, side, ask)
            break
        if entry is None:
            nskip += 1
            continue
        t_ent, side, ask = entry
        shares = STAKE / ask
        buy_fee = fee_ps(ask) * shares
        # --- exit: first sample with t in [sell_lo,sell_hi], bid >= ask+exit_spread, t>t_ent ---
        pnl = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            if t is None or t <= t_ent: continue
            if not (sell_lo <= t <= sell_hi): continue
            a2, b2 = side_prices(d, side)
            if b2 is not None and b2 >= ask + exit_spread:
                sell_fee = fee_ps(b2) * shares
                pnl = (b2 - ask) * shares - buy_fee - sell_fee
                nscalp += 1
                break
        if pnl is None:
            won = (side == winner)
            pnl = (shares if won else 0.0) - STAKE - buy_fee
            if won: nholdwin += 1
            else: nholdlose += 1
        total += pnl
    ntrade = nscalp + nholdwin + nholdlose
    cov = ntrade / nwin * 100
    print(f"buy[{buy_lo:3d}-{buy_hi:3d}] mp>={min_mp:.2f} ask<={max_ask:.2f} exit_sp={exit_spread:.2f} sell[{sell_lo:3d}-{sell_hi:3d}]")
    print(f"   trades={ntrade:4d} ({ntrade/HR:4.2f}/hr)  scalp={nscalp:4d} holdWin={nholdwin:3d} holdLose={nholdlose:3d} skip={nskip:4d} cov={cov:3.0f}%")
    print(f"   total=${total:+9.1f}  $/trade=${total/ntrade if ntrade else 0:+6.3f}  $/window=${total/nwin:+6.3f}  $/hr=${total/HR:+7.2f}")
    print()


print("=" * 90)
print("HONEST FIRST-SIGNAL SCALP+HOLD (fees 5% per side), $10 stake")
print("=" * 90)
print()
# Sweep: vary min_mp and max_ask to maximize coverage while staying profitable
for min_mp in [0.50, 0.52, 0.55, 0.58, 0.60]:
    for max_ask in [0.60, 0.70, 0.80, 0.90]:
        sim(90, 150, min_mp, max_ask, 0.05, 150, 280)
