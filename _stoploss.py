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


def sim(buy_lo, buy_hi, min_mp, max_ask, exit_spread, sell_lo, sell_hi,
        stop_spread, stop_lo, stop_hi, use_stop=True):
    """
    Entry: first sample with t in [buy_lo,buy_hi], fav_mp>=min_mp, fav_ask<=max_ask.
    Exit (first that fires, in time order):
      - SCALP: bid >= entry_ask + exit_spread, t in [sell_lo, sell_hi]
      - STOP : bid <= entry_ask - stop_spread, t in [stop_lo, stop_hi]  (tight-exit loss)
      - HOLD : otherwise, hold to settlement
    """
    nwin = nscalp = nstop = nholdwin = nholdlose = nskip = 0
    total = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
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
        # scan forward, track first scalp OR first stop
        scalp_t = None; scalp_p = None; stop_t = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            if t is None or t <= t_ent: continue
            a2, b2 = side_prices(d, side)
            if b2 is None: continue
            if scalp_t is None and sell_lo <= t <= sell_hi and b2 >= ask + exit_spread:
                scalp_t = t; scalp_p = b2
            if use_stop and stop_t is None and stop_lo <= t <= stop_hi and b2 <= ask - stop_spread:
                stop_t = t
            if scalp_t is not None and stop_t is not None:
                break
        if scalp_t is not None:
            sell_fee = fee_ps(scalp_p) * shares
            pnl = (scalp_p - ask) * shares - buy_fee - sell_fee
            nscalp += 1
        elif use_stop and stop_t is not None:
            # sell at the bid at stop time (approx: ask - stop_spread)
            sell_p = ask - stop_spread
            sell_fee = fee_ps(sell_p) * shares
            pnl = (sell_p - ask) * shares - buy_fee - sell_fee
            nstop += 1
        else:
            won = (side == winner)
            pnl = (shares if won else 0.0) - STAKE - buy_fee
            if won: nholdwin += 1
            else: nholdlose += 1
        total += pnl
    ntrade = nscalp + nstop + nholdwin + nholdlose
    cov = ntrade / nwin * 100
    print(f"buy[{buy_lo:3d}-{buy_hi:3d}] mp>={min_mp:.2f} ask<={max_ask:.2f} sp={exit_spread:.2f} "
          f"sell[{sell_lo:3d}-{sell_hi:3d}] stop={stop_spread:.2f}[{stop_lo:3d}-{stop_hi:3d}] useStop={int(use_stop)}")
    print(f"   trades={ntrade:4d} ({ntrade/HR:4.2f}/hr) scalp={nscalp:4d} stop={nstop:3d} "
          f"holdW={nholdwin:3d} holdL={nholdlose:3d} skip={nskip:4d} cov={cov:3.0f}%")
    print(f"   total=${total:+9.1f}  $/trade=${total/ntrade if ntrade else 0:+6.3f}  "
          f"$/window=${total/nwin:+6.3f}  $/hr=${total/HR:+7.2f}")
    print()


print("=" * 95)
print("SCALP + TIGHT-STOP (no hold), $10 stake, 5% fee")
print("=" * 95)
print()
# Best scalp base, add stop
sim(90, 150, 0.55, 0.80, 0.05, 150, 280, 0.15, 150, 280, True)
sim(90, 150, 0.55, 0.80, 0.05, 150, 280, 0.20, 150, 280, True)
sim(90, 150, 0.55, 0.80, 0.05, 150, 280, 0.25, 150, 280, True)
sim(90, 150, 0.55, 0.90, 0.05, 150, 280, 0.20, 150, 280, True)
sim(90, 150, 0.50, 0.90, 0.05, 150, 280, 0.20, 150, 280, True)
sim(90, 150, 0.50, 0.90, 0.05, 150, 280, 0.25, 150, 280, True)
sim(80, 160, 0.50, 0.90, 0.05, 160, 290, 0.20, 160, 290, True)
sim(80, 160, 0.50, 0.90, 0.05, 160, 290, 0.30, 160, 290, True)
# wider buy + stop to maximize coverage
sim(70, 170, 0.50, 0.92, 0.05, 150, 290, 0.25, 150, 290, True)
sim(70, 170, 0.48, 0.95, 0.05, 150, 290, 0.25, 150, 290, True)
# compare: no stop (hold) baseline
sim(90, 150, 0.50, 0.90, 0.05, 150, 280, 0.20, 150, 280, False)
