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

def fee_ps(p): return FEE_R * min(p, 1.0 - p)

def side_prices(d, side):
    """Return (ask, bid) for a side."""
    if side == "Up":
        a = d.get("ask_Up"); b = d.get("bid_Up")
        if a is None and d.get("bid_Down") is not None: a = 1.0 - d["bid_Down"]
        if b is None and d.get("ask_Down") is not None: b = 1.0 - d["ask_Down"]
    else:
        a = d.get("ask_Down"); b = d.get("bid_Down")
        if a is None and d.get("bid_Up") is not None: a = 1.0 - d["bid_Up"]
        if b is None and d.get("ask_Up") is not None: b = 1.0 - d["ask_Up"]
    return a, b

def sim(buy_t_lo, buy_t_hi, exit_spread, sell_t_lo, sell_t_hi, min_mp, max_ask):
    """
    For each window:
    1. Find first sample with t in [buy_t_lo, buy_t_hi], model mp >= min_mp, ask <= max_ask
    2. Buy STAKE dollars of the fav side at that ask
    3. Scan forward: first sample with t in [sell_t_lo, sell_t_hi] and bid >= entry_ask + exit_spread -> SELL
    4. If no early exit: hold to settlement (winner from clob)
    """
    nwin = nscalp = nhold_win = nhold_lose = nskip = 0
    total_pnl = 0.0
    hourly = []  # (hour_idx, pnl)
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok": continue
        nwin += 1
        winner = c["winner"]
        hour_idx = slug[-5:]  # last 5 chars = timestamp portion

        # Find entry
        entry = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            mp = d.get("p_up_model")
            if t is None or mp is None: continue
            if not (buy_t_lo <= t <= buy_t_hi): continue
            if mp < min_mp: continue
            side = "Up" if mp >= 0.5 else "Down"
            fav_mp = mp if mp >= 0.5 else 1.0 - mp
            if fav_mp < min_mp: continue
            ask, bid = side_prices(d, side)
            if ask is None or ask > max_ask or ask < 0.15: continue
            entry = (t, side, ask, fav_mp)
            break
        if entry is None:
            nskip += 1
            continue
        t_ent, side, ask, mp = entry
        shares = STAKE / ask
        buy_fee = fee_ps(ask) * shares

        # Find early exit
        exited = False
        pnl = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            if t is None or t <= t_ent: continue
            if not (sell_t_lo <= t <= sell_t_hi): continue
            a2, b2 = side_prices(d, side)
            if b2 is not None and b2 >= ask + exit_spread:
                sell_fee = fee_ps(b2) * shares
                pnl = (b2 - ask) * shares - buy_fee - sell_fee
                nscalp += 1
                exited = True
                break
        if not exited:
            # Hold to settlement
            won = (side == winner)
            gross = shares if won else 0.0
            pnl = gross - STAKE - buy_fee
            if won: nhold_win += 1
            else: nhold_lose += 1
        total_pnl += pnl
        hourly.append((hour_idx, pnl))

    ntrade = nscalp + nhold_win + nhold_lose
    # Group by hour
    from collections import Counter
    hour_pnl = defaultdict(float)
    for h, p in hourly:
        hour_pnl[h] += p
    n_hours = len(hour_pnl)
    pos_hours = sum(1 for v in hour_pnl.values() if v > 0)
    print(f"buy_t=[{buy_t_lo}-{buy_t_hi}] exit_sp={exit_spread:.2f} sell_t=[{sell_t_lo}-{sell_t_hi}] min_mp={min_mp:.2f} max_ask={max_ask:.2f}")
    print(f"  windows={nwin} trades={ntrade} scalp={nscalp} hold_win={nhold_win} hold_lose={nhold_lose} skip={nskip}")
    print(f"  total=${total_pnl:+.1f}  $/trade=${total_pnl/ntrade if ntrade else 0:+.3f}  $/window=${total_pnl/nwin:+.3f}  $/hr=${total_pnl/HR:+.2f}")
    print(f"  hours={n_hours} pos_hours={pos_hours} ({pos_hours/n_hours*100:.0f}%)  avg $/hr=${total_pnl/n_hours if n_hours else 0:+.2f}")
    if nscalp + nhold_win + nhold_lose > 0:
        win_rate = (nscalp + nhold_win) / (nscalp + nhold_win + nhold_lose) * 100
        print(f"  overall win rate={win_rate:.1f}%")
    print()

print("=" * 80)
print("STRATEGY: Buy model-favoured side, scalp exit or hold to settlement")
print("=" * 80)
print()
# Baseline: buy at t=100, exit if bid rises 0.05 by t=200, min_mp=0.55, max_ask=0.65
sim(90, 130, 0.05, 150, 250, 0.55, 0.65)
sim(90, 130, 0.08, 150, 250, 0.55, 0.65)
sim(90, 130, 0.10, 150, 250, 0.55, 0.65)
sim(100, 140, 0.05, 160, 260, 0.55, 0.65)
sim(100, 140, 0.08, 160, 260, 0.55, 0.65)
sim(100, 140, 0.10, 160, 260, 0.55, 0.65)
# Lower min_mp for more coverage
sim(90, 140, 0.05, 160, 260, 0.50, 0.70)
sim(90, 140, 0.08, 160, 260, 0.50, 0.70)
sim(90, 140, 0.10, 160, 260, 0.50, 0.70)
# Wider buy window
sim(80, 150, 0.05, 160, 270, 0.50, 0.70)
sim(80, 150, 0.08, 160, 270, 0.50, 0.70)
sim(80, 150, 0.10, 160, 270, 0.50, 0.70)
# No scalp (pure hold)
sim(90, 140, 0.50, 200, 290, 0.50, 0.70)
sim(90, 140, 0.50, 200, 290, 0.55, 0.65)
