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
STAKE = 10.0
HR = 154.25


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


# Pre-extract per-window timelines (t, side, ask, bid) once
timelines = {}
for slug, samples in wins.items():
    c = clob.get(slug)
    if not c or c.get("status") != "ok":
        continue
    tl = []
    for d in samples:
        info = d.get("info"); t = info.get("t") if info else None
        mp = d.get("p_up_model")
        if t is None or mp is None: continue
        side = "Up" if mp >= 0.5 else "Down"
        ask, bid = side_prices(d, side)
        tl.append((t, side, mp, ask, bid))
    timelines[slug] = (c["winner"], tl)
NW = len(timelines)


def sim(FEE_R, buy_lo, buy_hi, min_mp, max_ask, exit_sp, sell_lo, sell_hi,
        stop_sp, stop_lo, stop_hi):
    nscalp = nstop = nholdwin = nholdlose = nskip = 0
    total = 0.0
    for slug, (winner, tl) in timelines.items():
        entry = None
        for (t, side, mp, ask, bid) in tl:
            if not (buy_lo <= t <= buy_hi): continue
            fav_mp = mp if mp >= 0.5 else 1.0 - mp
            if fav_mp < min_mp: continue
            if ask is None or ask < 0.20 or ask > max_ask: continue
            entry = (t, side, ask)
            break
        if entry is None:
            nskip += 1
            continue
        t_ent, side, ask = entry
        shares = STAKE / ask
        buy_fee = FEE_R * min(ask, 1.0 - ask) * shares
        scalp_t = None; scalp_p = None; stop_t = None
        for (t, s2, mp2, a2, b2) in tl:
            if t <= t_ent or s2 != side or b2 is None: continue
            if scalp_t is None and sell_lo <= t <= sell_hi and b2 >= ask + exit_sp:
                scalp_t, scalp_p = t, b2
            if stop_t is None and stop_lo <= t <= stop_hi and b2 <= ask - stop_sp:
                stop_t = t
            if scalp_t is not None and stop_t is not None:
                break
        if scalp_t is not None:
            total += (scalp_p - ask) * shares - buy_fee - FEE_R * min(scalp_p, 1 - scalp_p) * shares
            nscalp += 1
        elif stop_t is not None:
            sp = max(ask - stop_sp, 0.01)
            total += (sp - ask) * shares - buy_fee - FEE_R * min(sp, 1 - sp) * shares
            nstop += 1
        else:
            won = side == winner
            total += (shares if won else 0.0) - STAKE - buy_fee
            nholdwin += won
            nholdlose += (not won)
    ntr = nscalp + nstop + nholdwin + nholdlose
    return (ntr, nscalp, nstop, nholdwin + nholdlose, nskip, total,
            total / HR, total / NW)


print("=" * 100)
print(f"WINDOWS={NW}  ($/hr normalised over {HR} windows; $/window over all {NW} windows)")
print("=" * 100)
print()
print("Fee sensitivity for the best-ish config (mp>=0.55 ask<=0.80 sp=0.05 sell[150-280] stop0.15):")
for F in [0.00, 0.01, 0.025, 0.05]:
    r = sim(F, 90, 150, 0.55, 0.80, 0.05, 150, 280, 0.15, 150, 280)
    print(f"  FEE_R={F:.3f}: trades={r[0]:3d} scalp={r[1]:3d} stop={r[2]:3d} hold={r[3]:2d} skip={r[4]:3d} "
          f"total=${r[5]:+9.1f} $/hr=${r[6]:+6.2f} $/win=${r[7]:+5.3f}")
print()
print("Fine grid (FEE_R=0.025), targeting max $/win:")
rows = []
for min_mp in [0.55, 0.58, 0.60, 0.62]:
    for max_ask in [0.75, 0.80, 0.85, 0.90]:
        for exit_sp in [0.04, 0.06, 0.08]:
            for stop_sp in [0.12, 0.18]:
                r = sim(0.025, 90, 150, min_mp, max_ask, exit_sp, 150, 280, stop_sp, 150, 280)
                rows.append((r[7], r[5]/HR, min_mp, max_ask, exit_sp, stop_sp, r[0], r[1], r[2], r[3], r[4]))
rows.sort(key=lambda x: -x[0])
print(f"{'$/win':>7} {'$/hr':>7} {'mp':>4} {'ask':>5} {'sp':>5} {'stop':>5} {'tr':>4} {'scalp':>5} {'stopN':>5} {'hold':>4} {'skip':>4}")
for rw in rows[:15]:
    print(f"{rw[0]:+7.3f} {rw[1]:+7.2f} {rw[2]:4.2f} {rw[3]:5.2f} {rw[4]:5.2f} {rw[5]:5.2f} "
          f"{rw[6]:4d} {rw[7]:5d} {rw[8]:5d} {rw[9]:4d} {rw[10]:4d}")
print()
print("Same grid with FEE_R=0.05 (conservative):")
rows = []
for min_mp in [0.55, 0.58, 0.60, 0.62]:
    for max_ask in [0.75, 0.80, 0.85, 0.90]:
        for exit_sp in [0.04, 0.06, 0.08]:
            for stop_sp in [0.12, 0.18]:
                r = sim(0.05, 90, 150, min_mp, max_ask, exit_sp, 150, 280, stop_sp, 150, 280)
                rows.append((r[7], r[5]/HR, min_mp, max_ask, exit_sp, stop_sp, r[0], r[1], r[2], r[3], r[4]))
rows.sort(key=lambda x: -x[0])
for rw in rows[:15]:
    print(f"{rw[0]:+7.3f} {rw[1]:+7.2f} {rw[2]:4.2f} {rw[3]:5.2f} {rw[4]:5.2f} {rw[5]:5.2f} "
          f"{rw[6]:4d} {rw[7]:5d} {rw[8]:5d} {rw[9]:4d} {rw[10]:4d}")
