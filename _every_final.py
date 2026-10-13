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
NW = 0


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


def up_ask(d):
    a = d.get("ask_Up")
    if a is not None: return a
    b = d.get("bid_Down")
    return (1.0 - b) if b is not None else None


def down_ask(d):
    a = d.get("ask_Down")
    if a is not None: return a
    b = d.get("bid_Up")
    return (1.0 - b) if b is not None else None


def sim(FEE_R, min_mp, max_ask, exit_sp, sell_lo, sell_hi, stop_sp,
        fallback="cheap"):
    """EVERY window gets one $10 action.
    Signal: first sample t in [90,150], fav_mp>=min_mp, fav_ask<=max_ask -> scalp fav side.
      Exit: first scalp (bid>=ask+exit_sp in [sell_lo,sell_hi]) else first stop (bid<=ask-stop_sp) else hold.
    No signal: fallback buy (cheap side OR both-sides arb) at first t>=120 sample.
    """
    global NW
    NW = 0
    tot = 0.0
    n_sig = 0; n_fb = 0
    sig_pnl = 0.0; fb_pnl = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok": continue
        NW += 1
        winner = c["winner"]
        # --- signal entry ---
        entry = None
        for d in samples:
            info = d.get("info"); t = info.get("t") if info else None
            mp = d.get("p_up_model")
            if t is None or mp is None: continue
            if not (90 <= t <= 150): continue
            fav_mp = mp if mp >= 0.5 else 1.0 - mp
            if fav_mp < min_mp: continue
            side = "Up" if mp >= 0.5 else "Down"
            ask, bid = side_prices(d, side)
            if ask is None or ask < 0.20 or ask > max_ask: continue
            entry = (t, side, ask)
            break
        if entry is not None:
            t_ent, side, ask = entry
            shares = STAKE / ask
            buy_fee = FEE_R * min(ask, 1 - ask) * shares
            scalp = None; stop = None
            for d in samples:
                info = d.get("info"); t = info.get("t") if info else None
                if t is None or t <= t_ent: continue
                a2, b2 = side_prices(d, side)
                if b2 is None: continue
                if scalp is None and sell_lo <= t <= sell_hi and b2 >= ask + exit_sp:
                    scalp = b2
                if stop is None and sell_lo <= t <= sell_hi and b2 <= ask - stop_sp:
                    stop = t
                if scalp is not None and stop is not None: break
            if scalp is not None:
                pnl = (scalp - ask) * shares - buy_fee - FEE_R * min(scalp, 1 - scalp) * shares
            elif stop is not None:
                sp = max(ask - stop_sp, 0.01)
                pnl = (sp - ask) * shares - buy_fee - FEE_R * min(sp, 1 - sp) * shares
            else:
                won = side == winner
                pnl = (shares if won else 0.0) - STAKE - buy_fee
            tot += pnl; sig_pnl += pnl; n_sig += 1
            continue
        # --- fallback (no signal) ---
        d = None
        for s in samples:
            info = s.get("info"); t = info.get("t") if info else None
            if t is not None and t >= 120:
                d = s; break
        if d is None:
            fb = None
        else:
            pU, pD = up_ask(d), down_ask(d)
            if fallback == "both" and pU and pD and (pU + pD + 2 * FEE_R * 0.5) < 1.0:
                S = STAKE / (pU + pD)
                fb = S - STAKE - FEE_R * min(pU, 1 - pU) * S - FEE_R * min(pD, 1 - pD) * S
            elif pU and pD:
                if pU <= pD:
                    ask, sd = pU, "Up"
                else:
                    ask, sd = pD, "Down"
                shares = STAKE / ask
                won = sd == winner
                fb = (shares if won else 0.0) - STAKE - FEE_R * min(ask, 1 - ask) * shares
            else:
                fb = None
        if fb is None:
            # last resort: no book -> count as $0 (no trade)
            fb = 0.0
        tot += fb; fb_pnl += fb; n_fb += 1
    cov = (n_sig + n_fb) / NW
    return dict(NW=NW, tot=tot, per_win=tot / NW, per_hr=tot / NW * 12,
                n_sig=n_sig, sig_avg=sig_pnl / n_sig if n_sig else 0,
                n_fb=n_fb, fb_avg=fb_pnl / n_fb if n_fb else 0)


for F in [0.025, 0.05]:
    for fb in ["cheap", "both"]:
        r = sim(F, 0.58, 0.90, 0.08, 150, 280, 0.12, fb)
        print(f"FEE={F:.3f} fb={fb:5} | wins={r['NW']:4d} sig={r['n_sig']:3d} (avg ${r['sig_avg']:+.2f}) "
              f"fb={r['n_fb']:3d} (avg ${r['fb_avg']:+.2f}) | total=${r['tot']:+9.1f} "
              f"$/win=${r['per_win']:+.3f} $/hr=${r['per_hr']:+.2f}")
