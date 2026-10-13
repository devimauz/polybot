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


def fee_ps(price):
    return FEE_R * min(price, 1.0 - price)


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


def side_pnl(price, side, winner):
    shares = STAKE / price
    gross = shares if side == winner else 0.0
    return (gross - STAKE) - fee_ps(price) * shares


def arb_profit(pU, pD, stake):
    S = stake / (pU + pD)
    fees = fee_ps(pU) * S + fee_ps(pD) * S
    return S - stake - fees


def combined_sim(arb_th, dir_gap, dec_t, fallback):
    """Every window, exactly one $10 action.
    Priority: arb (any time t) -> directional (fav if p-ask >= dir_gap, at dec_t) -> fallback."""
    nwin = n_arb = n_dir = n_fb = 0
    tot = 0.0
    for slug, samples in wins.items():
        c = clob.get(slug)
        if not c or c.get("status") != "ok":
            continue
        nwin += 1
        winner = c["winner"]
        action = None
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
                action = "arb"
                break
            if t >= dec_t:
                mp = d.get("p_up_model")
                if mp is None:
                    continue
                fav_p = mp if mp >= 0.5 else 1.0 - mp
                side = "Up" if mp >= 0.5 else "Down"
                ask = pU if side == "Up" else pD
                if (fav_p - ask) >= dir_gap:
                    tot += side_pnl(ask, side, winner)
                    n_dir += 1
                    action = "dir"
                    break
                # fallback: buy cheap side
                if fallback == "cheap":
                    if pU <= pD:
                        tot += side_pnl(pU, "Up", winner)
                        n_fb += 1
                    else:
                        tot += side_pnl(pD, "Down", winner)
                        n_fb += 1
                    action = "fb"
                    break
                elif fallback == "fav":
                    tot += side_pnl(ask, side, winner)
                    n_fb += 1
                    action = "fb"
                    break
                elif fallback == "both":
                    S = STAKE / (pU + pD)
                    fees = fee_ps(pU) * S + fee_ps(pD) * S
                    tot += S - STAKE - fees  # may be negative
                    n_fb += 1
                    action = "fb"
                    break
    n_trade = n_arb + n_dir + n_fb
    cov = n_trade / nwin * 100
    print(f"arb_th={arb_th:.3f} gap>={dir_gap:+.2f} dec_t={dec_t:3d} fb={fallback:6}  trades={n_trade:4d} ({n_trade/HR:4.2f}/hr)  arb={n_arb:3d} dir={n_dir:4d} fb={n_fb:4d}  cover={cov:3.0f}%  total=${tot:+9.2f}  $/hr=${tot/HR:+6.3f}  $/trade=${tot/n_trade if n_trade else 0:+5.3f}  $/window=${tot/nwin:+5.3f}")


print("=== COMBINED: arb + directional (gap) + fallback (trades every window) ===")
print("--- arb_th=1.0, dec_t=120, gap>=0.10, fallbacks ---")
for fb in ["cheap", "fav", "both"]:
    combined_sim(1.0, 0.10, 120, fb)
print()
print("--- sweep dir_gap (arb_th=1.0, dec_t=120, fb=cheap) ---")
for g in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    combined_sim(1.0, g, 120, "cheap")
print()
print("--- sweep dec_t (arb_th=1.0, gap=0.10, fb=cheap) ---")
for dec_t in [90, 120, 150, 180, 210]:
    combined_sim(1.0, 0.10, dec_t, "cheap")
print()
print("--- sweep arb_th (gap=0.10, dec_t=120, fb=cheap) ---")
for th in [0.99, 0.995, 1.0, 1.005, 1.01]:
    combined_sim(th, 0.10, 120, "cheap")
