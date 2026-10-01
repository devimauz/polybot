#!/usr/bin/env python
"""Backtest the NEW twap_bot gate against the 51 historical paper entries.

Applies:  model_p >= 0.80, ASK_LO<=ask<=ASK_HI, gap>=eff_gap (Up 1.5x),
          move-stalling (z spike->decay + leader-ask drawdown), ev_frac>edge.
Sizes by fractional Kelly, exits on TP_BID / STOP_BID before settlement.
Compares each config vs hold-to-settlement.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = ROOT / "price_logs" / "twap"

FEE_R = 0.05
# new-gate params (must match twap_bot.py)
CONF_MIN_FAV = 0.80
ASK_LO, ASK_HI = 0.60, 0.88
GAP_MIN = 0.08
UP_GAP_MULT = 1.5
Z_SPIKE_MIN, Z_DECAY_MIN = 0.50, 0.10
STALL_FRAC, STALL_ABS = 0.05, 0.02
TP_BID, STOP_BID = 0.95, 0.70
KELLY_FRAC, BANKROLL = 0.25, 1000.0
MIN_STAKE, MAX_STAKE = 10.0, 100.0
EDGE = 0.02  # args.edge default

def fee_for(price, shares):
    return FEE_R * min(price, 1 - price) * shares

def ev_frac(model_p, price):
    return model_p * ((1 - price) / price) - (1 - model_p) - FEE_R * min(price, 1 - price) / price

def kelly_frac(model_p, price):
    return max((model_p - price) / (1 - price), 0.0)

# ---- load windows (time-ordered per slug) ----
rows = [json.loads(l) for l in (T / "windows.jsonl").open(encoding="utf-8")]
by_slug = defaultdict(list)
for r in rows:
    info = r.get("info") or {}
    by_slug[r["window"]].append({
        "ts": r["t"],
        "z": info.get("z"),
        "p_up": r.get("p_up_model"),
        "ask_up": r.get("ask_Up"), "ask_dn": r.get("ask_Down"),
        "bid_up": r.get("bid_Up"), "bid_dn": r.get("bid_Down"),
    })
for slug in by_slug:
    by_slug[slug].sort(key=lambda x: x["ts"])

# ---- load entries + outcomes ----
jrows = [json.loads(l) for l in (T / "trades.jsonl").open(encoding="utf-8")]
entries = [r for r in jrows if "model_p" in r and "slug" in r]
outcome = {}
for r in jrows:
    if r.get("event") == "settle":
        outcome[r["slug"]] = r.get("won")
    elif r.get("event") == "exit":  # in case a prior run already wrote exits
        pass

def ts_of(s):
    # r['t'] is ISO; entries store offset in 't' but ts iso; compare via slug+iso
    return s

print(f"windows={len(by_slug)}  entries={len(entries)}  settled_outcomes={len(outcome)}\n")

def build_state(slug):
    """Per-row state: peak (max ask both sides), z_hi, z_lo (signed), and
    whether STALLED holds for the Up leader and the Down leader."""
    seq = by_slug.get(slug, [])
    out = []
    z_lo = z_hi = None
    peak = None
    for s in seq:
        if s["z"] is not None:
            z_lo = s["z"] if z_lo is None else min(z_lo, s["z"])
            z_hi = s["z"] if z_hi is None else max(z_hi, s["z"])
        asks = [a for a in (s["ask_up"], s["ask_dn"]) if a is not None]
        if asks:
            peak = max(peak or 0, max(asks))
        out.append((s, z_lo, z_hi, peak))
    return out

def stall_check(z, z_lo, z_hi, peak, ba):
    if z is None or peak is None:
        return False
    spiked = abs(z_hi) >= Z_SPIKE_MIN or abs(z_lo) >= Z_SPIKE_MIN
    decayed = (z_hi - z >= Z_DECAY_MIN) if z > 0 else (z - z_lo >= Z_DECAY_MIN)
    drawdown = (ba <= peak - STALL_FRAC * peak) and (peak - ba >= STALL_ABS)
    return spiked and decayed and drawdown

def kelly_stake(model_p, price):
    return min(max(KELLY_FRAC * BANKROLL * kelly_frac(model_p, price), MIN_STAKE), MAX_STAKE)

def compute_exit(side, slug, entry_ts, shares, entry_cost):
    """Walk the leader bid path after entry; TP/STOP or hold to settlement."""
    seq = by_slug.get(slug, [])
    for s in seq:
        if s["ts"] <= entry_ts:
            continue
        bid = s["bid_up"] if side == "Up" else s["bid_dn"]
        if bid is None:
            continue
        if bid >= TP_BID:
            proceeds = shares * bid
            return (proceeds - fee_for(bid, shares)) - entry_cost, "tp"
        if bid <= STOP_BID:
            proceeds = shares * bid
            return (proceeds - fee_for(bid, shares)) - entry_cost, "stop"
    # hold to settlement
    won = outcome.get(slug)
    if won is None:
        return None, "unresolved"
    return (shares if won else 0.0) - entry_cost, "settle"

results = []
for e in entries:
    slug, side = e["slug"], e["side"]
    model_p, ask = e["model_p"], e["ask"]
    z = e.get("z")
    gap = model_p - ask
    eff_gap = GAP_MIN * (UP_GAP_MULT if side == "Up" else 1.0)
    evf = ev_frac(model_p, ask)
    gate = {
        "conf": model_p >= CONF_MIN_FAV,
        "askband": ASK_LO <= ask <= ASK_HI,
        "gap": gap >= eff_gap,
        "ev": evf > EDGE,
    }
    # reconstruct stalling state at the entry moment
    state = build_state(slug)
    stalled = False
    # find the row closest at/before entry ts
    best_row = None
    for (s, z_lo, z_hi, peak) in state:
        if s["ts"] <= e["ts"]:
            best_row = (s, z_lo, z_hi, peak)
        else:
            break
    if best_row:
        s, z_lo, z_hi, peak = best_row
        ba = (s["ask_up"] if side == "Up" else s["ask_dn"]) or ask
        zz = (s["z"] if s["z"] is not None else z)
        stalled = stall_check(zz, z_lo, z_hi, peak, ba)
    gate["stall"] = stalled
    gate["ALL"] = all(gate.values())
    stake = kelly_stake(model_p, ask)
    shares = stake / ask
    entry_cost = stake + fee_for(ask, shares)
    hold_won = outcome.get(slug)
    hold_pnl = ((shares if hold_won else 0.0) - entry_cost) if hold_won is not None else None
    exit_pnl, how = (None, "skipped")
    if gate["ALL"] and hold_won is not None:
        exit_pnl, how = compute_exit(side, slug, e["ts"], shares, entry_cost)
    results.append(dict(slug=slug, side=side, model_p=model_p, ask=ask, z=z,
                        gap=round(gap, 3), evf=round(evf, 4), gate=gate,
                        stake=round(stake, 2), won=hold_won,
                        hold_pnl=round(hold_pnl, 2) if hold_pnl is not None else None,
                        exit_pnl=round(exit_pnl, 2) if exit_pnl is not None else None,
                        how=how))

# ---- reporting ----
def report(name, sel):
    n = len(sel)
    if n == 0:
        print(f"{name}: 0 trades")
        return
    pnl = [r for r in sel if r[1] is not None]
    tot = sum(r[1] for r in pnl)
    wins = sum(1 for r in pnl if r[1] > 0)
    print(f"{name}: {n} trades, {len(pnl)} resolved, wins={wins}, total=${tot:.2f}")

print("=" * 88)
print(f"{'slug':34} {'side':4} {'mp':>5} {'ask':>5} {'gap':>5} {'evf':>6} "
      f"{'stall':>5} {'stake':>6} {'hold':>7} {'exit':>7} {'how':>6}  gatefails")
for r in sorted(results, key=lambda x: x["ts"] if "ts" in x else ""):
    fails = [k for k in ("conf", "askband", "gap", "ev", "stall") if not r["gate"][k]]
    mark = "PASS" if r["gate"]["ALL"] else "  "
    print(f"{r['slug'][3:]:34} {r['side']:4} {r['model_p']:5.3f} {r['ask']:5.3f} "
          f"{r['gap']:5.3f} {r['evf']:6.3f} {'Y' if r['gate']['stall'] else 'n':>5} "
          f"{r['stake']:6.2f} "
          f"{str(r['hold_pnl']):>7} {str(r['exit_pnl']):>7} {r['how']:>6}  {','.join(fails) or '-'} {mark}")

# aggregate
print("=" * 88)
allpass = [r["exit_pnl"] for r in results if r["gate"]["ALL"] and r["exit_pnl"] is not None]
allhold = [r["hold_pnl"] for r in results if r["hold_pnl"] is not None]
print(f"HOLD-TO-SETTLEMENT (all {len(allhold)}): ${sum(allhold):.2f}, "
      f"win rate {sum(1 for p in allhold if p>0)}/{len(allhold)}")
print(f"NEW GATE -> EXITS   (passing {len(allpass)}): ${sum(allpass):.2f}")

# Up vs Down among passing
up = [r["exit_pnl"] for r in results if r["gate"]["ALL"] and r["side"]=="Up" and r["exit_pnl"] is not None]
dn = [r["exit_pnl"] for r in results if r["gate"]["ALL"] and r["side"]=="Down" and r["exit_pnl"] is not None]
if up: print(f"  passing Up bets: {len(up)}  ${sum(up):.2f}")
if dn: print(f"  passing Down bets: {len(dn)}  ${sum(dn):.2f}")
# how exits resolved
how_count = defaultdict(list)
for r in results:
    if r["gate"]["ALL"] and r["exit_pnl"] is not None:
        how_count[r["how"]].append(r["exit_pnl"])
for h, v in how_count.items():
    print(f"  exit={h}: {len(v)}  ${sum(v):.2f}")
