#!/usr/bin/env python
"""Grid-search the new-gate constants against the 51 historical entries to
find a config that (a) passes few trades, (b) is net positive, (c) blocks the
big -100 losses. Sizing = fractional Kelly; exits = TP/STOP before settlement."""
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = ROOT / "price_logs" / "twap"
FEE_R = 0.05
CONF_MIN_FAV = 0.80
ASK_LO, ASK_HI = 0.60, 0.88
TP_BID, STOP_BID = 0.95, 0.70
KELLY_FRAC, BANKROLL = 0.25, 1000.0
MIN_STAKE, MAX_STAKE = 10.0, 100.0

def fee_for(p, sh): return FEE_R * min(p, 1 - p) * sh
def ev_frac(mp, pr): return mp * ((1 - pr) / pr) - (1 - mp) - FEE_R * min(pr, 1 - pr) / pr
def kelly_frac(mp, pr): return max((mp - pr) / (1 - pr), 0.0)

rows = [json.loads(l) for l in (T / "windows.jsonl").open(encoding="utf-8")]
by_slug = defaultdict(list)
for r in rows:
    info = r.get("info") or {}
    by_slug[r["window"]].append({
        "ts": r["t"], "z": info.get("z"), "p_up": r.get("p_up_model"),
        "ask_up": r.get("ask_Up"), "ask_dn": r.get("ask_Down"),
        "bid_up": r.get("bid_Up"), "bid_dn": r.get("bid_Down")})
for s in by_slug: by_slug[s].sort(key=lambda x: x["ts"])

jrows = [json.loads(l) for l in (T / "trades.jsonl").open(encoding="utf-8")]
entries = [r for r in jrows if "model_p" in r and "slug" in r]
outcome = {r["slug"]: r.get("won") for r in jrows if r.get("event") == "settle"}

def build(slug):
    seq = by_slug.get(slug, []); out=[]; z_lo=z_hi=peak=None
    for s in seq:
        if s["z"] is not None:
            z_lo = s["z"] if z_lo is None else min(z_lo, s["z"])
            z_hi = s["z"] if z_hi is None else max(z_hi, s["z"])
        asks=[a for a in (s["ask_up"],s["ask_dn"]) if a is not None]
        if asks: peak = max(peak or 0, max(asks))
        out.append((s,z_lo,z_hi,peak))
    return out

def state_at(slug, ets):
    best=None
    for (s,zl,zh,peak) in build(slug):
        if s["ts"] <= ets: best=(s,zl,zh,peak)
        else: break
    return best

def compute_exit(side, slug, ets, shares, cost):
    for s in by_slug.get(slug,[]):
        if s["ts"] <= ets: continue
        bid = s["bid_up"] if side=="Up" else s["bid_dn"]
        if bid is None: continue
        if bid >= TP_BID: return (shares*bid - fee_for(bid,shares)) - cost, "tp"
        if bid <= STOP_BID: return (shares*bid - fee_for(bid,shares)) - cost, "stop"
    won = outcome.get(slug)
    if won is None: return None, "unres"
    return (shares if won else 0.0) - cost, "settle"

# precompute per-entry derived values
data=[]
for e in entries:
    side, mp, ask, z, slug, ets = e["side"], e["model_p"], e["ask"], e.get("z"), e["slug"], e["ts"]
    gap = mp - ask
    evf = ev_frac(mp, ask)
    eff_gap_base = gap / (1.5 if side=="Up" else 1.0)  # gap in "base gap units"
    st = state_at(slug, ets)
    z_lo=z_hi=peak=None
    if st: _, z_lo, z_hi, peak = st
    data.append(dict(slug=slug, side=side, mp=mp, ask=ask, z=z, ets=ets,
                     gap=gap, evf=evf, z_lo=z_lo, z_hi=z_hi, peak=peak,
                     won=outcome.get(slug)))

def eval_cfg(gap_min, up_mult, z_spike, z_decay, st_frac, st_abs, edge):
    total=0.0; n=0; wins=0
    for d in data:
        if d["mp"] < CONF_MIN_FAV or not (ASK_LO <= d["ask"] <= ASK_HI): continue
        eff = gap_min * (up_mult if d["side"]=="Up" else 1.0)
        if d["gap"] < eff: continue
        if d["evf"] <= edge: continue
        # stalling
        z=d["z"];
        if z is None: continue
        spiked = abs(d["z_hi"]) >= z_spike or abs(d["z_lo"]) >= z_spike
        decayed = (d["z_hi"] - z >= z_decay) if z > 0 else (z - d["z_lo"] >= z_decay)
        ba = d["ask"]; pk = d["peak"]
        drawdown = (pk is not None) and (ba <= pk - st_frac*pk) and (pk - ba >= st_abs)
        if not (spiked and decayed and drawdown): continue
        stake = min(max(KELLY_FRAC*BANKROLL*kelly_frac(d["mp"], ba), MIN_STAKE), MAX_STAKE)
        shares = stake / ba; cost = stake + fee_for(ba, shares)
        pnl, _ = compute_exit(d["side"], d["slug"], d["ets"], shares, cost)
        if pnl is None: continue
        total += pnl; n += 1; wins += pnl > 0
    return n, wins, total

grid = product(
    [0.08, 0.10, 0.12],     # gap_min
    [1.0, 1.5, 2.0],        # up_mult
    [0.5, 1.0],             # z_spike
    [0.05, 0.10, 0.20],     # z_decay
    [0.03, 0.05, 0.08],     # st_frac
    [0.02, 0.03],           # st_abs
)
best=[]
for gm,um,zs,zd,sf,sa in grid:
    for edge in [0.02, 0.05, 0.10]:
        n,w,t = eval_cfg(gm,um,zs,zd,sf,sa,edge)
        best.append((t,n,w,gm,um,zs,zd,sf,sa,edge))
best.sort(key=lambda x:(x[0]<0, -x[0], -x[1]))
print(f"{'total':>8} {'n':>3} {'w':>3}  gap up x zsp zdc sfrac sabs edge")
for t,n,w,gm,um,zs,zd,sf,sa,edge in best[:25]:
    print(f"{t:8.2f} {n:3d} {w:3d}  {gm:.2f} {um:.1f} {zs:.1f} {zd:.2f} {sf:.2f} {sa:.2f} {edge:.2f}")
print("...\nworst:")
for t,n,w,gm,um,zs,zd,sf,sa,edge in best[-5:]:
    print(f"{t:8.2f} {n:3d} {w:3d}  {gm:.2f} {um:.1f} {zs:.1f} {zd:.2f} {sf:.2f} {sa:.2f} {edge:.2f}")

# baseline: old hold-to-settlement on all
tot=0;n=0
for d in data:
    if d["mp"] < CONF_MIN_FAV: continue
    stake=100.0; sh=stake/d["ask"]; cost=stake+fee_for(d["ask"],sh)
    if d["won"] is None: continue
    tot += (sh if d["won"] else 0.0)-cost; n+=1
print(f"\nOLD all-0.80 hold-to-settle: {n} trades ${tot:.2f}")
