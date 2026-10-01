#!/usr/bin/env python
"""Full-window simulation of the NEW twap_bot across all recorded 5m windows.
Replays windows.jsonl (per ~3s sample: z, ask_up/dn, bid_up/dn) and enters any
time the new gate fires (conf 0.80 + ask band + gap + ev + STALLING), sizes by
Kelly, exits on TP/STOP before settlement. This is the only way to validate a
gate that trades *after* the spike peak (the 51 historical entries predate it)."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = ROOT / "price_logs" / "twap"
FEE_R = 0.05
CONF_MIN_FAV = 0.80
ASK_LO, ASK_HI = 0.60, 0.88
MIN_T, MAX_T = 90, 295
SKIP_LO, SKIP_HI = 100, 110
TP_BID, STOP_BID = 0.95, 0.70
KELLY_FRAC, BANKROLL = 0.25, 1000.0
MIN_STAKE, MAX_STAKE = 10.0, 100.0
EDGE = 0.02

def fee_for(p, sh): return FEE_R * min(p, 1 - p) * sh
def ev_frac(mp, pr): return mp * ((1 - pr) / pr) - (1 - mp) - FEE_R * min(pr, 1 - pr) / pr
def kelly_frac(mp, pr): return max((mp - pr) / (1 - pr), 0.0)
def stake_for(mp, pr): return min(max(KELLY_FRAC*BANKROLL*kelly_frac(mp, pr), MIN_STAKE), MAX_STAKE)

rows = [json.loads(l) for l in (T / "windows.jsonl").open(encoding="utf-8")]
by_slug = defaultdict(list)
for r in rows:
    info = r.get("info") or {}
    by_slug[r["window"]].append({
        "ts": r["t"], "t": info.get("t"), "z": info.get("z"), "p_up": r.get("p_up_model"),
        "ask_up": r.get("ask_Up"), "ask_dn": r.get("ask_Down"),
        "bid_up": r.get("bid_Up"), "bid_dn": r.get("bid_Dn") if False else r.get("bid_Down")})
for s in by_slug: by_slug[s].sort(key=lambda x: x["ts"])
jrows = [json.loads(l) for l in (T / "trades.jsonl").open(encoding="utf-8")]
outcome = {r["slug"]: r.get("won") for r in jrows if r.get("event") == "settle"}

def simulate(gap_min, up_mult, z_spike, z_decay, st_frac, st_abs, edge=EDGE, detail=False):
    total = 0.0; n = 0; wins = 0; up_pnl = 0.0; dn_pnl = 0.0
    how_count = defaultdict(lambda: [0, 0.0])
    details = []
    for slug, seq in by_slug.items():
        if slug not in outcome or outcome[slug] is None:
            continue
        won = outcome[slug]
        traded = set(); side_count = defaultdict(int)
        z_lo = z_hi = peak_up = peak_dn = None
        entered = False
        i = 0
        while i < len(seq):
            s = seq[i]
            t = s.get("t")
            tradeable = s["p_up"] is not None and t is not None \
                and MIN_T <= t <= MAX_T and not (SKIP_LO <= t < SKIP_HI)
            # update extremes (per-side ask peaks)
            if s["ask_up"] is not None:
                peak_up = s["ask_up"] if peak_up is None else max(peak_up, s["ask_up"])
            if s["ask_dn"] is not None:
                peak_dn = s["ask_dn"] if peak_dn is None else max(peak_dn, s["ask_dn"])
            if s["z"] is not None:
                z_lo = s["z"] if z_lo is None else min(z_lo, s["z"])
                z_hi = s["z"] if z_hi is None else max(z_hi, s["z"])
            if tradeable and not entered:
                z = s["z"]; spiked = z_lo is not None and (abs(z_hi) >= z_spike or abs(z_lo) >= z_spike)
                for side in ("Up", "Down"):
                    ask = s["ask_up"] if side == "Up" else s["ask_dn"]
                    if ask is None or not (ASK_LO <= ask <= ASK_HI): continue
                    mp = s["p_up"] if side == "Up" else 1 - s["p_up"]
                    if mp is None or mp < CONF_MIN_FAV: continue
                    eff = gap_min * (up_mult if side == "Up" else 1.0)
                    if (mp - ask) < eff: continue
                    if ev_frac(mp, ask) <= edge: continue
                    # stalling for this side
                    if z is None: continue
                    decayed = (z_hi - z >= z_decay) if z > 0 else (z - z_lo >= z_decay)
                    if side == "Up" and z > 0:
                        pk = peak_up
                        dd = pk is not None and ask <= pk - st_frac*pk and (pk - ask) >= st_abs
                    elif side == "Down" and z < 0:
                        pk = peak_dn
                        dd = pk is not None and ask <= pk - st_frac*pk and (pk - ask) >= st_abs
                    else:
                        continue
                    if not (spiked and decayed and dd): continue
                    if side in traded or side_count[side] >= 2: continue
                    # ENTER
                    stake = stake_for(mp, ask); shares = stake / ask
                    cost = stake + fee_for(ask, shares)
                    traded.add(side); side_count[side] += 1
                    entered = True
                    # walk forward for exit
                    pnl = None; how = None
                    for s2 in seq[i+1:]:
                        b2 = s2["bid_up"] if side == "Up" else s2["bid_dn"]
                        if b2 is None: continue
                        if b2 >= TP_BID:
                            pnl = shares*b2 - fee_for(b2, shares) - cost; how = "tp"; break
                        if b2 <= STOP_BID:
                            pnl = shares*b2 - fee_for(b2, shares) - cost; how = "stop"; break
                    if pnl is None:
                        pnl = (shares if won else 0.0) - cost; how = "settle"
                    total += pnl; n += 1; wins += pnl > 0
                    if side == "Up": up_pnl += pnl
                    else: dn_pnl += pnl
                    how_count[how][0] += 1; how_count[how][1] += pnl
                    if detail:
                        details.append(dict(slug=slug, side=side, mp=round(mp,3), ask=ask,
                                            z=round(z,2) if z is not None else None,
                                            stake=round(stake,1), won=won, pnl=round(pnl,2),
                                            how=how, t=round(t,0)))
                    i = len(seq); break
                if entered: break
            i += 1
    return dict(total=total, n=n, wins=wins, up=up_pnl, dn=dn_pnl, how=dict(how_count), details=details)

# --- sweep to see the landscape ---
from itertools import product
print(f"{'total':>8} {'n':>3} {'w':>3}  gap upm zsp zdc sfrac sabs   (up_pnl / dn_pnl / by-exit)")
results = []
for gm, um, zs, zd, sf, sa in product(
        [0.08, 0.10, 0.12, 0.15], [1.0, 1.5, 2.0], [0.5, 0.8, 1.0],
        [0.10, 0.20, 0.40], [0.05, 0.10, 0.15], [0.02, 0.04]):
    r = simulate(gm, um, zs, zd, sf, sa)
    results.append((r["total"], gm, um, zs, zd, sf, sa, r))
results.sort(key=lambda x: -x[0])
for total, gm, um, zs, zd, sf, sa, r in results[:18]:
    tp = r["how"].get("tp", [0,0]); st = r["how"].get("stop", [0,0]); se = r["how"].get("settle",[0,0])
    print(f"{total:8.2f} {r['n']:3d} {r['wins']:3d}  {gm:.2f} {um:.1f} {zs:.1f} {zd:.2f} {sf:.2f} {sa:.2f}   "
          f"tp{tp[0]:2d}/${tp[1]:6.1f} stop{st[0]:2d}/${st[1]:6.1f} settle{se[0]:2d}/${se[1]:6.1f}")

# --- full detail for the top config ---
best_total, gm, um, zs, zd, sf, sa, _ = results[0]
r = simulate(gm, um, zs, zd, sf, sa, detail=True)
print(f"\n=== BEST CONFIG DETAIL  gap>={gm} Up x{um}  z_spike>={zs} z_decay>={zd}  stall_frac>={sf} stall_abs>={sa} ===")
print(f"{'slug':30} {'side':4} {'mp':>5} {'ask':>5} {'z':>6} {'stake':>6} {'pnl':>7}  exit")
for d in sorted(r["details"], key=lambda x: x["slug"]):
    print(f"{d['slug'][3:]:30} {d['side']:4} {d['mp']:5.3f} {d['ask']:5.3f} {d['z']:6.2f} {d['stake']:6.1f} {d['pnl']:7.2f}  {d['how']}")
print(f"\nBEST: {r['n']} trades  net ${r['total']:.2f}  (Up ${r['up']:.2f} / Down ${r['dn']:.2f})")
