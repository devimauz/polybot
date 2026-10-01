#!/usr/bin/env python
"""Grid-search LOOSER gates: trade MORE windows, keep P&L >= 0.
Outcomes for every recorded window are derived from the official
group_item_threshold (resolved/window_outcomes jsons) vs the final
recorded TWAP estimate (e_final at max t), with fallbacks to recorded
settlements in trades.jsonl / resolved jsons.
"""
import json, math
from collections import defaultdict
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent
T = ROOT / "price_logs" / "twap"
FEE_R = 0.05
MIN_T, MAX_T = 90, 295
SKIP_LO, SKIP_HI = 100, 110
TP_BID, STOP_BID = 0.95, 0.70
KELLY_FRAC, BANKROLL = 0.25, 1000.0
MIN_STAKE, MAX_STAKE = 10.0, 100.0

# ---------- outcomes ----------
thr = {}          # slug -> threshold (group_item_threshold)
try:
    rw = json.load(open(ROOT / "resolved_windows.json", encoding="utf-8"))
    items = rw if isinstance(rw, list) else list(rw.values())
    for x in items:
        if isinstance(x, dict) and x.get("slug") and x.get("thr"):
            thr[x["slug"]] = float(x["thr"])
except Exception as e:
    print("resolved err", e)
try:
    wo = json.load(open(ROOT / "window_outcomes.json", encoding="utf-8"))
    items = wo if isinstance(wo, list) else list(wo.values())
    for x in items:
        if not isinstance(x, dict): continue
        slug = x.get("slug")
        if not slug:
            # asset/start form: start epoch = int of ISO
            a, st = x.get("asset"), x.get("start")
            if a and st and x.get("thr"):
                import re
                m = re.search(r"(\d{10})", st)
                if m: slug = f"{a.lower()}-updown-5m-{m.group(1)}"
        if slug and x.get("thr"):
            thr[slug] = float(x["thr"])
except Exception as e:
    print("outcomes err", e)

winner = {}       # slug -> "Up"/"Down"
for l in (T / "trades.jsonl").open(encoding="utf-8"):
    d = json.loads(l)
    if d.get("event") == "settle":
        if d.get("winner") is not None:
            winner[d["slug"]] = d["winner"]
        elif "won" in d:
            winner[d["slug"]] = d["side"] if d["won"] else ("Down" if d["side"] == "Up" else "Up")
try:
    rw = json.load(open(ROOT / "resolved_windows.json", encoding="utf-8"))
    items = rw if isinstance(rw, list) else list(rw.values())
    for x in items:
        if isinstance(x, dict) and x.get("slug") and x.get("winner"):
            winner[x["slug"]] = x["winner"]
except Exception:
    pass

rows = [json.loads(l) for l in (T / "windows.jsonl").open(encoding="utf-8")]
by_slug = defaultdict(list)
for r in rows:
    info = r.get("info") or {}
    by_slug[r["window"]].append({
        "ts": r["t"], "t": info.get("t"), "z": info.get("z"), "p_up": r.get("p_up_model"),
        "e_final": info.get("e_final"),
        "ask_up": r.get("ask_Up"), "ask_dn": r.get("ask_Down"),
        "bid_up": r.get("bid_Up"), "bid_dn": r.get("bid_Down")})
for s in by_slug: by_slug[s].sort(key=lambda x: x["ts"])

def outcome_for(slug):
    if slug in winner and winner[slug]:
        return winner[slug]
    if slug in thr:
        # final recorded e_final (or spot) at max t
        best = None
        for s in by_slug[slug]:
            if s["t"] is None: continue
            if best is None or s["t"] > best[0]:
                best = (s["t"], s["e_final"] or s.get("spot"))
        if best and best[1] is not None:
            return "Up" if best[1] >= thr[slug] else "Down"
    return None

# ---------- fees / sizing ----------
def fee_for(p, sh): return FEE_R * min(p, 1 - p) * sh
def ev_frac(mp, pr): return mp * ((1 - pr) / pr) - (1 - mp) - FEE_R * min(pr, 1 - pr) / pr
def kelly_frac(mp, pr): return max((mp - pr) / (1 - pr), 0.0)
def stake_for(mp, pr): return min(max(KELLY_FRAC * BANKROLL * kelly_frac(mp, pr), MIN_STAKE), MAX_STAKE)

def simulate(conf, ask_lo, ask_hi, gap_min, up_mult, z_spike, z_decay, st_frac, st_abs, edge, require_stall):
    total = 0.0; n = 0; wins = 0; up_pnl = 0.0; dn_pnl = 0.0
    how = defaultdict(lambda: [0, 0.0]); worst = 0.0
    for slug, seq in by_slug.items():
        oc = outcome_for(slug)
        if oc is None: continue
        won = (oc == "Up")
        traded = set(); side_count = defaultdict(int)
        z_lo = z_hi = peak_up = peak_dn = None
        i = 0
        while i < len(seq):
            s = seq[i]
            t = s.get("t")
            tradeable = s["p_up"] is not None and t is not None \
                and MIN_T <= t <= MAX_T and not (SKIP_LO <= t < SKIP_HI)
            if s["ask_up"] is not None:
                peak_up = s["ask_up"] if peak_up is None else max(peak_up, s["ask_up"])
            if s["ask_dn"] is not None:
                peak_dn = s["ask_dn"] if peak_dn is None else max(peak_dn, s["ask_dn"])
            if s["z"] is not None:
                z_lo = s["z"] if z_lo is None else min(z_lo, s["z"])
                z_hi = s["z"] if z_hi is None else max(z_hi, s["z"])
            if not (tradeable):
                i += 1; continue
            entered = False
            z = s["z"]
            for side in ("Up", "Down"):
                if side in traded or side_count[side] >= 1: continue
                ask = s["ask_up"] if side == "Up" else s["ask_dn"]
                if ask is None or not (ask_lo <= ask <= ask_hi): continue
                mp = s["p_up"] if side == "Up" else 1 - s["p_up"]
                if mp is None or mp < conf: continue
                eff = gap_min * (up_mult if side == "Up" else 1.0)
                if (mp - ask) < eff: continue
                if ev_frac(mp, ask) <= edge: continue
                if require_stall:
                    if z is None: continue
                    spiked = z_lo is not None and (abs(z_hi) >= z_spike or abs(z_lo) >= z_spike)
                    decayed = (z_hi - z >= z_decay) if z > 0 else (z - z_lo >= z_decay)
                    pk = peak_up if side == "Up" else peak_dn
                    dd = pk is not None and ask <= pk - st_frac * pk and (pk - ask) >= st_abs
                    dir_ok = (z > 0 and side == "Up") or (z < 0 and side == "Down")
                    if not (spiked and decayed and dd and dir_ok): continue
                stake = stake_for(mp, ask); shares = stake / ask
                cost = stake + fee_for(ask, shares)
                traded.add(side); side_count[side] += 1; entered = True
                pnl = None; howx = None
                for s2 in seq[i + 1:]:
                    b2 = s2["bid_up"] if side == "Up" else s2["bid_dn"]
                    if b2 is None: continue
                    if b2 >= TP_BID:
                        pnl = shares * b2 - fee_for(b2, shares) - cost; howx = "tp"; break
                    if b2 <= STOP_BID:
                        pnl = shares * b2 - fee_for(b2, shares) - cost; howx = "stop"; break
                if pnl is None:
                    pnl = (shares if won else 0.0) - cost; howx = "settle"
                total += pnl; n += 1; wins += pnl > 0
                worst = min(worst, pnl)
                if side == "Up": up_pnl += pnl
                else: dn_pnl += pnl
                how[howx][0] += 1; how[howx][1] += pnl
                break
            if entered: break
            i += 1
    return dict(total=total, n=n, wins=wins, up=up_pnl, dn=dn_pnl, how=dict(how), worst=worst)

# ---------- how many windows can we even score? ----------
scored = sum(1 for s in by_slug if outcome_for(s) is not None)
print(f"windows with known outcome: {scored} / {len(by_slug)}")

print()
print("== A) keep stall, loosen everything else ==")
res = []
for conf, gm, um, zs, zd, sf, sa, edge in product(
        [0.80, 0.75, 0.70], [0.06, 0.08, 0.10], [1.0, 1.25, 1.5],
        [0.5, 0.3], [0.2, 0.4], [0.10, 0.15], [0.02], [0.02]):
    r = simulate(conf, 0.60, 0.95, gm, um, zs, zd, sf, sa, edge, True)
    res.append((r, conf, gm, um, zs, zd, sf, sa))
res.sort(key=lambda x: (x[0]["total"] > 0, x[0]["n"]), reverse=True)
print(f"{'net$':>8} {'n':>4} {'wr':>5} {'worst':>7}  conf gap upm zsp zdc sfrac sabs  exits")
for r, conf, gm, um, zs, zd, sf, sa in res[:25]:
    wr = 100 * r["wins"] / r["n"] if r["n"] else 0
    h = " ".join(f"{k}{v[0]}/${v[1]:.0f}" for k, v in r["how"].items())
    print(f"{r['total']:8.1f} {r['n']:4d} {wr:4.0f}% {r['worst']:7.1f}  {conf:.2f} {gm:.2f} {um:.2f} {zs:.1f} {zd:.2f} {sf:.2f} {sa:.2f}  {h}")

print()
print("== B) NO stall requirement (conf/gap/ev only) ==")
res2 = []
for conf, gm, um, edge in product([0.80, 0.85], [0.04, 0.06, 0.08, 0.10], [1.0, 1.25, 1.5], [0.02]):
    r = simulate(conf, 0.60, 0.95, gm, um, 0.5, 0.4, 0.15, 0.02, edge, False)
    res2.append((r, conf, gm, um, edge))
res2.sort(key=lambda x: (x[0]["total"] > 0, x[0]["n"]), reverse=True)
for r, conf, gm, um, edge in res2[:15]:
    wr = 100 * r["wins"] / r["n"] if r["n"] else 0
    h = " ".join(f"{k}{v[0]}/${v[1]:.0f}" for k, v in r["how"].items())
    print(f"{r['total']:8.1f} {r['n']:4d} {wr:4.0f}% {r['worst']:7.1f}  conf={conf:.2f} gap>={gm:.2f} up x{um:.2f} ev>{edge}  {h}")

# per-night trade rate for the current live config
print()
cur = simulate(0.80, 0.60, 0.95, 0.12, 1.5, 0.5, 0.4, 0.15, 0.02, 0.02, True)
print(f"current live config: {cur['n']} trades over scored windows, net ${cur['total']:.1f}")
