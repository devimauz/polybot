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

# For each window: walk samples in time order. For each sample, compute combined cost (both asks + fees).
# Find the MIN combined cost per window (that's the best arb moment).
# Then for various thresholds, report how many windows qualify and avg profit.
per_window_best = []
for slug, samples in wins.items():
    c = clob.get(slug)
    if not c or c.get("status") != "ok": continue
    best_comb = None
    best_pnl = 0.0
    best_t = None
    for d in samples:
        info = d.get("info"); t = info.get("t") if info else None
        if t is None: continue
        pU, pD = up_ask(d), down_ask(d)
        if not (pU and pD and 0.05 <= pU <= 0.95 and 0.05 <= pD <= 0.95): continue
        comb = pU + pD + fee_ps(pU) + fee_ps(pD)
        if best_comb is None or comb < best_comb:
            # profit at this moment with STAKE split equally
            S = STAKE / (pU + pD)  # total shares per side
            fees = fee_ps(pU) * S + fee_ps(pD) * S
            pnl = S - STAKE - fees
            best_comb = comb
            best_pnl = pnl
            best_t = t
    per_window_best.append((slug, best_comb, best_pnl, best_t))

nwin = len(per_window_best)
print(f"total windows with ok winner: {nwin}")
print()
print("=== combined (both-sides) arb: best moment per window ===")
print(f"{'comb<=th':>9} {'n_win':>6} {'n/hr':>6} {'avg $/win':>10} {'total $':>10} {'$/hr':>8}")
for th in [1.05, 1.03, 1.02, 1.01, 1.00, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]:
    sel = [(c, p, t) for (slug, c, p, t) in per_window_best if c is not None and c <= th]
    if not sel: continue
    n = len(sel)
    avg = sum(p for c, p, t in sel) / n
    tot = sum(p for c, p, t in sel)
    print(f"{th:9.3f} {n:6d} {n/HR:6.2f} {avg:+10.3f} {tot:+10.1f} {tot/HR:+8.2f}")

print()
print("=== histogram of best-combined-cost per window ===")
buckets = defaultdict(int)
for slug, c, p, t in per_window_best:
    if c is None: continue
    b = round(c * 100) / 100
    buckets[b] += 1
for k in sorted(buckets):
    print(f"  {k:.2f}: {buckets[k]:4d}")

# Also: min combined cost per window distribution
print()
print("=== best combined cost: P25/P50/P75/P90/P95/P99 ===")
combs = sorted(c for slug, c, p, t in per_window_best if c is not None)
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  P{int(q*100):2d}: {combs[int(q*len(combs))]:.3f}")
