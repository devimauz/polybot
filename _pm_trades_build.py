"""_pm_trades_build.py - build a compact local cache of Polymarket trade prints.

Reads price_logs/trades/<slug>.jsonl.gz (fetched via the AWS box from
data-api.polymarket.com), writes _pm_cache/trades.npz keyed by slug with arrays
[ts, px, sz, taker_is_sell(1/0), outcome_is_down(1/0)] and prints sanity stats
against the book snapshots (print price vs displayed bid/ask).
"""
import glob, gzip, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
TD = os.path.join(ROOT, "price_logs", "trades")
OUT = os.path.join(ROOT, "_pm_cache", "trades.npz")

import btc5m_endgame_backtest as B
W = B.load_all()
by_slug = {d["_slug"]: d for d in W}

data = {}
n_files = 0
mis = []
for fp in sorted(glob.glob(os.path.join(TD, "*.jsonl.gz"))):
    slug = os.path.basename(fp)[:-len(".jsonl.gz")]
    d = by_slug.get(slug)
    if d is None:
        continue
    ts, px, sz, sell, dn = [], [], [], [], []
    with gzip.open(fp, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts.append(r["ts"]); px.append(r["p"]); sz.append(r["sz"])
            sell.append(1 if r["side"] == "SELL" else 0)
            dn.append(1 if r["oc"] == "Down" else 0)
    if not ts:
        continue
    a = np.array([ts, px, sz, sell, dn], float)
    data[slug] = a
    n_files += 1
    # sanity: each print price should lie within [bid - 3c, ask + 3c] of the
    # nearest book frame for the SAME outcome token
    t_frame = np.searchsorted(d["t"], a[0] - d["ws"])
    tf = np.clip(t_frame, 0, len(d["t"]) - 1)
    up = a[4] == 0
    bid = np.where(up, d["Up_bid"][tf], d["Down_bid"][tf])
    ask = np.where(up, d["Up_ask"][tf], d["Down_ask"][tf])
    bad = (a[1] < bid - 0.03) | (a[1] > ask + 0.03)
    mis.append(float(np.mean(bad)))

if data:
    keys = list(data)
    np.savez_compressed(
        OUT,
        slugs=np.array(keys),
        starts=np.cumsum([0] + [data[k].shape[1] for k in keys[:-1]]),
        lens=np.array([data[k].shape[1] for k in keys]),
        flat=np.concatenate([data[k] for k in keys], axis=1),
    )

print(f"windows with prints: {n_files}")
if mis:
    print(f"mean fraction of prints outside displayed touch +-3c: {np.mean(mis):.4f}")
    print(f"worst window: {np.max(mis):.3f}")
tot = sum(int(v.shape[1]) for v in data.values())
print(f"total prints: {tot}  mean/window: {tot / max(1, n_files):.0f}")
# sell-side share by outcome
if data:
    f = np.concatenate([data[k] for k in data], axis=1)
    for o in (0, 1):
        m = f[4] == o
        name = "Down" if o else "Up"
        print(f"outcome {name}: prints={int(m.sum())} "
              f"taker-SELL share={f[3][m].mean():.3f}")
