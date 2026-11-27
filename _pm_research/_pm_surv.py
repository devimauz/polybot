"""Stage 31: SURVIVORSHIP AUDIT + live-logic simulation.

TWO CONCERNS WITH THE EDGE FOUND SO FAR:
 (1) SURVIVORSHIP: near expiry one side of the book dies (ask -> 0.01,
     bid -> NaN).  If the logger only emits snapshots while a side is alive,
     windows where the outcome was decided EARLY would be missing from the
     late-window sample -> the measured win rate would be inflated.
 (2) LOGIC: the backtest enters at the first qualifying snapshot; the live
     bot needs the quote SUSTAINED for N seconds and must not buy a
     near-certain 0.99 when the loser side is already dead.
"""
import os, json, glob, pickle
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
LOGDIR = r"C:/Users/user/Desktop/bot/price_logs/polymarket"
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh
STAKE = 10.0


def ffill(a):
    a = np.array(a, dtype=float).copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]): last = a[i]
        elif np.isfinite(last): a[i] = last
    nxt = np.nan
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]): nxt = a[i]
        elif np.isfinite(nxt): a[i] = nxt
    return a


def cluster_t(wi, pnl):
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se > 0 else np.nan), len(uq)


# ------------------------------------------------------------------ raw dump
def raw_dump(slug):
    p = os.path.join(LOGDIR, slug + ".jsonl")
    out = []
    for line in open(p, encoding="utf-8", errors="replace"):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("event") != "snapshot": continue
        t = r.get("seconds_into_window")
        if t is None or t < 250: continue
        u, d = r.get("Up") or {}, r.get("Down") or {}
        out.append((t, u.get("bid"), u.get("ask"), u.get("mid"),
                    d.get("bid"), d.get("ask"), d.get("mid")))
    return out


def main():
    print("=== A) RAW SNAPSHOT DUMP near expiry (3 windows) ===")
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    slugs = [m["_slug"] for m in man if m.get("_winner")][:3]
    for s in slugs:
        rows = raw_dump(s)
        print(f"\n-- {s}  ({len(rows)} snapshots with t>=250) --")
        print("      t     Up_bid Up_ask Up_mid | Down_bid Down_ask Down_mid")
        for r in rows[:4] + rows[len(rows)//2:len(rows)//2+2] + rows[-4:]:
            print("  " + " ".join(f"{(v if v is not None else float('nan')):7.3f}" for v in r))

    # ------------------------------------------------- coverage / survivorship
    print("\n=== B) COVERAGE: how many windows have usable data at T>=260? ===")
    nfiles = 0; nlate = 0; nlate_both = 0; nsnap_total = 0
    per_window_late = {}
    for m in man:
        p = os.path.join(LOGDIR, m["_slug"] + ".jsonl")
        if not os.path.exists(p): continue
        nfiles += 1
        n_late = 0; n_both = 0
        for line in open(p, encoding="utf-8", errors="replace"):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("event") != "snapshot": continue
            nsnap_total += 1
            t = r.get("seconds_into_window")
            if t is None or t < 260: continue
            n_late += 1
            u, d = r.get("Up") or {}, r.get("Down") or {}
            if u.get("ask") is not None and d.get("ask") is not None:
                n_both += 1
        per_window_late[m["_slug"]] = (n_late, n_both)
        if n_late: nlate += 1
        if n_both: nlate_both += 1
    print(f"  files scanned              : {nfiles}")
    print(f"  total snapshots            : {nsnap_total}")
    print(f"  windows with >=1 snap T>=260: {nlate}  ({nlate/max(nfiles,1)*100:.1f}%)")
    print(f"  windows with BOTH sides at T>=260: {nlate_both} ({nlate_both/max(nfiles,1)*100:.1f}%)")
    print("  -> if a window has NO late snapshots, it is EXCLUDED from the sample:")
    print("     check whether excluded windows are systematically 'decided early'.")

    # ---------------------------------- are excluded windows decided early?
    print("\n=== C) SURVIVORSHIP: outcome of windows WITHOUT late data ===")
    cached = {}
    for m in man:
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        cached[m["_slug"]] = dict(
            t=z["t"], upask=z["Up_ask"], dnask=z["Down_ask"],
            upmid=z["Up_mid"], dnmid=z["Down_mid"], w=m.get("_winner"))
    inc, exc = [], []
    for m in man:
        s = m["_slug"]
        if s not in cached: continue
        c = cached[s]
        nl, nb = per_window_late.get(s, (0, 0))
        # decided early: one side's ask pinned at <=0.02 while the other >= 0.98
        ua, da = c["upask"], c["dnask"]
        decided = False
        for i in range(len(c["t"])):
            if 200 <= c["t"][i] < 260:
                if (np.isfinite(ua[i]) and ua[i] <= 0.02 and np.isfinite(da[i]) and da[i] >= 0.98):
                    decided = True; break
                if (np.isfinite(da[i]) and da[i] <= 0.02 and np.isfinite(ua[i]) and ua[i] >= 0.98):
                    decided = True; break
        (inc if nb > 0 else exc).append(decided)
    if inc:
        print(f"  INCLUDED windows: n={len(inc)}  decided-early before T=260: {sum(inc)} "
              f"({sum(inc)/len(inc)*100:.1f}%)")
    if exc:
        print(f"  EXCLUDED windows: n={len(exc)}  decided-early before T=260: {sum(exc)} "
              f"({sum(exc)/len(exc)*100:.1f}%)")
        print("  (if excluded windows are LESS decided-early, the sample is biased EASY)")


if __name__ == "__main__":
    main()
