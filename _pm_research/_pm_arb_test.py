"""Stage 3: Is the complete-set 'arb' (ask_up+ask_down<1) REAL or STALE-BOOK NOISE?

The two legs are separate CLOB books with independent `book_timestamp`s. If the
sum dips below 1 only because one leg updated and the other didn't, the dip
reverts on the next tick -> stale, untradeable. If the dip PERSISTS (and
deepens) with depth on both legs, a FOK arbitrageur could take it.

Test: at the moment the sum first goes < threshold, look at the next K
snapshots. Does it stay < threshold? How much depth is really there?
"""
import os, pickle
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        out.append(d)
    return out


def main():
    data = load_all()
    print(f"windows {len(data)}\n")

    # ---- persistence of the dip ----
    for thr in (1.00, 0.98, 0.96, 0.94):
        pers1 = pers5 = pers20 = 0
        hits = 0
        widths = []
        for d in data:
            au, ad = d["Up_ask"], d["Down_ask"]
            ok = np.isfinite(au) & np.isfinite(ad)
            ssum = np.where(ok, au + ad, np.nan)
            idx = np.where(ok & (ssum < thr))[0]
            if len(idx) == 0:
                continue
            hits += 1
            i = idx[0]
            nxt = ssum[i + 1:i + 21]
            nxt = nxt[np.isfinite(nxt)]
            if len(nxt) == 0:
                continue
            if nxt[0] < thr:
                pers1 += 1
            if len(nxt) >= 5 and np.all(nxt[:5] < thr):
                pers5 += 1
            if len(nxt) >= 20 and np.all(nxt[:20] < thr):
                pers20 += 1
            # how wide is the dip in time (seconds)?
            after = ssum[i:]
            after = after[np.isfinite(after)]
            k = np.where(after >= thr)[0]
            widths.append((k[0] if len(k) else len(after)) * 0.28)
        w = np.array(widths) if widths else np.array([0.0])
        print(f"sum<{thr:.2f}: hits={hits:>3}  persists next tick={pers1:>3} "
              f"next 5 ticks(1.4s)={pers5:>3}  next 20 ticks(5.6s)={pers20:>3}  "
              f"median dip width={np.median(w):.2f}s  p90={np.percentile(w,90):.2f}s")

    # ---- depth at the dip: is it tradeable with size? ----
    print("\n--- depth at the FIRST dip below 0.98 (both legs) ---")
    print(f"{'slug':<28} {'t':>7} {'a_up':>6} {'a_dn':>6} {'sum':>6} {'$up':>8} {'$dn':>8} {'persist20':>9}")
    n_depth = 0
    rows = []
    for d in data:
        au, ad = d["Up_ask"], d["Down_ask"]
        su, sd = d["Up_asksz"], d["Down_asksz"]
        ok = np.isfinite(au) & np.isfinite(ad) & np.isfinite(su) & np.isfinite(sd)
        ssum = np.where(ok, au + ad, np.nan)
        idx = np.where(ok & (ssum < 0.98))[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        after = ssum[i:]
        after = after[np.isfinite(after)]
        k = np.where(after >= 0.98)[0]
        p20 = (len(k) == 0) or (k[0] >= 20)
        # $ of depth at top level
        du = su[i] * au[i]
        dd = sd[i] * ad[i]
        rows.append((d["_slug"], d["t"][i], au[i], ad[i], ssum[i], du, dd, p20))
        if du >= 20 and dd >= 20:
            n_depth += 1
    for r in rows[:20]:
        print(f"{r[0]:<28} {r[1]:>7.1f} {r[2]:>6.2f} {r[3]:>6.2f} {r[4]:>6.3f} {r[5]:>8.1f} {r[6]:>8.1f} {str(r[7]):>9}")
    print(f"\ntotal dip events: {len(rows)}   with >=$20 depth on BOTH legs: {n_depth}")

    # ---- what would the arb P&L be? ----
    print("\n--- arb economics (buy both legs, hold to settlement, fee 0.05*min(p,1-p)*shares) ---")
    for stake in (10, 25, 50):
        tot_pnl = 0.0
        cnt = 0
        for d in data:
            au, ad = d["Up_ask"], d["Down_ask"]
            su, sd = d["Up_asksz"], d["Down_asksz"]
            ok = np.isfinite(au) & np.isfinite(ad) & np.isfinite(su) & np.isfinite(sd)
            ssum = np.where(ok, au + ad, np.nan)
            idx = np.where(ok & (ssum < 0.98))[0]
            if len(idx) == 0:
                continue
            i = idx[0]
            # need enough depth on both legs for `stake` dollars each
            if su[i] * au[i] < stake or sd[i] * ad[i] < stake:
                continue
            # buy equal share counts: q shares each, cost = q*(au+ad)
            q = stake / au[i]
            if q * ad[i] > stake * 1.5:
                continue
            cost = q * au[i] + q * ad[i]
            fee = 0.05 * min(au[i], 1 - au[i]) * q + 0.05 * min(ad[i], 1 - ad[i]) * q
            # settlement: exactly one leg pays $1
            pnl = q * 1.0 - cost - fee
            tot_pnl += pnl
            cnt += 1
        print(f"  stake=${stake}: fills={cnt}  total P&L=${tot_pnl:+.2f}  avg=${tot_pnl/max(cnt,1):+.3f}")

    # ---- CRITICAL: does the dip coincide with the book being about to vanish? ----
    print("\n--- is the dip an artifact of the book about to go one-sided? ---")
    # compare pair-sum dips against each leg's own book state
    dips_t = []
    for d in data:
        au, ad = d["Up_ask"], d["Down_ask"]
        ok = np.isfinite(au) & np.isfinite(ad)
        ssum = np.where(ok, au + ad, np.nan)
        idx = np.where(ok & (ssum < 0.98))[0]
        if len(idx):
            i = idx[0]
            dips_t.append(d["t"][i])
    allt = np.concatenate([d["t"] for d in data])
    print(f"  dip events at t: median={np.median(dips_t):.0f}s  "
          f"frac in last 60s={np.mean(np.array(dips_t)>240):.3f}  frac in last 120s={np.mean(np.array(dips_t)>180):.3f}")
    hist, edges = np.histogram(dips_t, bins=np.arange(0, 301, 30))
    print("  dip-count by 30s bucket:", dict(zip([int(e) for e in edges[:-1]], hist.tolist())))


if __name__ == "__main__":
    main()
