"""Stage 37: FINAL VALIDATION + LOSS FORENSIC.

Remaining concerns before shipping:
  1. "first qualifying snapshot" gives the rule freedom to pick a good
     moment.  Test FIXED-T entry (T=260, 265, 270) - no freedom at all.
  2. The single losing trade must be a genuine loss (forensic check on the
     raw log), not a data bug.
  3. Power: how many windows are needed to confirm the edge at 5%?
  4. Does the edge hold for a DIFFERENT asset? (only BTC available)
"""
import os, json, pickle
import numpy as np
from scipy.stats import binom, norm

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


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]; d["ws"] = int(d["ws"][0])
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out


def cluster_t(wi, pnl):
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    c = np.array([(wi == w).sum() for w in uq])
    m = s.sum() / c.sum()
    se = s.std(ddof=1) / np.sqrt(len(uq)) if len(uq) > 1 else np.nan
    return m, (m / se if se > 0 else np.nan), len(uq)


def boot_ci(wi, pnl, n=6000, seed=11):
    rng = np.random.default_rng(seed)
    wi = np.asarray(wi); pnl = np.asarray(pnl, float)
    uq = np.unique(wi)
    tot = {w: pnl[wi == w].sum() for w in uq}
    cnt = {w: (wi == w).sum() for w in uq}
    keys = list(tot); out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        out.append(sum(tot[k] for k in pick) / sum(cnt[k] for k in pick))
    return np.percentile(out, [2.5, 97.5])


def pnl_of(ask, won):
    ask = np.asarray(ask, float); won = np.asarray(won, float)
    sh = STAKE / ask
    return sh * won - STAKE - FEE(ask, sh)


def build_rows(W):
    rows = []
    for wi, d in enumerate(W):
        t = d["t"]; n = len(t)
        for T in range(240, 300, 2):
            i = int(np.searchsorted(t, T))
            if i >= n: continue
            mu, md = d["Up_mid"][i], d["Down_mid"][i]
            if not (np.isfinite(mu) and np.isfinite(md)): continue
            p_up = 0.5 * (mu + (1 - md))
            if not (0.002 < p_up < 0.998): continue
            for side in ("Up", "Down"):
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                asz = d[side + "_asksz"][i]
                if not all(np.isfinite(v) for v in (ask, bid, asz)): continue
                if not (0 < bid <= ask < 1): continue
                if (ask - bid) > 0.04: continue
                if asz * ask < 10.0: continue
                o = d[("Down_ask" if side == "Up" else "Up_ask")][i]
                if not (np.isfinite(o) and 0.001 < o < 0.999): continue
                mkt = p_up if side == "Up" else 1.0 - p_up
                rows.append(dict(wi=wi, ws=d["ws"], T=float(T), side=side, ask=float(ask),
                                 bid=float(bid), mkt=float(mkt),
                                 won=1 if d["_winner"] == side else 0))
    return rows


def main():
    W = load_all()
    rows = build_rows(W)
    by_ws = {d["ws"]: d for d in W}

    # ---------------- 1. FIXED-T entry (no timing freedom) ----------------
    print("=== 1) FIXED-T ENTRY (no freedom in when to trade) ===")
    print("  T   band        N   win     ask     tot$    avg$     t    binom_p")
    for T in (255, 260, 265, 270, 275, 280):
        for lo, hi in ((0.80, 0.99), (0.85, 0.98)):
            s = [r for r in rows if r["T"] == T and r["mkt"] >= 0.5 and lo <= r["ask"] < hi]
            if len(s) < 8: continue
            ask = np.array([r["ask"] for r in s]); won = np.array([r["won"] for r in s])
            p = pnl_of(ask, won); m, t, w_ = cluster_t([r["wi"] for r in s], p)
            pv = float(binom.sf(int(won.sum()) - 1, len(won), ask.mean()))
            print(f"  {T:>3} [{lo:.2f},{hi:.2f}) {len(s):>4} {won.mean():.4f} {ask.mean():>6.4f} "
                  f"{p.sum():+7.2f} {p.mean():+.3f} {t:+6.2f}  {pv:.4f}")

    # ---------------- 2. LOSING TRADE FORENSIC ------------------------------
    print("\n=== 2) FORENSIC: the single losing trade (raw log) ===")
    sel = []
    best = {}
    for r in sorted([r for r in rows if r["T"] >= 260 and r["mkt"] >= 0.5 and 0.80 <= r["ask"] < 0.99],
                    key=lambda r: (r["wi"], r["T"])):
        best.setdefault(r["wi"], r)
    sel = list(best.values())
    losses = [r for r in sel if r["won"] == 0]
    for r in losses:
        d = by_ws[r["ws"]]
        print(f"  window {r['ws']}  side={r['side']} entry T={r['T']:.0f} ask={r['ask']:.4f} "
              f"mkt={r['mkt']:.4f}  RESOLUTION WINNER = {d['_winner']}")
        t = d["t"]
        i0 = int(np.searchsorted(t, r["T"]))
        print("   t     Up_bid Up_ask | Down_bid Down_ask   (raw cached book)")
        for i in range(i0, len(t), max(1, (len(t) - i0) // 8)):
            print(f"  {t[i]:6.1f}  {d['Up_bid'][i]:7.3f} {d['Up_ask'][i]:7.3f} | "
                  f"{d['Down_bid'][i]:8.3f} {d['Down_ask'][i]:8.3f}")
        # also confirm the resolution event in the raw file
        p = os.path.join(LOGDIR, d["_slug"] + ".jsonl")
        for line in open(p, encoding="utf-8", errors="replace"):
            if '"resolution"' in line:
                j = json.loads(line)
                print(f"   RAW resolution event: winner={j.get('winner')} at {j.get('ts')}")
                break

    # ---------------- 3. POWER ----------------------------------------------
    print("\n=== 3) POWER ANALYSIS ===")
    ask = np.array([r["ask"] for r in sel]); won = np.array([r["won"] for r in sel])
    p_hat = won.mean(); p0 = ask.mean()
    print(f"  observed win rate {p_hat:.4f} vs market-implied {p0:.4f} "
          f"(break-even {((STAKE+FEE(p0,STAKE/p0))/(STAKE/p0)):.4f})")
    for target in (0.95, 0.97, 0.99):
        # n needed to reject H0: p = p0 with 80% power, one-sided alpha=0.05
        se = np.sqrt(p0 * (1 - p0) / 200)
        n_needed = ((norm.ppf(0.95) + norm.ppf(0.80)) * np.sqrt(p0 * (1 - p0)) /
                    (target - p0)) ** 2
        print(f"  to detect a true win rate of {target:.2f} vs {p0:.4f}: "
              f"need ~{n_needed:.0f} independent windows "
              f"(have {len(sel)}; at 269 windows/22.5h that is "
              f"{n_needed/269*22.5/24:.1f} days of logging)")
    print(f"  current evidence: {int(won.sum())}/{len(won)} wins, "
          f"binom_p={float(binom.sf(int(won.sum())-1, len(won), p0)):.4f}")

    # ---------------- 4. what the bot would have done over the full period ---
    print("\n=== 4) FULL-PERIOD YIELD (269 windows = 22.5h) ===")
    p = pnl_of(ask, won)
    m, t, w_ = cluster_t([r["wi"] for r in sel], p)
    lo, hi = boot_ci([r["wi"] for r in sel], p)
    print(f"  fires: {len(sel)} times in 22.5h = 1 per {22.5*3600/len(sel)/60:.1f} min")
    print(f"  total net ${p.sum():+.2f}  ROI on stake {p.sum()/(STAKE*len(sel))*100:+.2f}%  "
          f"cluster-t {t:+.2f}  CI95 [{lo:+.2f},{hi:+.2f}]")
    print(f"  worst trade ${p.min():+.2f}   best ${p.max():+.2f}")
    print(f"  Kelly fraction on the observed win rate: "
          f"{(p_hat*(1-p0)-(1-p_hat)*p0)/((1-p0)*p0+ (p_hat*p0 + (1-p_hat)*(1-p0))*0):.4f} (informational)")


if __name__ == "__main__":
    main()
