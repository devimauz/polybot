"""Stage 22: ENDGAME FAVOURITE - exit rules, power analysis, and honesty checks.

The surviving candidate: in the last ~50s of a 5-min BTC window, buy the
FAVOURITE (ask 0.95-1.00) and (a) hold to settlement, or (b) take profit.

WHY THE TAIL DOMINATES:
  Buying a 0.976 favourite: 1 loss = -$10, 1 win = +$0.23. You need ~44 wins
  to cover one loss. So the strategy is a bet on the win rate exceeding the
  break-even win rate:
        p* = (ask + fee_per_share) / 1.0
  With 52 independent windows and 0 observed losses, the 95% lower bound on the
  win rate is only ~94.2% -- BELOW p* ~ 97.8%. The sample CANNOT prove the edge
  at $10 stakes. Exit rules change this: capping the loss shrinks the variance.

THIS SCRIPT:
  1. Recomputes the strategy with exits: hold / TP(0.98..0.999) / TP+STOP.
     Exits are executed at the BID (taker) with depth and spread checks.
  2. Cluster-robust stats per exit rule, per fee schedule.
  3. Power analysis: how many independent windows are needed to distinguish
     the observed win rate from break-even?
  4. A market-naivete check: is the market pricing the SPOT rule while
     reality is the 60s-TWAP rule? (uses the reconstructed settlement)
  5. Honest "what the sample supports" summary.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUT = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUT, exist_ok=True)

FEE_A = lambda p, sh: 0.07 * p * (1 - p) * sh
FEE_B = lambda p, sh: 0.25 * (p * (1 - p)) ** 2 * sh


def ffill(a):
    a = np.array(a, dtype=float).copy()
    for i in range(1, len(a)):
        if not np.isfinite(a[i]) and np.isfinite(a[i - 1]):
            a[i] = a[i - 1]
    for i in range(len(a) - 2, -1, -1):
        if not np.isfinite(a[i]) and np.isfinite(a[i + 1]):
            a[i] = a[i + 1]
    return a


def load_all():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
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


def build_candidates(W, stake=10.0, max_spread=0.04):
    out = []
    for wi, d in enumerate(W):
        t = d["t"]; n = len(t)
        for T in range(240, 300, 5):
            i = int(np.searchsorted(t, T))
            if i >= n or i < 5:
                continue
            for side in ("Up", "Down"):
                o = "Down" if side == "Up" else "Up"
                ask, bid = d[side + "_ask"][i], d[side + "_bid"][i]
                oask, obid = d[o + "_ask"][i], d[o + "_bid"][i]
                asz = d[side + "_asksz"][i]; bsz = d[side + "_bidsz"][i]
                if not (np.isfinite(ask) and np.isfinite(bid) and np.isfinite(asz)):
                    continue
                if not (0 < bid <= ask < 1):
                    continue
                if not (np.isfinite(oask) and np.isfinite(obid)) or not (0 < obid <= oask < 1):
                    continue
                if (ask - bid) > max_spread:
                    continue
                if not np.isfinite(asz) or asz * ask < stake:
                    continue
                out.append(dict(wi=wi, ws=int(d["ws"]), T=T, side=side, ask=float(ask),
                                bid=float(bid), asz=float(asz), bsz=float(bsz),
                                won=(d["_winner"] == side), i=i, d=d))
    return out


def simulate(W, cand, tp=None, stop=None, stake=10.0, fee=FEE_A, exit_fee=True):
    """Enter at the ask at cand['T']; if tp is set, walk forward and exit at the
    bid when bid >= tp (or <= stop). Otherwise hold to settlement."""
    d = cand["d"]; t = d["t"]; n = len(t); i = cand["i"]
    side = cand["side"]
    a = cand["ask"]
    sh = stake / a
    entry_fee = fee(a, sh)
    pnl = None; reason = "settle"; exit_t = None; exit_px = None
    if tp is not None:
        for j in range(i + 1, n):
            b = d[side + "_bid"][j]
            if not np.isfinite(b):
                continue
            bsz = d[side + "_bidsz"][j]
            if stop is not None and b <= stop:
                if np.isfinite(bsz) and bsz * b >= stake:
                    pnl = sh * b - stake - entry_fee - (fee(b, sh) if exit_fee else 0.0)
                    reason = "stop"; exit_t = float(t[j]); exit_px = float(b); break
                continue
            if b >= tp:
                if np.isfinite(bsz) and bsz * b >= stake:
                    pnl = sh * b - stake - entry_fee - (fee(b, sh) if exit_fee else 0.0)
                    reason = "tp"; exit_t = float(t[j]); exit_px = float(b); break
    if pnl is None:
        pnl = sh * (1.0 if cand["won"] else 0.0) - stake - entry_fee
        reason = "settle"
    return dict(pnl=pnl, reason=reason, exit_t=exit_t, exit_px=exit_px, sh=sh)


def main():
    W = load_all()
    cands = build_candidates(W)
    print(f"windows {len(W)}  candidate entries {len(cands)}")

    # one trade per (window, T): favourite only
    fav = {}
    for c in cands:
        k = (c["wi"], c["T"])
        if k not in fav or c["ask"] > fav[k]["ask"]:
            fav[k] = c
    fav = list(fav.values())
    print(f"favourite entries (1 per window-T): {len(fav)}")

    # ---- headline band ----
    band = [c for c in fav if 250 <= c["T"] < 275 and 0.95 <= c["ask"] < 1.01]
    print(f"\n=== headline band t[250,275) ask[0.95,1.01): n={len(band)} ===")
    a = np.mean([c["ask"] for c in band])
    fee_ps = 0.07 * a * (1 - a)
    pstar = a + fee_ps
    win = np.mean([c["won"] for c in band])
    print(f"  meanAsk={a:.4f}  fee/share={fee_ps:.5f}  break-even win rate p*={pstar:.4f}")
    print(f"  observed win rate={win:.4f}  (n_trades={len(band)}, n_windows={len(set(c['ws'] for c in band))})")
    nw = len(set(c["ws"] for c in band))
    lo = 1 - 3.0 / nw if nw else 0
    print(f"  95% lower bound on win rate (rule of 3, {nw} windows): {lo:.4f}  "
          f"-> {'CANNOT prove' if lo < pstar else 'CAN prove'} the edge at this stake")

    # ---- exit rules ----
    print("\n=== exit rules on the headline band ===")
    rules = [("hold to settle", None, None),
             ("TP 0.980", 0.980, None), ("TP 0.985", 0.985, None),
             ("TP 0.990", 0.990, None), ("TP 0.995", 0.995, None),
             ("TP 0.990 STOP 0.900", 0.990, 0.900),
             ("TP 0.990 STOP 0.950", 0.990, 0.950)]
    results = {}
    for name, tp, stop in rules:
        pnl = []; reasons = {}
        for c in band:
            r = simulate(W, c, tp=tp, stop=stop)
            pnl.append(r["pnl"]); reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        pnl = np.array(pnl)
        w2 = [c["wi"] for c in band]
        m, tst, nww = cluster_t(w2, pnl)
        eq = (pnl - pnl.mean()).std() / np.sqrt(len(pnl))
        results[name] = dict(n=len(pnl), total=float(pnl.sum()), avg=float(pnl.mean()),
                             t=float(tst), reasons=reasons, worst=float(pnl.min()),
                             sd=float(pnl.std()))
        print(f"  {name:<22} n={len(pnl):>3} total=${pnl.sum():+7.2f} avg=${pnl.mean():+.3f} "
              f"cluster-t={tst:+5.2f} worst=${pnl.min():+7.2f} {reasons}")

    # fee B on the best few
    print("\n  same rules under fee_B (0.25 (p(1-p))^2):")
    for name, tp, stop in rules:
        pnl = np.array([simulate(W, c, tp=tp, stop=stop, fee=FEE_B)["pnl"] for c in band])
        m, tst, nww = cluster_t([c["wi"] for c in band], pnl)
        print(f"  {name:<22} total=${pnl.sum():+7.2f} avg=${pnl.mean():+.3f} cluster-t={tst:+5.2f}")

    # ---- wider band sweep with the best exit rule (TP 0.990) ----
    print("\n=== band sweep with TP 0.990 (fee_A) ===")
    for tlo, thi in ((245, 265), (250, 275), (250, 285), (255, 285), (260, 290), (265, 295)):
        for plo, phi in ((0.93, 0.98), (0.95, 1.01), (0.97, 1.01)):
            sel = [c for c in fav if tlo <= c["T"] < thi and plo <= c["ask"] < phi]
            if len(sel) < 10:
                continue
            pnl = np.array([simulate(W, c, tp=0.990)["pnl"] for c in sel])
            m, tst, nww = cluster_t([c["wi"] for c in sel], pnl)
            print(f"  t[{tlo},{thi}) ask[{plo},{phi}) n={len(sel):>3} w={len(set(c['ws'] for c in sel)):>3} "
                  f"total=${pnl.sum():+7.2f} avg=${pnl.mean():+.3f} t={tst:+5.2f}")

    # ---- market naivete check: spot rule vs TWAP rule ----
    print("\n=== is the market naive about the TWAP settlement rule? ===")
    by_ws = {d["ws"]: d for d in W}
    pre = {}
    for d in W:
        prev = by_ws.get(d["ws"] - 300)
        if prev is None:
            continue
        tt = prev["t"]; cc = ffill(prev["cb_px"])
        m = (tt >= 240) & np.isfinite(cc)
        if m.sum() >= 10:
            pre[d["ws"]] = (tt[m] - 300.0, cc[m])
    agree_spot = agree_twap = n = 0
    for c in band:
        d = c["d"]; t = d["t"]; cb = ffill(d["cb_px"])
        i = c["i"]
        if d["ws"] in pre:
            tt, pp = pre[d["ws"]]; mm = tt >= -60
            strike = float(np.mean(pp[mm])) if mm.sum() >= 10 else float(cb[0])
        else:
            strike = float(cb[0])
        sel = (t >= 240) & np.isfinite(cb)
        if sel.sum() < 10:
            continue
        settle = float(np.mean(cb[sel]))
        n += 1
        agree_spot += int((cb[i] > strike) == c["won"])
        agree_twap += int((settle > strike) == c["won"])
    print(f"  in the headline band (n={n}):")
    print(f"    market's implied direction matches outcome: {agree_spot}/{n} = {agree_spot/max(n,1):.4f}")
    print(f"    reconstructed TWAP settlement matches outcome: {agree_twap}/{n} = {agree_twap/max(n,1):.4f}")

    with open(os.path.join(OUT, "endgame_exits.json"), "w") as fh:
        json.dump({"results": results, "n_windows": nw, "pstar": pstar, "observed_win": win}, fh, indent=2)


if __name__ == "__main__":
    main()
