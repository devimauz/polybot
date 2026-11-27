#!/usr/bin/env python3
"""_pm_hunt_printflow.py — does TRADE-PRINT FLOW imbalance predict settlement?

The trade prints (trades.npz, 1.28M real taker prints) have only been used for
MAKER-FILL simulation. Open question: is the IMBALANCE of taker flow (aggressive
buy vs sell share, or signed volume) in the first part of a 5m window predictive
of the settlement — i.e., is there information in the tape that the mid-price
hasn't absorbed? This is distinct from tape-FOLLOW (buy the side the tape just
moved to, falsified) and from book-trend (falsified): here we ask whether
*net taker aggression* forecasts the FINAL winner beyond the current price.

DESIGN slice only; freeze by plateau + both-halves; one holdout shot.
"""
import os, pickle, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def load_trades():
    tp = os.path.join(CACHE, "trades.npz")
    z = np.load(tp, allow_pickle=True)
    slugs = list(z["slugs"]); starts = z["starts"]; lens = z["lens"]; flat = z["flat"]
    TR = {}
    for k, s0, ln in zip(slugs, starts, lens):
        a = flat[:, s0:s0+ln]
        o = np.argsort(a[0], kind="stable")
        TR[str(k)] = a[:, o]
    return TR

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def flow_features(TR, d, T):
    """Signed taker flow in [0, T] of the window.
    flat rows: [ts_epoch, price, size, taker_is_SELL(1/0), outcome_is_DOWN(1/0)]
    """
    a = TR.get(d["_slug"])
    if a is None: return None
    t = a[0]; side = a[3]; oc = a[4]; sz = a[2]
    m = (t >= d["ws"]) & (t < d["ws"] + T)
    if not m.any(): return None
    t = t[m]; side = side[m]; oc = oc[m]; sz = sz[m]
    # Up-flow: buying Up or selling Down is bullish Up
    up_buy = sz[(oc == 0) & (side == 0)].sum()
    up_sell = sz[(oc == 0) & (side == 1)].sum()
    dn_buy = sz[(oc == 1) & (side == 0)].sum()
    dn_sell = sz[(oc == 1) & (side == 1)].sum()
    signed_up = up_buy - up_sell - dn_buy + dn_sell
    total = up_buy + up_sell + dn_buy + dn_sell
    if total <= 0: return None
    imbalance = signed_up / total  # in [-1, 1]
    n_prints = len(t)
    return dict(imbalance=imbalance, signed_up=signed_up, total=total,
                n_prints=n_prints)

def frame_at(d, T):
    i = int(np.searchsorted(d["t"], T, side="right")) - 1
    return i if i >= 0 else None

def collect(W, TR, T):
    rows = []
    for d in W:
        i = frame_at(d, T)
        if i is None: continue
        f = flow_features(TR, d, T)
        if f is None or f["n_prints"] < 20: continue
        for side in ("Up", "Down"):
            ask = d[f"{side}_ask"][i]; asz = d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (0.10 <= ask <= 0.90): continue
            sh = int(STAKE/ask)
            if sh < 5 or not np.isfinite(asz) or asz < sh: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            imb = f["imbalance"] if side == "Up" else -f["imbalance"]
            rows.append(dict(ws=d["ws"], side=side, ask=ask, won=won, pnl=pnl,
                             imbalance=imb, n_prints=f["n_prints"], total=f["total"]))
    return rows

def stats(rows):
    if not rows: return None
    pnl=np.array([r["pnl"] for r in rows])
    return dict(n=len(rows), wr=np.mean([r["won"] for r in rows]),
                ask=np.mean([r["ask"] for r in rows]), net=pnl.sum(),
                mean=pnl.mean(), be=np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows]))

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} be={st['be']:.3f} "
          f"net=${st['net']:+.0f} mean=${st['mean']:+.3f}")

def main():
    W = load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")
    TR = load_trades()
    print(f"trades loaded: {len(TR)} slugs")

    T = 120
    d_rows = collect(design, TR, T)
    print(f"\n=== DESIGN T={T}: {len(d_rows)} tradable rows ===")
    show("baseline all", stats(d_rows))

    # Does signed flow imbalance separate winners within a price band?
    print("\n-- flow imbalance terciles within ask[0.30,0.50) (DESIGN) --")
    band = [r for r in d_rows if 0.30 <= r["ask"] < 0.50]
    if len(band) >= 30:
        vals=np.array([r["imbalance"] for r in band])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low(against)",[r for r in band if r["imbalance"]<=q1]),
                        ("mid",[r for r in band if q1<r["imbalance"]<=q2]),
                        ("high(with)",[r for r in band if r["imbalance"]>q2])]:
            show(f"imb {lab}", stats(sub))
        print(f"  (cuts {q1:+.3f}, {q2:+.3f})")

    print("\n-- flow imbalance terciles across ALL asks (DESIGN) --")
    if len(d_rows) >= 60:
        vals=np.array([r["imbalance"] for r in d_rows])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low(against)",[r for r in d_rows if r["imbalance"]<=q1]),
                        ("mid",[r for r in d_rows if q1<r["imbalance"]<=q2]),
                        ("high(with)",[r for r in d_rows if r["imbalance"]>q2])]:
            show(f"imb {lab}", stats(sub))
        print(f"  (cuts {q1:+.3f}, {q2:+.3f})")

    # Strong flow only: |imb| > 0.3 — does strong directional pressure predict?
    print("\n-- strong flow |imb|>0.3 (DESIGN) --")
    strong = [r for r in d_rows if abs(r["imbalance"]) > 0.3]
    show("strong-with (buy side flow agrees)", stats([r for r in strong if r["imbalance"]>0.3]))
    show("strong-against (fade the flow)", stats([r for r in strong if r["imbalance"]<-0.3]))

    # ---- HONEST BATTERY on the 'fade the flow' lead ----
    # The skill: a monotone fade result must survive halves + FOLLOW control +
    # placebo before it counts. Otherwise it's the sample's base rate.
    print("\n-- HONEST BATTERY: fade-the-flow (imb<0) across ALL asks (DESIGN) --")
    fade = [r for r in d_rows if r["imbalance"] < 0]
    follow = [r for r in d_rows if r["imbalance"] > 0]
    show("FADE (imb<0)", stats(fade))
    show("FOLLOW (imb>0)", stats(follow))
    # halves
    fade.sort(key=lambda r: r["ws"]); m=len(fade)//2
    show("fade 1st half", stats(fade[:m]))
    show("fade 2nd half", stats(fade[m:]))
    follow.sort(key=lambda r: r["ws"]); m2=len(follow)//2
    show("follow 1st half", stats(follow[:m2]))
    show("follow 2nd half", stats(follow[m2:]))

    # PRICE-BAND control: is the fade edge just the cheap-side base rate?
    # Compare fade vs follow WITHIN each ask band.
    print("\n-- fade-vs-follow within price bands (DESIGN) --")
    for lo,hi in [(0.10,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.90)]:
        f=[r for r in fade if lo<=r["ask"]<hi]
        g=[r for r in follow if lo<=r["ask"]<hi]
        show(f"ask[{lo:.2f},{hi:.2f}) FADE", stats(f))
        show(f"ask[{lo:.2f},{hi:.2f}) FOLLOW", stats(g))

    # PLACEBO: shuffle the win flag within each ask band, keep imbalance.
    # If the shuffled 'fade' still reads positive, the edge is the base rate.
    print("\n-- PLACEBO: shuffle winners within ask band, 200 draws (DESIGN) --")
    rng = np.random.default_rng(42)
    bands = [(0.10,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.90)]
    # precompute per-band win arrays and fade indices
    band_data = []
    for lo,hi in bands:
        idx = [i for i,r in enumerate(d_rows) if lo<=r["ask"]<hi]
        wins = np.array([d_rows[i]["won"] for i in idx], dtype=float)
        fade_idx = [k for k,i in enumerate(idx) if d_rows[i]["imbalance"]<0]
        band_data.append((wins, np.array(fade_idx, dtype=int)))
    real_net = np.array([r["pnl"] for r in fade]).sum()
    cnt=0; ge=0
    for _ in range(200):
        tot=0.0
        for wins, fidx in band_data:
            if len(fidx)==0: continue
            w = rng.permutation(wins)
            tot += w[fidx].sum()
        if tot >= real_net: ge+=1
        cnt+=1
    print(f"  real fade net (win-count proxy): {real_net:+.0f}")
    print(f"  placebo P(shuffled >= real) = {ge/cnt:.3f}  (n={cnt} draws)")

if __name__ == "__main__":
    main()