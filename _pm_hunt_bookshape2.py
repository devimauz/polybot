#!/usr/bin/env python3
"""_pm_hunt_bookshape2.py — does order-book SHAPE predict settlement beyond price?

Per the skill: a feature only counts if it separates (win - ask - fee) WITHIN a
price band, on the design slice, with a plateau (adjacent tiles agree), then one
holdout shot. We test the depth-imbalance and quote-asymmetry features measured
at a fixed mid-window T, controlling for the ask band.

Falsified families are NOT re-run: this is book-shape, not tape-follow /
trigger / favourite-band / model / maker / scalp / complete-set.
"""
import os, pickle
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

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def frame_at(d, T):
    i = int(np.searchsorted(d["t"], T, side="right")) - 1
    return i if i >= 0 else None

def features(d, i):
    up_ask=d["Up_ask"][i]; up_bid=d["Up_bid"][i]
    dn_ask=d["Down_ask"][i]; dn_bid=d["Down_bid"][i]
    up_bsz=d["Up_bidsz"][i]; dn_bsz=d["Down_bidsz"][i]
    up_asz=d["Up_asksz"][i]; dn_asz=d["Down_asksz"][i]
    if not all(np.isfinite(x) and x>0 for x in (up_ask,up_bid,dn_ask,dn_bid)): return None
    ub=np.nansum(d["Up_bids_sz"][i]); ua=np.nansum(d["Up_asks_sz"][i])
    db=np.nansum(d["Down_bids_sz"][i]); da=np.nansum(d["Down_asks_sz"][i])
    up_liq=ub+ua; dn_liq=db+da; tot=up_liq+dn_liq
    if tot<=0: return None
    depth_imb=(up_liq-dn_liq)/tot          # Up share of total book depth
    bid_imb=(ub-db)/(ub+db+1e-9)           # bid-side depth imbalance
    top_imb=(up_bsz-dn_bsz)/(up_bsz+dn_bsz+1e-9)  # top-of-book size imbalance
    up_spr=up_ask-up_bid; dn_spr=dn_ask-dn_bid
    spr_asym=up_spr-dn_spr
    # weighted mid: depth-weighted mid across both books
    return dict(up_ask=up_ask, dn_ask=dn_ask, up_bid=up_bid, dn_bid=dn_bid,
                up_asz=up_asz, dn_asz=dn_asz,
                depth_imb=depth_imb, bid_imb=bid_imb, top_imb=top_imb,
                spr_asym=spr_asym, overround=up_ask+dn_ask,
                up_mid=(up_ask+up_bid)/2)

def collect(W, T):
    """All tradable (side, ask, won, pnl, feature) rows at clock T."""
    rows = []
    for d in W:
        i = frame_at(d, T)
        if i is None: continue
        f = features(d, i)
        if f is None: continue
        for side, ask in (("Up", f["up_ask"]), ("Down", f["dn_ask"])):
            if not (0.10 <= ask <= 0.90): continue
            asz = f["up_asz"] if side=="Up" else f["dn_asz"]
            sh = int(STAKE/ask)
            if sh < 5 or asz < sh: continue
            won = (d["_winner"] == side)
            pnl = sh*(1.0 if won else 0.0) - sh*ask - fee(ask, sh)
            # sign of depth_imb relative to the side we're buying
            # depth_imb>0 => Up book heavier. Buying Up when Up-heavy is
            # "with pressure"; buying Down when Up-heavy is "against".
            di = f["depth_imb"] if side=="Up" else -f["depth_imb"]
            bi = f["bid_imb"] if side=="Up" else -f["bid_imb"]
            ti = f["top_imb"] if side=="Up" else -f["top_imb"]
            sa = f["spr_asym"] if side=="Up" else -f["spr_asym"]
            rows.append(dict(ask=ask, won=won, pnl=pnl, depth_imb=di, bid_imb=bi,
                             top_imb=ti, spr_asym=sa, ws=d["ws"], side=side,
                             overround=f["overround"]))
    return rows

def edge_stats(rows):
    n=len(rows)
    if n==0: return None
    wr=np.mean([r["won"] for r in rows])
    ask=np.mean([r["ask"] for r in rows])
    feepct=np.mean([FEE*r["ask"]*(1-r["ask"]) for r in rows])
    pnl=np.array([r["pnl"] for r in rows])
    return dict(n=n, wr=wr, ask=ask, edge=wr-ask-feepct, net=pnl.sum(),
                mean=pnl.mean())

def show(tag, st):
    if st is None: print(f"  {tag}: N=0"); return
    print(f"  {tag}: N={st['n']} wr={st['wr']:.3f} ask={st['ask']:.3f} "
          f"edge={st['edge']:+.4f} net=${st['net']:+.0f} mean=${st['mean']:+.3f}")

def main():
    W=load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")

    T = 180  # the only T where the cheap band showed a hint; test feature there
    d_rows = collect(design, T)
    print(f"\n=== DESIGN T={T}: {len(d_rows)} tradable rows ===")

    # Baseline: edge by ask band (already known: all ~0)
    print("-- baseline by ask band --")
    for lo,hi in [(0.10,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.90)]:
        r=[x for x in d_rows if lo<=x["ask"]<hi]
        show(f"ask[{lo:.2f},{hi:.2f})", edge_stats(r))

    # Does depth_imb separate winners within the 0.30-0.50 band?
    # (the one band that showed a hint). Split by depth_imb terciles.
    print("\n-- depth_imb terciles within ask[0.30,0.50) (DESIGN) --")
    band=[x for x in d_rows if 0.30<=x["ask"]<0.50]
    if len(band) >= 30:
        vals=np.array([x["depth_imb"] for x in band])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low",[x for x in band if x["depth_imb"]<=q1]),
                        ("mid",[x for x in band if q1<x["depth_imb"]<=q2]),
                        ("high",[x for x in band if x["depth_imb"]>q2])]:
            show(f"depth_imb {lab}", edge_stats(sub))
        print(f"  (tercile cuts: {q1:+.3f}, {q2:+.3f})")

    # bid_imb terciles in the same band
    print("\n-- bid_imb terciles within ask[0.30,0.50) (DESIGN) --")
    if len(band) >= 30:
        vals=np.array([x["bid_imb"] for x in band])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low",[x for x in band if x["bid_imb"]<=q1]),
                        ("mid",[x for x in band if q1<x["bid_imb"]<=q2]),
                        ("high",[x for x in band if x["bid_imb"]>q2])]:
            show(f"bid_imb {lab}", edge_stats(sub))
        print(f"  (tercile cuts: {q1:+.3f}, {q2:+.3f})")

    # top_imb terciles
    print("\n-- top_imb terciles within ask[0.30,0.50) (DESIGN) --")
    if len(band) >= 30:
        vals=np.array([x["top_imb"] for x in band])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low",[x for x in band if x["top_imb"]<=q1]),
                        ("mid",[x for x in band if q1<x["top_imb"]<=q2]),
                        ("high",[x for x in band if x["top_imb"]>q2])]:
            show(f"top_imb {lab}", edge_stats(sub))
        print(f"  (tercile cuts: {q1:+.3f}, {q2:+.3f})")

    # Also test depth_imb across the WHOLE book (not just cheap band) —
    # a structural pressure signal should show up everywhere.
    print("\n-- depth_imb terciles across ALL asks (DESIGN) --")
    if len(d_rows) >= 60:
        vals=np.array([x["depth_imb"] for x in d_rows])
        q1,q2=np.quantile(vals,[1/3,2/3])
        for lab,sub in [("low",[x for x in d_rows if x["depth_imb"]<=q1]),
                        ("mid",[x for x in d_rows if q1<x["depth_imb"]<=q2]),
                        ("high",[x for x in d_rows if x["depth_imb"]>q2])]:
            show(f"depth_imb {lab}", edge_stats(sub))
        print(f"  (tercile cuts: {q1:+.3f}, {q2:+.3f})")

if __name__ == "__main__":
    main()