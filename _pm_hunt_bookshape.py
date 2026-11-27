#!/usr/bin/env python3
"""_pm_hunt_bookshape.py — hunt a NEW non-latency family on the 5m cache.

Family: ORDER-BOOK SHAPE predictors of settlement, measured at a fixed
mid-window time (no tape-following, no triggers, no latency).

For a chosen clock time T and price band, we test whether book-derived
features (depth imbalance, cumulative depth skew, quote asymmetry) identify
frames where (realised_win_rate - ask - fee) is positive, beyond what the
price level alone gives. This is a price-surface study per the skill:
design slice only, plateau check (adjacent bands agree), freeze, then one
holdout evaluation.

No falsified family is re-run: this is not tape-follow / book-trend /
streak / favourite-band / model-vs-market / maker / complete-set / scalp.
"""
import os, sys, pickle, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0])
        d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def fee(p, sh):
    return sh * FEE * p * (1.0 - p)

def frame_at(d, T):
    """Index of the last frame with causal clock t <= T (seconds into window)."""
    t = d["t"]  # already seconds-into-window (t_src)
    i = int(np.searchsorted(t, T, side="right")) - 1
    if i < 0:
        return None
    return i

def book_features(d, i):
    """Compute book-shape features from the 12-level ladders at frame i."""
    up_ask = d["Up_ask"][i]; up_bid = d["Up_bid"][i]
    dn_ask = d["Down_ask"][i]; dn_bid = d["Down_bid"][i]
    up_asz = d["Up_asksz"][i]; up_bsz = d["Up_bidsz"][i]
    dn_asz = d["Down_asksz"][i]; dn_bsz = d["Down_bidsz"][i]
    if not all(np.isfinite(x) and x > 0 for x in (up_ask, up_bid, dn_ask, dn_bid)):
        return None
    # cumulative depth on each side of the Up book
    up_bid_lad = d["Up_bids_sz"][i]; up_ask_lad = d["Up_asks_sz"][i]
    dn_bid_lad = d["Down_bids_sz"][i]; dn_ask_lad = d["Down_asks_sz"][i]
    up_bid_cum = np.nansum(up_bid_lad); up_ask_cum = np.nansum(up_ask_lad)
    dn_bid_cum = np.nansum(dn_bid_lad); dn_ask_cum = np.nansum(dn_ask_lad)
    # total liquidity on each outcome (bid+ask depth)
    up_liq = up_bid_cum + up_ask_cum
    dn_liq = dn_bid_cum + dn_ask_cum
    tot = up_liq + dn_liq
    if tot <= 0:
        return None
    # depth imbalance: Up share of total book depth
    depth_imb = (up_liq - dn_liq) / tot
    # top-of-book size imbalance (Up bid vs Down bid at touch)
    top_imb = (up_bsz - dn_bsz) / (up_bsz + dn_bsz + 1e-9)
    # bid-side depth imbalance
    bid_imb = (up_bid_cum - dn_bid_cum) / (up_bid_cum + dn_bid_cum + 1e-9)
    # spread asymmetry
    up_spr = up_ask - up_bid; dn_spr = dn_ask - dn_bid
    spr_asym = (up_spr - dn_spr)
    # overround
    overround = up_ask + dn_ask
    return dict(up_ask=up_ask, up_bid=up_bid, dn_ask=dn_ask, dn_bid=dn_bid,
                up_asz=up_asz, dn_asz=dn_asz,
                depth_imb=depth_imb, top_imb=top_imb, bid_imb=bid_imb,
                spr_asym=spr_asym, overround=overround,
                up_liq=up_liq, dn_liq=dn_liq, up_mid=(up_ask+up_bid)/2)

def evaluate(rows, label):
    """rows: list of (win_bool, ask, pnl). Report edge = win - ask - fee."""
    n = len(rows)
    if n == 0:
        print(f"  {label}: N=0"); return
    wins = sum(1 for w, a, p in rows if w)
    asks = np.array([a for w, a, p in rows]); pnls = np.array([p for w, a, p in rows])
    mean_ask = asks.mean()
    wr = wins / n
    fee_pct = (FEE * asks * (1 - asks)).mean()  # fee per $1 notional ~ per share
    edge = wr - mean_ask - fee_pct
    print(f"  {label}: N={n} wr={wr:.3f} ask={mean_ask:.3f} fee~{fee_pct:.4f} "
          f"edge={edge:+.4f} net=${pnls.sum():+.0f} mean/trade=${pnls.mean():+.3f}")

def main():
    W = load_all()
    n = len(W)
    split = int(n * 0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} windows | design={len(design)} holdout={len(holdout)}")

    # scan T candidates and bands on DESIGN only
    for T in (120, 180, 210, 240, 260):
        print(f"\n=== T={T}s (DESIGN) ===")
        # collect all frames at this T across design windows
        samples = []
        for d in design:
            i = frame_at(d, T)
            if i is None: continue
            f = book_features(d, i)
            if f is None: continue
            # a trade = buy the side whose ask is in band; we test BOTH sides
            for side, ask in (("Up", f["up_ask"]), ("Down", f["dn_ask"])):
                if not (0.10 <= ask <= 0.90): continue
                # depth gate: need >= stake/ask shares at the ask touch
                asz = f["up_asz"] if side == "Up" else f["dn_asz"]
                sh = int(STAKE / ask)
                if sh < 5 or asz < sh: continue
                won = (d["_winner"] == side)
                pnl = sh * (1.0 if won else 0.0) - sh * ask - fee(ask, sh)
                samples.append((T, side, ask, won, pnl, f, d["ws"]))

        # price-band x feature-tile surface (design): which book-shape feature
        # separates positive-edge frames from negative, controlling for price?
        # We report edge by ask-band first (the baseline surface).
        print(f"  frames sampled: {len(samples)}")
        if not samples: continue
        bands = [(0.10,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.90)]
        for lo, hi in bands:
            rows = [(w, a, p) for (T_, s, a, w, p, f, ws) in samples if lo <= a < hi]
            evaluate(rows, f"ask[{lo:.2f},{hi:.2f})")

if __name__ == "__main__":
    main()