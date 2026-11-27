"""Stage 12: THE WICK-REVERSION TEST (pre-registered, time-holdout).

MECHANISM HYPOTHESIS (H-wick):
  Polymarket 5-min crypto markets resolve on the Chainlink 60s-TWAP over the
  window vs. the open price -- NOT on a transient spot tick. A fast wick in the
  underlying therefore reprices the market's book HARD, but does not change the
  eventual resolution. The market that wicks to 0.15 and reverts is then
  offered Up at 0.15 when Up's true win probability is materially higher.
  => BUY THE SIDE THAT JUST WICKED DOWN, then either hold to settlement or
     sell back into the reversion.

Why this should be more robust than a curve-fit: it predicts a MECHANISM, and
it makes a falsifiable prediction (the market's own price should revert after
the entry; reversion should be larger for winners than losers).

PRE-REGISTERED CONFIG (chosen from the plateau in _pm_stability, not the peak):
  band 0.06-0.20, t in [30,150], first entry per (window, side), stake $10,
  hold to settlement, doc fee 0.07*p*(1-p).

TESTS:
  1. Time holdout: fit/choose on windows 0-134, REPORT ONLY on 135-268.
  2. Block sign test: 8 chronological blocks -> how many positive?
  3. Mechanistic falsification: is the price actually reverting after entry?
     (a) mean ask change +10/+30/+60s for winners vs losers
     (b) if there is no reversion, H-wick is WRONG.
  4. Entry-condition split: wick (ask falling into the band) vs. flat/rising.
     H-wick predicts the EDGE IS CONCENTRATED IN THE WICK CASES.
  5. Exit variants: hold vs. reversion-TP sell-back vs. stop.
  6. Two-sided consistency + placebo.
"""
import os, pickle, json
import numpy as np

CACHE = r"C:/Users/user/Desktop/bot/_pm_cache"
OUTDIR = r"C:/Users/user/Desktop/bot/_pm_out"
os.makedirs(OUTDIR, exist_ok=True)

BAND = (0.06, 0.20)
T_LO, T_HI = 30, 150
STAKE = 10.0
FEE = lambda p, sh: 0.07 * p * (1 - p) * sh


def load_windows():
    with open(os.path.join(CACHE, "manifest.pkl"), "rb") as fh:
        man = pickle.load(fh)
    wins = []
    for m in man:
        if not m["_winner"]:
            continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]
        d["_slug"] = m["_slug"]
        wins.append(d)
    wins.sort(key=lambda d: int(d["ws"][0]))
    return wins


def find_entry(d, side, t_lo, t_hi, p_lo, p_hi, stake, max_spread=0.05):
    t, bid, ask, asz = d["t"], d[side + "_bid"], d[side + "_ask"], d[side + "_asksz"]
    for i in range(len(t)):
        ti = t[i]
        if ti < t_lo:
            continue
        if ti > t_hi:
            break
        b, a = bid[i], ask[i]
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        if b <= 0 or a >= 1 or b > a + 1e-9 or (a - b) > max_spread:
            continue
        if a < p_lo or a > p_hi:
            continue
        if not np.isfinite(asz[i]) or asz[i] * a < stake:
            continue
        # entry context: was the ask falling into the band?
        j = int(np.searchsorted(t, ti - 6.0))
        prev_ask = ask[j] if j < i and np.isfinite(ask[j]) else np.nan
        prev_bid = bid[j] if j < i and np.isfinite(bid[j]) else np.nan
        return dict(i=i, t=float(ti), ask=float(a), bid=float(b),
                    prev_ask=float(prev_ask), prev_bid=float(prev_bid))
    return None


def build(W):
    trades = []
    for wi, d in enumerate(W):
        for side in ("Up", "Down"):
            e = find_entry(d, side, T_LO, T_HI, BAND[0], BAND[1], STAKE)
            if e is None:
                continue
            a = e["ask"]
            shares = STAKE / a
            fee = FEE(a, shares)
            won = (d["_winner"] == side)
            hold_pnl = shares * (1.0 if won else 0.0) - STAKE - fee
            # reversion path of the SAME side's ask after entry
            j10 = int(np.searchsorted(d["t"], e["t"] + 10))
            j30 = int(np.searchsorted(d["t"], e["t"] + 30))
            j60 = int(np.searchsorted(d["t"], e["t"] + 60))
            ask_ser = d[side + "_ask"]
            bid_ser = d[side + "_bid"]
            def at(ser, j):
                return float(ser[j]) if j < len(ser) and np.isfinite(ser[j]) else np.nan
            trades.append(dict(wi=wi, slug=d["_slug"], ws=int(d["ws"][0]), side=side,
                               t=e["t"], ask=a, bid=e["bid"], _i=e["i"],
                               prev_ask=e["prev_ask"], prev_bid=e["prev_bid"],
                               won=won, hold_pnl=hold_pnl,
                               ask10=at(ask_ser, j10), ask30=at(ask_ser, j30), ask60=at(ask_ser, j60),
                               bid10=at(bid_ser, j10), bid30=at(bid_ser, j30), bid60=at(bid_ser, j60),
                               _d=d))
    return trades


def block_signs(trades, nblocks=8):
    wss = sorted(set(t["ws"] for t in trades))
    edges = np.linspace(wss[0], wss[-1], nblocks + 1)
    out = []
    for b in range(nblocks):
        sel = [t for t in trades if edges[b] <= t["ws"] <= edges[b + 1]]
        out.append((len(sel), float(np.mean([t["hold_pnl"] for t in sel])) if sel else np.nan))
    return out


def main():
    W = load_windows()
    print(f"windows {len(W)}   pre-registered: band {BAND} t[{T_LO},{T_HI}] stake ${STAKE}\n")
    tr = build(W)
    print(f"trades: {len(tr)}")
    pnl = np.array([t["hold_pnl"] for t in tr])
    wins = np.array([t["won"] for t in tr], dtype=float)
    print(f"  overall: total=${pnl.sum():+.2f} avg=${pnl.mean():+.3f} win={wins.mean():.3f} "
          f"worst=${pnl.min():+.2f}")

    # ---------------- 1. TIME HOLDOUT ----------------
    print("\n" + "=" * 100)
    print("1) TIME HOLDOUT (train = first half, REPORT = second half)")
    print("=" * 100)
    half = tr[len(tr) // 2]
    wcut = sorted(set(t["ws"] for t in tr))[len(set(t["ws"] for t in tr)) // 2]
    A = [t for t in tr if t["ws"] <= wcut]
    B = [t for t in tr if t["ws"] > wcut]
    for nm, S in (("first half (in-sample)", A), ("SECOND HALF (HELD OUT)", B)):
        p = np.array([t["hold_pnl"] for t in S])
        w = np.array([t["won"] for t in S], dtype=float)
        print(f"  {nm:<24} N={len(S):>3} total=${p.sum():+7.2f} avg=${p.mean():+.3f} win={w.mean():.3f}")

    # ---------------- 2. BLOCK SIGN TEST ----------------
    print("\n" + "=" * 100)
    print("2) BLOCK SIGN TEST (8 chronological blocks)")
    print("=" * 100)
    bs = block_signs(tr, 8)
    pos = sum(1 for n, m in bs if n and m > 0)
    print("  " + "  ".join(f"[{i+1}: n={n} ${m:+.2f}]" for i, (n, m) in enumerate(bs)))
    print(f"  positive blocks: {pos}/{len([b for b in bs if b[0]])}")

    # ---------------- 3. MECHANISM: does the price revert? ----------------
    print("\n" + "=" * 100)
    print("3) MECHANISM CHECK: does the market's price revert after the entry?")
    print("=" * 100)
    for hz in (10, 30, 60):
        for grp, lbl in ((1.0, "winners"), (0.0, "losers")):
            sel = [t for t in tr if t["won"] == bool(grp)]
            v = np.array([t[f"ask{hz}"] - t["ask"] for t in sel if np.isfinite(t[f"ask{hz}"])])
            b = np.array([t[f"bid{hz}"] - t["ask"] for t in sel if np.isfinite(t[f"bid{hz}"])])
            if len(v):
                print(f"  +{hz:>2}s {lbl:<8} n={len(v):>3} mean ask change={v.mean():+.4f} "
                      f"median={np.median(v):+.4f} | mean bid change={b.mean():+.4f}")
    # and the entry-context split
    wick = [t for t in tr if np.isfinite(t["prev_ask"]) and t["prev_ask"] > t["ask"] + 0.01]
    flat = [t for t in tr if not (np.isfinite(t["prev_ask"]) and t["prev_ask"] > t["ask"] + 0.01)]
    print(f"\n  entry context: WICK (ask fell >1c in last 6s) n={len(wick)}   FLAT/RISING n={len(flat)}")
    for nm, S in (("wick", wick), ("flat/rising", flat)):
        if not S:
            continue
        p = np.array([t["hold_pnl"] for t in S])
        w = np.array([t["won"] for t in S], dtype=float)
        ma = np.mean([t["ask"] for t in S])
        print(f"    {nm:<12} N={len(S):>3} avg=${p.mean():+.3f} win={w.mean():.3f} meanAsk={ma:.3f} "
              f"edge={w.mean()-ma:+.4f}")

    # ---------------- 4. EXIT VARIANTS ----------------
    print("\n" + "=" * 100)
    print("4) EXIT VARIANTS")
    print("=" * 100)
    def exit_variant(tp_delta, hold_min=5.0, max_hold=180.0):
        d = t["_d"]
        side = t["side"]
        bid_ser = d[side + "_bid"]
        bsz = d[side + "_bidsz"]
        a = t["ask"]
        shares = STAKE / a
        entry_fee = FEE(a, shares)
        tp = t["ask"] + tp_delta
        for j in range(t["_i"] + 1, len(bid_ser)):
            if d["t"][j] - t["t"] > max_hold:
                break
            b = bid_ser[j]
            if not np.isfinite(b):
                continue
            if d["t"][j] - t["t"] < hold_min:
                continue
            if b >= tp:
                if np.isfinite(bsz[j]) and bsz[j] >= shares:
                    return shares * b - FEE(b, shares) - STAKE - entry_fee, "tp"
                break
        won = t["won"]
        return shares * (1.0 if won else 0.0) - STAKE - entry_fee, "settle"
    for tp_delta in (0.02, 0.04, 0.06, 0.10):
        res = []
        for t in tr:
            p, r = exit_variant(tp_delta)
            res.append((p, r))
        allp = np.array([r[0] for r in res])
        tps = [r[0] for r in res if r[1] == "tp"]
        st = [r[0] for r in res if r[1] == "settle"]
        print(f"  TP=+{tp_delta:.2f}: total=${allp.sum():+8.2f} avg=${allp.mean():+.3f} "
              f"| tp n={len(tps)} ${sum(tps):+.2f} | settle n={len(st)} ${sum(st):+.2f}")

    # ---------------- 5. TWO-SIDED + PLACEBO ----------------
    print("\n" + "=" * 100)
    print("5) TWO-SIDED CONSISTENCY + PLACEBO")
    print("=" * 100)
    # buy the opposite side at the same moment
    opp = []
    for t in tr:
        d = t["_d"]
        o = "Down" if t["side"] == "Up" else "Up"
        oa = d[o + "_ask"][t["_i"]]
        ob = d[o + "_bid"][t["_i"]]
        if not (np.isfinite(oa) and np.isfinite(ob) and 0 < ob <= oa < 1):
            continue
        if (oa - ob) > 0.05:
            continue
        shares = STAKE / oa
        fee = FEE(oa, shares)
        won = (d["_winner"] == o)
        opp.append(shares * (1.0 if won else 0.0) - STAKE - fee)
    print(f"  BUY THE BAND (longshot) avg=${np.mean([t['hold_pnl'] for t in tr]):+.3f}")
    print(f"  BUY THE OPPOSITE      avg=${np.mean(opp):+.3f}   (if the market is fair, both < 0 after fees)")
    # placebo: shuffle winners across windows
    rng = np.random.default_rng(5)
    perm = rng.permutation(len(W))
    W2 = [dict(d) for d in W]
    for i, d in enumerate(W2):
        d["_winner"] = W[perm[i]]["_winner"]
    tr2 = build(W2)
    print(f"  PLACEBO (shuffled winners) avg=${np.mean([t['hold_pnl'] for t in tr2]):+.3f} "
          f"(this destroys the price<->outcome link; a positive placebo means the METHOD is biased)")

    with open(os.path.join(OUTDIR, "wick_test.json"), "w") as fh:
        json.dump({"band": BAND, "t": [T_LO, T_HI], "n": len(tr),
                   "total": float(pnl.sum()), "avg": float(pnl.mean()),
                   "win": float(wins.mean()),
                   "blocks": [(int(n), float(m)) for n, m in bs]}, fh, indent=2)


if __name__ == "__main__":
    main()
