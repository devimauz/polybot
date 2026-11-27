"""_up15_battery.py - print-based taker/maker battery on btc-updown-15m (+4h) windows.

Same discipline as the 5m/sports studies, executed prices only (taker prints):
 - normalize every print into UP-token frame: Down BUY@x == Up SELL@1-x,
   Down SELL@x == Up BUY@1-x. So we know each aggressor's action on the Up side.
 - Taker Up-BUY edge: win - p - 0.07*p*(1-p), banded by price decile x minutes
   remaining (900-ts window). Taker Up-SELL = mirror (1-win - (1-p) - fee).
 - Maker via print-through: rest Up bid at q, certain fill iff a later UP-frame
   SELL print at price < q (aggressor selling through our level); P&L = win - q.
 - Settlement lag: last Up-frame price on the eventual winner side, and the
   two-sided cost proxy pUp_buy + pDn_buy (buy both = complete set).
 - Chronological halves (design/holdout style, though same regime).
"""
import json, os, sys, zlib, gzip
import numpy as np

ROOT = r"C:\Users\user\Desktop\bot"
WL = {x["cond"]: x for x in json.load(open(os.path.join(ROOT, "_pm_out/updown_worklist.json")))}

def load(fp):
    raw = open(fp, "rb").read()
    try:
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode("utf-8", "replace")
    except zlib.error:
        return gzip.decompress(raw).decode("utf-8", "replace")

PRINTS = {}
for line in load(os.path.join(ROOT, "price_logs/up15/prints.jsonl.gz")).splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    PRINTS.setdefault(r["c"], []).append(r)
print("windows with prints:", len(PRINTS), flush=True)

FEE = lambda p: 0.07 * p * (1 - p)

def upframe(cond):
    """return arrays: t (up-buy time), px (Up-token price), upbuy (1 aggressor bought Up),
    upsell(1 aggressor sold Up), sz"""
    rows = sorted(PRINTS[cond], key=lambda r: r["ts"])
    ts = np.array([r["ts"] for r in rows], float)
    pxu = np.where([r["u"] for r in rows],
                   [r["p"] for r in rows],
                   [1 - r["p"] for r in rows])
    upbuy = np.array([bool(r["b"]) == bool(r["u"]) for r in rows])   # buy Up or sell Down
    sz = np.array([r["sz"] for r in rows])
    return ts, np.array(pxu), upbuy, ~upbuy, sz

def minutes_left(cond, t):
    ws = WL[cond]["ws"]
    per = 900 if WL[cond]["tag"] == "15m" else 14400
    return (ws + per - t) / 60.0

# ---------- TAKER calibration ----------
for tag in ("15m", "4h"):
    recs = []
    for cond in PRINTS:
        if WL[cond]["tag"] != tag or cond not in WL:
            continue
        ts, px, upb, ups, sz = upframe(cond)
        win = float(WL[cond]["win"])
        for j in range(len(ts)):
            if not (0 <= minutes_left(cond, ts[j]) <= (15 if tag == "15m" else 240)):
                continue
            p = px[j]
            if not (0.02 < p < 0.98):
                continue
            if upb[j]:
                edge = win - p - FEE(p)          # per share, net of taker fee
            else:
                edge = (1 - win) - (1 - p) - FEE(1 - p)
            recs.append((cond, minutes_left(cond, ts[j]), p, 1 if upb[j] else 0, edge))
    if not recs:
        continue
    A = np.array([(r[1], r[2], r[3], r[4]) for r in recs])
    conds = np.array([r[0] for r in recs])
    print(f"\n############ {tag}: taker edge per share, by price x time-left (n={len(A)}) ############")
    print("band        mins-left  n     mean_p  net_edge_c  t(win-clustered)")
    for mlo, mhi in ((0, 2), (2, 5), (5, 15), (15, 60), (60, 241)):
        if tag == "15m" and mlo >= 15:
            continue
        for lo, hi in ((0.02, 0.15), (0.15, 0.35), (0.35, 0.60), (0.60, 0.80), (0.80, 0.98)):
            m = (A[:, 0] >= mlo) & (A[:, 0] < mhi) & (A[:, 1] >= lo) & (A[:, 1] < hi)
            n = int(m.sum())
            if n < 120:
                continue
            e = A[m, 3]
            uq = np.unique(conds[m])
            per_w = np.array([e[conds[m] == u].sum() / max((conds[m] == u).sum(), 1) for u in uq])
            t = e.mean() / (per_w.std(ddof=1) / np.sqrt(len(uq))) if len(uq) > 1 else np.nan
            halves = ""
            ml = A[m, 0]
            print(f"[{lo:.2f},{hi:.2f})  {mlo:3d}-{mhi:3d}m  {n:5d}  {A[m,1].mean():.3f}  {e.mean()*100:+6.2f}c  t={t:+5.2f}")

# ---------- settlement lag ----------
print("\n### settlement lag: last print on the eventual winner side ###")
for tag in ("15m", "4h"):
    vals = []
    for cond in PRINTS:
        if WL[cond]["tag"] != tag:
            continue
        ts, px, upb, ups, sz = upframe(cond)
        pw = np.where(WL[cond]["win"] == 1, px, 1 - px)
        inw = ts >= WL[cond]["ws"]
        if inw.sum() and 0.01 < pw[-1] < 1.0:
            vals.append(pw[-1])
    v = np.array(vals)
    if len(v):
        disc = 1 - v
        th = 0.07
        net = disc - th * v * (1 - v)
        print(f"{tag}: n={len(v)} median_last={np.median(v):.4f} share<=0.99={np.mean(v<=0.99)*100:.1f}% "
              f"share<=0.98={np.mean(v<=0.98)*100:.1f}% mean_net_disc={net.mean()*100:+.2f}c")

# ---------- maker print-through on 15m ----------
print("\n### maker: rest Up bid at a touch price; certain fill iff later SELL print strictly below ###")
recs = []
for cond in PRINTS:
    if WL[cond]["tag"] != "15m":
        continue
    ts, px, upb, ups, sz = upframe(cond)
    win = float(WL[cond]["win"])
    # entry candidates: prints at 2-6 min left; rest bid at that price q
    for j in range(len(ts)):
        ml = minutes_left(cond, ts[j])
        if not (2 <= ml <= 6) or not (0.30 < px[j] < 0.95):
            continue
        q = px[j]
        later = np.where((ts > ts[j]) & (ts <= WL[cond]["ws"] + 900))[0]
        through = [k for k in later if ups[k] and px[k] < q]
        if through:
            sh = max(5.0, np.floor(5 / q))
            recs.append((cond, q, win, (win - q) * sh))
if recs:
    q = np.array([r[1] for r in recs]); win = np.array([r[2] for r in recs])
    pnl = np.array([r[3] for r in recs])
    conds = np.unique([r[0] for r in recs])
    per_w = np.array([pnl[[i for i, r in enumerate(recs) if r[0] == u]].sum() for u in conds])
    t = pnl.mean() / (per_w.mean(axis=0).std(ddof=1) / np.sqrt(len(conds))) if len(conds) > 1 else np.nan
    print(f"fills={len(recs)} windows={len(conds)} pay={q.mean():.3f} wr={win.mean():.3f} "
          f"net/share={( (win-q).mean() )*100:+.2f}c (maker fee 0) tot=${pnl.sum():+.0f}")
