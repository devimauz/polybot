"""pm_lag_analyze.py - Coinbase-move -> Polymarket-reprice lag study + stale-ask EV.

Input: laglog/lag-<ws>.jsonl(.gz) files fetched from the AWS box (--dir), plus a
winners map (slug -> 'Up'|'Down', from gamma; --winners json, built by the caller
or fetched here). Prints, pooled over all supplied windows:

  1) pipeline sanity: PM network delay (local recv - PM exchange stamp), RTT p50.
  2) repricing-lag table: for Coinbase moves of >= theta $ (detected on a 200 ms
     lookback), time until the aligned-side PM ask has raised by >= kappa, by
     theta band.  p10/50/90.
  3) actionability: at t_det + T_ACT (T_ACT = rtt_p50 + 25 ms), is the aligned
     ask STILL <= a0 + kappa (i.e. a stale-able fill)? fraction + N.
  4) crude taker EV at that stale ask: buy aligned side at the ask observable at
     t_det+T_ACT (FOK-gated by displayed depth >= 5 shares), settle to the TRUE
     winner.  Reports gross, taker fee 0.07*p*(1-p)/sh, net per trade and total
     at 5 shares, with a cluster t over windows. If repricing is fast, EV <= 0
     automatically because the fill price has already adjusted.

This is the decision table for whether sub-second latency arb exists for us.
"""
import argparse, gzip, json, os, sys, glob
import numpy as np
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc5m_endgame_backtest as B

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir", required=True, help="dir with lag-*.jsonl[.gz]")
PARSER.add_argument("--winners", default=None, help="json map slug->winner")
PARSER.add_argument("--taus", default="200", help="move lookback ms")
PARSER.add_argument("--thetas", default="2,4,8,16,32", help="move thresholds $")
PARSER.add_argument("--kappas", default="0.01,0.02,0.05", help="reprice size")
A = PARSER.parse_args()
TAU = int(A.taus) * 10**6
THETAS = [float(x) for x in A.thetas.split(",")]
KAPPAS = [float(x) for x in A.kappas.split(",")]
SIZE = 5.0

win = json.load(open(A.winners)) if A.winners else {}
# normalize keys: slug -> int(window start)
win = {int(str(k).split("-")[-1]): v for k, v in win.items() if str(k).startswith("btc-updown")}

def fetch_winner(w):
    slug = f"btc-updown-5m-{w}"
    try:
        req = urllib.request.Request(f"https://gamma-api.polymarket.com/events?slug={slug}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        m = json.load(urllib.request.urlopen(req, timeout=15))[0]["markets"][0]
        prs = [float(x) for x in json.loads(m["outcomePrices"])]
        if sorted(prs) == [0.0, 1.0]:
            return "Up" if prs[0] == 1.0 else "Down"
    except Exception:
        pass
    return None

# ---------- load ----------
net_delay, rtts = [], []
events_all = {}   # ws -> dict of arrays
for fp in sorted(glob.glob(os.path.join(A.dir, "lag-*.jsonl*")) + glob.glob(os.path.join(A.dir, "*.jsonl*"))):
    base = os.path.basename(fp)
    if not base.startswith("lag-"):
        continue
    ws = int(base.split("lag-")[1].split(".")[0])
    op = gzip.open if fp.endswith(".gz") else open
    pm_t, pm_baU, pm_baD = [], [], []
    cb_t, cb_mid = [], []
    n_pm = 0
    with op(fp, "rt") as fh:
        for line in fh:
            if '"t_ns"' not in line:
                continue
            r = json.loads(line)
            s = r["src"]
            t = r["t_ns"]
            if s == "pm":
                pm_t.append(t); pm_baU.append(r["baU"]); pm_baD.append(r["baD"])
                if r.get("ets"):
                    net_delay.append(t / 1e6 - float(r["ets"]))
                n_pm += 1
            elif s == "cb" and r["tp"] == "t":
                if r["bb"] and r["ba"]:
                    cb_t.append(t); cb_mid.append((r["bb"] + r["ba"]) / 2)
            elif s == "cb" and r["tp"] == "m":
                cb_t.append(t); cb_mid.append(r["p"])
            elif s == "rtt" and r.get("ms"):
                rtts.append(r["ms"])
    if n_pm < 200 or len(cb_t) < 200:
        print(f"# skip {base}: pm={n_pm} cb={len(cb_t)}")
        continue
    o = np.argsort(cb_t); cb_t = np.array(cb_t)[o]; cb_mid = np.array(cb_mid)[o]
    o = np.argsort(pm_t)
    events_all[ws] = dict(cb_t=cb_t, cb_mid=cb_mid,
                          pm_t=np.array(pm_t)[o], pm_ask={"Up": np.array(pm_baU)[o],
                                                          "Down": np.array(pm_baD)[o]})
    win.setdefault(ws, None)

if not events_all:
    sys.exit("no usable windows (need >=200 pm and cb lines each)")

print(f"# windows analyzed: {len(events_all)}; "
      f"pm net delay ms: p50={np.percentile(net_delay,50):.0f} p90={np.percentile(net_delay,90):.0f}; "
      f"CLOB REST RTT ms: p50={np.percentile(rtts,50):.0f} p90={np.percentile(rtts,90):.0f}")
T_ACT = (np.percentile(rtts, 50) + 25) * 1e6   # detect->order-in-book estimate
print(f"# assumed action pipeline T_ACT = {T_ACT/1e6:.0f} ms after detection\n")

# ---------- event study ----------
rng = np.random.default_rng(7)
tab = {}
ev_store = []
for ws, E in events_all.items():
    cb_t, cb_mid = E["cb_t"], E["cb_mid"]
    # 50ms grid for lookback reference
    grid = np.arange(cb_t[0], cb_t[-1], 50 * 10**6)
    gm = np.interp(grid, cb_t, cb_mid)
    lb = max(1, TAU // (50 * 10**6))
    last_fire = -10**18
    for i in range(lb, len(gm)):
        d = gm[i] - gm[i - lb]
        ad = abs(d)
        if ad < THETAS[0] or grid[i] - last_fire < 2 * 10**9:
            continue
        # bucket = LARGEST theta threshold satisfied
        th = max(t for t in THETAS if ad >= t)
        last_fire = grid[i]
        t_det = grid[i]
        side = "Up" if d > 0 else "Down"
        pt, asks = E["pm_t"], E["pm_ask"][side]
        k = np.searchsorted(pt, t_det, side="right") - 1
        if k < 0 or k >= len(pt):
            continue
        a0 = asks[k]
        rec = dict(ws=ws, th=th, t_det=t_det, a0=a0, side=side, d=d)
        for ka in KAPPAS:
            m = np.where(asks[k:] >= a0 + ka)[0]
            rec["lag_" + str(ka)] = float(pt[k + m[0]] - t_det) if len(m) else np.nan
        for lab, extra in (("act", T_ACT), ("act150", 150e6), ("act250", 250e6), ("act400", 400e6)):
            ka_act = np.searchsorted(pt, t_det + extra, side="right") - 1
            rec["a_" + lab] = asks[ka_act] if ka_act >= 0 else np.nan
        rec["stale"] = (rec["a_act"] <= a0 + 0.005)
        k3 = np.searchsorted(pt, t_det + 3 * 10**9, side="right") - 1
        rec["a_3s"] = asks[k3] if 0 <= k3 < len(asks) else np.nan
        ev_store.append(rec)

print("== 1) PM repricing lag vs Coinbase move (ms from detection) ==")
for th in THETAS:
    for ka in KAPPAS:
        lags = np.array([e["lag_" + str(ka)] for e in ev_store
                         if e["th"] == th and np.isfinite(e["lag_" + str(ka)])])
        if len(lags) < 5:
            print(f"move>={th:5.0f}$  reprice>={ka:.2f}: N={len(lags):5d}   (thin)")
            continue
        p10, p50, p90 = np.percentile(lags, [10, 50, 90]) / 1e6
        frac_slow = float(np.mean(lags > T_ACT))
        print(f"move>={th:5.0f}$  reprice>={ka:.2f}: N={len(lags):5d}  "
              f"lag p10={p10:6.0f} p50={p50:6.0f} p90={p90:6.0f} ms   "
              f"share lag > T_ACT({T_ACT/1e6:.0f}ms): {frac_slow:.2f}")

print("\n== 2) stale-ask fills and settlement EV (5 shares, aligned side, buy at ask observable at t_det+T_ACT) ==")
for th in THETAS:
    sub = [e for e in ev_store if e["th"] == th and np.isfinite(e["a_act"]) and 0.02 <= e["a_act"] <= 0.98]
    if len(sub) < 8:
        print(f"move>={th:5.0f}$: N={len(sub)} (thin)")
        continue
    won = []
    acc = []
    pnl, wi, fees, fills = [], [], [], []
    for e in sub:
        w = win.get(e["ws"])
        if w is None:
            w = win[e["ws"]] = fetch_winner(e["ws"])
        if w is None:
            continue
        acc.append(e)
        p = 1.0 if w == e["side"] else 0.0
        sh = SIZE / e["a_act"]
        fee = sh * 0.07 * e["a_act"] * (1 - e["a_act"])
        pnl.append(sh * p - sh * e["a_act"] - fee)
        wi.append(e["ws"]); won.append(p); fees.append(fee)
    if not pnl:
        print(f"move>={th:5.0f}$: winners missing (build --winners map)")
        continue
    pnl = np.array(pnl); wi = np.array(wi); won = np.array(won)
    uq = np.unique(wi)
    s = np.array([pnl[wi == w].sum() for w in uq])
    tstat = s.mean() / (s.std(ddof=1) / np.sqrt(len(uq))) if len(uq) > 2 and s.std(ddof=1) > 0 else np.nan
    nstale = sum(e["stale"] for e in sub)
    order = np.argsort(pnl)[::-1]
    top5 = pnl[order[:5]].sum() / pnl.sum() if pnl.sum() != 0 else np.nan
    trimmed = np.sort(pnl)[2:-2].mean() if len(pnl) >= 8 else np.nan
    cont = np.mean([1.0 if np.isfinite(e["a_3s"]) and e["a_3s"] >= e["a0"] + 0.01 else 0.0
                    for e in acc])
    order = np.argsort(pnl)[::-1]
    top5s = pnl[order[:5]].sum() / pnl.sum() if pnl.sum() != 0 else np.nan
    wi2 = np.array([e["ws"] for e in acc])
    med_w = np.median(np.unique(wi2))
    h1 = pnl[wi2 <= med_w].sum(); h2 = pnl[wi2 > med_w].sum()
    upm = np.array([e["side"] == "Up" for e in acc])
    su, sd = pnl[upm].sum(), pnl[~upm].sum()
    print(f"move>={th:5.0f}$: N={len(pnl)}  wr={won.mean():.3f}  cont3s={cont:.2f}  "
          f"stale={nstale}/{len(sub)}  net=${pnl.sum():+.2f} mean={pnl.mean():+.3f} MED=${np.median(pnl):+.3f} "
          f"trim={np.sort(pnl)[2:-2].mean():+.3f} top5={top5s:+.2f} t={tstat:+.2f}  "
          f"halves={h1:+.0f}/{h2:+.0f} arms U{su:+.0f}({upm.sum()}) D{sd:+.0f}({(~upm).sum()})")
    # latency sweep: same events, action at 94/150/250/400 ms
    for lab in ("act", "act150", "act250", "act400"):
        pn2 = []
        wi3 = []
        for e in acc:
            a = e["a_" + lab]
            if not (np.isfinite(a) and 0.02 <= a <= 0.98):
                continue
            sh = SIZE / a
            w = win.get(e["ws"]) or fetch_winner(e["ws"])
            win[e["ws"]] = w
            if w is None:
                continue
            p = 1.0 if w == e["side"] else 0.0
            pn2.append(sh * p - sh * a - sh * 0.07 * a * (1 - a))
            wi3.append(e["ws"])
        if len(pn2) >= 8:
            arr = np.array(pn2); w3 = np.array(wi3)
            u3 = np.unique(w3)
            s3 = np.array([arr[w3 == u].sum() for u in u3])
            tt = s3.mean() / (s3.std(ddof=1) / np.sqrt(len(u3))) if len(u3) > 2 and s3.std(ddof=1) > 0 else float("nan")
            print(f"    latency {lab:7s}: N={len(arr)} net=${arr.sum():+.2f} /tr=${arr.mean():+.3f} t={tt:+.2f}")
    # timing-shift placebo: shift the PM book series by a random offset per
    # window (5-60 s, signed) -> destroys signal->book causality, keeps winners.
    obs = pnl.sum()
    beats = 0
    rng2 = np.random.default_rng(5)
    byw = {}
    for e in acc:
        byw.setdefault(e["ws"], []).append(e)
    Ew = events_all
    for _ in range(200):
        tot = 0.0
        for ws_e, es in byw.items():
            dE = Ew[ws_e]
            for e in es:
                pt_, asks_ = dE["pm_t"], dE["pm_ask"][e["side"]]
                shift = rng2.uniform(5, 60) * 1e9 * rng2.choice([-1, 1])
                ka = np.searchsorted(pt_, e["t_det"] + T_ACT + shift, side="right") - 1
                if ka < 0 or ka >= len(asks_):
                    continue
                a = asks_[ka]
                if not (0.02 <= a <= 0.98):
                    continue
                sh_ = SIZE / a
                w_ = win.get(ws_e)
                p_ = 1.0 if w_ == e["side"] else 0.0
                tot += sh_ * p_ - SIZE - sh_ * 0.07 * a * (1 - a)
        if tot >= obs:
            beats += 1
    print(f"    timing-shift placebo: p(shifted book >= observed) = {beats/200:.3f}")
    _ = trimmed, top5
