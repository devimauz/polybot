"""pm_rule_median.py - user rule: BTC crosses the STRIKE (settlement reference,
Chainlink TWAP at window open) by +/- $10 -> buy Up/Down at the ask (1s
reaction), take profit at +20% of entry price, else hold to settlement.
Importable by the viewer for the overlay; runnable for the full-sample stats.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import btc5m_endgame_backtest as B

TH = 0.0         # trade on crossing the strike line itself (was 10.0)
MIN_T = 60.0     # no entries in the first 60 s of the window
TP = 1.20        # take profit: sell when bid >= 1.20 x entry ask
L = 1.0          # reaction delay, seconds
STAKE = 5.0
FEE = lambda p, sh: sh * 0.07 * p * (1 - p)


def running_median(d, step=5):
    t, cb = d["t"], d["cb_px"]
    n = len(t)
    med = np.full(n, np.nan)
    last_m = np.nan
    for i in range(n):
        if i % step == 0 and np.isfinite(cb[i]):
            v = cb[:i + 1][np.isfinite(cb[:i + 1])]
            if len(v):
                last_m = float(np.median(v))
        med[i] = last_m
    return med


def window_trades(d, strike, step=5):
    """Return dict or None. Trigger: cb crosses strike +/- TH (causal frames)."""
    t, cb = d["t"], d["cb_px"]
    n = len(t)
    med = None
    if strike is None:
        return None
    for i in range(10, n - 4):
        if not np.isfinite(cb[i]) or t[i] > 290 or t[i] < MIN_T:
            continue
        side = None
        if cb[i] >= strike + TH:
            side = "Up"
        elif cb[i] <= strike - TH:
            side = "Down"
        if side is None:
            continue
        j = int(np.searchsorted(t, float(t[i]) + L))
        if j >= n - 1:
            return None
        pe = d[side + "_ask"][j]
        ze = d[side + "_asksz"][j]
        if not (np.isfinite(pe) and 0.02 <= pe <= 0.98 and np.isfinite(ze) and ze >= 5 and ze * pe >= STAKE):
            return None
        sh = STAKE / pe
        bid = d[side + "_bid"]; bsz = d[side + "_bidsz"]
        target = pe * TP
        k = j + 1
        while k < n - 1 and t[k] < 297:
            if np.isfinite(bid[k]) and bid[k] >= target and np.isfinite(bsz[k]) and bsz[k] >= sh:
                jf = int(np.searchsorted(t, float(t[k]) + L))
                if jf < n and np.isfinite(bid[jf]) and bid[jf] >= target * 0.99:
                    ps = float(bid[jf])
                    pnl = sh * (ps - pe) - FEE(pe, sh) - FEE(ps, sh)
                    return dict(side=side, te=float(t[j]), pe=pe, tx=float(t[jf]), px=ps,
                                kind="tp", pnl=pnl, med=float(strike), trig=float(t[i]))
                break
            k += 1
        won = d["_winner"] == side
        pnl = sh * (1.0 if won else 0.0) - STAKE - FEE(pe, sh)
        return dict(side=side, te=float(t[j]), pe=pe, tx=None, px=None,
                    kind="settle-win" if won else "settle-lose", pnl=pnl,
                    med=float(strike), trig=float(t[i]))
    return None


def strikes_for(W):
    by_ws = {d["ws"]: d for d in W}
    out = {}
    for d in W:
        cl = d.get("cl_px")
        if cl is not None and np.isfinite(cl).sum() > 5:
            out[d["ws"]] = float(cl[np.isfinite(cl)][0]); continue
        prev = by_ws.get(d["ws"] - 300)
        if prev is not None:
            m = (prev["t"] >= 240) & np.isfinite(prev["cb_px"])
            if m.sum() >= 20:
                out[d["ws"]] = float(np.mean(prev["cb_px"][m])); continue
        out[d["ws"]] = None
    return out


def replay(W=None):
    if W is None:
        W = [d for d in B.load_all() if d["_winner"] in ("Up", "Down")]
    S = strikes_for(W)
    rows = []
    for wi, d in enumerate(W):
        r = window_trades(d, S[d["ws"]])
        if r:
            r["wi"] = wi; r["ws"] = d["ws"]
            rows.append(r)
    return W, rows


if __name__ == "__main__":
    W, rows = replay()
    print(f"windows={len(W)}  trades={len(rows)}")
    if rows:
        pnl = np.array([r["pnl"] for r in rows])
        wi = np.array([r["wi"] for r in rows])
        tp = sum(1 for r in rows if r["kind"] == "tp")
        win = sum(1 for r in rows if r["pnl"] > 0)
        m, t, nw = B.cluster_t(wi, pnl)
        lo, hi = B.boot_ci(wi, pnl)
        paid = np.mean([r["pe"] for r in rows])
        print(f"TP-hits={tp}/{len(rows)}  win$={win}/{len(rows)}  avg entry={paid:.3f}")
        print(f"net=${pnl.sum():+.2f}  /trade=${pnl.mean():+.3f}  median=${np.median(pnl):+.3f}  "
              f"ROI={100*pnl.sum()/(STAKE*len(pnl)):+.1f}%  t={t:+.2f}  CI=[{lo:+.2f},{hi:+.2f}]")
        for s in ("Up", "Down"):
            sub = [r for r in rows if r["side"] == s]
            if sub:
                a = np.array([r["pnl"] for r in sub])
                print(f"  arm {s}: N={len(sub)} net=${a.sum():+.2f} /trade=${a.mean():+.3f}")
        h = max(wi) // 2
        print(f"  halves: ${pnl[wi<=h].sum():+.2f} / ${pnl[wi>h].sum():+.2f}")
