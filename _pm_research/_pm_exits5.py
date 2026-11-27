"""Take-profit / stop exit study on the SHIPPING rule, at the $5 stake.

The report quotes a take-profit-at-0.95 figure that was originally computed
at a $10 stake.  Because the fee is proportional and the exit is a price
target, the P&L scales linearly with the stake - BUT the exit depth check
does not: a $5 exit needs less displayed size than a $10 exit, so at $5 some
trades that failed the depth test at $10 now succeed.  That is a real
difference, so it has to be recomputed rather than scaled by hand.

Exits are executed at the BID (taker), with a depth check, exactly like the
live bot would have to.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import btc5m_endgame_backtest as B


def simulate_one(d, side, ask, ei, tp=None, stop=None, stake=5.0):
    """P&L for a single trade with an optional take-profit / stop exit.

    Exits are executed at the BID (taker), with a depth check: if the book
    does not show enough size at the exit price the exit does NOT happen and
    the trade is held to settlement.  The exit fee is charged on the shares
    actually sold.
    """
    bids = d[f"{side}_bid"]
    bszs = d[f"{side}_bidsz"]
    n = min(len(bids), len(bszs))
    sh = stake / ask
    entry_fee = B.fee(ask, sh)
    for j in range(ei + 1, n):
        b = bids[j]
        if not np.isfinite(b) or b <= 0:
            continue
        bsz = bszs[j]
        hit = ((tp is not None and b >= tp) or (stop is not None and b <= stop))
        if not hit:
            continue
        if np.isfinite(bsz) and bsz * b >= stake:
            return sh * b - stake - entry_fee - B.fee(b, sh)
        # not enough size at the exit price -> keep holding
        break
    won = 1 if d["_winner"] == side else 0
    return sh * won - stake - entry_fee


def simulate(W, sel_rows, tp=None, stop=None, stake=5.0):
    """Re-simulate the shipping rule's exits over all selected trades."""
    pnl = []
    for d, side, ask, ei in sel_rows:
        pnl.append(simulate_one(d, side, ask, ei, tp=tp, stop=stop, stake=stake))
    return np.array(pnl)


def main():
    W = B.load_all()
    rows = B.build_rows(W)
    sel = B.first_per_window([r for r in rows
                              if r["T"] >= B.SHIP_T_MIN and r["mkt"] >= 0.5
                              and B.SHIP_BAND[0] <= r["ask"] < B.SHIP_BAND[1]])
    # reconstruct the entry frames so we can walk the book forward
    sel_rows = []
    for r in sel:
        d = W[r["wi"]]
        sel_rows.append((d, r["side"], r["ask"], r["i"]))

    print(f"shipping rule: {len(sel_rows)} trades")
    ask = np.array([r["ask"] for r in sel])
    won = np.array([r["won"] for r in sel])
    wi = [r["wi"] for r in sel]

    p_hold, _ = B.pnl_sized(ask, won)
    _, t, _ = B.cluster_t(wi, p_hold)
    print(f"  hold to settlement (sized) : net ${p_hold.sum():+.2f}  t {t:+.2f}  "
          f"worst ${p_hold.min():+.2f}")

    for stake in (5.0, 10.0):
        ph, _ = B.pnl_sized(ask, won, stake=stake, stake_cheap=stake * 3.0)
        _, th, _ = B.cluster_t(wi, ph)
        print(f"  SIZED ${stake:.0f} hold         : net ${ph.sum():+.2f}  "
              f"t {th:+.2f}  worst ${ph.min():+.2f}")
        for tp in (0.95, 0.97):
            p = simulate(W, sel_rows, tp=tp, stake=stake)
            _, t2, _ = B.cluster_t(wi, p)
            print(f"  flat  ${stake:.0f} TP {tp:.2f}          : net ${p.sum():+.2f}  "
                  f"t {t2:+.2f}  worst ${p.min():+.2f}  n={len(p)}")
        for stop in (0.60, 0.80):
            p = simulate(W, sel_rows, stop=stop, stake=stake)
            _, t2, _ = B.cluster_t(wi, p)
            print(f"  flat  ${stake:.0f} STOP {stop:.2f}        : net ${p.sum():+.2f}  "
                  f"t {t2:+.2f}  worst ${p.min():+.2f}  n={len(p)}")

    # the SHIPPED configuration uses the conditional sizing overlay, so the
    # exit study has to be repeated with the overlay applied per trade
    print()
    for stake in (5.0, 10.0):
        for tp in (None, 0.95):
            p = []
            for (d, side, a, ei), r in zip(sel_rows, sel):
                st = B.stake_for_ask(a) if stake == 5.0 else (
                    stake * 3.0 if a < 0.85 else stake)
                p.append(simulate_one(d, side, a, ei, tp=tp, stake=st))
            p = np.array(p)
            _, t2, _ = B.cluster_t(wi, p)
            lbl = "hold    " if tp is None else f"TP {tp:.2f} "
            print(f"  SIZED ${stake:.0f} {lbl}          : net ${p.sum():+.2f}  "
                  f"t {t2:+.2f}  worst ${p.min():+.2f}  n={len(p)}")


if __name__ == "__main__":
    main()
