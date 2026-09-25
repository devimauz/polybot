"""Measure the twap_bot's realized P&L from the Polysimulator account.

Usage:
  python twap_pnl.py --baseline 9826.30 [--filter btc-updown-5m-1789...]

Reads /v1/account/balance and /v1/account/positions and /v1/orders, and prints
realized + unrealized P&L, win/loss record, and per-trade detail for the
up/down markets this bot trades.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
API = "https://api.polysimulator.com/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=10000.0)
    ap.add_argument("--tail", action="store_true", help="show only recent trades")
    args = ap.parse_args()

    key = os.environ["POLYSIM_API_KEY"].strip()
    c = httpx.Client(timeout=15, headers={"X-API-Key": key})
    bal = c.get(API + "/account/balance").json()
    print(f"=== Balance ===\n  balance={bal['balance']}  starting={bal['starting_balance']}  "
          f"unrealized={bal['unrealized_pnl']}  total_value={bal['total_value']}")

    # recent updown trades
    orders = c.get(API + "/orders", params={"limit": 200}).json()
    rows = orders if isinstance(orders, list) else orders.get("orders", [])
    trades = [o for o in rows if "updown-5m" in str(o.get("market_slug", "")) or
              "updown" in str(o.get("market_question", ""))]
    print(f"\n=== Recent up/down orders ({len(trades)}) ===")
    realized = 0.0
    for o in trades[:30]:
        st = o.get("status")
        pnl = o.get("realized_pnl", o.get("pnl"))
        print(f"  {str(o.get('created_at'))[:19]}  {o.get('market_slug','?')[:28]:28} "
              f"{o.get('side')} {o.get('outcome')} qty={o.get('quantity')} "
              f"price={o.get('price')} status={st} pnl={pnl}")

    # positions (open + recently closed)
    pos = c.get(API + "/account/positions", params={"limit": 300}).json()
    items = pos if isinstance(pos, list) else pos.get("positions", [])
    updown = [p for p in items if "Up or Down" in str(p.get("market_question", ""))]
    open_p = [p for p in updown if p.get("status") == "OPEN" and float(p.get("quantity") or 0) > 0]
    closed = [p for p in updown if p.get("status") == "CLOSED"]
    print(f"\n=== Positions (up/down) ===  open={len(open_p)}  closed_recent={len(closed)}")
    up = down = 0
    win_amt = loss_amt = 0.0
    for p in closed:
        q = float(p.get("quantity") or 0)
        if q <= 0:
            continue
        entry = float(p.get("avg_entry_price") or 0)
        outcome = p.get("outcome")
        # closed up/down: payout is $1 if won, $0 if lost
        qty = q
        if "CLOSED" in str(p.get("status")):
            pass
        # infer win/loss from avg entry vs payout: pnl = q*(1-entry) if win else -q*entry
        # The API may give us realized_pnl on the position; else infer
        rp = p.get("realized_pnl")
        if rp is None:
            # heuristic: if current_price is 0 -> lost, 1 -> won
            cp = p.get("current_price")
            won = str(cp) in ("1", "1.0", "1.00")
        else:
            won = float(rp) > 0
        if won:
            up += 1
            win_amt += abs(float(rp or 0))
        else:
            down += 1
            loss_amt += abs(float(rp or 0))
    if up + down:
        print(f"  recent closed: {up} wins / {down} losses, "
              f"win$={win_amt:.2f} loss$={loss_amt:.2f}")
    print(f"\n=== Net vs baseline {args.baseline} ===")
    print(f"  total_value {bal['total_value']} -> "
          f"{'+' if float(bal['total_value'])-args.baseline>=0 else ''}"
          f"{float(bal['total_value'])-args.baseline:.2f}")


if __name__ == "__main__":
    main()
