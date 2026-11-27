"""Reconcile momentum-bot journal fills with actual CLOB response amounts.

This is a read-only audit.  For a BUY, ``makingAmount`` is the USDC notional
and ``takingAmount`` is the matched share quantity.  Platform taker fees are
charged on top of the notional, so resolved P&L is calculated from those
actual amounts rather than the quote cached in the journal's ``signal``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                yield value


def winners(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in directory.glob("btc-updown-5m-*.jsonl"):
        for row in rows(path):
            if row.get("event") == "resolution" and row.get("winner") in ("Up", "Down"):
                result[str(row.get("slug") or path.stem)] = row["winner"]
    return result


def timestamp(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def crypto_fee(price: float, shares: float) -> float:
    return round(shares * 0.07 * price * (1.0 - price), 5)


def audit(journal: Path, resolutions: dict[str, str]):
    attempts: dict[str, dict] = {}
    fills = []
    fok_failures = 0
    for row in rows(journal):
        event = row.get("event")
        slug = str(row.get("slug") or "")
        if event == "order_attempt":
            attempts[slug] = row
            continue
        detail = str(row.get("detail") or row.get("reason") or "").lower()
        if "fully filled" in detail or "fok" in detail:
            fok_failures += 1
        if event != "fill":
            continue
        response = row.get("response") or {}
        paid = number(response.get("makingAmount"))
        shares = number(response.get("takingAmount"))
        if paid is None or shares is None or paid <= 0 or shares <= 0:
            continue
        signal = row.get("signal") or {}
        side = signal.get("side") or row.get("side")
        winner = resolutions.get(slug)
        actual_price = paid / shares
        fee = crypto_fee(actual_price, shares)
        actual_pnl = None
        if winner in ("Up", "Down") and side in ("Up", "Down"):
            actual_pnl = (shares if side == winner else 0.0) - paid - fee

        quote_price = number(signal.get("avg_price"))
        quote_shares = number(signal.get("shares"))
        quote_cost = number(signal.get("cost"))
        quote_pnl = None
        if (
            winner in ("Up", "Down")
            and side in ("Up", "Down")
            and quote_shares is not None
            and quote_cost is not None
        ):
            quote_pnl = (quote_shares if side == winner else 0.0) - quote_cost

        response_ms = None
        attempt = attempts.get(slug)
        if attempt:
            start, end = timestamp(attempt.get("ts")), timestamp(row.get("ts"))
            if start is not None and end is not None:
                response_ms = (end - start).total_seconds() * 1000.0
        fills.append({
            "slug": slug,
            "side": side,
            "winner": winner,
            "paid": paid,
            "shares": shares,
            "fee": fee,
            "quote_price": quote_price,
            "move_bps": number(signal.get("move_bps")),
            "actual_price": actual_price,
            "slip_cents": (
                100.0 * (actual_price - quote_price)
                if quote_price is not None else None
            ),
            "response_ms": response_ms,
            "actual_pnl": actual_pnl,
            "quote_pnl": quote_pnl,
        })
    return fills, fok_failures


def fmt_money(value: float) -> str:
    return f"${value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal", type=Path,
        default=ROOT / "price_logs" / "momentum-polymarket" / "live.jsonl",
    )
    parser.add_argument(
        "--logs", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    args = parser.parse_args()

    fills, fok_failures = audit(args.journal, winners(args.logs))
    resolved = [row for row in fills if row["actual_pnl"] is not None]
    slips = [row["slip_cents"] for row in fills if row["slip_cents"] is not None]
    response_times = [row["response_ms"] for row in fills if row["response_ms"] is not None]
    actual_pnl = sum(row["actual_pnl"] for row in resolved)
    quoted_pnl = sum(row["quote_pnl"] for row in resolved if row["quote_pnl"] is not None)
    print(
        f"fills={len(fills)} resolved={len(resolved)} "
        f"wins={sum(row['side'] == row['winner'] for row in resolved)} "
        f"losses={sum(row['side'] != row['winner'] for row in resolved)} "
        f"FOK_fail_events={fok_failures}"
    )
    print(
        f"actual_response_amount_PnL={fmt_money(actual_pnl)} "
        f"journal_quote_PnL={fmt_money(quoted_pnl)} "
        f"difference={fmt_money(actual_pnl - quoted_pnl)} "
        f"fees=${sum(row['fee'] for row in resolved):.4f}"
    )
    if slips:
        print(
            f"fill-vs-quote slippage cents: mean={statistics.mean(slips):+.3f} "
            f"median={statistics.median(slips):+.3f} "
            f"best={min(slips):+.3f} worst={max(slips):+.3f}"
        )
    if response_times:
        print(
            f"POST-response time ms (not exchange fill latency): "
            f"mean={statistics.mean(response_times):.1f} "
            f"median={statistics.median(response_times):.1f} "
            f"max={max(response_times):.1f}"
        )
    print("\nPost-hoc diagnostics only (not strategy validation)")
    for cap in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        chosen = [
            row for row in resolved
            if row["quote_price"] is not None and row["quote_price"] <= cap
        ]
        if chosen:
            print(
                f"  quote <= {cap:.2f}: n={len(chosen):2d} "
                f"wins={sum(row['side'] == row['winner'] for row in chosen):2d} "
                f"pnl={fmt_money(sum(row['actual_pnl'] for row in chosen))}"
            )
    for threshold in (2.0, 3.0, 4.0, 5.0, 6.0):
        chosen = [
            row for row in resolved
            if row["move_bps"] is not None and abs(row["move_bps"]) >= threshold
        ]
        if chosen:
            print(
                f"  |move| >= {threshold:.0f} bps: n={len(chosen):2d} "
                f"wins={sum(row['side'] == row['winner'] for row in chosen):2d} "
                f"pnl={fmt_money(sum(row['actual_pnl'] for row in chosen))}"
            )
    print("\nWorst resolved fills by actual P&L")
    for row in sorted(resolved, key=lambda item: item["actual_pnl"])[:10]:
        slip = row["slip_cents"]
        print(
            f"  {row['slug']} {row['side']:>4} winner={row['winner']:<4} "
            f"quote={row['quote_price']!s:<6} actual={row['actual_price']:.4f} "
            f"slip={slip:+.2f}c pnl={fmt_money(row['actual_pnl'])}"
        )


if __name__ == "__main__":
    main()
