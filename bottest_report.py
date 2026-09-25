"""Report logged fills against authoritative settlement events, without trading."""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


def number(value):
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def group_label(row, group_by):
    if group_by == "strategy":
        settings = row.get("settings", {})
        return row.get("strategy_version", "legacy") + " " + json.dumps(settings, sort_keys=True, separators=(",", ":"))
    if group_by == "entry-time":
        try:
            elapsed = datetime.fromisoformat(row["timestamp"]).timestamp() - int(row["market"].rsplit("-", 1)[1])
            return "<60s" if elapsed < 60 else "60-120s" if elapsed < 120 else "120-240s" if elapsed < 240 else ">=240s"
        except (KeyError, TypeError, ValueError):
            return "unknown"
    return str(row.get("asset", "unknown"))


def summarize(rows, group_by="asset"):
    winners, conflicting = {}, set()
    for row in rows:
        if row.get("event") == "market_resolved" and row.get("winner") in ("Up", "Down"):
            market = row.get("market")
            if market in winners and winners[market] != row["winner"]:
                conflicting.add(market)
            winners[market] = row["winner"]
    for market in conflicting:
        winners.pop(market, None)
    groups = defaultdict(lambda: {"fills": 0, "resolved": 0, "wins": 0, "gross": Decimal(0),
                                  "priced": 0, "net": Decimal(0), "with_fees": 0})
    seen = set()
    for row in rows:
        if row.get("event") != "buy_matched" or row.get("mode") != "live":
            continue
        key = row.get("order_id") or (row.get("market"), row.get("outcome"), row.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        group = groups[group_label(row, group_by)]
        group["fills"] += 1
        winner = winners.get(row.get("market"))
        if winner is None:
            continue
        group["resolved"] += 1
        won = row.get("outcome") == winner
        group["wins"] += int(won)
        paid, shares = number(row.get("paid")), number(row.get("shares"))
        if paid is not None and shares is not None and paid > 0 and shares > 0:
            gross = (shares if won else Decimal(0)) - paid
            group["gross"] += gross
            group["priced"] += 1
            fee = number(row.get("fees_paid"))
            if fee is not None and fee >= 0:
                group["net"] += gross - fee
                group["with_fees"] += 1
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent / "price_logs")
    parser.add_argument("--group-by", choices=("asset", "entry-time", "strategy"), default="asset")
    args = parser.parse_args()
    rows, skipped = [], 0
    for path in sorted(args.directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected object")
                    rows.append(row)
                except ValueError:
                    skipped += 1
    groups = summarize(rows, args.group_by)
    print("Group | Fills | Resolved | Wins | Win rate | Gross P&L (priced fills) | Net P&L (fills with fees)")
    for name, group in sorted(groups.items()):
        rate = f"{100 * group['wins'] / group['resolved']:.1f}%" if group["resolved"] else "unknown"
        gross = f"${group['gross']:.2f} ({group['priced']})" if group["priced"] else "unknown"
        net = f"${group['net']:.2f} ({group['with_fees']})" if group["with_fees"] else "unknown"
        print(f"{name} | {group['fills']} | {group['resolved']} | {group['wins']} | {rate} | {gross} | {net}")
    if not groups:
        print("No logged live fills found.")
    print(f"Malformed records skipped: {skipped}")
    print("Missing settlement data is not a loss. P&L assumes holding to resolution and excludes external sells.")
    print("Legacy logs lack post-entry quotes and may lack paid/share amounts; they cannot establish an optimal entry rule.")


if __name__ == "__main__":
    main()
