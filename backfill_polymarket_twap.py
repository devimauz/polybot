"""Backfill exact Polymarket Chainlink-TWAP open/close prices.

The raw high-frequency JSONL logs are never modified.  This script reads their
resolved BTC five-minute slugs and writes a separate, restartable JSON cache.
Polymarket's web application uses the same public endpoint to display its
``Price to Beat`` and final price.

Example:
    python backfill_polymarket_twap.py price_logs/polymarket
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import httpx


ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://polymarket.com/api/crypto/crypto-price"
DEFAULT_OUTPUT = ROOT / "price_logs" / "polymarket_chainlink_twap.json"


def resolved_winner(path: Path) -> str | None:
    winner = None
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if '"event":"resolution"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row.get("winner") in ("Up", "Down"):
                    winner = row["winner"]
    except OSError:
        return None
    return winner


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finite_price(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def fetch_one(
    start: int,
    winner: str,
    client: httpx.Client,
    retries: int,
) -> dict:
    params = {
        "symbol": "BTC",
        "eventStartTime": iso(start),
        "variant": "fiveminute",
        "endDate": iso(start + 300),
        "twapEnabled": "true",
        "twapLookbackSeconds": "60",
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.get(ENDPOINT, params=params)
            response.raise_for_status()
            payload = response.json()
            opening = finite_price(payload.get("openPrice"))
            closing = finite_price(payload.get("closePrice"))
            if opening is None or closing is None:
                raise ValueError(f"missing TWAP price: {payload!r}")
            inferred = "Up" if closing >= opening else "Down"
            return {
                "slug": f"btc-updown-5m-{start}",
                "window_start": start,
                "window_end": start + 300,
                "open_price": opening,
                "close_price": closing,
                "winner": winner,
                "inferred_winner": inferred,
                "winner_matches": inferred == winner,
                "completed": bool(payload.get("completed")),
                "incomplete": bool(payload.get("incomplete")),
                "source": ENDPOINT,
                "twap_lookback_seconds": 60,
            }
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(4.0, (0.4 * 2**attempt) + random.random() * 0.2))
    raise RuntimeError(f"{start}: {type(last_error).__name__}: {last_error}")


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rows = raw.get("windows") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return {}
    return {
        row["slug"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("slug"), str)
    }


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(cache.values(), key=lambda row: int(row["window_start"]))
    matches = sum(row.get("winner_matches") is True for row in rows)
    payload = {
        "schema": "polymarket-chainlink-twap-window-v1",
        "source": ENDPOINT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "winner_matches": matches,
        "windows": rows,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--through", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("--workers must be between 1 and 16")

    targets: list[tuple[int, str]] = []
    for path in sorted(args.directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if args.through is not None and start > args.through:
            continue
        winner = resolved_winner(path)
        if winner is not None:
            targets.append((start, winner))

    cache = {} if args.refresh else load_cache(args.output)
    missing = [
        item for item in targets
        if f"btc-updown-5m-{item[0]}" not in cache
    ]
    print(
        f"resolved={len(targets)} cached={len(targets)-len(missing)} "
        f"to_fetch={len(missing)} workers={args.workers}",
        flush=True,
    )
    failures = []
    completed = 0
    with httpx.Client(
        timeout=args.timeout,
        headers={"User-Agent": "btc5m-chainlink-backfill/1.0"},
        follow_redirects=True,
        limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers),
    ) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_one, start, winner, client, args.retries): start
                for start, winner in missing
            }
            for future in as_completed(futures):
                start = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # keep other independent windows progressing
                    failures.append(str(exc))
                else:
                    cache[row["slug"]] = row
                    completed += 1
                    if completed % 10 == 0:
                        save_cache(args.output, cache)
                        print(f"fetched={completed}/{len(missing)}", flush=True)
    save_cache(args.output, cache)

    selected = [cache.get(f"btc-updown-5m-{start}") for start, _ in targets]
    selected = [row for row in selected if row is not None]
    mismatches = [row for row in selected if not row.get("winner_matches")]
    print(
        f"saved={len(selected)}/{len(targets)} output={args.output.resolve()} "
        f"winner_mismatches={len(mismatches)} failures={len(failures)}"
    )
    for row in mismatches[:10]:
        print(
            f"MISMATCH {row['slug']} recorded={row['winner']} "
            f"prices={row['open_price']:.8f}->{row['close_price']:.8f}"
        )
    for error in failures[:10]:
        print(f"ERROR {error}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
