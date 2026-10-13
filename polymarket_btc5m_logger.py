"""Log BTC 5-minute markets directly from Polymarket's public APIs.

Market metadata comes from Gamma and order books/resolutions come from the
Polymarket CLOB. No wallet, API key, or Polysimulator account is required.

Examples:
    python polymarket_btc5m_logger.py
    python polymarket_btc5m_logger.py --poll 0.5 --full-book
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import websocket


ROOT = Path(__file__).resolve().parent
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
WINDOW_SECONDS = 300
LOG = logging.getLogger("polymarket-btc5m-logger")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_list(value: Any) -> list:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, list) else []


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_market(row: dict, expected_slug: str, start: int) -> dict | None:
    """Validate a Gamma row and return the fields needed by the logger."""
    if row.get("slug") != expected_slug:
        return None
    outcomes = decode_list(row.get("outcomes"))
    token_ids = decode_list(row.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2 or set(outcomes) != {"Up", "Down"}:
        raise ValueError(f"unexpected outcome/token mapping for {expected_slug}")
    return {
        "slug": expected_slug,
        "start": start,
        "end": start + WINDOW_SECONDS,
        "condition_id": row.get("conditionId"),
        "question_id": row.get("questionID"),
        "tokens": dict(zip(outcomes, map(str, token_ids))),
        "active": bool(row.get("active")),
        "closed": bool(row.get("closed")),
        "accepting_orders": bool(row.get("acceptingOrders")),
    }


def summarize_book(book: dict, full_book: bool = False) -> dict:
    """Normalize a CLOB book and calculate best prices and sizes."""
    sides: dict[str, list[dict[str, float]]] = {}
    for side in ("bids", "asks"):
        levels = []
        for item in book.get(side, []):
            price = finite_number(item.get("price"))
            size = finite_number(item.get("size"))
            if price is not None and size is not None and 0 <= price <= 1 and size > 0:
                levels.append({"price": price, "size": size})
        levels.sort(key=lambda x: x["price"], reverse=(side == "bids"))
        sides[side] = levels

    bid = sides["bids"][0] if sides["bids"] else None
    ask = sides["asks"][0] if sides["asks"] else None
    result = {
        "bid": bid["price"] if bid else None,
        "bid_size": bid["size"] if bid else None,
        "ask": ask["price"] if ask else None,
        "ask_size": ask["size"] if ask else None,
        "mid": (bid["price"] + ask["price"]) / 2 if bid and ask else None,
        "spread": ask["price"] - bid["price"] if bid and ask else None,
        "book_timestamp": book.get("timestamp"),
        "hash": book.get("hash"),
    }
    if full_book:
        result["bids"] = sides["bids"]
        result["asks"] = sides["asks"]
    return result


class JsonlWriter:
    """Route each record to btc-updown-5m-START.jsonl."""

    def __init__(self, directory: Path):
        # Accept the old --output ...jsonl form by using its parent directory.
        self.directory = directory.parent if directory.suffix.lower() == ".jsonl" else directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path: Path | None = None
        self.stream = None

    def path_for(self, slug: str) -> Path:
        return self.directory / f"{slug}.jsonl"

    def write(self, row: dict) -> None:
        slug = row.get("slug") or row.get("window")
        if not isinstance(slug, str) or not slug.startswith("btc-updown-5m-"):
            raise ValueError("log record is missing a valid BTC 5-minute slug")
        path = self.path_for(slug)
        encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
        if path == self.path and self.stream is not None:
            self.stream.write(encoded)
            return
        # A resolution may arrive for an older window. Append it without
        # rotating the active high-frequency stream away from the current file.
        if self.path is not None and path != self.path and row.get("event") == "resolution":
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            return
        if self.stream is not None:
            self.stream.close()
        self.path = path
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.stream.write(encoded)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


class CoinbaseFeed:
    """Keep the latest public Coinbase Exchange BTC-USD ticker in memory."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.data = {"product": "BTC-USD", "price": None, "exchange_ts": None,
                     "received_ts": None, "age_ms": None}
        self.thread = threading.Thread(target=self._run, name="coinbase-btc-feed", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def handle(self, message: dict, received: float | None = None) -> bool:
        if message.get("type") != "ticker" or message.get("product_id") != "BTC-USD":
            return False
        price = finite_number(message.get("price"))
        if price is None or price <= 0:
            return False
        received = time.time() if received is None else received
        with self.lock:
            self.data = {"product": "BTC-USD", "price": price,
                         "exchange_ts": message.get("time"),
                         "received_ts": received, "age_ms": 0.0}
        return True

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            result = dict(self.data)
        received = result.get("received_ts")
        result["age_ms"] = round(max(0.0, now - received) * 1000, 3) if received else None
        return result

    def _run(self) -> None:
        while not self.stop.is_set():
            sock = None
            try:
                sock = websocket.create_connection(COINBASE_WS, timeout=5)
                sock.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"],
                                      "channels": ["ticker"]}))
                sock.settimeout(1)
                while not self.stop.is_set():
                    try:
                        raw = sock.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if raw:
                        self.handle(json.loads(raw))
            except Exception as exc:
                if not self.stop.is_set():
                    LOG.warning("Coinbase feed reconnecting (%s)", type(exc).__name__)
                    self.stop.wait(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


class Logger:
    def __init__(self, client: httpx.Client, writer: JsonlWriter, full_book: bool,
                 coinbase: CoinbaseFeed | None = None):
        self.client = client
        self.writer = writer
        self.full_book = full_book
        self.coinbase = coinbase
        self.market: dict | None = None
        self.pending_resolutions: dict[str, str] = {}
        self.last_resolution_check = 0.0

    def get_json(self, url: str, **params) -> Any:
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def discover(self, now: float) -> dict | None:
        start = int(now) // WINDOW_SECONDS * WINDOW_SECONDS
        slug = f"btc-updown-5m-{start}"
        if self.market and self.market["slug"] == slug:
            return self.market
        if self.market and self.market.get("condition_id"):
            self.pending_resolutions[self.market["slug"]] = self.market["condition_id"]
        rows = self.get_json(GAMMA_MARKETS, slug=slug)
        for row in rows if isinstance(rows, list) else []:
            market = parse_market(row, slug, start)
            if market:
                self.market = market
                self.writer.write({"ts": utc_now(), "event": "market", **market,
                                   "source": "polymarket"})
                LOG.info("market %s", slug)
                return market
        self.market = None
        return None

    def snapshot(self, market: dict) -> None:
        received = time.time()
        books = {}
        for outcome, token_id in market["tokens"].items():
            raw = self.get_json(f"{CLOB}/book", token_id=token_id)
            books[outcome] = summarize_book(raw, self.full_book)
        row = {
            "ts": datetime.fromtimestamp(received, timezone.utc).isoformat(),
            "ts_epoch": received,
            "event": "snapshot",
            "source": "polymarket",
            "slug": market["slug"],
            "window_start": market["start"],
            "window_end": market["end"],
            "seconds_into_window": round(received - market["start"], 3),
            "condition_id": market["condition_id"],
            "tokens": market["tokens"],
            "Up": books["Up"],
            "Down": books["Down"],
        }
        if self.coinbase:
            row["coinbase"] = self.coinbase.snapshot(received)
        self.writer.write(row)

    def check_resolutions(self, now: float) -> None:
        if now - self.last_resolution_check < 30:
            return
        self.last_resolution_check = now
        for slug, condition_id in list(self.pending_resolutions.items()):
            data = self.get_json(f"{CLOB}/markets/{condition_id}")
            winners = [token.get("outcome") for token in data.get("tokens", [])
                       if token.get("winner") is True]
            if len(winners) == 1:
                self.writer.write({"ts": utc_now(), "event": "resolution",
                                   "source": "polymarket", "slug": slug,
                                   "condition_id": condition_id, "winner": winners[0]})
                del self.pending_resolutions[slug]
                LOG.info("resolved %s: %s", slug, winners[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between snapshots (default: 1.0)")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path,
                        default=ROOT / "price_logs" / "polymarket",
                        help="directory for one JSONL file per five-minute window")
    parser.add_argument("--full-book", action="store_true",
                        help="store every order-book level, not only the best bid/ask")
    parser.add_argument("--once", action="store_true", help="write one snapshot and exit")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not math.isfinite(args.poll) or args.poll <= 0:
        raise SystemExit("--poll must be greater than zero")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    writer = JsonlWriter(args.output_dir.resolve())
    coinbase = CoinbaseFeed()
    coinbase.start()
    LOG.info("writing one direct Polymarket log per window to %s", writer.directory)
    try:
        with httpx.Client(timeout=args.timeout,
                          headers={"User-Agent": "btc5m-research-logger/1.0"}) as client:
            logger = Logger(client, writer, args.full_book, coinbase)
            while True:
                loop_started = time.monotonic()
                try:
                    market = logger.discover(time.time())
                    if market:
                        logger.snapshot(market)
                    logger.check_resolutions(time.time())
                    if args.once and market:
                        break
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    LOG.warning("read failed (%s): %s", type(exc).__name__, exc)
                    if args.once:
                        raise SystemExit(1) from exc
                delay = max(0.05, args.poll - (time.monotonic() - loop_started))
                time.sleep(delay)
    except KeyboardInterrupt:
        LOG.info("stopped")
    finally:
        coinbase.close()
        writer.close()


if __name__ == "__main__":
    main()
