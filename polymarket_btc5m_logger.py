"""Log BTC 5-minute markets directly from Polymarket's public feeds.

Market metadata comes from Gamma, order-book updates come from Polymarket's
public market WebSocket, and resolutions come from the Polymarket CLOB. No
Polysimulator account is used. CLOB API credentials enable the preferred
authenticated PolyBolt settlement feed; without them the logger warns and
falls back to deprecated public RTDS.

Examples:
    python polymarket_btc5m_logger.py
    python polymarket_btc5m_logger.py --poll 0.5 --full-book
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import websocket
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
POLYMARKET_RTDS_WS = "wss://ws-live-data.polymarket.com"
POLYMARKET_POLYBOLT_WS = "wss://ws-live-v2.polymarket.com/ws"
POLYMARKET_MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CHAINLINK_TWAP_TOPIC = "crypto_prices_twap_sixty"
POLYBOLT_TWAP_CHANNEL = "price.crypto.twap"
WINDOW_SECONDS = 300
DISCOVERY_RETRY_SECONDS = 5.0
RESOLUTION_POLL_SECONDS = 30.0
RESOLUTION_STATE_FILENAME = ".btc5m-pending-resolutions.json"
RESOLUTION_STATE_MAX_BYTES = 1_000_000
MARKET_WS_PING_SECONDS = 10.0
MARKET_WS_LIVENESS_SECONDS = 30.0
CHAINLINK_LEGACY_PING_SECONDS = 5.0
CHAINLINK_LIVENESS_SECONDS = 30.0
RETRY_MIN_SECONDS = 0.5
RETRY_MAX_SECONDS = 30.0
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


def btc5m_window_start(slug: Any) -> int | None:
    """Return a validated BTC 5-minute slug epoch, otherwise ``None``."""
    prefix = "btc-updown-5m-"
    if not isinstance(slug, str) or not slug.startswith(prefix):
        return None
    suffix = slug[len(prefix):]
    if not 1 <= len(suffix) <= 20 or not suffix.isdigit():
        return None
    start = int(suffix)
    if str(start) != suffix or start % WINDOW_SECONDS != 0:
        return None
    return start


def valid_condition_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and 0 < len(value) <= 256
        and value.isascii()
        and all(character.isalnum() or character in "-_" for character in value)
    )


def retry_delay_seconds(consecutive_failures: int, poll_seconds: float) -> float:
    """Return a bounded exponential retry delay without shortening ``poll``.

    The first failed attempt waits at least half a second, then doubles up to
    30 seconds.  A deliberately slower normal polling interval remains the
    lower bound.
    """
    normal_delay = max(0.05, poll_seconds)
    if consecutive_failures <= 0:
        return normal_delay
    exponent = min(consecutive_failures - 1, 10)
    backoff = min(RETRY_MAX_SECONDS, RETRY_MIN_SECONDS * (2 ** exponent))
    return max(normal_delay, backoff)


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


class ChainlinkTwapFeed:
    """Keep Polymarket's BTC/USD 60-second settlement TWAP in memory.

    BTC five-minute markets resolve from this stream, not from Coinbase spot.
    Authenticated PolyBolt is preferred.  Legacy public RTDS remains a fallback
    so logging can still start before CLOB API credentials have been exported.
    """

    def __init__(self, api_key: str | None = None,
                 api_secret: str | None = None,
                 api_passphrase: str | None = None):
        credentials = tuple(
            value.strip() if isinstance(value, str) else ""
            for value in (api_key, api_secret, api_passphrase)
        )
        if any(credentials) and not all(credentials):
            raise ValueError(
                "PolyBolt requires POLY_API_KEY, POLY_API_SECRET, and "
                "POLY_API_PASSPHRASE together"
            )
        self.api_key, self.api_secret, self.api_passphrase = credentials
        self.use_polybolt = all(credentials)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.data = {
            "source": "polymarket_polybolt" if self.use_polybolt else "polymarket_rtds",
            "topic": POLYBOLT_TWAP_CHANNEL if self.use_polybolt else CHAINLINK_TWAP_TOPIC,
            "symbol": "btcusd" if self.use_polybolt else "btc/usd",
            "window_seconds": 60,
            "price": None,
            "full_accuracy_value": None,
            "source_ts": None,
            "source_ts_epoch": None,
            "received_ts": None,
            "age_ms": None,
        }
        self.thread = threading.Thread(
            target=self._run, name="chainlink-btc-twap-feed", daemon=True
        )

    @staticmethod
    def subscription() -> dict:
        """Return the legacy RTDS subscription frame."""
        filters = json.dumps({"symbol": "btc/usd"}, separators=(",", ":"))
        return {
            "action": "subscribe",
            "subscriptions": [{
                "topic": CHAINLINK_TWAP_TOPIC,
                "type": "update",
                "filters": filters,
            }],
        }

    def polybolt_auth(self) -> dict:
        if not self.use_polybolt:
            raise ValueError("PolyBolt credentials are not configured")
        return {
            "op": "auth",
            "rid": "chainlink-auth",
            "auth": {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "passphrase": self.api_passphrase,
            },
        }

    @staticmethod
    def polybolt_subscription() -> dict:
        return {
            "op": "subscribe",
            "rid": "chainlink-twap",
            "subscriptions": [{
                "channel": POLYBOLT_TWAP_CHANNEL,
                "filter": {"symbol": "btcusd", "window_seconds": 60},
            }],
        }

    @staticmethod
    def _timestamp_seconds(value: Any) -> float | None:
        number = finite_number(value)
        if number is None or number <= 0:
            return None
        # RTDS currently sends milliseconds.  Repeated division also handles
        # micro/nanosecond variants without discarding the original value.
        while number > 10_000_000_000:
            number /= 1000.0
        return number

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def handle(self, message: dict, received: float | None = None) -> bool:
        polybolt = message.get("channel") == POLYBOLT_TWAP_CHANNEL
        legacy = message.get("topic") == CHAINLINK_TWAP_TOPIC
        if not polybolt and not legacy:
            return False
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return False
        symbol = str(payload.get("symbol") or "").lower()
        expected_symbol = "btcusd" if polybolt else "btc/usd"
        if symbol != expected_symbol:
            return False
        try:
            window_seconds = int(
                payload.get("window_seconds") if polybolt else payload.get("window_s") or 60
            )
        except (TypeError, ValueError):
            return False
        if window_seconds != 60:
            return False
        # PolyBolt may seed a subscription with a short history.  Keep only
        # the newest point; live updates use the scalar payload directly.
        point = payload
        if polybolt and isinstance(payload.get("data"), list):
            candidates = [
                item for item in payload["data"]
                if isinstance(item, dict)
                and finite_number(item.get("timestamp")) is not None
            ]
            if not candidates:
                return False
            point = max(candidates, key=lambda item: float(item["timestamp"]))
        price = finite_number(point.get("value"))
        if price is None:
            full_accuracy = finite_number(point.get("full_accuracy_value"))
            # Legacy RTDS uses an E18 integer; PolyBolt documents an ordinary
            # decimal string.  Applying the legacy scaling to PolyBolt would
            # silently turn a $70k BTC price into ~7e-14 dollars.
            if full_accuracy is not None:
                price = full_accuracy if polybolt else full_accuracy / 1e18
        if price is None or price <= 0:
            return False
        source_ts = point.get("timestamp", message.get("ts", message.get("timestamp")))
        received = time.time() if received is None else received
        with self.lock:
            self.data = {
                "source": "polymarket_polybolt" if polybolt else "polymarket_rtds",
                "topic": POLYBOLT_TWAP_CHANNEL if polybolt else CHAINLINK_TWAP_TOPIC,
                "symbol": symbol,
                "window_seconds": window_seconds,
                "price": price,
                "full_accuracy_value": point.get("full_accuracy_value"),
                "source_ts": source_ts,
                "source_ts_epoch": self._timestamp_seconds(source_ts),
                "received_ts": received,
                "age_ms": 0.0,
                "seq": message.get("seq") if polybolt else None,
                "dropped": message.get("dropped") if polybolt else None,
                "snapshot": bool(message.get("snapshot")) if polybolt else None,
            }
        return True

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.lock:
            result = dict(self.data)
        received = result.get("received_ts")
        result["age_ms"] = (
            round(max(0.0, now - received) * 1000, 3) if received else None
        )
        return result

    def _run(self) -> None:
        if self.use_polybolt:
            LOG.info("Chainlink TWAP source=authenticated PolyBolt")
        else:
            LOG.warning(
                "Chainlink TWAP source=legacy public RTDS; set POLY_API_KEY, "
                "POLY_API_SECRET, and POLY_API_PASSPHRASE for PolyBolt"
            )
        failure_streak = 0
        while not self.stop.is_set():
            sock = None
            try:
                endpoint = (
                    POLYMARKET_POLYBOLT_WS if self.use_polybolt else POLYMARKET_RTDS_WS
                )
                sock = websocket.create_connection(endpoint, timeout=5)
                if self.use_polybolt:
                    sock.send(json.dumps(self.polybolt_auth(), separators=(",", ":")))
                    subscribed = False
                else:
                    sock.send(json.dumps(self.subscription(), separators=(",", ":")))
                    subscribed = True
                sock.settimeout(1)
                last_traffic = last_ping = time.monotonic()
                while not self.stop.is_set():
                    now_monotonic = time.monotonic()
                    if now_monotonic - last_traffic >= CHAINLINK_LIVENESS_SECONDS:
                        raise TimeoutError(
                            "Chainlink TWAP WebSocket received no traffic for "
                            f"{CHAINLINK_LIVENESS_SECONDS:.0f}s"
                        )
                    if (not self.use_polybolt
                            and now_monotonic - last_ping >= CHAINLINK_LEGACY_PING_SECONDS):
                        sock.send("PING")
                        last_ping = now_monotonic
                    try:
                        raw = sock.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        # The RTDS endpoint emits an EMPTY keepalive frame right
                        # after the subscribe (and occasionally later). An empty
                        # frame is not a closed socket: a real closure surfaces as
                        # WebSocketConnectionClosedException from recv(), and a
                        # silent peer is caught by the liveness timeout above.
                        last_traffic = time.monotonic()
                        continue
                    last_traffic = time.monotonic()
                    if raw == "PING":
                        sock.send("PONG")
                        continue
                    if raw == "PONG":
                        continue
                    message = json.loads(raw)
                    if isinstance(message, dict):
                        if self.use_polybolt and message.get("op") == "error":
                            raise RuntimeError(
                                f"PolyBolt error code={message.get('code', 'unknown')}"
                            )
                        if (self.use_polybolt and message.get("op") == "authed"
                                and not subscribed):
                            sock.send(json.dumps(
                                self.polybolt_subscription(), separators=(",", ":")
                            ))
                            subscribed = True
                            continue
                        if message.get("message"):
                            raise RuntimeError(str(message["message"]))
                        if self.handle(message):
                            failure_streak = 0
            except Exception as exc:
                if not self.stop.is_set():
                    failure_streak += 1
                    delay = retry_delay_seconds(failure_streak, 1.0)
                    LOG.warning(
                        "Chainlink TWAP feed reconnecting in %.1fs (%s)",
                        delay,
                        type(exc).__name__,
                    )
                    self.stop.wait(delay)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


class PolymarketBookFeed:
    """Maintain full books for the current market from the public CLOB WS.

    A subscription starts with a complete ``book`` event for each asset.
    Subsequent ``price_change`` rows replace (or, for size zero, remove) one
    aggregate price level.  All mutations and snapshots share one lock, so a
    log row can never observe half of a multi-level update.

    Changing the two token IDs increments a generation and clears the cache
    synchronously.  The reader then reconnects and subscribes to the new
    generation; late packets belonging to the previous window are ignored.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self._ready_condition = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.changed = threading.Event()
        self.tokens: dict[str, str] = {}
        self.books: dict[str, dict[str, Any]] = {}
        self.generation = 0
        # ``generation`` identifies the selected market. ``session`` also
        # changes on a transport reconnect within that same market, preventing
        # packets decoded from an old socket from repopulating a fresh cache.
        self.session = 0
        self.thread = threading.Thread(
            target=self._run, name="polymarket-book-feed", daemon=True
        )
        self._started = False
        self._socket_lock = threading.Lock()
        self._socket = None

    @staticmethod
    def subscription(token_ids: list[str]) -> dict:
        return {
            "assets_ids": [str(token_id) for token_id in token_ids],
            "type": "market",
            # Enables the current batched price_change schema, including
            # best_bid/best_ask on each changed asset.
            "custom_feature_enabled": True,
        }

    def start(self) -> None:
        if not self._started:
            self._started = True
            self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.changed.set()
        with self._ready_condition:
            self._ready_condition.notify_all()
        # Interrupt recv/create cleanup promptly instead of waiting for its
        # timeout during Ctrl+C or process shutdown.
        with self._socket_lock:
            sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if self._started:
            self.thread.join(timeout=6)

    def subscribe(self, outcome_tokens: dict[str, str] | None) -> bool:
        """Select a market, returning True only when its token set changed."""
        token_map = {
            str(token_id): str(outcome)
            for outcome, token_id in (outcome_tokens or {}).items()
        }
        with self._ready_condition:
            if token_map == self.tokens:
                return False
            self.tokens = token_map
            self.books = {}
            self.generation += 1
            self.session += 1
            self._ready_condition.notify_all()
        self.changed.set()
        return True

    @staticmethod
    def _levels(rows: Any) -> dict[float, float]:
        levels: dict[float, float] = {}
        if not isinstance(rows, list):
            return levels
        for row in rows:
            if not isinstance(row, dict):
                continue
            price = finite_number(row.get("price"))
            size = finite_number(row.get("size"))
            if price is not None and size is not None and 0 <= price <= 1 and size > 0:
                levels[price] = size
        return levels

    def handle(self, message: Any, received: float | None = None) -> bool:
        """Apply decoded WS data to the current subscription.

        This public entry point is also useful for deterministic replay tests.
        The live reader supplies an explicit generation to reject packets that
        were decoded just as the logger switched five-minute windows.
        """
        with self.lock:
            generation = self.generation
            tokens = dict(self.tokens)
            session = self.session
        return self._handle(message, tokens, generation, session, received)

    def _handle(self, message: Any, tokens: dict[str, str], generation: int,
                session: int, received: float | None = None) -> bool:
        if isinstance(message, list):
            received = time.time() if received is None else received
            handled = False
            # Keep wire order. A delta following its initial book in the same
            # frame must be visible after, never before, that initial state.
            # Holding the re-entrant lock also prevents a snapshot from seeing
            # only part of one decoded WebSocket frame.
            with self.lock:
                if generation != self.generation or session != self.session:
                    return False
                for item in message:
                    handled = self._handle(
                        item, tokens, generation, session, received
                    ) or handled
            return handled
        if not isinstance(message, dict):
            return False

        event = message.get("event_type")
        if event is None and "bids" in message and "asks" in message:
            event = "book"
        received = time.time() if received is None else received

        if event == "book":
            token = str(message.get("asset_id", ""))
            if token not in tokens:
                return False
            book = {
                "bids": self._levels(message.get("bids")),
                "asks": self._levels(message.get("asks")),
                "timestamp": message.get("timestamp"),
                "hash": message.get("hash"),
                "received_ts": received,
            }
            with self._ready_condition:
                if (generation != self.generation or session != self.session
                        or token not in self.tokens):
                    return False
                self.books[token] = book
                self._ready_condition.notify_all()
            return True

        if event != "price_change":
            return False
        changes = message.get("price_changes")
        if not isinstance(changes, list):
            changes = [message]
        handled = False
        # One lock makes the complete batch atomic to snapshot().
        with self.lock:
            if generation != self.generation or session != self.session:
                return False
            for row in changes:
                if not isinstance(row, dict):
                    continue
                token = str(row.get("asset_id", ""))
                if token not in tokens or token not in self.tokens:
                    continue
                book = self.books.get(token)
                # Never build a deceptively partial book from a delta received
                # before the subscription's initial full-book image.
                if book is None:
                    continue
                side = str(row.get("side", "")).upper()
                side_name = "bids" if side == "BUY" else "asks" if side == "SELL" else None
                price = finite_number(row.get("price"))
                size = finite_number(row.get("size"))
                if side_name is None or price is None or size is None or not 0 <= price <= 1:
                    continue
                if size > 0:
                    book[side_name][price] = size
                elif size == 0:
                    book[side_name].pop(price, None)
                else:
                    continue
                book["timestamp"] = row.get("timestamp", message.get("timestamp"))
                book["hash"] = row.get("hash", message.get("hash", book.get("hash")))
                book["received_ts"] = received
                handled = True
        return handled

    def snapshot(self, full_book: bool = False, now: float | None = None) -> dict:
        """Return both outcomes, reusing the latest state between WS changes."""
        now = time.time() if now is None else now
        with self.lock:
            token_items = list(self.tokens.items())
            copied = {
                token: {
                    "bids": [
                        {"price": price, "size": size}
                        for price, size in book.get("bids", {}).items()
                    ],
                    "asks": [
                        {"price": price, "size": size}
                        for price, size in book.get("asks", {}).items()
                    ],
                    "timestamp": book.get("timestamp"),
                    "hash": book.get("hash"),
                    "received_ts": book.get("received_ts"),
                }
                for token, book in self.books.items()
            }

        result = {}
        for token, outcome in token_items:
            raw = copied.get(token, {})
            summary = summarize_book(raw, full_book)
            book_received = raw.get("received_ts")
            summary["book_received_ts"] = book_received
            summary["book_age_ms"] = (
                round(max(0.0, now - book_received) * 1000, 3)
                if book_received is not None else None
            )
            summary["book_ready"] = token in copied
            result[outcome] = summary
        return result

    @staticmethod
    def _token_map(outcome_tokens: dict[str, str] | None) -> dict[str, str]:
        return {
            str(token_id): str(outcome)
            for outcome, token_id in (outcome_tokens or {}).items()
        }

    def _books_ready_locked(self, outcome_tokens: dict[str, str]) -> bool:
        expected = self._token_map(outcome_tokens)
        return (
            len(expected) == 2
            and expected == self.tokens
            and all(token in self.books for token in expected)
        )

    def books_ready(self, outcome_tokens: dict[str, str]) -> bool:
        """Return whether every token has a full image for this exact market."""
        with self._ready_condition:
            return self._books_ready_locked(outcome_tokens)

    def wait_for_books(self, outcome_tokens: dict[str, str], timeout: float) -> bool:
        """Wait at most ``timeout`` seconds for both current full-book images."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._ready_condition:
            while not self._books_ready_locked(outcome_tokens):
                if self.stop.is_set():
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._ready_condition.wait(remaining)
            return True

    def wait_for_snapshot(self, outcome_tokens: dict[str, str], full_book: bool,
                          timeout: float) -> dict | None:
        """Atomically wait for and copy both current full-book images."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._ready_condition:
            while not self._books_ready_locked(outcome_tokens):
                if self.stop.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._ready_condition.wait(remaining)
            # RLock recursion keeps the readiness condition held until the
            # complete two-book copy is made, closing the reconnect TOCTOU.
            return self.snapshot(full_book, time.time())

    def _subscription_state(self) -> tuple[dict[str, str], int, int]:
        with self.lock:
            return dict(self.tokens), self.generation, self.session

    def _begin_session(self, generation: int) -> int | None:
        """Clear readiness and open a new socket session for one generation."""
        with self._ready_condition:
            if generation != self.generation:
                return None
            self.books = {}
            self.session += 1
            self._ready_condition.notify_all()
            return self.session

    def _invalidate_session(self, generation: int, session: int) -> bool:
        """Atomically make a failed socket's cache and late packets unusable."""
        with self._ready_condition:
            if generation != self.generation or session != self.session:
                return False
            self.books = {}
            self.session += 1
            self._ready_condition.notify_all()
            return True

    @staticmethod
    def traffic_timed_out(last_traffic: float, now: float,
                          timeout: float = MARKET_WS_LIVENESS_SECONDS) -> bool:
        """Return whether a connection has gone too long without inbound data."""
        return now - last_traffic >= timeout

    def _set_socket(self, sock) -> None:
        with self._socket_lock:
            self._socket = sock

    def _clear_socket(self, sock) -> None:
        with self._socket_lock:
            if self._socket is sock:
                self._socket = None

    def _run(self) -> None:
        failure_streak = 0
        while not self.stop.is_set():
            self.changed.clear()
            tokens, generation, _ = self._subscription_state()
            if not tokens:
                self.changed.wait(0.5)
                continue

            # A reconnect in the same five-minute market must not expose the
            # previous socket's book while this socket awaits its initial full
            # image. Do this before connecting/subscribing so snapshots become
            # unready for the entire reconnect attempt.
            session = self._begin_session(generation)
            if session is None:
                continue

            sock = None
            reconfigured = False
            try:
                sock = websocket.create_connection(POLYMARKET_MARKET_WS, timeout=5)
                self._set_socket(sock)
                sock.send(json.dumps(
                    self.subscription(list(tokens)), separators=(",", ":")
                ))
                sock.settimeout(0.25)
                last_ping = last_traffic = time.monotonic()
                while not self.stop.is_set():
                    _, current_generation, _ = self._subscription_state()
                    if current_generation != generation:
                        reconfigured = True
                        break
                    now_monotonic = time.monotonic()
                    if self.traffic_timed_out(last_traffic, now_monotonic):
                        raise TimeoutError(
                            "market WebSocket received no traffic for "
                            f"{MARKET_WS_LIVENESS_SECONDS:.0f}s"
                        )
                    if now_monotonic - last_ping >= MARKET_WS_PING_SECONDS:
                        sock.send("PING")
                        last_ping = now_monotonic
                    try:
                        raw = sock.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        # empty keepalive frame, not a closure (see the TWAP feed)
                        last_traffic = time.monotonic()
                        continue
                    # Any inbound frame proves that the peer-to-client path is
                    # live. In quiet books this will normally be a PONG.
                    last_traffic = time.monotonic()
                    if raw == "PING":
                        sock.send("PONG")
                        continue
                    if raw == "PONG":
                        continue
                    if self._handle(
                            json.loads(raw), tokens, generation, session):
                        failure_streak = 0
            except Exception as exc:
                active_session = self._invalidate_session(generation, session)
                if not self.stop.is_set() and not reconfigured and active_session:
                    failure_streak += 1
                    delay = retry_delay_seconds(failure_streak, 1.0)
                    LOG.warning(
                        "Polymarket book feed reconnecting in %.1fs (%s)",
                        delay,
                        type(exc).__name__,
                    )
                    self.changed.clear()
                    _, current_generation, _ = self._subscription_state()
                    if current_generation == generation:
                        self.changed.wait(delay)
            finally:
                if sock is not None:
                    self._clear_socket(sock)
                    try:
                        sock.close()
                    except Exception:
                        pass


class Logger:
    def __init__(self, client: httpx.Client, writer: JsonlWriter, full_book: bool,
                 coinbase: CoinbaseFeed | None = None,
                 chainlink_twap: ChainlinkTwapFeed | None = None,
                 market_books: PolymarketBookFeed | None = None,
                 resolution_client: httpx.Client | None = None):
        self.client = client
        self.writer = writer
        self.full_book = full_book
        self.coinbase = coinbase
        self.chainlink_twap = chainlink_twap
        self.market_books = market_books
        self.resolution_client = resolution_client or client
        self.market: dict | None = None
        self.resolution_state_path = self._resolution_state_path(writer)
        self.pending_resolutions = self._load_pending_resolutions()
        self.last_resolution_check = 0.0
        self.last_discovery_slug: str | None = None
        self.last_discovery_attempt = 0.0
        self._resolution_results: queue.SimpleQueue = queue.SimpleQueue()
        self._resolution_thread: threading.Thread | None = None
        self._resolution_stop = threading.Event()

    @staticmethod
    def _resolution_state_path(writer: Any) -> Path | None:
        """Return a sidecar path only for a writer with a real directory."""
        directory = getattr(writer, "directory", None)
        if not isinstance(directory, (str, os.PathLike)):
            return None
        return Path(directory) / RESOLUTION_STATE_FILENAME

    def _load_pending_resolutions(self) -> dict[str, str]:
        path = self.resolution_state_path
        if path is None:
            return {}
        try:
            if not path.exists():
                return {}
            if path.stat().st_size > RESOLUTION_STATE_MAX_BYTES:
                raise ValueError("sidecar exceeds size limit")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict)
                    or payload.get("version") != 1
                    or not isinstance(payload.get("pending"), dict)):
                raise ValueError("invalid sidecar schema")
            pending = {
                slug: condition_id
                for slug, condition_id in payload["pending"].items()
                if (btc5m_window_start(slug) is not None
                    and valid_condition_id(condition_id))
            }
            ignored = len(payload["pending"]) - len(pending)
            if ignored:
                LOG.warning(
                    "ignored %d invalid pending-resolution sidecar entries",
                    ignored,
                )
            return pending
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOG.warning(
                "ignoring invalid pending-resolution sidecar %s (%s): %s",
                path,
                type(exc).__name__,
                exc,
            )
            return {}

    def _persist_pending_resolutions(self) -> None:
        path = self.resolution_state_path
        if path is None:
            return
        payload = {
            "version": 1,
            "pending": dict(sorted(self.pending_resolutions.items())),
        }
        encoded = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ) + "\n"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _remember_resolution(self, slug: Any, condition_id: Any) -> bool:
        if btc5m_window_start(slug) is None or not valid_condition_id(condition_id):
            LOG.warning("cannot persist invalid resolution identity for %r", slug)
            return False
        previous = self.pending_resolutions.get(slug)
        if previous == condition_id:
            return False
        self.pending_resolutions[slug] = condition_id
        try:
            self._persist_pending_resolutions()
        except Exception:
            if previous is None:
                self.pending_resolutions.pop(slug, None)
            else:
                self.pending_resolutions[slug] = previous
            raise
        return True

    def _forget_resolution(self, slug: str, condition_id: str) -> bool:
        if self.pending_resolutions.get(slug) != condition_id:
            return False
        del self.pending_resolutions[slug]
        try:
            self._persist_pending_resolutions()
        except Exception:
            self.pending_resolutions[slug] = condition_id
            raise
        return True

    def get_json(self, url: str, **params) -> Any:
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_resolution_json(self, url: str) -> Any:
        response = self.resolution_client.get(url)
        response.raise_for_status()
        return response.json()

    def discover(self, now: float) -> dict | None:
        start = int(now) // WINDOW_SECONDS * WINDOW_SECONDS
        slug = f"btc-updown-5m-{start}"
        if self.market and self.market["slug"] == slug:
            return self.market
        if self.market and self.market.get("condition_id"):
            self._remember_resolution(
                self.market["slug"], self.market["condition_id"]
            )
        if self.market is not None:
            self.market = None
            if self.market_books:
                self.market_books.subscribe({})
        # Gamma is discovery, not a 100 ms data source. If a just-opened
        # market has not appeared yet, retry at a deliberately low rate.
        if (self.last_discovery_slug == slug
                and now - self.last_discovery_attempt < DISCOVERY_RETRY_SECONDS):
            return None
        self.last_discovery_slug = slug
        self.last_discovery_attempt = now
        rows = self.get_json(GAMMA_MARKETS, slug=slug)
        for row in rows if isinstance(rows, list) else []:
            market = parse_market(row, slug, start)
            if market:
                self.market = market
                self._remember_resolution(slug, market.get("condition_id"))
                if self.market_books:
                    self.market_books.subscribe(market["tokens"])
                self.writer.write({"ts": utc_now(), "event": "market", **market,
                                   "source": "polymarket"})
                LOG.info("market %s", slug)
                return market
        return None

    def snapshot(self, market: dict, books: dict | None = None) -> None:
        # Capture each feed first. The row timestamp is chosen only afterward,
        # so it can never precede the receipt time of data included in the row.
        capture_started = time.time()
        if books is None:
            books = (
                self.market_books.snapshot(self.full_book, capture_started)
                if self.market_books else {}
            )
        coinbase = self.coinbase.snapshot(capture_started) if self.coinbase else None
        chainlink_twap = (
            self.chainlink_twap.snapshot(capture_started)
            if self.chainlink_twap else None
        )
        # Preserve the stable Up/Down schema while the initial WebSocket book
        # images are in flight. This row is still valuable for Coinbase/TWAP
        # time alignment and will be marked book_ready=False.
        for outcome in ("Up", "Down"):
            if outcome not in books:
                empty = summarize_book({}, self.full_book)
                empty.update({
                    "book_received_ts": None,
                    "book_age_ms": None,
                    "book_ready": False,
                })
                books[outcome] = empty

        received_times = [time.time()]
        for outcome in ("Up", "Down"):
            book_received = finite_number(books[outcome].get("book_received_ts"))
            if book_received is not None:
                received_times.append(book_received)
        for feed_snapshot in (coinbase, chainlink_twap):
            if feed_snapshot is not None:
                feed_received = finite_number(feed_snapshot.get("received_ts"))
                if feed_received is not None:
                    received_times.append(feed_received)
        received = max(received_times)

        # Ages must use the final causal row timestamp, including the unusual
        # but possible case where another feed's receive clock is ahead.
        for outcome in ("Up", "Down"):
            book_received = finite_number(books[outcome].get("book_received_ts"))
            books[outcome]["book_age_ms"] = (
                round(max(0.0, received - book_received) * 1000, 3)
                if book_received is not None else None
            )
        for feed_snapshot in (coinbase, chainlink_twap):
            if feed_snapshot is not None:
                feed_received = finite_number(feed_snapshot.get("received_ts"))
                feed_snapshot["age_ms"] = (
                    round(max(0.0, received - feed_received) * 1000, 3)
                    if feed_received is not None else None
                )
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
        if coinbase is not None:
            row["coinbase"] = coinbase
        if chainlink_twap is not None:
            row["chainlink_twap_60s"] = chainlink_twap
        self.writer.write(row)

    def snapshot_once_when_ready(self, market: dict, timeout: float) -> bool:
        """Write a one-shot row only after both exact-market books are ready."""
        if self.market_books is None:
            return False
        books = self.market_books.wait_for_snapshot(
            market["tokens"], self.full_book, timeout
        )
        if books is None:
            return False
        self.snapshot(market, books)
        return True

    def _poll_resolution_batch(self, pending: list[tuple[str, str]],
                               poll_started: float) -> None:
        """Fetch a resolution batch off the high-frequency snapshot thread."""
        for slug, condition_id in pending:
            if self._resolution_stop.is_set():
                return
            try:
                data = self.get_resolution_json(f"{CLOB}/markets/{condition_id}")
                winners = [
                    token.get("outcome")
                    for token in data.get("tokens", [])
                    if token.get("winner") is True
                ]
                if len(winners) == 1:
                    self._resolution_results.put((
                        "resolved", slug, condition_id, winners[0], poll_started
                    ))
            except Exception as exc:
                # The exception is data for the main loop to report. Never let
                # settlement HTTP latency or failure escape into snapshot or
                # market-discovery scheduling.
                self._resolution_results.put((
                    "error", slug, condition_id,
                    f"{type(exc).__name__}: {exc}", poll_started
                ))

    def _apply_resolution_results(self) -> None:
        """Apply completed worker results; all JSONL writes stay on this thread."""
        while True:
            try:
                kind, slug, condition_id, value, poll_started = (
                    self._resolution_results.get_nowait()
                )
            except queue.Empty:
                return
            if kind == "error":
                LOG.warning(
                    "resolution read failed for %s; next batch eligible at "
                    "%.3f: %s",
                    slug,
                    poll_started + RESOLUTION_POLL_SECONDS,
                    value,
                )
                continue
            if self.pending_resolutions.get(slug) != condition_id:
                continue
            self.writer.write({"ts": utc_now(), "event": "resolution",
                               "source": "polymarket", "slug": slug,
                               "condition_id": condition_id, "winner": value})
            if self._forget_resolution(slug, condition_id):
                LOG.info("resolved %s: %s", slug, value)

    def check_resolutions(self, now: float) -> bool:
        """Start a due nonblocking batch; return True only when one was started."""
        self._apply_resolution_results()
        if (self._resolution_thread is not None
                and self._resolution_thread.is_alive()):
            return False
        if now - self.last_resolution_check < RESOLUTION_POLL_SECONDS:
            return False
        pending = [
            (slug, condition_id)
            for slug, condition_id in self.pending_resolutions.items()
            if (btc5m_window_start(slug) is not None
                and now >= btc5m_window_start(slug) + WINDOW_SECONDS)
        ]
        if not pending:
            return False
        self.last_resolution_check = now
        self._resolution_thread = threading.Thread(
            target=self._poll_resolution_batch,
            args=(pending, now),
            name="polymarket-resolution-poll",
            daemon=True,
        )
        self._resolution_thread.start()
        return True

    def close(self, timeout: float = 0.0) -> None:
        """Stop settlement polling, optionally waiting for its current request."""
        self._resolution_stop.set()
        thread = self._resolution_thread
        if thread is not None and thread.is_alive() and timeout > 0:
            thread.join(timeout)
        self._apply_resolution_results()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=float, default=1.0,
                        help="seconds between snapshots (default: 1.0)")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path,
                        default=ROOT / "price_logs" / "polymarket",
                        help="directory for one JSONL file per five-minute window")
    parser.add_argument("--full-book", action="store_true",
                        help="store every order-book level, not only the best bid/ask")
    parser.add_argument(
        "--once", action="store_true",
        help="wait for both full-book images, write one snapshot, and exit",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="HTTP and --once book-readiness timeout in seconds (default: 10)",
    )
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = build_parser().parse_args()
    if not math.isfinite(args.poll) or args.poll <= 0:
        raise SystemExit("--poll must be greater than zero")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    writer = JsonlWriter(args.output_dir.resolve())
    coinbase = CoinbaseFeed()
    chainlink_twap = ChainlinkTwapFeed(
        os.getenv("POLY_API_KEY"),
        os.getenv("POLY_API_SECRET"),
        os.getenv("POLY_API_PASSPHRASE"),
    )
    market_books = PolymarketBookFeed()
    coinbase.start()
    chainlink_twap.start()
    market_books.start()
    LOG.info("writing one direct Polymarket log per window to %s", writer.directory)
    try:
        with httpx.Client(timeout=args.timeout,
                          headers={"User-Agent": "btc5m-research-logger/1.0"}) as client, \
                httpx.Client(
                    timeout=args.timeout,
                    headers={"User-Agent": "btc5m-research-logger/1.0"},
                ) as resolution_client:
            logger = Logger(
                client, writer, args.full_book, coinbase, chainlink_twap,
                market_books, resolution_client,
            )
            try:
                while True:
                    loop_started = time.monotonic()
                    market = None
                    try:
                        market = logger.discover(time.time())
                    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                        retry_in = max(
                            0.0,
                            logger.last_discovery_attempt
                            + DISCOVERY_RETRY_SECONDS
                            - time.time(),
                        )
                        LOG.warning(
                            "market discovery failed; next Gamma attempt eligible "
                            "in %.1fs (%s): %s",
                            retry_in,
                            type(exc).__name__,
                            exc,
                        )
                        if args.once:
                            raise SystemExit(1) from exc

                    if market:
                        if args.once:
                            if not logger.snapshot_once_when_ready(
                                    market, args.timeout):
                                LOG.error(
                                    "--once timed out after %.1fs waiting for both "
                                    "full-book images for %s; no snapshot written",
                                    args.timeout,
                                    market["slug"],
                                )
                                raise SystemExit(1)
                        else:
                            logger.snapshot(market)
                    logger.check_resolutions(time.time())
                    if args.once and market:
                        break
                    delay = max(0.05, args.poll - (time.monotonic() - loop_started))
                    time.sleep(delay)
            finally:
                # The resolution worker uses its own client. Give its current
                # bounded request time to finish before that client is closed.
                logger.close(args.timeout + 0.5)
    except KeyboardInterrupt:
        LOG.info("stopped")
    finally:
        market_books.close()
        coinbase.close()
        chainlink_twap.close()
        writer.close()


if __name__ == "__main__":
    main()
