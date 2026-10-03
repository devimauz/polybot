"""Threaded public Polymarket best-bid/ask WebSocket feed."""

import json
import logging
import threading
import time
from collections import deque

import websocket

from bot import D, Quote

LOG = logging.getLogger("btc-bot")
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class MarketFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = {}
        self.books = {}
        self.generation = 0
        self.thread = None
        self.trades = deque(maxlen=200)
        self._trade_keys = set()

    def subscribe(self, tokens):
        token_map = {str(token): outcome for outcome, token in tokens.items()}
        with self.lock:
            if token_map == self.tokens:
                return
            self.tokens = token_map
            self.books = {}
            self.trades.clear()
            self._trade_keys.clear()
            self.generation += 1
            generation = self.generation
        self.thread = threading.Thread(target=self._run, args=(generation, token_map),
                                       name="polymarket-market-feed", daemon=True)
        self.thread.start()

    def _run(self, generation, tokens):
        while generation == self.generation:
            ws = None
            try:
                ws = websocket.create_connection(WS_URL, timeout=5)
                ws.send(json.dumps({"assets_ids": list(tokens), "type": "market",
                                    "custom_feature_enabled": True}))
                ws.settimeout(1)
                last_ping = time.monotonic()
                while generation == self.generation:
                    if time.monotonic() - last_ping >= 10:
                        ws.send("PING")
                        last_ping = time.monotonic()
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw or raw == "PONG":
                        continue
                    self._handle(json.loads(raw), tokens, generation)
            except Exception as exc:
                if generation == self.generation:
                    LOG.warning("Live market feed reconnecting (%s)", type(exc).__name__)
                    time.sleep(1)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _handle(self, message, tokens, generation):
        if isinstance(message, list):
            for item in message:
                self._handle(item, tokens, generation)
            return
        if not isinstance(message, dict):
            return
        event = message.get("event_type")
        if event is None and "bids" in message and "asks" in message:
            event = "book"
        changes = message.get("price_changes", []) if event == "price_change" else [message]
        now = time.monotonic()
        with self.lock:
            if generation != self.generation:
                return
            for row in changes:
                token = str(row.get("asset_id", row.get("market", "")))
                if token not in tokens:
                    continue
                book = self.books.setdefault(token, {"bid": None, "ask": None,
                                                      "tick": D("0.01"), "stamp": 0.0,
                                                      "bids": {}, "asks": {}})
                if event == "book":
                    book["bids"] = {
                        D(str(x["price"])): D(str(x["size"]))
                        for x in row.get("bids", []) if D(str(x["size"])) > 0}
                    book["asks"] = {
                        D(str(x["price"])): D(str(x["size"]))
                        for x in row.get("asks", []) if D(str(x["size"])) > 0}
                    book["bid"] = max(book["bids"], default=None)
                    book["ask"] = min(book["asks"], default=None)
                elif event in ("price_change", "best_bid_ask"):
                    # price_change rows contain the changed level as well as
                    # best_bid/best_ask. Retain sizes for the dashboard while
                    # quotes() remains the latency-sensitive trading API.
                    side = str(row.get("side", "")).upper()
                    if side in {"BUY", "SELL"} and row.get("price") not in (None, ""):
                        price = D(str(row["price"]))
                        size = D(str(row.get("size", "0") or "0"))
                        levels = book["bids" if side == "BUY" else "asks"]
                        if size > 0:
                            levels[price] = size
                        else:
                            levels.pop(price, None)
                    if row.get("best_bid") not in (None, ""):
                        book["bid"] = D(str(row["best_bid"]))
                    if row.get("best_ask") not in (None, ""):
                        book["ask"] = D(str(row["best_ask"]))
                elif event == "last_trade_price":
                    price = D(str(row.get("price", "0") or "0"))
                    size = D(str(row.get("size", "0") or "0"))
                    timestamp = row.get("timestamp") or int(time.time() * 1000)
                    try:
                        timestamp = float(timestamp)
                        if timestamp > 10_000_000_000:
                            timestamp /= 1000.0
                    except (TypeError, ValueError):
                        timestamp = time.time()
                    side = str(row.get("side", "")).upper() or "UNKNOWN"
                    key = (token, timestamp, str(price), str(size), side)
                    if price > 0 and size > 0 and key not in self._trade_keys:
                        self.trades.append({
                            "timestamp": timestamp, "outcome": tokens[token],
                            "side": side, "price": float(price),
                            "size": float(size),
                        })
                        self._trade_keys.add(key)
                        while len(self._trade_keys) > 400:
                            # Rebuild occasionally; deque is authoritative.
                            self._trade_keys = {
                                (token_id, t["timestamp"], str(D(str(t["price"]))),
                                 str(D(str(t["size"]))), t["side"])
                                for token_id, outcome in self.tokens.items()
                                for t in self.trades if t["outcome"] == outcome
                            }
                elif event == "tick_size_change" and row.get("new_tick_size"):
                    book["tick"] = D(str(row["new_tick_size"]))
                book["stamp"] = now

    def quotes(self):
        with self.lock:
            result = {}
            for token, outcome in self.tokens.items():
                book = self.books.get(token)
                if not book or book["bid"] is None or book["ask"] is None:
                    continue
                bid, ask = book["bid"], book["ask"]
                if not D(0) <= bid <= ask <= D(1) or ask - bid > D("0.10"):
                    continue
                result[outcome] = Quote((bid + ask) / 2, ask, ask, str(book["tick"]),
                                        False, bid)
            return result

    def market_activity(self, depth_levels=12, trade_seconds=60):
        """Return a JSON-safe depth and recent-trade snapshot.

        This is intentionally separate from quotes() so dashboard work cannot
        alter the execution path used by the trader.
        """
        cutoff = time.time() - trade_seconds
        with self.lock:
            depth = {}
            for token, outcome in self.tokens.items():
                book = self.books.get(token)
                if not book:
                    continue
                bids = sorted(book.get("bids", {}).items(), reverse=True)
                asks = sorted(book.get("asks", {}).items())
                depth[outcome] = {
                    "bids": [[float(p), float(s)] for p, s in bids[:depth_levels]],
                    "asks": [[float(p), float(s)] for p, s in asks[:depth_levels]],
                }
            trades = [dict(t) for t in self.trades if t["timestamp"] >= cutoff]
        return {"depth": depth, "trades": trades}
