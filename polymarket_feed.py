"""Threaded public Polymarket best-bid/ask WebSocket feed."""

import json
import logging
import threading
import time

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

    def subscribe(self, tokens):
        token_map = {str(token): outcome for outcome, token in tokens.items()}
        with self.lock:
            if token_map == self.tokens:
                return
            self.tokens = token_map
            self.books = {}
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
                                                      "tick": D("0.01"), "stamp": 0.0})
                if event == "book":
                    bids = [D(x["price"]) for x in row.get("bids", []) if D(x["size"]) > 0]
                    asks = [D(x["price"]) for x in row.get("asks", []) if D(x["size"]) > 0]
                    book["bid"] = max(bids, default=None)
                    book["ask"] = min(asks, default=None)
                elif event in ("price_change", "best_bid_ask"):
                    if row.get("best_bid") not in (None, ""):
                        book["bid"] = D(str(row["best_bid"]))
                    if row.get("best_ask") not in (None, ""):
                        book["ask"] = D(str(row["best_ask"]))
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
