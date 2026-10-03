"""Standalone BTC chart and real Polymarket $1 jump trader.

Run without --live for signal-only dry runs. Add --live to submit real orders.
"""

import argparse
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    # Headless terminal mode reuses this module's feeds and trading strategy.
    tk = None
    ttk = None

import httpx
import websocket
from dotenv import load_dotenv

from bot import ROOT
from polymarket_feed import MarketFeed

GAMMA_API = "https://gamma-api.polymarket.com/markets"
RTDS_API = "wss://ws-live-data.polymarket.com"
COINBASE_API = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
BINANCE_API = "https://api.binance.com/api/v3/ticker/price"
KRAKEN_API = "https://api.kraken.com/0/public/Ticker"
BITSTAMP_API = "https://www.bitstamp.net/api/v2/ticker/btcusd/"
OKX_API = "https://www.okx.com/api/v5/market/ticker"
BYBIT_API = "https://api.bybit.com/v5/market/tickers"
GEMINI_API = "https://api.gemini.com/v1/pubticker/btcusd"
WINDOW_SECONDS = 300.0
TRADE_START_DELAY_SECONDS = 5.0
TRADE_COOLDOWN_SECONDS = 0.0
MAX_TRADES_PER_WINDOW = 3
JUMP_TP_RATE = Decimal("0.10")
JUMP_NOISE_USD = 0.05
MIN_ENTRY_PRICE = Decimal("0.30")
MAX_ENTRY_PRICE = Decimal("0.70")
STAKE = Decimal("1.10")
TAKE_PROFIT = Decimal("0.05")
# Stop-loss: exit the whole position when the bid falls this far below the
# entry price. A resting limit SELL cannot act as a stop on the CLOB (a sell
# limit at or below the current bid crosses the book and fills immediately),
# so the stop is a software trigger: observe() force-sells at market the
# moment bid <= stop, cancelling the resting TP first.
STOP_LOSS = Decimal("0.05")
# FAK buy limit = quoted ask + this band. Wider band = more aggressive fill
# on thin/moving books; you only ever pay the actual best available ask.
ENTRY_SLIPPAGE = Decimal("0.06")
# Polymarket allows ~9,000 req/10s per key, so every balance poll / retry
# below can run as fast as the network round-trip. These caps only bound a
# failed lookup; the happy path never waits for them.
# Hard exit clock: this long after a fill the position is force-sold at the
# market. Polymarket's CTF balance only becomes sellable once its indexer
# catches the fill (lags a beat), so a dedicated worker hammers
# update_balance_allowance from fill time and posts the FAK market sell the
# instant the balance is live — the 0.5s sell actually fires.
RESTING_TIMEOUT_SECONDS = 0.5
REINDEX_POLL_SECONDS = 0.02
SELL_RETRY_SECONDS = 0.02
SELL_RETRY_WINDOW_SECONDS = 8.0
RESTING_CANCEL_LEAD_SECONDS = 5.0
RESTING_FILL_POLL_SECONDS = 0.02
TREND_SECONDS = 10.0
TREND_MIN_MOVE = 1.0
FRAME_MS = 20
BG, PANEL, GRID = "#0b1020", "#121a2d", "#26324b"
TEXT, MUTED = "#dce7ff", "#8290aa"
UP_COLOR, DOWN_COLOR, BTC_COLOR = "#39d98a", "#ff6577", "#f5b942"
COINBASE_COLOR, BINANCE_COLOR = "#4d8dff", "#f08a24"
KRAKEN_COLOR, BITSTAMP_COLOR = "#a970ff", "#45d6d0"
OKX_COLOR, BYBIT_COLOR, GEMINI_COLOR = "#eeeeee", "#f4d03f", "#e056fd"
POSITION_COLOR = "#5bc0ff"


def jump_direction(samples, threshold):
    """Return (Up/Down, move) when the sampled price move reaches threshold."""
    if len(samples) < 2:
        return None
    move = samples[-1][1] - samples[0][1]
    if abs(move) < threshold:
        return None
    return ("Up" if move > 0 else "Down"), move


def directional_jump(moves, threshold):
    """Detect one large move or 2-3 same-direction moves over the threshold."""
    if not moves:
        return None
    recent = list(moves)[-3:]
    last = recent[-1]
    if abs(last) >= threshold:
        return ("Up" if last > 0 else "Down"), last
    sign = 1 if last > 0 else -1
    same_direction = []
    for move in reversed(recent):
        if move == 0 or (1 if move > 0 else -1) != sign:
            break
        same_direction.append(move)
    if len(same_direction) < 2:
        return None
    total = sum(same_direction)
    if abs(total) < threshold:
        return None
    return ("Up" if total > 0 else "Down"), total


def position_on_up_axis(outcome, price):
    """Project either contract's price onto the Polymarket UP axis."""
    price = float(price)
    return price if outcome == "Up" else 1.0 - price


def price_trend(points, min_move=TREND_MIN_MOVE):
    """Return direction, fitted move, and fitted endpoints for timestamped prices."""
    if len(points) < 2:
        return None
    origin = points[0][0]
    xs = [stamp - origin for stamp, _ in points]
    ys = [price for _, price in points]
    span = xs[-1] - xs[0]
    if span <= 0:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean)
                for x, y in zip(xs, ys)) / denominator
    start_price = y_mean + slope * (xs[0] - x_mean)
    end_price = y_mean + slope * (xs[-1] - x_mean)
    move = end_price - start_price
    direction = "FLAT" if abs(move) < min_move else ("UP" if move > 0 else "DOWN")
    return direction, move, (points[0][0], start_price), (points[-1][0], end_price)


def market_from_payload(payload, now):
    """Return current BTC (slug, end, token map), or None."""
    rows = payload if isinstance(payload, list) else payload.get(
        "markets", payload.get("items", []))
    start = int(now) // 300 * 300
    slug = f"btc-updown-5m-{start}"
    for market in rows:
        if (market.get("slug") != slug or market.get("closed")
                or not market.get("active")):
            continue
        condition = market.get("conditionId") or market.get("condition_id")
        nested = next((item for item in market.get("markets", [])
                       if item.get("condition_id") == condition), None)
        source = nested or market
        outcomes = source.get("outcomes")
        tokens = source.get("clobTokenIds", source.get("token_ids"))
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
        if (set(outcomes or []) != {"Up", "Down"} or len(tokens or []) != 2
                or len(set(tokens)) != 2):
            continue
        return slug, start + 300, dict(zip(outcomes, tokens))
    return None


def threshold_from_history(data, window):
    """Return the first TWAP at a window boundary, never a later substitute."""
    if not data:
        return None
    boundary_ms = int(window) * 1000
    if int(data[0].get("timestamp", boundary_ms + 1)) > boundary_ms:
        return None
    opening = next((item for item in data
                    if int(item.get("timestamp", 0)) >= boundary_ms), None)
    if not opening:
        return None
    value = float(opening.get("value", 0))
    return value if value > 0 else None


class PolymarketBTCFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.price = None
        self.price_to_beat = None
        self.threshold_window = None
        self.error = "Connecting to Polymarket BTC feed..."
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.price, self.price_to_beat, self.error

    def run(self):
        subscription = {"action": "subscribe", "subscriptions": [{
            "topic": "crypto_prices_twap_sixty", "type": "*",
            "filters": json.dumps({"symbol": "btc/usd"})}]}
        while not self.stop.is_set():
            socket = None
            try:
                socket = websocket.create_connection(RTDS_API, timeout=8)
                socket.send(json.dumps(subscription))
                socket.settimeout(3)
                while not self.stop.is_set():
                    raw = socket.recv()
                    if not raw or raw == "PONG":
                        continue
                    message = json.loads(raw)
                    payload = message.get("payload", {})
                    data = payload.get("data", [])
                    value = data[-1].get("value") if data else payload.get("value")
                    if value is None:
                        continue
                    price = float(value)
                    if price <= 0:
                        continue
                    latest_timestamp = (data[-1].get("timestamp") if data else
                                        payload.get("timestamp"))
                    threshold = None
                    window = None
                    if latest_timestamp is not None:
                        window = int(latest_timestamp) // 1000 // 300 * 300
                    if data and window is not None:
                        threshold = threshold_from_history(data, window)
                    with self.lock:
                        self.price, self.error = price, ""
                        if window is not None and window != self.threshold_window:
                            self.price_to_beat = threshold
                            self.threshold_window = window
                    # RTDS currently supplies the latest ticks as a snapshot
                    # on subscription but may not push subsequent frames.
                    # Reconnect for the next snapshot instead of displaying a
                    # stale last value indefinitely.
                    break
            except Exception as exc:
                with self.lock:
                    self.error = f"Polymarket BTC {type(exc).__name__}"
                self.stop.wait(1)
            finally:
                if socket is not None:
                    try:
                        socket.close()
                    except Exception:
                        pass
            self.stop.wait(0.5)


class ExchangePriceFeed:
    """Poll one public exchange ticker without blocking the chart UI."""

    def __init__(self, name, url, params, parser, headers=None):
        self.name, self.url, self.params, self.parser = name, url, params, parser
        self.headers = headers or {}
        self.lock = threading.Lock()
        self.price = None
        self.last_changed = None
        self.error = f"Connecting to {name}..."
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.price, self.error, self.last_changed

    def run(self):
        headers = {"User-Agent": "btc-live-chart/1.0", **self.headers}
        with httpx.Client(headers=headers,
                          timeout=5) as client:
            while not self.stop.is_set():
                try:
                    params = dict(self.params)
                    params["_"] = int(time.time() * 1000)
                    response = client.get(self.url, params=params)
                    response.raise_for_status()
                    price = float(self.parser(response.json()))
                    if price <= 0:
                        raise ValueError("invalid price")
                    with self.lock:
                        if price != self.price:
                            self.last_changed = time.monotonic()
                        self.price, self.error = price, ""
                except Exception as exc:
                    with self.lock:
                        self.error = f"{self.name} {type(exc).__name__}"
                self.stop.wait(0.5)


class CoinbasePriceFeed:
    """Stream Coinbase Exchange BTC-USD ticker prices over WebSocket."""

    def __init__(self):
        self.lock = threading.Lock()
        self.price = None
        self.error = "Connecting to Coinbase Exchange..."
        self.stop = threading.Event()
        self.on_price = None
        self.thread = threading.Thread(
            target=self.run, name="coinbase-btc-feed", daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.price, self.error

    def set_callback(self, callback):
        self.on_price = callback

    def run(self):
        subscription = {"type": "subscribe", "product_ids": ["BTC-USD"],
                        "channels": ["ticker"]}
        while not self.stop.is_set():
            socket = None
            try:
                socket = websocket.create_connection(COINBASE_WS, timeout=8)
                socket.send(json.dumps(subscription))
                socket.settimeout(2)
                while not self.stop.is_set():
                    try:
                        message = json.loads(socket.recv())
                    except websocket.WebSocketTimeoutException:
                        socket.ping()
                        continue
                    if (message.get("type") != "ticker" or
                            message.get("product_id") != "BTC-USD"):
                        continue
                    price = float(message["price"])
                    if price <= 0:
                        continue
                    with self.lock:
                        self.price, self.error = price, ""
                    callback = self.on_price
                    if callback is not None:
                        try:
                            callback(price)
                        except Exception:
                            # A strategy callback must never disconnect the
                            # Coinbase market-data socket.
                            pass
            except Exception as exc:
                with self.lock:
                    self.error = f"Coinbase Exchange {type(exc).__name__}"
                self.stop.wait(0.5)
            finally:
                if socket is not None:
                    try:
                        socket.close()
                    except Exception:
                        pass


class EventDrivenJumpDetector:
    """Evaluate jump signals directly on each Coinbase WebSocket tick."""

    def __init__(self, discovery, market_feed, trader, threshold, on_event=None):
        self.discovery = discovery
        self.market_feed = market_feed
        self.trader = trader
        self.threshold = threshold
        self.on_event = on_event
        self.lock = threading.Lock()
        self.current_slug = None
        self.last_price = None
        self.last_move = None
        self.moves = deque(maxlen=3)
        self.history = deque(maxlen=200)

    def _event(self, kind, message):
        if self.on_event:
            self.on_event(kind, message)

    def reset(self, slug, tokens=None):
        changed = False
        with self.lock:
            if slug != self.current_slug:
                changed = True
                self.current_slug = slug
                self.last_price = None
                self.last_move = None
                self.moves.clear()
        if changed:
            self.trader.prepare(slug, tokens)

    def snapshot(self):
        with self.lock:
            return self.last_price, self.last_move, list(self.moves)

    def jump_history(self):
        with self.lock:
            return [dict(item) for item in self.history]

    def on_price(self, price):
        detected_at = time.monotonic()
        market, _ = self.discovery.snapshot()
        if not market:
            return
        slug = market[0]
        self.reset(slug, market[2])
        with self.lock:
            previous = self.last_price
            self.last_price = price
            self.last_move = None if previous is None else price - previous
            if self.last_move is None or self.last_move == 0:
                return
            market_age = time.time() - (market[1] - 300)
            if market_age < TRADE_START_DELAY_SECONDS:
                self.moves.clear()
                return
            if abs(self.last_move) <= JUMP_NOISE_USD + 1e-9:
                quotes = self.market_feed.quotes()
                up_quote, down_quote = quotes.get("Up"), quotes.get("Down")
                self.history.append({
                    "timestamp": time.time(), "price": price,
                    "delta": self.last_move,
                    "direction": "NOISE",
                    "streak_count": len(self.moves),
                    "streak_total": float(sum(self.moves)),
                    "up_ask": None if up_quote is None else float(up_quote.ask),
                    "down_ask": None if down_quote is None else float(down_quote.ask),
                    "action": "ignored noise", "request_id": None,
                })
                return
            if self.last_move is not None:
                if self.moves and self.moves[-1] * self.last_move < 0:
                    self.moves.clear()
                self.moves.append(self.last_move)
            quotes = self.market_feed.quotes()
            up_quote, down_quote = quotes.get("Up"), quotes.get("Down")
            jump_record = {
                "timestamp": time.time(), "price": price,
                "delta": self.last_move,
                "direction": ("UP" if (self.last_move or 0) > 0 else
                              "DOWN" if (self.last_move or 0) < 0 else "FLAT"),
                "streak_count": len(self.moves),
                "streak_total": float(sum(self.moves)),
                "up_ask": None if up_quote is None else float(up_quote.ask),
                "down_ask": None if down_quote is None else float(down_quote.ask),
                "action": "accumulating", "request_id": None,
            }
            self.history.append(jump_record)
            signal = directional_jump(self.moves, self.threshold)
            if not signal:
                return
            outcome, move = signal
            jump_record["action"] = f"{outcome} signal"
            self._event("SIGNAL", f"{outcome} jump {move:+.2f} USD")
            quote = quotes.get(outcome)
            if quote is None:
                with self.trader.lock:
                    self.trader.status = f"skipped {outcome}: live ask unavailable"
                self._event("SKIP", f"{outcome}: live ask unavailable")
                jump_record["action"] = "skipped: ask unavailable"
                self.moves.clear()
                return
            if not MIN_ENTRY_PRICE <= quote.ask <= MAX_ENTRY_PRICE:
                with self.trader.lock:
                    self.trader.status = (
                        f"skipped {outcome}: ask {quote.ask:.3f} outside entry filter")
                self._event("SKIP", f"{outcome}: ask {quote.ask:.3f} outside "
                            f"{MIN_ENTRY_PRICE:.2f}-{MAX_ENTRY_PRICE:.2f}")
                jump_record["action"] = "skipped: entry filter"
                self.moves.clear()
                return
            request_id = self.trader.submit(
                slug, outcome, move, quote.tick, token=market[2][outcome],
                ask=quote.ask, detected_at=detected_at)
            if request_id:
                jump_record["request_id"] = request_id
                jump_record["action"] = f"{outcome} order queued"
                self._event("ORDER", f"{outcome} queued at ask {quote.ask:.3f}")
                self.moves.clear()


class PolymarketJumpTrader:
    """One jump entry with a resting limit take-profit sell on Polymarket."""

    def __init__(self, live=False):
        self.live = live
        self.venue_name = "POLYMARKET LIVE" if live else "POLYMARKET DRY RUN"
        self.db_path = ROOT / (
            "chart-jump-polymarket-live.sqlite3" if live
            else "chart-jump-polymarket-dry-run.sqlite3")
        with sqlite3.connect(self.db_path) as db:
            columns = [row[1] for row in db.execute(
                "PRAGMA table_info(trades)").fetchall()]
            if columns and "id" not in columns:
                db.execute("ALTER TABLE trades RENAME TO trades_single")
                db.execute("""CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL, outcome TEXT, move REAL, token TEXT,
                    tick TEXT, entry_state TEXT, price TEXT, shares TEXT,
                    tp_state TEXT, target TEXT, created REAL, detail TEXT,
                    filled_at REAL, sold_at REAL)""")
                db.execute("""INSERT INTO trades
                    (market,outcome,move,token,tick,entry_state,price,shares,
                     tp_state,target,created,detail)
                    SELECT market,outcome,move,token,tick,entry_state,price,
                           shares,tp_state,target,created,detail
                    FROM trades_single""")
                db.execute("DROP TABLE trades_single")
            else:
                db.execute("""CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL, outcome TEXT, move REAL, token TEXT,
                    tick TEXT, entry_state TEXT, price TEXT, shares TEXT,
                    tp_state TEXT, target TEXT, created REAL, detail TEXT,
                    filled_at REAL, sold_at REAL)""")
            db.execute("CREATE INDEX IF NOT EXISTS trades_market_idx "
                       "ON trades(market, created)")
            columns = [row[1] for row in db.execute(
                "PRAGMA table_info(trades)").fetchall()]
            if "request_id" not in columns:
                db.execute("ALTER TABLE trades ADD COLUMN request_id TEXT")
                db.execute("UPDATE trades SET request_id='legacy-' || id "
                           "WHERE request_id IS NULL")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS trades_request_idx "
                       "ON trades(request_id)")
            # Timestamps for when the bet executed (filled_at) and when the
            # position was sold (sold_at). Back-filled for any pre-existing
            # rows from the closest column already available.
            for column, backfill in (
                    ("filled_at", "created"), ("sold_at", "created")):
                if column not in columns:
                    db.execute(f"ALTER TABLE trades ADD COLUMN {column} REAL")
                    db.execute(f"UPDATE trades SET {column}={backfill} "
                               f"WHERE {column} IS NULL")
            db.execute("""UPDATE trades SET entry_state='rejected'
                WHERE entry_state='unknown' AND price IS NULL
                  AND LOWER(detail) LIKE '%fully filled or killed%'""")
        self.lock = threading.Lock()
        self.status = "ready; waiting for jump"
        self.stop = threading.Event()
        self.position_cache = (None, 0.0, None)
        self.latest_cache = (None, 0.0, None)
        self.open_position = None
        self.next_sell_attempt = 0.0
        # Trade IDs closed by the resting-TP fill watch / settle-cancel. Their
        # DB row still reads "watching" for a moment before the writer thread
        # commits "sold", so position() uses this set to not show a phantom
        # open position during that lag.
        self.settled_trades = set()
        self.window_counts = {}
        self.pending_markets = set()
        self.prepared_markets = set()
        self.warming_markets = set()
        self.warmed_markets = set()
        self.warm_status = "not started"
        self.last_entry_times = {}
        self.execution_timings = deque(maxlen=200)
        self.timing_by_request = {}
        self.persist_queue = queue.Queue()
        self.order_queue = queue.Queue()
        self.persist_thread = threading.Thread(
            target=self._persist_loop, name="polymarket-db-writer", daemon=True)
        self.order_thread = threading.Thread(
            target=self._order_loop, name="polymarket-order-worker", daemon=True)
        self.client = None
        self.order_api = None
        if live:
            self._connect()
            self._recover_position()
        self.persist_thread.start()
        self.order_thread.start()

    def _connect(self):
        response = httpx.get("https://polymarket.com/api/geoblock", timeout=8)
        response.raise_for_status()
        if response.json().get("blocked") is not False:
            raise RuntimeError("Polymarket trading is unavailable from this location")
        from py_clob_client_v2 import ClobClient
        key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
        if not key:
            raise ValueError("Set POLY_PRIVATE_KEY in .env before using --live")
        signature = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
        funder = os.environ.get("POLY_FUNDER", "").strip() or None
        if signature not in (0, 1, 2, 3) or (signature != 0 and not funder):
            raise ValueError("Set a valid POLY_SIGNATURE_TYPE and POLY_FUNDER")
        self.client = ClobClient(
            "https://clob.polymarket.com", chain_id=137, key=key,
            signature_type=signature, funder=funder)
        self.client.set_api_creds(self.client.create_or_derive_api_key())
        # Import and cache the signing classes before the first jump. Credential
        # derivation above also warms the authenticated CLOB connection.
        from py_clob_client_v2 import (MarketOrderArgs, OrderType,
                                       PartialCreateOrderOptions, Side)
        self.order_api = (MarketOrderArgs, OrderType,
                          PartialCreateOrderOptions, Side)

    def snapshot(self):
        with self.lock:
            return self.status

    def warmup_snapshot(self):
        with self.lock:
            return self.warm_status

    def _recover_position(self):
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT * FROM trades
                WHERE entry_state='filled' AND tp_state IN ('watching','selling')
                ORDER BY created DESC LIMIT 1""").fetchone()
        if not row:
            return
        try:
            active = int(row["market"].rsplit("-", 1)[1]) + 300 > time.time()
        except (ValueError, IndexError):
            active = False
        if active:
            self.open_position = {
                "id": row["request_id"], "market": row["market"],
                "outcome": row["outcome"],
                "token": row["token"], "tick": Decimal(row["tick"]),
                "shares": Decimal(row["shares"]),
                "entry": Decimal(row["price"]),
                "target": Decimal(row["target"]), "selling": False,
                "sell_deadline": None,
            }
            self._update(row["request_id"], tp_state="watching")
            self.status = (f"recovered {row['outcome']}; watching bid for "
                           f"software sell at {Decimal(row['target']):.4f}")

    def prepare(self, slug, tokens=None):
        with self.lock:
            needs_database = slug not in self.prepared_markets
        if needs_database:
            with sqlite3.connect(self.db_path) as db:
                count = db.execute("""SELECT COUNT(*) FROM trades
                    WHERE market=? AND entry_state IN ('submitting','unknown',
                                                        'filled','simulated')""",
                                   (slug,)).fetchone()[0]
                pending = db.execute("""SELECT 1 FROM trades WHERE market=?
                    AND entry_state IN ('submitting','unknown') LIMIT 1""",
                                     (slug,)).fetchone()
            with self.lock:
                self.window_counts[slug] = count
                if pending:
                    self.pending_markets.add(slug)
                self.prepared_markets.add(slug)
        if self.live and tokens:
            with self.lock:
                should_warm = (slug not in self.warming_markets and
                               slug not in self.warmed_markets)
                if should_warm:
                    self.warming_markets.add(slug)
                    self.warm_status = f"warming {slug} UP/DOWN metadata"
            if should_warm:
                threading.Thread(
                    target=self._warm_tokens,
                    args=(slug, tuple(str(token) for token in tokens.values())),
                    name="polymarket-token-warmer", daemon=True).start()

    def _warm_tokens(self, slug, tokens):
        """Populate SDK caches and signing paths without posting an order."""
        try:
            MarketOrderArgs, OrderType, _, Side = self.order_api
            for token in tokens:
                self.client.create_market_order(
                    MarketOrderArgs(token_id=token, amount=float(STAKE),
                                    side=Side.BUY, price=0.5,
                                    order_type=OrderType.FOK))
            with self.lock:
                self.warmed_markets.add(slug)
                self.warm_status = f"{slug} UP/DOWN metadata ready"
        except Exception as exc:
            with self.lock:
                self.warm_status = f"warmup {type(exc).__name__}: {str(exc)[:160]}"
        finally:
            with self.lock:
                self.warming_markets.discard(slug)

    def trade_count(self, slug):
        with sqlite3.connect(self.db_path) as db:
            return db.execute("""SELECT COUNT(*) FROM trades
                WHERE market=? AND entry_state IN ('submitting','unknown',
                                                    'filled','simulated')""",
                              (slug,)).fetchone()[0]

    def submit(self, slug, outcome, move, tick="0.01", token=None, ask=None,
               detected_at=None):
        if not token or ask is None:
            with self.lock:
                self.status = "trade skipped: missing Polymarket token/ask"
            return False
        with self.lock:
            accepted = self.window_counts.get(slug, 0)
            since_entry = time.monotonic() - self.last_entry_times.get(slug, 0.0)
            if since_entry < TRADE_COOLDOWN_SECONDS:
                self.status = (f"entry cooldown: "
                               f"{TRADE_COOLDOWN_SECONDS - since_entry:.2f}s")
                return False
            if accepted >= MAX_TRADES_PER_WINDOW:
                self.status = (f"{MAX_TRADES_PER_WINDOW}-trade limit "
                               "reached for this window")
                return False
            if self.open_position and self.open_position["market"] == slug:
                self.status = "waiting for open position to sell"
                return False
            if slug in self.pending_markets:
                self.status = "entry already pending; duplicate blocked"
                return False
            request_id = uuid.uuid4().hex
            self.pending_markets.add(slug)
            self.last_entry_times[slug] = time.monotonic()
            queued_at = time.monotonic()
            timing = {
                "request_id": request_id, "timestamp": time.time(),
                "outcome": outcome, "move": float(move), "result": "queued",
                "detected_to_queue_ms": None if detected_at is None else
                    (queued_at - detected_at) * 1000,
                "queue_wait_ms": None, "request_ms": None, "total_ms": None,
                "_detected_at": detected_at or queued_at,
                "_queued_at": queued_at,
            }
            self.execution_timings.append(timing)
            self.timing_by_request[request_id] = timing
            self.status = f"{outcome} jump {move:+.2f}; submitting ${STAKE}"
        self.persist_queue.put(("insert", request_id, slug, outcome, move,
                                str(token), str(tick), time.time()))
        self.order_queue.put((request_id, slug, outcome, str(token),
                              str(tick), str(ask)))
        return request_id

    def timing_history(self):
        with self.lock:
            result = []
            for timing in self.execution_timings:
                result.append({key: value for key, value in timing.items()
                               if not key.startswith("_")})
            return result

    def _timing_update(self, request_id, result=None, response_at=None):
        now = response_at or time.monotonic()
        with self.lock:
            timing = self.timing_by_request.get(request_id)
            if not timing:
                return
            if result is not None:
                timing["result"] = result
            if response_at is not None:
                started = timing.get("_worker_at", timing["_queued_at"])
                timing["request_ms"] = (now - started) * 1000
                timing["total_ms"] = (now - timing["_detected_at"]) * 1000

    def _persist_loop(self):
        with sqlite3.connect(self.db_path) as db:
            while True:
                task = self.persist_queue.get()
                if task is None:
                    return
                if task[0] == "insert":
                    _, request_id, slug, outcome, move, token, tick, created = task
                    db.execute("""INSERT INTO trades
                        (request_id,market,outcome,move,token,tick,entry_state,
                         created,detail) VALUES (?,?,?,?,?,?,'submitting',?,'')""",
                               (request_id, slug, outcome, move, token, tick, created))
                else:
                    _, request_id, fields = task
                    assignments = ",".join(f"{name}=?" for name in fields)
                    db.execute(f"UPDATE trades SET {assignments} WHERE request_id=?",
                               (*fields.values(), request_id))
                db.commit()

    def _order_loop(self):
        while True:
            task = self.order_queue.get()
            if task is None:
                return
            self._execute(*task)

    def _update(self, request_id, **fields):
        self.persist_queue.put(("update", request_id, fields))
        with self.lock:
            self.position_cache = (None, 0.0, None)
            self.latest_cache = (None, 0.0, None)

    def _entry_order_terms(self, ask, tick):
        """Return limit, USDC buy amount, and order type for an entry."""
        limit = (min(Decimal("0.99"), ask + ENTRY_SLIPPAGE) /
                 tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        return limit, Decimal(STAKE), "FAK"

    def _create_entry_order(self, token, limit, buy_amount,
                            entry_order_type, options):
        """Create the default USDC-amount market entry order.

        Specialized front ends may override this hook when they need an exact
        share quantity instead of spending an exact USDC amount.
        """
        MarketOrderArgs, OrderType, _, Side = self.order_api
        order_type = getattr(OrderType, entry_order_type)
        signed = self.client.create_market_order(
            MarketOrderArgs(token_id=token, amount=float(buy_amount),
                            side=Side.BUY, price=float(limit),
                            order_type=order_type), options)
        return signed, order_type

    def _execute(self, request_id, slug, outcome, token, tick_text, ask_text):
        worker_at = time.monotonic()
        with self.lock:
            timing = self.timing_by_request.get(request_id)
            if timing:
                timing["_worker_at"] = worker_at
                timing["queue_wait_ms"] = (
                    worker_at - timing["_queued_at"]) * 1000
                timing["result"] = "submitting"
        tick = Decimal(tick_text)
        ask = Decimal(ask_text)
        # WIDE limit: a FAK market buy fills every ask at or below the limit,
        # so a higher limit is more aggressive and more likely to fill on a
        # thin book. We still only pay the actual best available ask. The
        # entry filter (MIN/MAX_ENTRY_PRICE) caps the price we ever pay.
        limit, buy_amount, entry_order_type = self._entry_order_terms(ask, tick)
        if not self.live:
            self._update(request_id, entry_state="simulated",
                         detail=(f"dry run ${buy_amount}, ask {ask}, "
                                 f"limit {limit}"))
            with self.lock:
                self.pending_markets.discard(slug)
                self.window_counts[slug] = self.window_counts.get(slug, 0) + 1
                self.status = (f"DRY RUN {outcome}; ${buy_amount}, ask {ask}, "
                               f"limit {limit}")
            self._timing_update(request_id, "simulated", time.monotonic())
            return
        filled = False
        try:
            _, _, PartialCreateOrderOptions, _ = self.order_api
            options = PartialCreateOrderOptions(tick_size=tick_text,
                                                neg_risk=False)
            # The default profile uses FAK market orders. Web/live profiles
            # can override the creation hook for exact-share FOK orders.
            signed, order_type = self._create_entry_order(
                token, limit, buy_amount, entry_order_type, options)
            response = self.client.post_order(
                signed, order_type, defer_exec=True)
            if (not response.get("success") or
                    str(response.get("status", "")).lower() != "matched"):
                self._update(request_id, entry_state="rejected",
                             detail=json.dumps(response)[:4000])
                with self.lock:
                    self.pending_markets.discard(slug)
                    self.status = f"{outcome} order rejected"
                self._timing_update(request_id, "rejected", time.monotonic())
                return
            paid = Decimal(str(response["makingAmount"]))
            shares = Decimal(str(response["takingAmount"]))
            if shares <= 0 or paid <= 0:
                self._update(request_id, entry_state="rejected",
                             detail=json.dumps(response)[:4000])
                with self.lock:
                    self.pending_markets.discard(slug)
                    self.status = f"{outcome} order filled nothing (thin book)"
                self._timing_update(request_id, "rejected", time.monotonic())
                return
            price = paid / shares
            target = (price * (Decimal("1") + TAKE_PROFIT) / tick).to_integral_value(
                rounding=ROUND_CEILING) * tick
            # Stop price: floor to tick so the trigger is bid < stop (the bid
            # must drop below this level). Mirrors the TP (ceiling to tick)
            # on the other side of the entry.
            stop = (price * (Decimal("1") - STOP_LOSS) / tick).to_integral_value(
                rounding=ROUND_FLOOR) * tick
            self._update(request_id, entry_state="filled", price=str(price),
                         shares=str(shares), tp_state="watching",
                         target=str(target), filled_at=time.time(),
                         detail=json.dumps(response)[:4000])
            filled = True
            position = {
                "id": request_id, "market": slug, "outcome": outcome, "token": token,
                "tick": tick, "shares": shares, "entry": price, "target": target,
                "stop": stop,
                "selling": False, "sell_deadline": None,
                # Hard exit clock: 0.5s after this fill, if the resting TP
                # has not filled at target, the position is force-sold at the
                # market bid (observe/_sell). Never rides a dead trend.
                "deadline": time.monotonic() + RESTING_TIMEOUT_SECONDS,
                # The TP arm retries placement while the balance indexer lags
                # (it may only be able to land a beat after the 0.5s deadline).
                # Once placed, this is how long the resting TP is allowed to
                # sit in the book before the software force-sell takes over.
                "tp_deadline": time.monotonic() + RESTING_TIMEOUT_SECONDS * 2,
                "exit_mode": "software",
                "resting_order_id": None, "resting_placed": False,
            }
            with self.lock:
                self.open_position = position
                self.pending_markets.discard(slug)
                self.window_counts[slug] = self.window_counts.get(slug, 0) + 1
                self.status = (f"{outcome} filled {price:.4f}; TP order "
                               f"{target:.4f} outgoing")
            self._timing_update(request_id, "filled", time.monotonic())
            # Fire-and-forget exit workers, started the instant the fill
            # returns: (1) a reindex thread hammers update_balance_allowance
            # so Polymarket's CTF balance indexer catches the fill fast (this
            # is what stops both the resting-TP arm and the 0.5s force-sell
            # failing with "not enough balance"); (2) the arm thread posts the
            # take-profit resting limit SELL at target the moment it can.
            threading.Thread(
                target=self._reindex_balance, args=(dict(position),),
                name="polymarket-reindex", daemon=True).start()
            threading.Thread(
                target=self._arm_take_profit, args=(dict(position),),
                name="polymarket-arm-take-profit", daemon=True).start()
            if target >= Decimal("1"):
                self._update(request_id, tp_state="unreachable")
                with self.lock:
                    self.open_position = None
                    self.status = f"{outcome} filled {price:.4f}; sell target unreachable"
                return
        except Exception as exc:
            fields = {"detail": f"{type(exc).__name__}: {str(exc)[:500]}"}
            if filled:
                fields["tp_state"] = "unknown"
            elif ("no orders found to match with FAK" in str(exc)
                    or "partially filled or killed" in str(exc).lower()):
                fields["entry_state"] = "rejected"
                fields["detail"] = ("no book depth at posted price (FAK "
                                    "killed): " + str(exc)[:300])
            elif "fully filled or killed" in str(exc).lower():
                fields["entry_state"] = "rejected"
            else:
                fields["entry_state"] = "unknown"
            self._update(request_id, **fields)
            self._timing_update(
                request_id, "post-fill error" if filled else fields["entry_state"],
                time.monotonic())
            with self.lock:
                if filled:
                    self.status = f"post-fill error: {type(exc).__name__}; check Polymarket"
                elif fields["entry_state"] == "rejected":
                    self.pending_markets.discard(slug)
                    self.status = f"{outcome} FOK rejected; no fill"
                else:
                    self.status = (f"order error: {type(exc).__name__}; "
                                   "check Polymarket")

    def observe(self, slug, quotes):
        """Software exit: FAK market sell at the available bid.

        Fires when:
        1. The TP deadline has passed — the resting TP is not filling at
           target fast enough, so sell NOW at whatever the bid is (no target
           check) and never ride a dead trend. When a resting TP is live it
           gets the full tp_deadline window; otherwise the shorter 0.5s
           deadline applies.
        2. No resting TP is armed yet and the bid already sits at/above
           target.

        Retries are capped by sell_deadline (SELL_RETRY_WINDOW_SECONDS); the
        exit is a FAK market sell, so it takes whatever depth is available.
        """
        if not self.live:
            return
        now = time.monotonic()
        stop_hit = False
        with self.lock:
            position = self.open_position
            if (not position or position["market"] != slug or
                    position["selling"]):
                return
            resting_live = (position.get("resting_placed") and
                            position.get("exit_mode", "software") == "resting")
            deadline = (position.get("tp_deadline") if resting_live
                        else position.get("deadline"))
            quote = quotes.get(position["outcome"])
            bid = Decimal(quote.bid) if quote is not None else None
            stop = position.get("stop")
            stop_hit = (bid is not None and stop is not None
                        and bid <= stop)
            if deadline is not None and now >= deadline:
                force = True
            elif stop_hit:
                # Stop-loss: the bid has dropped to entry minus STOP_LOSS.
                # Exit at market NOW instead of waiting for the exit clock
                # or the TP window (a resting limit can't stop a falling
                # bid on the CLOB — a sell limit below the bid would just
                # cross and fill, so the stop is this software trigger).
                force = True
            elif resting_live:
                return  # resting TP live, inside its window, stop not hit
            else:
                if bid is None or bid < position["target"]:
                    return
                force = False
            if position.get("sell_deadline") is None:
                position["sell_deadline"] = now + SELL_RETRY_WINDOW_SECONDS
            elif now >= position["sell_deadline"]:
                self.open_position = None
                self.status = "software sell stopped after window"
                self._update(position["id"], tp_state="sell_timeout")
                return
            position["selling"] = True
            trade_id = position["id"]
        if force and position.get("exit_mode") == "resting":
            # Pull the resting maker TP so the forced market sell is the only
            # live exit order (no orphaned limit, no double-offer).
            self._cancel_resting(trade_id)
        self._update(trade_id, tp_state="selling")
        with self.lock:
            self.status = (
                f"{'stop-loss ' if stop_hit else 'forced ' if force else ''}"
                f"software sell triggered {position['outcome']}"
                + (f" bid {bid:.3f}" if bid is not None
                   else " (no bid quote)"))
        threading.Thread(target=self._sell,
                         args=(dict(position), bid),
                         name="chart-polymarket-sell", daemon=True).start()

    def _sell(self, position, bid):
        slug = position["market"]
        trade_id = position["id"]
        try:
            from py_clob_client_v2 import (MarketOrderArgs, OrderType,
                                           PartialCreateOrderOptions, Side)
            token, shares, tick = (position["token"], position["shares"],
                                   position["tick"])
            # No balance-read gate: the reindex thread hammers
            # update_balance_allowance in parallel, so we simply POST the FAK
            # market sell back-to-back and let the server validate the balance.
            # The first post after the indexer catches the fill matches; until
            # then the post is rejected ("not enough balance") and we retry.
            # FAK (fill-and-kill) sells the available depth at the floor rather
            # than hard-failing the whole exit on a thin book.
            deadline = position["sell_deadline"]
            if deadline is None:
                deadline = time.monotonic() + SELL_RETRY_WINDOW_SECONDS
            options = PartialCreateOrderOptions(tick_size=str(tick),
                                                neg_risk=False)
            limit = tick
            last_err = None
            while not self.stop.is_set() and time.monotonic() < deadline:
                with self.lock:
                    closed = (not self.open_position or
                              self.open_position.get("id") != trade_id)
                if closed:
                    return
                try:
                    signed = self.client.create_market_order(
                        MarketOrderArgs(token_id=token, amount=float(shares),
                                        side=Side.SELL, price=float(limit),
                                        order_type=OrderType.FAK), options)
                    response = self.client.post_order(
                        signed, OrderType.FAK, defer_exec=True)
                    if (response.get("success") and
                            str(response.get("status", "")).lower() == "matched"):
                        self._update(trade_id, tp_state="sold",
                                     sold_at=time.time(),
                                     detail=json.dumps(response)[:4000])
                        with self.lock:
                            self.open_position = None
                            self.settled_trades.add(trade_id)
                            self.status = (f"{position['outcome']} aggressive "
                                           f"software sell filled (floor "
                                           f"{limit:.3f})")
                        return
                    last_err = ValueError(
                        f"sell not matched: {str(response)[:200]}")
                except Exception as exc:
                    last_err = exc
                self.stop.wait(0.01)
            if last_err:
                raise last_err
            raise TimeoutError("software sell not confirmed before window")
        except Exception as exc:
            timed_out = time.monotonic() >= position["sell_deadline"]
            self._update(trade_id,
                         tp_state="sell_timeout" if timed_out else "watching",
                         detail=f"{type(exc).__name__}: {str(exc)[:500]}")
            with self.lock:
                if timed_out:
                    self.open_position = None
                    self.status = "software sell stopped after 2 seconds"
                elif self.open_position:
                    self.open_position["selling"] = False
                    self.next_sell_attempt = time.monotonic() + SELL_RETRY_SECONDS
                    self.status = (f"software sell retry in 20ms: "
                                   f"{type(exc).__name__}")

    def _reindex_balance(self, position):
        """Hammer update_balance_allowance from the fill instant.

        Polymarket's CTF balance for a token only becomes spendable once its
        indexer catches the fill — which lags a few hundred ms. Both the
        resting-TP arm and the 0.5s force-sell fail with "not enough balance"
        until that indexer catches up. This dedicated thread re-triggers the
        on-chain balance refresh as fast as the network allows (warm RTT ~64ms,
        ~15/s), running in parallel with the arm thread and the software sell,
        so the balance is sellable the instant any of them needs it. It stops
        once the position is closed or superseded.
        """
        trade_id = position["id"]
        token = position["token"]
        try:
            from py_clob_client_v2 import (AssetType, BalanceAllowanceParams)
            balance_params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token)
        except Exception:
            return
        while not self.stop.is_set():
            with self.lock:
                pos = self.open_position
                # Keep hammering until this position is actually closed
                # (None) or superseded by a newer trade. Crucially, do NOT stop
                # when selling starts — the software FAK sell needs the balance
                # to be indexed to match, so keep re-indexing through the sell.
                if not pos or pos.get("id") != trade_id:
                    return
            try:
                self.client.update_balance_allowance(balance_params)
            except Exception:
                pass
            self.stop.wait(REINDEX_POLL_SECONDS)

    def _arm_take_profit(self, position):
        """Post the take-profit as a resting maker SELL on Polymarket.

        A software TP watches the bid in Python and, by the time it posts the
        market sell, the whipsaw is usually already back — so fills land at or
        below target. Posting a limit SELL at the target instead puts the exit
        in the book the moment the balance is indexed; the exchange then fills
        us at target-or-better with zero reaction delay, and it can never fail
        on a thin book because it is already resting in it.

        The FAK software sell (observe/_sell) remains as the safety net: if
        the price reaches target but the resting limit has no bids in front of
        it to cross, the fallback takes the available depth at the floor.
        """
        trade_id = position["id"]
        token, shares, tick, target = (position["token"], position["shares"],
                                       position["tick"], position["target"])
        if target >= Decimal("1"):
            return
        # Window end: slug = btc-updown-5m-{start}; market settles at start+300.
        settle_at = position.get("settle_at") or self._settle_at(position["market"])
        from py_clob_client_v2 import (AssetType, BalanceAllowanceParams,
                                       OrderArgs, OrderType,
                                       PartialCreateOrderOptions, Side)
        balance_params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL, token_id=token)
        options = PartialCreateOrderOptions(tick_size=str(tick),
                                            neg_risk=False)
        # Wait on the book, then protect. We cannot attach a TP to a fill
        # that Polymarket has not indexed yet — the entry matched, but the
        # sellable balance only appears in the book once the indexer catches
        # the fill (lags a few hundred ms). So: poll (update + get) until the
        # indexed balance covers our shares, and ONLY then post the resting
        # limit SELL. No fixed wait cap — poll as fast as the network allows;
        # the only real boundary is settlement (don't post a sell in the last
        # 2s). The reindex thread hammers update_balance_allowance in
        # parallel, so the book shows the fill quickly.
        deadline = settle_at - 2
        placed = False
        order_id = None
        while not self.stop.is_set() and time.monotonic() < deadline:
            # Bail if the position was closed/superseded (or already placed).
            # Do NOT bail when selling=True: the book may only show the fill
            # a beat after the 0.5s force-sell deadline — keep waiting.
            with self.lock:
                pos = self.open_position
                if (not pos or pos["id"] != trade_id or pos.get("resting_placed")):
                    return
            # 1) WAIT: re-index, then read the book. Keep polling until the
            # balance is sellable (indexed >= our shares).
            try:
                self.client.update_balance_allowance(balance_params)
                balance = self.client.get_balance_allowance(balance_params)
                remaining = (Decimal(str(balance.get("balance", "0"))) /
                             Decimal("1000000"))
            except Exception:
                remaining = Decimal("0")
            if remaining + Decimal("0.001") < shares:
                with self.lock:
                    self.status = (f"{position['outcome']} TP waiting for "
                                   f"fill in book ({remaining:.3f}/"
                                   f"{float(shares):.3f} sh)")
                self.stop.wait(0.02)
                continue
            # 2) PROTECT: the fill is live in the book — post the resting
            # limit SELL at target (shares, not dollars).
            try:
                signed = self.client.create_order(
                    OrderArgs(token_id=token, price=float(target),
                              size=float(shares), side=Side.SELL), options)
                response = self.client.post_order(signed, OrderType.GTC)
                order_id = self._order_id_from(response)
                if not order_id:
                    raise ValueError(f"no order id: {str(response)[:120]}")
                with self.lock:
                    pos = self.open_position
                    still_open = bool(pos and pos["id"] == trade_id)
                    if still_open:
                        pos["resting_order_id"] = order_id
                        pos["resting_placed"] = True
                        pos["exit_mode"] = "resting"
                if not still_open:
                    # The position was already closed by the software force
                    # sell while we were waiting, so this just-posted limit
                    # would be orphaned. Pull it and stop — it cannot fill
                    # (the shares are gone) and leaving it would be noise.
                    self._cancel_order(order_id)
                    return
                self._update(trade_id, tp_state="watching")
                with self.lock:
                    self.status = (f"{position['outcome']} TP limit resting "
                                   f"@ {target:.4f} ({float(shares):.3f} sh)")
                placed = True
                break
            except Exception as exc:
                err = str(exc)
                # If the book changed under us (partial consumption, balance
                # dropped), re-check on the next pass — keep waiting.
                with self.lock:
                    self.status = (f"TP arm {type(exc).__name__}: {err[:120]}")
                self.stop.wait(0.02)
        # Order is live. The whole exit has RESTING_TIMEOUT_SECONDS from the
        # buy (position["deadline"]): if the resting limit fills at target in
        # time (balance drops) we finalize the win immediately. Otherwise the
        # watch loop ends and the software sell loop (observe) force-sells the
        # position at the current bid on its next poll.
        if not (placed and not self.stop.is_set()):
            return
        threshold = shares - Decimal("0.001")
        with self.lock:
            pos = self.open_position
            if not (pos and pos["id"] == trade_id):
                return  # closed/superseded while we were (re)posting
            tp_deadline = (pos.get("tp_deadline")
                           or pos.get("deadline") or time.monotonic())
        watch_until = max(time.monotonic(),
                          min(tp_deadline, settle_at - RESTING_CANCEL_LEAD_SECONDS))
        while not self.stop.is_set() and time.monotonic() < watch_until:
            self.stop.wait(min(RESTING_FILL_POLL_SECONDS,
                               max(0.0, watch_until - time.monotonic())))
            with self.lock:
                pos = self.open_position
                still_mine = bool(pos and pos.get("id") == trade_id)
            if not still_mine:
                # The software force-sell already closed this position (or a
                # newer trade superseded it): pull the resting TP so it does
                # not linger in the book with no shares behind it.
                self._cancel_order(order_id)
                return
            # update (re-index) then get: two sequential RTTs.
            try:
                self.client.update_balance_allowance(balance_params)
                balance = self.client.get_balance_allowance(balance_params)
                remaining = (Decimal(str(balance.get("balance", "0"))) /
                             Decimal("1000000"))
            except Exception:
                continue
            if remaining < threshold:
                self._finalize_sold(trade_id, position, target, by="resting")
                return
        # Deadline reached with the resting TP unfilled: pull the limit so the
        # shares are free for the forced market sell (observe handles it).
        self._cancel_resting(trade_id)

    def _finalize_sold(self, trade_id, position, target, by):
        """Close a position out after its take-profit filled (resting limit or
        software FAK). Clear the open slot so new entries in this window are
        allowed and the DB row reflects the win."""
        with self.lock:
            pos = self.open_position
            if pos is not None and pos.get("id") == trade_id:
                self.open_position = None
                self.status = (f"{position['outcome']} take-profit filled "
                               f"(target {target:.4f}, via {by})")
        self._update(trade_id, tp_state="sold", sold_at=time.time())
        with self.lock:
            self.settled_trades.add(trade_id)

    def _settle_at(self, slug):
        try:
            return int(slug.rsplit("-", 1)[1]) + WINDOW_SECONDS
        except (ValueError, IndexError):
            return time.time() + WINDOW_SECONDS

    def _order_id_from(self, response):
        for key in ("orderID", "orderId", "order_id", "id"):
            value = response.get(key) if isinstance(response, dict) else None
            if value:
                return str(value)
        return None

    def _cancel_order(self, order_id):
        """Cancel a single order by id, ignoring failures (e.g. it already
        filled or was never live)."""
        if not order_id:
            return
        try:
            from py_clob_client_v2 import OrderPayload
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception:
            pass

    def _cancel_resting(self, trade_id):
        """Pull the resting TP limit so the shares are free to be force-sold.

        Keeps the position open in resting mode with its deadline set, so the
        software sell loop (observe) force-sells it at the current bid on its
        next poll. Called when the resting limit did not fill in time (or the
        window is closing).
        """
        with self.lock:
            pos = self.open_position
            if not pos or pos.get("id") != trade_id:
                return
            order_id = pos.get("resting_order_id")
        if not order_id:
            return
        try:
            from py_clob_client_v2 import OrderPayload
            self.client.cancel_order(OrderPayload(orderID=order_id))
        except Exception:
            pass
        with self.lock:
            pos = self.open_position
            if pos and pos.get("id") == trade_id:
                pos["resting_placed"] = False
                pos["resting_order_id"] = None
                if pos.get("deadline") is None:
                    pos["deadline"] = time.monotonic()
                self.status = (f"TP limit not filled in time; forcing sell "
                               f"({pos['outcome']})")

    def _row(self, slug=None):
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            if slug:
                return db.execute("""SELECT * FROM trades WHERE market=?
                    ORDER BY created DESC LIMIT 1""",
                                  (slug,)).fetchone()
            return db.execute(
                "SELECT * FROM trades ORDER BY created DESC LIMIT 1").fetchone()

    def position(self, slug):
        now = time.monotonic()
        with self.lock:
            cached_slug, checked, cached = self.position_cache
            if cached_slug == slug and now - checked < 0.25:
                return cached
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT * FROM trades
                WHERE market=? AND entry_state='filled'
                  AND COALESCE(tp_state,'') != 'sold'
                ORDER BY created DESC LIMIT 1""", (slug,)).fetchone()
        # A trade closed by the resting-TP fill watch (or the settle-cancel)
        # has no live position any more, even though its DB row may still read
        # "watching" for a moment before the writer thread commits "sold".
        if row is not None and row["request_id"] in self.settled_trades:
            row = None
        if not row or row["entry_state"] != "filled" or not row["price"]:
            position = None
        else:
            price = Decimal(row["price"])
            target = Decimal(row["target"])
            with self.lock:
                live = self.open_position
                exit_mode = (live.get("exit_mode", "software") if live
                             and live.get("id") == row["request_id"] else "software")
            position = {
                "outcome": row["outcome"],
                "shares": float(row["shares"]),
                "entry": position_on_up_axis(row["outcome"], price),
                "target": position_on_up_axis(row["outcome"], target),
                "tp_state": row["tp_state"] or "unknown",
                "exit_mode": exit_mode,
            }
        with self.lock:
            self.position_cache = (slug, now, position)
        return position

    def latest_trade(self, slug=None):
        now = time.monotonic()
        with self.lock:
            cached_slug, checked, cached = self.latest_cache
            if slug is not None and cached_slug == slug and now - checked < 0.25:
                return cached
        row = self._row(slug)
        if not row:
            trade = None
        else:
            trade = {
                "market": row["market"], "outcome": row["outcome"],
                "move": row["move"], "entry_state": row["entry_state"],
                "price": row["price"], "shares": row["shares"],
                "target": None if row["target"] is None else float(row["target"]),
                "tp_state": row["tp_state"], "detail": row["detail"] or "",
            }
        with self.lock:
            self.latest_cache = (slug, now, trade)
        return trade

    def close(self):
        self.stop.set()
        self.order_queue.put(None)
        self.order_thread.join(timeout=5)
        self.persist_queue.put(None)
        self.persist_thread.join(timeout=5)


class Discovery:
    def __init__(self, feed):
        self.feed = feed
        self.lock = threading.Lock()
        self.market = None
        self.error = "Discovering Polymarket BTC market..."
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.market, self.error

    def run(self):
        with httpx.Client(timeout=8) as client:
            while not self.stop.is_set():
                try:
                    now = time.time()
                    slug = f"btc-updown-5m-{int(now) // 300 * 300}"
                    response = client.get(GAMMA_API, params={"slug": slug})
                    response.raise_for_status()
                    found = market_from_payload(response.json(), now)
                    with self.lock:
                        changed = found and (not self.market or found[0] != self.market[0])
                        self.market = found
                        self.error = "" if found else "Waiting for active market..."
                    if changed:
                        self.feed.subscribe(found[2])
                except Exception as exc:
                    with self.lock:
                        self.error = f"Polymarket {type(exc).__name__}"
                self.stop.wait(1)


class PriceChart:
    def __init__(self, root, trader_factory=PolymarketJumpTrader):
        self.root = root
        root.title("BTC Live - Polymarket Price + 5-Minute Up")
        root.geometry("1100x720")
        root.minsize(760, 520)
        root.configure(bg=BG)
        self.feed, self.btc_feed = MarketFeed(), PolymarketBTCFeed()
        self.coinbase_feed = CoinbasePriceFeed()
        self.jump_trader = trader_factory()
        self.jump_threshold = float(os.environ.get("JUMP_THRESHOLD_USD", "10"))
        if self.jump_threshold <= 0:
            raise ValueError("Invalid jump trader threshold")
        self.last_jump_price = None
        self.last_coinbase_move = None
        self.directional_moves = deque(maxlen=3)
        self.discovery = Discovery(self.feed)
        self.jump_detector = EventDrivenJumpDetector(
            self.discovery, self.feed, self.jump_trader, self.jump_threshold)
        self.coinbase_feed.set_callback(self.jump_detector.on_price)
        self.samples, self.current_slug = deque(), None
        self.coinbase_offset = None
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 11))
        top = ttk.Frame(root, padding=(18, 14, 18, 8))
        top.pack(fill="x")
        ttk.Label(top, text="BTC LIVE", font=("Segoe UI Semibold", 18)).pack(side="left")
        self.status = ttk.Label(top, text="Connecting...", foreground=MUTED)
        self.status.pack(side="right")
        cards = ttk.Frame(root, padding=(18, 0, 18, 10))
        cards.pack(fill="x")
        self.btc_card = self.card(cards, "POLYMARKET BTC-USD", BTC_COLOR)
        self.coinbase_card = self.card(
            cards, "COINBASE EXCHANGE BTC-USD", COINBASE_COLOR)
        self.up_card = self.card(cards, "POLYMARKET UP", UP_COLOR)
        all_cards = (self.btc_card, self.coinbase_card, self.up_card)
        for index, card in enumerate(all_cards):
            card[0].pack(side="left", fill="x", expand=True,
                         padx=(0 if index == 0 else 3,
                               0 if index == len(all_cards) - 1 else 3))
        self.trade_banner = tk.Label(
            root, text=f"{self.jump_trader.venue_name}: no trade yet", bg=BG, fg=MUTED,
            anchor="w", padx=20, pady=3, font=("Consolas", 10, "bold"))
        self.trade_banner.pack(fill="x")
        self.canvas = tk.Canvas(root, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.discovery.start()
        self.btc_feed.start()
        self.coinbase_feed.start()
        self.tick()

    @staticmethod
    def card(parent, name, color):
        frame = tk.Frame(parent, bg=PANEL, padx=16, pady=10)
        tk.Label(frame, text=name, bg=PANEL, fg=color,
                 font=("Segoe UI Semibold", 11)).pack(anchor="w")
        value = tk.Label(frame, text="--", bg=PANEL, fg=TEXT,
                         font=("Consolas", 23, "bold"))
        value.pack(anchor="w")
        detail = tk.Label(frame, text="waiting for data", bg=PANEL, fg=MUTED,
                          font=("Consolas", 10))
        detail.pack(anchor="w")
        return frame, value, detail

    def tick(self):
        market, poly_error = self.discovery.snapshot()
        btc, price_to_beat, btc_error = self.btc_feed.snapshot()
        coinbase_price, coinbase_error = self.coinbase_feed.snapshot()
        quotes, now = self.feed.quotes(), time.monotonic()
        if market and market[0] != self.current_slug:
            self.current_slug = market[0]
            self.jump_trader.prepare(market[0], market[2])
            self.samples.clear()
            self.last_jump_price = None
            self.last_coinbase_move = None
            self.directional_moves.clear()
            self.coinbase_offset = None
            self.jump_detector.reset(market[0], market[2])
        if (btc is not None and coinbase_price is not None and
                self.coinbase_offset is None):
            self.coinbase_offset = btc - coinbase_price
        self.last_jump_price, self.last_coinbase_move, detected_moves = (
            self.jump_detector.snapshot())
        self.directional_moves = deque(detected_moves, maxlen=3)
        if btc is not None:
            self.btc_card[1].configure(text=f"${btc:,.2f}")
            if price_to_beat is not None:
                difference = btc - price_to_beat
                self.btc_card[2].configure(
                    text=f"to beat ${price_to_beat:,.2f}   delta {difference:+,.2f}")
            else:
                self.btc_card[2].configure(text="waiting for price to beat")
        if coinbase_price is not None:
            self.coinbase_card[1].configure(
                text=f"${coinbase_price:,.2f}", fg=COINBASE_COLOR)
            move_text = ("waiting for next Coinbase tick"
                         if self.last_coinbase_move is None else
                         f"live move {self.last_coinbase_move:+,.2f} USD")
            offset_text = ("waiting for graph sync"
                           if self.coinbase_offset is None else
                           f"offset {self.coinbase_offset:+,.2f}")
            self.coinbase_card[2].configure(
                text=f"{move_text}   |   {offset_text}",
                fg=(UP_COLOR if (self.last_coinbase_move or 0) > 0 else
                    DOWN_COLOR if (self.last_coinbase_move or 0) < 0 else MUTED))
        up, down = quotes.get("Up"), quotes.get("Down")
        if market:
            self.jump_trader.observe(market[0], quotes)
        position = self.jump_trader.position(market[0]) if market else None
        latest_trade = self.jump_trader.latest_trade()
        if latest_trade:
            entry = (f" @ {float(latest_trade['price']):.4f}"
                     if latest_trade.get("price") else "")
            shares = (f" | {float(latest_trade['shares']):.3f} shares"
                      if latest_trade.get("shares") else "")
            target = (f" | SELL {latest_trade['target']:.4f} "
                      f"{str(latest_trade.get('tp_state') or '').upper()}"
                      if latest_trade.get("target") is not None else "")
            jump = (f" | JUMP {float(latest_trade['move']):+.2f} USD"
                    if latest_trade.get("move") is not None else "")
            state = str(latest_trade.get("entry_state") or "signal").upper()
            self.trade_banner.configure(
                text=(f"{self.jump_trader.venue_name}: {state} "
                      f"{latest_trade['outcome']}{entry}"
                      f"{shares}{target}{jump} | {latest_trade['market']}"),
                fg=UP_COLOR if state == "FILLED" else DOWN_COLOR)
        if up:
            self.up_card[1].configure(text=f"{up.probability:.3f}", fg=UP_COLOR)
            position_text = ("" if not position else
                f"   POS {position['outcome']} @ UP {position['entry']:.3f}")
            self.up_card[2].configure(
                text=f"bid {up.bid:.3f}   ask {up.ask:.3f}{position_text}")
        if btc is not None or coinbase_price is not None or up or down:
            synced_coinbase = (
                None if coinbase_price is None or self.coinbase_offset is None
                else coinbase_price + self.coinbase_offset)
            self.samples.append((now, synced_coinbase,
                                 None if up is None else float(up.probability),
                                 None if down is None else float(down.probability)))
            while self.samples and self.samples[0][0] < now - WINDOW_SECONDS:
                self.samples.popleft()
        errors = " | ".join(filter(
            None, (poly_error, btc_error, coinbase_error)))
        if market:
            remaining = max(0, int(market[1] - time.time()))
            trade_status = self.jump_trader.snapshot()
            elapsed = time.time() - (market[1] - 300)
            if elapsed < TRADE_START_DELAY_SECONDS:
                trade_status = (f"trading starts in "
                                f"{max(0, TRADE_START_DELAY_SECONDS - elapsed):.1f}s")
            status = (f"LIVE | {remaining}s left | {datetime.now():%H:%M:%S} | "
                      f"{trade_status}")
            self.status.configure(text=status + (f" | {errors}" if errors else ""),
                                  foreground=UP_COLOR if up and down else MUTED)
        else:
            self.status.configure(text=errors or "Waiting for live data...", foreground=MUTED)
        self.draw(now, price_to_beat, position)
        self.root.after(FRAME_MS, self.tick)

    def draw(self, now, price_to_beat=None, position=None):
        c = self.canvas
        c.delete("all")
        width, height = max(1, c.winfo_width()), max(1, c.winfo_height())
        left, right, top, bottom = 82, width - 62, 42, height - 30
        values = [sample[1] for sample in self.samples if sample[1] is not None]
        if price_to_beat is not None:
            values.append(price_to_beat)
        if values:
            low, high = min(values), max(values)
            padding = max(1., (high - low) * .12)
            low, high = low - padding, high + padding
        else:
            low, high = 0., 1.
        for index in range(5):
            y = top + (bottom - top) * index / 4
            btc_value = high - (high - low) * index / 4
            probability = 1 - index / 4
            c.create_line(left, y, right, y, fill=GRID)
            c.create_text(left - 10, y, text=f"${btc_value:,.0f}",
                          fill=COINBASE_COLOR, anchor="e")
            c.create_text(right + 10, y, text=f"{probability:.2f}",
                          fill=MUTED, anchor="w")
        c.create_text(left, 18, text="Coinbase BTC-USD (synced)",
                      fill=COINBASE_COLOR, anchor="w")
        c.create_text(left + 155, 18, text="Polymarket Up",
                      fill=UP_COLOR, anchor="w")
        trend_points = [
            (stamp, coinbase_price)
            for stamp, coinbase_price, _, _ in self.samples
            if coinbase_price is not None and stamp >= now - TREND_SECONDS]
        trend = price_trend(trend_points)
        if trend:
            direction, move, _, _ = trend
            trend_color = (UP_COLOR if direction == "UP" else
                           DOWN_COLOR if direction == "DOWN" else MUTED)
            arrow = "â†‘" if direction == "UP" else "â†“" if direction == "DOWN" else "â†’"
            c.create_text(left + 300, 18,
                          text=f"{arrow} {direction} {move:+.2f} / {TREND_SECONDS:.0f}s",
                          fill=trend_color, anchor="w",
                          font=("Consolas", 10, "bold"))
        c.create_text(left - 10, top - 8, text="BTC $",
                      fill=COINBASE_COLOR, anchor="e")
        c.create_text(right + 10, top - 8, text="Probability", fill=MUTED, anchor="w")
        if price_to_beat is not None:
            beat_y = bottom - (price_to_beat-low)/(high-low)*(bottom-top)
            c.create_line(left, beat_y, right, beat_y, fill="#ffffff",
                          width=1, dash=(7, 5))
            c.create_text(right - 5, beat_y - 5,
                          text=f"TO BEAT ${price_to_beat:,.2f}", fill="#ffffff",
                          anchor="se", font=("Consolas", 9, "bold"))
        if position:
            entry_y = bottom - position["entry"] * (bottom - top)
            target_y = bottom - position["target"] * (bottom - top)
            c.create_line(left, entry_y, right, entry_y,
                          fill=POSITION_COLOR, width=2, dash=(8, 4))
            c.create_text(left + 6, entry_y - 5,
                          text=(f"POSITION {position['outcome']}  "
                                f"UP={position['entry']:.3f}  "
                                f"{position['shares']:.2f} shares"),
                          fill=POSITION_COLOR, anchor="sw",
                          font=("Consolas", 9, "bold"))
            if 0 <= position["target"] <= 1:
                c.create_line(left, target_y, right, target_y,
                              fill=POSITION_COLOR, width=1, dash=(2, 5))
                c.create_text(right - 5, target_y - 4,
                              text=(f"SELL TARGET on UP axis {position['target']:.3f}  "
                                    f"[{position['tp_state']}]") ,
                              fill=POSITION_COLOR, anchor="se",
                              font=("Consolas", 9))
        if len(self.samples) < 2:
            return
        start = now - WINDOW_SECONDS
        points = {"coinbase": [], "up": []}
        for stamp, coinbase_price, up, down in self.samples:
            x = left + (stamp - start) / WINDOW_SECONDS * (right - left)
            if coinbase_price is not None:
                points["coinbase"] += [
                    x, bottom - (coinbase_price-low)/(high-low)*(bottom-top)]
            if up is not None:
                points["up"] += [x, bottom - up * (bottom-top)]
        for name, color in (("coinbase", COINBASE_COLOR), ("up", UP_COLOR)):
            if len(points[name]) >= 4:
                c.create_line(*points[name], fill=color, width=2, smooth=True)
        if trend:
            direction, _, fitted_start, fitted_end = trend
            trend_color = (UP_COLOR if direction == "UP" else
                           DOWN_COLOR if direction == "DOWN" else MUTED)
            trend_xy = []
            for stamp, price in (fitted_start, fitted_end):
                x = left + (stamp - start) / WINDOW_SECONDS * (right - left)
                y = bottom - (price - low) / (high - low) * (bottom - top)
                trend_xy.extend((x, y))
            c.create_line(*trend_xy, fill=trend_color, width=3, dash=(6, 4))

    def close(self):
        self.discovery.stop.set()
        self.btc_feed.stop.set()
        self.coinbase_feed.stop.set()
        self.jump_trader.close()
        self.feed.generation += 1
        self.root.destroy()


def main():
    if tk is None:
        raise RuntimeError(
            "Tkinter is unavailable. Run btc_polymarket_headless.py for the "
            "Rich terminal dashboard, or install your OS tkinter package."
        )
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Submit real-money Polymarket orders")
    args = parser.parse_args()
    root = tk.Tk()
    PriceChart(root, trader_factory=lambda: PolymarketJumpTrader(args.live))
    root.title("BTC Live - Polymarket $1 Software-Exit Trader")
    root.mainloop()


if __name__ == "__main__":
    main()
