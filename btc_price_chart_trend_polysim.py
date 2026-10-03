"""Live Polymarket BTC price, price-to-beat, and 5-minute Up graph."""

import json
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from decimal import Decimal
import tkinter as tk
from tkinter import ttk

import httpx
import websocket
from dotenv import load_dotenv

from bot import ROOT
from opening_bot import OpeningBot
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
# Rolling display history. The traded Polymarket market remains five minutes;
# this only controls how much data the chart retains and draws.
WINDOW_SECONDS = 60.0
TRADE_START_DELAY_SECONDS = 0.0
TRADE_COOLDOWN_SECONDS = 1.0
JUMP_TP_RATE = Decimal("0.10")
JUMP_STAKE = Decimal("100")
MIN_ENTRY_PRICE = Decimal("0.20")
MAX_ENTRY_PRICE = Decimal("0.80")
TREND_SECONDS = 10.0
TREND_MIN_MOVE = 1.0
TREND_ENTRY_THRESHOLD = 40.0
FAST_TREND_SECONDS = 3.0
FAST_TREND_THRESHOLD = 15.0
MIN_FAST_SPAN = 2.0
MIN_SLOW_SPAN = 6.0
MIN_DIRECTIONAL_RATIO = 0.70
MAX_CONTRACT_MOVE = 0.04
MAX_ENTRY_SPREAD = Decimal("0.03")
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


def confirmed_trend_signal(btc_points, up_points, price_to_beat, now):
    """Confirm BTC multi-horizon direction and a lagging UP-contract move."""
    slow_points = [point for point in btc_points
                   if point[0] >= now - TREND_SECONDS]
    fast_points = [point for point in btc_points
                   if point[0] >= now - FAST_TREND_SECONDS]
    contract_points = [point for point in up_points
                       if point[0] >= now - FAST_TREND_SECONDS]
    slow = price_trend(slow_points, min_move=0)
    fast = price_trend(fast_points, min_move=0)
    if price_to_beat is None:
        return None, "waiting for price to beat"
    if (not slow or not fast or len(contract_points) < 2 or
            slow_points[-1][0] - slow_points[0][0] < MIN_SLOW_SPAN or
            fast_points[-1][0] - fast_points[0][0] < MIN_FAST_SPAN):
        return None, "collecting trend data"
    slow_move, fast_move = slow[1], fast[1]
    up_move = contract_points[-1][1] - contract_points[0][1]
    btc_now = btc_points[-1][1]
    tick_moves = [right[1] - left[1]
                  for left, right in zip(fast_points, fast_points[1:])
                  if right[1] != left[1]]
    if not tick_moves:
        return None, "waiting for directional BTC ticks"
    if (slow_move >= TREND_ENTRY_THRESHOLD and
            fast_move >= FAST_TREND_THRESHOLD):
        outcome = "Up"
        directional_ratio = sum(move > 0 for move in tick_moves) / len(tick_moves)
        if directional_ratio < MIN_DIRECTIONAL_RATIO:
            return None, f"UP tick consistency {directional_ratio:.0%}"
        if price_to_beat is not None and btc_now <= price_to_beat:
            return None, "UP trend has not crossed price to beat"
        if not 0 <= up_move <= MAX_CONTRACT_MOVE:
            return None, f"UP contract already moved {up_move:+.3f}"
    elif (slow_move <= -TREND_ENTRY_THRESHOLD and
          fast_move <= -FAST_TREND_THRESHOLD):
        outcome = "Down"
        directional_ratio = sum(move < 0 for move in tick_moves) / len(tick_moves)
        if directional_ratio < MIN_DIRECTIONAL_RATIO:
            return None, f"DOWN tick consistency {directional_ratio:.0%}"
        if price_to_beat is not None and btc_now >= price_to_beat:
            return None, "DOWN trend has not crossed price to beat"
        if not -MAX_CONTRACT_MOVE <= up_move <= 0:
            return None, f"UP contract already moved {up_move:+.3f}"
    else:
        return None, "fast and slow BTC trends do not agree"
    return {
        "outcome": outcome, "move": slow_move, "fast_move": fast_move,
        "up_move": up_move, "directional_ratio": directional_ratio,
    }, "confirmed"


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
        self.thread = threading.Thread(
            target=self.run, name="coinbase-btc-feed", daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.price, self.error

    def run(self):
        subscription = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker"],
        }
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


class JumpPaperTrader:
    """PolySimulator trend entries, each with a 10% take-profit."""

    venue_name = "POLYSIM"

    def __init__(self):
        self.db_path = ROOT / "chart-trend-polysim.sqlite3"
        self.http = httpx.Client(timeout=10)
        setup = OpeningBot(
            asset="btc", amount=JUMP_STAKE, paper=True, entry_window=300,
            db_path=self.db_path, http_client=self.http,
            take_profit=JUMP_TP_RATE)
        setup.close()
        with sqlite3.connect(self.db_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jump_signals (
                market TEXT PRIMARY KEY, outcome TEXT, move REAL, created REAL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS jump_entries (
                market TEXT, slot INTEGER, trade_key TEXT UNIQUE, outcome TEXT,
                move REAL, created REAL, PRIMARY KEY(market,slot))""")
            db.execute("""INSERT OR IGNORE INTO jump_entries
                (market,slot,trade_key,outcome,move,created)
                SELECT market,1,market,outcome,move,created FROM jump_signals""")
        self.lock = threading.Lock()
        self.inflight = set()
        self.preparing = set()
        self.prepared = {}
        self.status = "trend trader ready; waiting for confirmed BTC/UP trend"
        self.position_cache = (None, 0.0, None)
        self.latest_cache = (0.0, None)
        self.stop = threading.Event()
        self.maintenance = threading.Thread(
            target=self._maintain, name="chart-jump-tp", daemon=True)
        self.maintenance.start()

    def snapshot(self):
        with self.lock:
            return self.status

    def position(self, slug):
        """Return the current fill and TP projected onto the UP-price axis."""
        now = time.monotonic()
        with self.lock:
            cached_slug, checked, cached = self.position_cache
            if cached_slug == slug and now - checked < 0.25:
                return cached
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT outcome,price,shares,tick,tp_state
                FROM trades WHERE market LIKE ? AND entry_state='filled'
                ORDER BY rowid DESC LIMIT 1""", (slug + "#jump-%",)).fetchone()
        position = None
        if row and row["price"]:
            price = Decimal(row["price"])
            tick = Decimal(row["tick"])
            target = (price * (Decimal("1") + JUMP_TP_RATE) / tick).to_integral_value(
                rounding="ROUND_CEILING") * tick
            position = {
                "outcome": row["outcome"],
                "shares": float(row["shares"] or 0),
                "entry": position_on_up_axis(row["outcome"], price),
                "target": position_on_up_axis(row["outcome"], target),
                "tp_state": row["tp_state"] or "needed",
            }
        with self.lock:
            self.position_cache = (slug, now, position)
        return position

    def latest_trade(self):
        """Return the newest journaled PolySimulator trade for persistent display."""
        now = time.monotonic()
        with self.lock:
            checked, cached = self.latest_cache
            if now - checked < 0.25:
                return cached
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("""SELECT j.market,j.outcome,j.move,t.entry_state,
                t.price,t.shares,t.tick,t.tp_state,t.detail
                FROM jump_entries AS j LEFT JOIN trades AS t
                  ON t.market=j.trade_key AND t.outcome=j.outcome
                ORDER BY j.created DESC LIMIT 1""").fetchone()
        trade = None
        if row:
            trade = dict(row)
            if row["price"] and row["tick"]:
                price, tick = Decimal(row["price"]), Decimal(row["tick"])
                trade["target"] = float(
                    (price * (Decimal("1") + JUMP_TP_RATE) / tick).to_integral_value(
                        rounding="ROUND_CEILING") * tick)
        with self.lock:
            self.latest_cache = (now, trade)
        return trade

    def trade_window_state(self, slug):
        with sqlite3.connect(self.db_path) as db:
            return db.execute("""SELECT COALESCE(MAX(slot),0),
                COALESCE(MAX(created),0) FROM jump_entries WHERE market=?""",
                              (slug,)).fetchone()

    def prepare(self, slug):
        """Warm the connection and cache simulator market metadata before signals."""
        with self.lock:
            if slug in self.prepared or slug in self.preparing:
                return
            self.preparing.add(slug)
        threading.Thread(target=self._prepare, args=(slug,),
                         name="chart-jump-prepare", daemon=True).start()

    def _prepare(self, expected_slug):
        bot = None
        try:
            bot = OpeningBot(
                asset="btc", amount=JUMP_STAKE, paper=True,
                entry_window=300, db_path=self.db_path,
                http_client=self.http, take_profit=JUMP_TP_RATE)
            for _ in range(10):
                if self.stop.is_set():
                    return
                market = bot.discover(time.time())
                if market and market[0] == expected_slug:
                    with self.lock:
                        self.prepared[expected_slug] = (
                            market, bot.condition_id)
                        self.status = "market prepared; waiting for confirmed trend"
                    return
                self.stop.wait(0.5)
        except Exception as exc:
            with self.lock:
                self.status = f"market prepare error: {type(exc).__name__}"
        finally:
            if bot is not None:
                bot.close()
            with self.lock:
                self.preparing.discard(expected_slug)

    def submit(self, slug, outcome, move, tick="0.01", token=None):
        with self.lock:
            last_slot, last_created = self.trade_window_state(slug)
            elapsed = time.time() - last_created
            if last_created and elapsed < TRADE_COOLDOWN_SECONDS:
                self.status = (f"trade cooldown "
                               f"{TRADE_COOLDOWN_SECONDS - elapsed:.2f}s")
                return False
            slot = last_slot + 1
            trade_key = f"{slug}#jump-{slot}"
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute(
                    "INSERT OR IGNORE INTO jump_entries VALUES (?,?,?,?,?,?)",
                    (slug, slot, trade_key, outcome, move, time.time()))
                if not cursor.rowcount:
                    return False
            self.inflight.add(trade_key)
            self.status = (f"trade {slot} signal "
                           f"{outcome} ({move:+.2f}); submitting")
            self.latest_cache = (0.0, None)
        threading.Thread(target=self._execute,
                         args=(slug, trade_key, outcome, tick),
                         name="chart-jump-order", daemon=True).start()
        return True

    def _execute(self, expected_slug, trade_key, outcome, tick):
        bot = None
        try:
            bot = OpeningBot(
                asset="btc", amount=JUMP_STAKE, paper=True, entry_window=300,
                db_path=self.db_path, http_client=self.http,
                take_profit=JUMP_TP_RATE)
            with self.lock:
                cached = self.prepared.get(expected_slug)
            if cached:
                market, bot.condition_id = cached
                bot.market = market
            else:
                market = bot.discover(time.time())
            if not market or market[0] != expected_slug:
                raise ValueError("simulator market unavailable")
            slug, end, tokens = market
            bot.enter(trade_key, end, outcome, tokens[outcome], direct_tick=tick,
                      direct_limit=MAX_ENTRY_PRICE)
            row = bot.db.execute(
                "SELECT entry_state,tp_state FROM trades WHERE market=? AND outcome=?",
                (trade_key, outcome)).fetchone()
            with self.lock:
                self.status = (f"{outcome} " +
                               (f"{row['entry_state']}; TP {row['tp_state']}"
                                if row else "entry skipped"))
                self.position_cache = (None, 0.0, None)
                self.latest_cache = (0.0, None)
            bot.place_profits()
            row = bot.db.execute(
                "SELECT entry_state,tp_state FROM trades WHERE market=? AND outcome=?",
                (trade_key, outcome)).fetchone()
            if row:
                with self.lock:
                    self.status = (f"{outcome} {row['entry_state']}; "
                                   f"TP {row['tp_state']}")
                    self.position_cache = (None, 0.0, None)
                    self.latest_cache = (0.0, None)
        except Exception as exc:
            with self.lock:
                self.status = f"jump trade error: {type(exc).__name__}"
        finally:
            if bot is not None:
                bot.close()
            with self.lock:
                self.inflight.discard(trade_key)

    def _maintain(self):
        while not self.stop.wait(5):
            bot = None
            try:
                bot = OpeningBot(
                    asset="btc", amount=JUMP_STAKE, paper=True,
                    entry_window=300, db_path=self.db_path,
                    http_client=self.http, take_profit=JUMP_TP_RATE)
                bot.place_profits()
            except Exception as exc:
                with self.lock:
                    self.status = f"TP maintenance error: {type(exc).__name__}"
            finally:
                if bot is not None:
                    bot.close()

    def close(self):
        self.stop.set()
        self.http.close()


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
    def __init__(self, root, trader_factory=JumpPaperTrader):
        self.root = root
        root.title("BTC + Polymarket Confirmation Trend - PolySimulator")
        root.geometry("1450x820")
        root.minsize(1050, 620)
        root.configure(bg=BG)
        self.feed, self.btc_feed = MarketFeed(), PolymarketBTCFeed()
        self.coinbase_feed = CoinbasePriceFeed()
        self.jump_trader = trader_factory()
        self.last_jump_price = None
        self.last_coinbase_move = None
        self.directional_moves = deque(maxlen=3)
        self.trend_signal_zone = None
        self.btc_trend_ticks = deque()
        self.up_trend_ticks = deque()
        self.bid_pressure_samples = deque()
        self.discovery = Discovery(self.feed)
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
        content = tk.Frame(root, bg=BG)
        content.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.canvas = tk.Canvas(content, bg=PANEL, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.activity_panel = tk.Frame(
            content, bg=PANEL, width=370, padx=12, pady=10)
        self.activity_panel.pack(side="right", fill="y", padx=(10, 0))
        self.activity_panel.pack_propagate(False)
        self.last_activity_render = 0.0
        self._build_market_activity_panel()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.discovery.start()
        self.btc_feed.start()
        self.coinbase_feed.start()
        self.tick()

    def _build_market_activity_panel(self):
        panel = self.activity_panel
        tk.Label(panel, text="LIVE POLYMARKET ACTIVITY", bg=PANEL, fg=TEXT,
                 font=("Segoe UI Semibold", 12)).pack(anchor="w")
        tk.Label(panel, text="Shares/orders, not unique people", bg=PANEL,
                 fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 8))

        self.depth_pressure_text = tk.StringVar(value="UP --     DOWN --")
        tk.Label(panel, text="RESTING BID PRESSURE · TOP 12 LEVELS",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        tk.Label(panel, textvariable=self.depth_pressure_text, bg=PANEL,
                 fg=TEXT, font=("Consolas", 9, "bold")).pack(anchor="w")
        self.depth_pressure_canvas = tk.Canvas(
            panel, height=16, bg=GRID, highlightthickness=0)
        self.depth_pressure_canvas.pack(fill="x", pady=(2, 9))

        self.flow_pressure_text = tk.StringVar(value="UP --     DOWN --")
        tk.Label(panel, text="EXECUTED FLOW · LAST 1 SECOND",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        tk.Label(panel, textvariable=self.flow_pressure_text, bg=PANEL,
                 fg=TEXT, font=("Consolas", 9, "bold")).pack(anchor="w")
        self.flow_pressure_canvas = tk.Canvas(
            panel, height=16, bg=GRID, highlightthickness=0)
        self.flow_pressure_canvas.pack(fill="x", pady=(2, 9))

        self.liquidity_text = tk.StringVar(value="Waiting for order-book depth…")
        tk.Label(panel, textvariable=self.liquidity_text, bg=PANEL, fg=TEXT,
                 justify="left", anchor="w", font=("Consolas", 9)).pack(
                     fill="x", pady=(0, 8))

        tk.Label(panel, text="TOP DEPTH", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")
        self.depth_table = ttk.Treeview(
            panel, columns=("outcome", "side", "price", "shares"),
            show="headings", height=10)
        for column, title, width in (("outcome", "Outcome", 58),
                                     ("side", "Side", 45),
                                     ("price", "Price", 58),
                                     ("shares", "Shares", 82)):
            self.depth_table.heading(column, text=title)
            self.depth_table.column(column, width=width, anchor="e")
        self.depth_table.pack(fill="x", pady=(2, 9))

        tk.Label(panel, text="RECENT TRADES", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")
        self.market_trade_table = ttk.Treeview(
            panel, columns=("time", "outcome", "side", "price", "shares"),
            show="headings", height=10)
        for column, title, width in (("time", "Time", 65),
                                     ("outcome", "Out", 38),
                                     ("side", "Side", 40),
                                     ("price", "Price", 50),
                                     ("shares", "Shares", 67)):
            self.market_trade_table.heading(column, text=title)
            self.market_trade_table.column(column, width=width, anchor="e")
        self.market_trade_table.pack(fill="both", expand=True, pady=(2, 0))

    @staticmethod
    def _paint_pressure(canvas, up_percent):
        canvas.update_idletasks()
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        split = width * max(0.0, min(100.0, up_percent)) / 100.0
        canvas.delete("all")
        canvas.create_rectangle(0, 0, split, height, fill=UP_COLOR, outline="")
        canvas.create_rectangle(split, 0, width, height, fill=DOWN_COLOR, outline="")

    def _update_market_activity(self, now):
        # The source is an event-driven WebSocket, so a faster UI refresh does
        # not generate extra Polymarket API requests.
        if now - self.last_activity_render < 0.05:
            return
        self.last_activity_render = now
        activity = self.feed.market_activity(depth_levels=12, trade_seconds=60)
        depth, trades = activity.get("depth", {}), activity.get("trades", [])
        totals = {}
        for outcome in ("Up", "Down"):
            book = depth.get(outcome, {})
            totals[outcome] = {
                "bid": sum(row[1] for row in book.get("bids", [])),
                "ask": sum(row[1] for row in book.get("asks", [])),
            }
        bid_total = totals["Up"]["bid"] + totals["Down"]["bid"]
        up_depth_pct = (50.0 if not bid_total else
                        totals["Up"]["bid"] / bid_total * 100)
        self.bid_pressure_samples.append((now, up_depth_pct / 100.0))
        while (self.bid_pressure_samples and
               self.bid_pressure_samples[0][0] < now - WINDOW_SECONDS):
            self.bid_pressure_samples.popleft()
        flow_trades = [trade for trade in trades
                       if float(trade.get("timestamp", 0)) >= time.time() - 1.0]
        up_flow = down_flow = 0.0
        for trade in flow_trades:
            size, outcome, side = (float(trade.get("size", 0)),
                                   trade.get("outcome"), trade.get("side"))
            if ((outcome == "Up" and side == "BUY") or
                    (outcome == "Down" and side == "SELL")):
                up_flow += size
            elif ((outcome == "Down" and side == "BUY") or
                  (outcome == "Up" and side == "SELL")):
                down_flow += size
        flow_total = up_flow + down_flow
        up_flow_pct = 50.0 if not flow_total else up_flow / flow_total * 100
        self.depth_pressure_text.set(
            f"UP {up_depth_pct:5.1f}%     DOWN {100-up_depth_pct:5.1f}%")
        self.flow_pressure_text.set(
            f"UP {up_flow:,.2f}     DOWN {down_flow:,.2f} shares")
        self._paint_pressure(self.depth_pressure_canvas, up_depth_pct)
        self._paint_pressure(self.flow_pressure_canvas, up_flow_pct)
        net = up_flow - down_flow
        self.liquidity_text.set(
            f"UP    bids {totals['Up']['bid']:>9,.2f}  asks {totals['Up']['ask']:>9,.2f}\n"
            f"DOWN  bids {totals['Down']['bid']:>9,.2f}  asks {totals['Down']['ask']:>9,.2f}\n"
            f"1s    trades {len(flow_trades):>4}       net {net:+9,.2f} UP")

        self.depth_table.delete(*self.depth_table.get_children())
        for outcome in ("Up", "Down"):
            book = depth.get(outcome, {})
            for side, rows in (("BID", book.get("bids", [])[:5]),
                               ("ASK", book.get("asks", [])[:5])):
                for price, size in rows:
                    self.depth_table.insert(
                        "", "end", values=(outcome.upper(), side,
                                           f"{price:.3f}", f"{size:,.2f}"))
        self.market_trade_table.delete(*self.market_trade_table.get_children())
        for trade in reversed(trades[-10:]):
            stamp = datetime.fromtimestamp(float(trade["timestamp"]))
            self.market_trade_table.insert(
                "", "end", values=(stamp.strftime("%H:%M:%S"),
                                   str(trade.get("outcome", "--")).upper(),
                                   trade.get("side", "--"),
                                   f"{float(trade.get('price', 0)):.3f}",
                                   f"{float(trade.get('size', 0)):,.2f}"))

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
            self.jump_trader.prepare(market[0])
            self.samples.clear()
            self.last_jump_price = None
            self.last_coinbase_move = None
            self.directional_moves.clear()
            self.trend_signal_zone = None
            self.btc_trend_ticks.clear()
            self.up_trend_ticks.clear()
            self.bid_pressure_samples.clear()
            self.coinbase_offset = None
        if (btc is not None and coinbase_price is not None and
                self.coinbase_offset is None):
            self.coinbase_offset = btc - coinbase_price
        current_up = quotes.get("Up")
        if current_up is not None:
            up_value = float(current_up.probability)
            self.up_trend_ticks.append((now, up_value))
        while (self.up_trend_ticks and
               self.up_trend_ticks[0][0] < now - TREND_SECONDS):
            self.up_trend_ticks.popleft()
        if coinbase_price is not None and coinbase_price != self.last_jump_price:
            previous_price = self.last_jump_price
            self.last_coinbase_move = (None if previous_price is None else
                                       coinbase_price - previous_price)
            self.last_jump_price = coinbase_price
            current_trend_price = (coinbase_price if self.coinbase_offset is None
                                   else coinbase_price + self.coinbase_offset)
            self.btc_trend_ticks.append((now, current_trend_price))
            while (self.btc_trend_ticks and
                   self.btc_trend_ticks[0][0] < now - TREND_SECONDS):
                self.btc_trend_ticks.popleft()
            market_open_for = (time.time() - (market[1] - 300)) if market else 0
            if market_open_for < TRADE_START_DELAY_SECONDS:
                self.directional_moves.clear()
            slow = price_trend(list(self.btc_trend_ticks), min_move=0)
            slow_move = slow[1] if slow else 0.0
            raw_zone = ("Up" if slow_move >= TREND_ENTRY_THRESHOLD else
                        "Down" if slow_move <= -TREND_ENTRY_THRESHOLD else None)
            if raw_zone is None:
                self.trend_signal_zone = None
            confirmation, reason = confirmed_trend_signal(
                list(self.btc_trend_ticks), list(self.up_trend_ticks),
                price_to_beat, now)
            signal = (confirmation is not None and
                      confirmation["outcome"] != self.trend_signal_zone)
            if (signal and market and
                    market_open_for >= TRADE_START_DELAY_SECONDS):
                outcome = confirmation["outcome"]
                move = confirmation["move"]
                quote = quotes.get(outcome)
                if quote is None:
                    with self.jump_trader.lock:
                        self.jump_trader.status = (
                            f"skipped {outcome}: live ask unavailable")
                    submitted = False
                    self.directional_moves.clear()
                elif quote.ask - quote.bid > MAX_ENTRY_SPREAD:
                    with self.jump_trader.lock:
                        self.jump_trader.status = (
                            f"skipped {outcome}: spread "
                            f"{quote.ask - quote.bid:.3f} > {MAX_ENTRY_SPREAD}")
                    submitted = False
                elif not MIN_ENTRY_PRICE <= quote.ask <= MAX_ENTRY_PRICE:
                    with self.jump_trader.lock:
                        self.jump_trader.status = (
                            f"skipped {outcome}: ask {quote.ask:.3f} outside "
                            f"{MIN_ENTRY_PRICE:.2f}-{MAX_ENTRY_PRICE:.2f}")
                    submitted = False
                    self.directional_moves.clear()
                else:
                    submitted = self.jump_trader.submit(
                        market[0], outcome, move, quote.tick,
                        token=market[2][outcome])
                if submitted:
                    self.trend_signal_zone = outcome
                    with self.jump_trader.lock:
                        self.jump_trader.status = (
                            f"{outcome} confirmed: slow {move:+.2f}, fast "
                            f"{confirmation['fast_move']:+.2f}, UP contract "
                            f"{confirmation['up_move']:+.3f}, ticks "
                            f"{confirmation['directional_ratio']:.0%}; "
                            "order submitted")
            elif raw_zone is not None and confirmation is None:
                with self.jump_trader.lock:
                    self.jump_trader.status = f"trend filtered: {reason}"
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
        position = self.jump_trader.position(market[0]) if market else None
        latest_trade = self.jump_trader.latest_trade()
        if latest_trade:
            entry = (f" @ {float(latest_trade['price']):.4f}"
                     if latest_trade.get("price") else "")
            shares = (f" | {float(latest_trade['shares']):.3f} shares"
                      if latest_trade.get("shares") else "")
            target = (f" | TP {latest_trade['target']:.4f} "
                      f"{str(latest_trade.get('tp_state') or '').upper()}"
                      if latest_trade.get("target") is not None else "")
            jump = (f" | TREND {float(latest_trade['move']):+.2f} USD"
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
        self._update_market_activity(now)
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
        c.create_text(left + 270, 18, text="UP bid pressure",
                      fill=BTC_COLOR, anchor="w")
        trend_points = [
            (stamp, coinbase_price)
            for stamp, coinbase_price, _, _ in self.samples
            if coinbase_price is not None and stamp >= now - TREND_SECONDS]
        trend = price_trend(trend_points)
        if trend:
            direction, move, _, _ = trend
            trend_color = (UP_COLOR if direction == "UP" else
                           DOWN_COLOR if direction == "DOWN" else MUTED)
            arrow = "↑" if direction == "UP" else "↓" if direction == "DOWN" else "→"
            c.create_text(left + 400, 18,
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
                              text=(f"TP on UP axis {position['target']:.3f}  "
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
                c.create_line(*points[name], fill=color, width=2)
        pressure_points = []
        for stamp, pressure in self.bid_pressure_samples:
            x = left + (stamp - start) / WINDOW_SECONDS * (right - left)
            pressure_points += [x, bottom - pressure * (bottom - top)]
        if len(pressure_points) >= 4:
            c.create_line(*pressure_points, fill=BTC_COLOR, width=2,
                          dash=(5, 3))
    def close(self):
        self.discovery.stop.set()
        self.btc_feed.stop.set()
        self.coinbase_feed.stop.set()
        self.jump_trader.close()
        self.feed.generation += 1
        self.root.destroy()


def main():
    load_dotenv(ROOT / ".env")
    root = tk.Tk()
    PriceChart(root)
    root.mainloop()


if __name__ == "__main__":
    main()
