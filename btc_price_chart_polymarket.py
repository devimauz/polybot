"""Standalone BTC chart and real Polymarket $1.10 jump trader.

Run without --live for signal-only dry runs. Add --live to submit real orders.
"""

import argparse
import json
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import tkinter as tk
from tkinter import ttk

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
TRADE_START_DELAY_SECONDS = 0.0
MAX_TRADES_PER_WINDOW = 1
JUMP_TP_RATE = Decimal("0.10")
MIN_ENTRY_PRICE = Decimal("0.30")
MAX_ENTRY_PRICE = Decimal("0.70")
STAKE = Decimal("1.10")
TAKE_PROFIT = Decimal("0.10")
ENTRY_SLIPPAGE = Decimal("0.03")
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
        self.thread = threading.Thread(
            target=self.run, name="coinbase-btc-feed", daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.price, self.error

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


class PolymarketJumpTrader:
    """One real Polymarket jump trade per five-minute window with 10% TP."""

    def __init__(self, live=False):
        self.live = live
        self.venue_name = "POLYMARKET LIVE" if live else "POLYMARKET DRY RUN"
        self.db_path = ROOT / (
            "chart-jump-polymarket-live.sqlite3" if live
            else "chart-jump-polymarket-dry-run.sqlite3")
        with sqlite3.connect(self.db_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS trades (
                market TEXT PRIMARY KEY, outcome TEXT, move REAL, token TEXT,
                tick TEXT, entry_state TEXT, price TEXT, shares TEXT,
                tp_state TEXT, target TEXT, created REAL, detail TEXT)""")
        self.lock = threading.Lock()
        self.status = "ready; waiting for jump"
        self.stop = threading.Event()
        self.position_cache = (None, 0.0, None)
        self.latest_cache = (0.0, None)
        self.client = None
        if live:
            self._connect()

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

    def snapshot(self):
        with self.lock:
            return self.status

    def prepare(self, slug):
        # Market tokens and live quotes already arrive through Polymarket's
        # Gamma API and CLOB WebSocket, so there is nothing else to preload.
        return None

    def trade_count(self, slug):
        with sqlite3.connect(self.db_path) as db:
            return db.execute("SELECT COUNT(*) FROM trades WHERE market=?",
                              (slug,)).fetchone()[0]

    def submit(self, slug, outcome, move, tick="0.01", token=None, ask=None):
        if not token or ask is None:
            with self.lock:
                self.status = "trade skipped: missing Polymarket token/ask"
            return False
        with self.lock:
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("""INSERT OR IGNORE INTO trades
                    (market,outcome,move,token,tick,entry_state,created,detail)
                    VALUES (?,?,?,?,?,'submitting',?,'')""",
                    (slug, outcome, move, str(token), str(tick), time.time()))
                if not cursor.rowcount:
                    self.status = "one-trade limit reached for this window"
                    return False
            self.status = f"{outcome} jump {move:+.2f}; submitting ${STAKE}"
        threading.Thread(target=self._execute,
                         args=(slug, outcome, str(token), str(tick), str(ask)),
                         name="chart-polymarket-order", daemon=True).start()
        return True

    def _update(self, slug, **fields):
        assignments = ",".join(f"{name}=?" for name in fields)
        with sqlite3.connect(self.db_path) as db:
            db.execute(f"UPDATE trades SET {assignments} WHERE market=?",
                       (*fields.values(), slug))
        with self.lock:
            self.position_cache = (None, 0.0, None)
            self.latest_cache = (0.0, None)

    def _execute(self, slug, outcome, token, tick_text, ask_text):
        tick = Decimal(tick_text)
        ask = Decimal(ask_text)
        limit = (min(Decimal("0.99"), ask + ENTRY_SLIPPAGE) /
                 tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        if not self.live:
            self._update(slug, entry_state="simulated",
                         detail=f"dry run ${STAKE}, ask {ask}, limit {limit}")
            with self.lock:
                self.status = (f"DRY RUN {outcome}; ${STAKE}, ask {ask}, "
                               f"limit {limit}")
            return
        filled = False
        try:
            from py_clob_client_v2 import (MarketOrderArgs, OrderArgs, OrderType,
                                           PartialCreateOrderOptions, Side)
            options = PartialCreateOrderOptions(tick_size=tick_text,
                                                neg_risk=False)
            signed = self.client.create_market_order(
                MarketOrderArgs(token_id=token, amount=float(STAKE),
                                side=Side.BUY, price=float(limit),
                                order_type=OrderType.FOK), options)
            response = self.client.post_order(signed, OrderType.FOK)
            if (not response.get("success") or
                    str(response.get("status", "")).lower() != "matched"):
                self._update(slug, entry_state="rejected",
                             detail=json.dumps(response)[:4000])
                with self.lock:
                    self.status = f"{outcome} order rejected"
                return
            paid = Decimal(str(response["makingAmount"]))
            shares = Decimal(str(response["takingAmount"]))
            price = paid / shares
            target = (price * (Decimal("1") + TAKE_PROFIT) / tick).to_integral_value(
                rounding=ROUND_CEILING) * tick
            self._update(slug, entry_state="filled", price=str(price),
                         shares=str(shares), tp_state="submitting",
                         target=str(target), detail=json.dumps(response)[:4000])
            filled = True
            with self.lock:
                self.status = f"{outcome} filled {price:.4f}; submitting TP {target:.4f}"
            if target >= Decimal("1"):
                self._update(slug, tp_state="unreachable")
                with self.lock:
                    self.status = f"{outcome} filled {price:.4f}; TP unreachable"
                return
            tp_signed = self.client.create_order(
                OrderArgs(token_id=token, price=float(target), size=float(shares),
                          side=Side.SELL), options)
            tp_response = self.client.post_order(tp_signed, OrderType.GTC)
            tp_status = str(tp_response.get("status", "")).lower()
            if tp_status not in ("live", "open", "matched", "pending"):
                raise ValueError("take-profit order not confirmed")
            self._update(slug, tp_state="submitted",
                         detail=json.dumps(tp_response)[:4000])
            with self.lock:
                self.status = f"{outcome} filled {price:.4f}; TP {target:.4f} submitted"
        except Exception as exc:
            fields = {"detail": f"{type(exc).__name__}: {str(exc)[:500]}"}
            if filled:
                fields["tp_state"] = "unknown"
            else:
                fields["entry_state"] = "unknown"
            self._update(slug, **fields)
            with self.lock:
                self.status = (("TP error" if filled else "order error") +
                               f": {type(exc).__name__}; check Polymarket")

    def _row(self, slug=None):
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            if slug:
                return db.execute("SELECT * FROM trades WHERE market=?",
                                  (slug,)).fetchone()
            return db.execute(
                "SELECT * FROM trades ORDER BY created DESC LIMIT 1").fetchone()

    def position(self, slug):
        now = time.monotonic()
        with self.lock:
            cached_slug, checked, cached = self.position_cache
            if cached_slug == slug and now - checked < 0.25:
                return cached
        row = self._row(slug)
        if not row or row["entry_state"] != "filled" or not row["price"]:
            position = None
        else:
            price = Decimal(row["price"])
            target = Decimal(row["target"])
            position = {
                "outcome": row["outcome"],
                "shares": float(row["shares"]),
                "entry": position_on_up_axis(row["outcome"], price),
                "target": position_on_up_axis(row["outcome"], target),
                "tp_state": row["tp_state"] or "unknown",
            }
        with self.lock:
            self.position_cache = (slug, now, position)
        return position

    def latest_trade(self):
        now = time.monotonic()
        with self.lock:
            checked, cached = self.latest_cache
            if now - checked < 0.25:
                return cached
        row = self._row()
        if not row:
            return None
        trade = {
            "market": row["market"], "outcome": row["outcome"],
            "move": row["move"], "entry_state": row["entry_state"],
            "price": row["price"], "shares": row["shares"],
            "target": None if row["target"] is None else float(row["target"]),
            "tp_state": row["tp_state"],
        }
        with self.lock:
            self.latest_cache = (now, trade)
        return trade

    def close(self):
        self.stop.set()


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
            self.jump_trader.prepare(market[0])
            self.samples.clear()
            self.last_jump_price = None
            self.last_coinbase_move = None
            self.directional_moves.clear()
            self.coinbase_offset = None
        if (btc is not None and coinbase_price is not None and
                self.coinbase_offset is None):
            self.coinbase_offset = btc - coinbase_price
        if coinbase_price is not None and coinbase_price != self.last_jump_price:
            previous_price = self.last_jump_price
            self.last_coinbase_move = (None if previous_price is None else
                                       coinbase_price - previous_price)
            self.last_jump_price = coinbase_price
            market_open_for = (time.time() - (market[1] - 300)) if market else 0
            if market_open_for < TRADE_START_DELAY_SECONDS:
                self.directional_moves.clear()
            elif self.last_coinbase_move is not None:
                if (self.directional_moves and
                        self.directional_moves[-1] * self.last_coinbase_move < 0):
                    self.directional_moves.clear()
                self.directional_moves.append(self.last_coinbase_move)
            signal = directional_jump(self.directional_moves,
                                      self.jump_threshold)
            if (signal and market and
                    market_open_for >= TRADE_START_DELAY_SECONDS):
                outcome, move = signal
                quote = quotes.get(outcome)
                if quote is None:
                    with self.jump_trader.lock:
                        self.jump_trader.status = (
                            f"skipped {outcome}: live ask unavailable")
                    submitted = False
                    self.directional_moves.clear()
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
                        token=market[2][outcome], ask=quote.ask)
                if submitted:
                    self.directional_moves.clear()
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
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Submit real-money Polymarket orders")
    args = parser.parse_args()
    root = tk.Tk()
    PriceChart(root, trader_factory=lambda: PolymarketJumpTrader(args.live))
    root.title("BTC Live - Polymarket $1.10 Jump Trader")
    root.mainloop()


if __name__ == "__main__":
    main()
