"""Read-only Polymarket BTC Up/Down WebSocket desktop analyzer."""

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass, replace

import httpx
import websocket

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def price(value):
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) and 0 <= result <= 1 else None


@dataclass
class Quote:
    bid: float | None = None
    ask: float | None = None
    received: float = 0
    timestamp: float = 0
    delay_ms: float | None = None
    updates: int = 0


class Feed:
    def __init__(self, minutes=5):
        self.minutes = minutes
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.quotes = {"Up": Quote(), "Down": Quote()}
        self.tokens = {}
        self.slug = ""
        self.end = 0
        self.status = "Discovering market…"
        self.connected = False

    def snapshot(self):
        with self.lock:
            return {"quotes": {k: replace(v) for k, v in self.quotes.items()},
                    "slug": self.slug, "end": self.end,
                    "status": self.status, "connected": self.connected}

    def discover(self, client):
        duration = self.minutes * 60
        start = int(time.time()) // duration * duration
        slug = f"btc-updown-{self.minutes}m-{start}"
        response = client.get("https://gamma-api.polymarket.com/markets", params={"slug": slug})
        response.raise_for_status()
        for market in response.json():
            if market.get("slug") != slug or market.get("closed") or not market.get("acceptingOrders"):
                continue
            outcomes, tokens = market["outcomes"], market["clobTokenIds"]
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
            if len(outcomes) != 2 or len(tokens) != 2 or set(outcomes) != {"Up", "Down"}:
                raise ValueError("Unexpected BTC outcome mapping")
            with self.lock:
                self.tokens = dict(zip(tokens, outcomes))
                self.slug, self.end = slug, start + duration
                self.quotes = {"Up": Quote(), "Down": Quote()}
            return
        raise ValueError(f"Waiting for active market: {slug}")

    def process(self, payload):
        events = payload if isinstance(payload, list) else [payload]
        with self.lock:
            for event in events:
                kind = event.get("event_type")
                if kind == "market_resolved":
                    if event.get("market") and any(t in self.tokens for t in event.get("assets_ids", [])):
                        self.connected = False
                        self.status = "Market resolved; waiting for next interval"
                    continue
                changes = event.get("price_changes", []) if kind == "price_change" else [event]
                if kind not in ("book", "price_change", "best_bid_ask"):
                    continue
                for change in changes:
                    outcome = self.tokens.get(change.get("asset_id"))
                    if not outcome:
                        continue
                    quote = self.quotes[outcome]
                    stamp = float(event.get("timestamp") or 0)
                    if stamp and stamp < quote.timestamp:
                        continue
                    if kind == "book":
                        bids = [price(x["price"]) for x in change.get("bids", []) if float(x["size"]) > 0]
                        asks = [price(x["price"]) for x in change.get("asks", []) if float(x["size"]) > 0]
                        quote.bid = max((x for x in bids if x is not None), default=None)
                        quote.ask = min((x for x in asks if x is not None), default=None)
                    else:
                        if "best_bid" not in change and "best_ask" not in change:
                            continue
                        if "best_bid" in change:
                            quote.bid = price(change["best_bid"])
                            # CLOB uses 0/1 as empty-side sentinels in top quotes.
                            if quote.bid == 0:
                                quote.bid = None
                        if "best_ask" in change:
                            quote.ask = price(change["best_ask"])
                            if quote.ask == 1:
                                quote.ask = None
                    quote.received = time.monotonic()
                    quote.timestamp = max(stamp, quote.timestamp)
                    quote.delay_ms = time.time() * 1000 - stamp if stamp else None
                    quote.updates += 1

    def run(self):
        retry = 0.5
        with httpx.Client(timeout=8) as client:
            while not self.stop.is_set():
                sock = None
                try:
                    with self.lock:
                        self.connected = False
                        self.status = "Connecting to Polymarket…"
                    self.discover(client)
                    sock = websocket.create_connection(WS_URL, timeout=5)
                    sock.send(json.dumps({"assets_ids": list(self.tokens), "type": "market", "custom_feature_enabled": True}))
                    sock.settimeout(0.5)
                    with self.lock:
                        self.connected = True
                        self.status = "LIVE · Polymarket WebSocket"
                    last_ping = last_message = time.monotonic()
                    while not self.stop.is_set() and time.time() < self.end:
                        now = time.monotonic()
                        if now - last_ping >= 10:
                            sock.send("PING")
                            last_ping = now
                        if now - last_message > 25:
                            raise TimeoutError("Feed heartbeat timed out")
                        try:
                            raw = sock.recv()
                        except websocket.WebSocketTimeoutException:
                            continue
                        if not raw:
                            raise ConnectionError("Feed disconnected")
                        last_message = time.monotonic()
                        retry = 0.5
                        if raw == "PONG":
                            continue
                        self.process(json.loads(raw))
                except Exception as exc:
                    with self.lock:
                        self.connected = False
                        self.status = f"Reconnecting: {exc}"
                    self.stop.wait(retry)
                    retry = min(retry * 2, 8)
                finally:
                    if sock:
                        sock.close()
                    with self.lock:
                        self.connected = False


def cents(value):
    return "—" if value is None else f"{value * 100:.2f}¢"


def gui(minutes):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Polymarket • BTC Up / Down")
    root.geometry("940x590")
    root.minsize(800, 530)
    bg, panel, muted = "#0b1120", "#141f32", "#8da2bf"
    root.configure(bg=bg)
    feed = None
    def label(parent, text="", size=12, color="#eaf1fc", background=bg):
        return tk.Label(parent, text=text, font=("Segoe UI", size), fg=color, bg=background)
    top = tk.Frame(root, bg=bg)
    top.pack(fill="x", padx=28, pady=(24, 10))
    label(top, "BTC / UP & DOWN", 24).pack(side="left")
    interval = ttk.Combobox(top, values=["5 minutes", "15 minutes"], state="readonly", width=12)
    interval.set(f"{minutes} minutes")
    interval.pack(side="right")
    status = label(root, "Connecting…", color=muted)
    status.pack(anchor="w", padx=30)
    market = label(root, "", size=10, color=muted)
    market.pack(anchor="w", padx=30, pady=(4, 15))
    cards = tk.Frame(root, bg=bg)
    cards.pack(fill="both", expand=True, padx=22)
    widgets = {}
    for column, (outcome, color) in enumerate((("Up", "#43e4b0"), ("Down", "#ff7f96"))):
        card = tk.Frame(cards, bg=panel, padx=20, pady=18)
        card.grid(row=0, column=column, sticky="nsew", padx=7)
        cards.columnconfigure(column, weight=1)
        label(card, outcome.upper(), 22, color, panel).pack(anchor="w")
        row = tk.Frame(card, bg=panel)
        row.pack(fill="x", pady=18)
        values = []
        for title in ("BID · sell", "ASK · buy"):
            side = tk.Frame(row, bg=panel)
            side.pack(side="left", expand=True, fill="x")
            label(side, title, 11, muted, panel).pack(anchor="w")
            value = label(side, "—", 32, color, panel)
            value.pack(anchor="w")
            values.append(value)
        detail = label(card, "Waiting for first quote", 12, muted, panel)
        detail.pack(anchor="w", pady=8)
        age = label(card, "", 10, muted, panel)
        age.pack(anchor="w")
        widgets[outcome] = (*values, detail, age)
    footer = label(root, "", 11, muted)
    footer.pack(anchor="w", padx=30, pady=(18, 6))
    label(root, "Contract prices in cents · Public market data · Display refresh: 25 ms", 10, muted).pack(anchor="w", padx=30)
    label(root, "Event delay uses your computer clock; it is not measured network latency.", 10, muted).pack(anchor="w", padx=30, pady=(4, 22))

    def start(_=None):
        nonlocal feed
        if feed:
            feed.stop.set()
        feed = Feed(int(interval.get().split()[0]))
        threading.Thread(target=feed.run, daemon=True).start()

    def render():
        state = feed.snapshot()
        expired = bool(state["end"] and time.time() >= state["end"])
        live = state["connected"] and not expired
        status.config(text="Rolling to next market…" if expired else state["status"], fg="#43e4b0" if live else "#ffc779")
        market.config(text=state["slug"])
        for outcome, quote in state["quotes"].items():
            bid, ask, detail, age = widgets[outcome]
            bid.config(text=cents(quote.bid) if live else "—")
            ask.config(text=cents(quote.ask) if live else "—")
            spread = quote.ask - quote.bid if quote.bid is not None and quote.ask is not None else None
            detail.config(text=f"Spread  {cents(spread)}" if live else "Waiting for connection / current market")
            seconds = time.monotonic() - quote.received if quote.received else None
            delay = f"{quote.delay_ms:.0f} ms" if quote.delay_ms is not None else "—"
            age.config(text=(f"Quote age {seconds:.2f}s · Event delay {delay}\n{quote.updates:,} quote events" if seconds is not None else "Waiting for first quote"), fg="#ffc779" if seconds and seconds > 10 else muted)
        remaining = max(0, int(state["end"] - time.time()))
        footer.config(text=f"Interval ends in {remaining // 60:02d}:{remaining % 60:02d} · Automatically follows the next BTC market")
        root.after(25, render)

    def close():
        feed.stop.set()
        root.destroy()

    interval.bind("<<ComboboxSelected>>", start)
    root.protocol("WM_DELETE_WINDOW", close)
    start()
    render()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, choices=(5, 15), default=5)
    parser.add_argument("--smoke", action="store_true", help="Verify live quotes without opening a window")
    args = parser.parse_args()
    if args.smoke:
        feed = Feed(args.minutes)
        threading.Thread(target=feed.run, daemon=True).start()
        deadline = time.monotonic() + 25
        try:
            while time.monotonic() < deadline:
                state = feed.snapshot()
                if state["connected"] and all(q.updates for q in state["quotes"].values()):
                    print(state)
                    break
                time.sleep(0.1)
            else:
                raise SystemExit(f"Live check failed: {feed.snapshot()}")
        finally:
            feed.stop.set()
    else:
        gui(args.minutes)
