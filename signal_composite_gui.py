"""Live graphical dashboard for the BTCUSD 1m weighted composite signal.

Dependency-free (tkinter only) and matches the dark theme of the other bot
GUIs in this folder. Data + indicators come from signal_composite.py (Binance
spot REST: 1m klines + recent trades).

Panels:
  - price chart: 1m closes + SMA9 / SMA21 / VWAP overlays
  - composite:   area chart of the weighted signal with buy/sell bands
  - RSI(14) and market skew (rolling, sampled live)
  - per-component gauges with weights and the composite verdict

Run:
  python signal_composite_gui.py [--interval 5] [--bars 180]
"""
import argparse
import json
import math
import threading
import time
from collections import deque
from tkinter import font as tkfont

import tkinter as tk
from tkinter import ttk

from signal_composite import (COMPONENTS, DEFAULT_WEIGHTS, KLINE_LIMIT,
                              LiveState, SYMBOL, snapshot, verdict)

# ---- theme (matches bot_gui.py) -------------------------------------------
BG = "#101217"
PANEL = "#181c24"
LINE = "#2c3340"
TEXT = "#e9edf5"
MUTED = "#8896ac"
GREEN = "#93e0be"
RED = "#f09caa"
BLUE = "#80baff"
PURPLE = "#c5a3ff"
AMBER = "#ffd080"
FONT = "Segoe UI"

COMPONENT_COLORS = {
    "rsi": BLUE,
    "momentum": PURPLE,
    "vwap": AMBER,
    "sma": GREEN,
    "skew": RED,
}
COMPONENT_LABELS = {
    "rsi": "RSI(14)",
    "momentum": "Momentum(10m)",
    "vwap": "VWAP dev",
    "sma": "SMA 9/21 cross",
    "skew": "Market skew",
}


def fmt(x, spec="{:,.2f}"):
    return "--" if x is None or not math.isfinite(x) else spec.format(x)


def sig_color(v, strong=0.6, weak=0.2):
    if v is None:
        return MUTED
    if v >= strong:
        return GREEN
    if v <= -strong:
        return RED
    if v >= weak:
        return "#b7ecd5"
    if v <= -weak:
        return "#f8c4cc"
    return MUTED


class LineChart(tk.Canvas):
    """Multi-series line/area chart, y auto-scaled, None values break the line."""

    def __init__(self, master, height=180, y_range=None, fill_below_zero=False,
                 hlines=None, bg=PANEL, **kw):
        super().__init__(master, height=height, bg=bg, highlightthickness=0, **kw)
        self.y_range = y_range          # fixed (lo, hi) or None for autoscale
        self.fill_below_zero = fill_below_zero
        self.hlines = hlines or []      # [(value, color, dashed)]
        self.series = []                # [(name, values, color, width, dashed, fill)]
        self.times = []
        self.padding = {"l": 62, "r": 10, "t": 10, "b": 20}
        self.bind("<Configure>", lambda e: self.draw())

    def set_data(self, times, series):
        self.times = times
        self.series = series
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 100 or h < 60 or not self.series:
            return
        p = self.padding
        n = len(self.times)
        if n < 2:
            return
        plot_w = w - p["l"] - p["r"]
        plot_h = h - p["t"] - p["b"]

        # y range
        if self.y_range:
            lo, hi = self.y_range
        else:
            vals = [v for _, s, *_ in self.series for v in s if v is not None and math.isfinite(v)]
            if not vals:
                return
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            lo -= span * 0.06
            hi += span * 0.06

        def x(i):
            return p["l"] + i / (n - 1) * plot_w

        def y(v):
            return p["t"] + (1 - (v - lo) / (hi - lo)) * plot_h

        # grid + y labels
        self.configure(bg=PANEL)
        f_small = tkfont.Font(family=FONT, size=8)
        for gi in range(5):
            gy = p["t"] + gi / 4 * plot_h
            gv = hi - gi / 4 * (hi - lo)
            self.create_line(p["l"], gy, w - p["r"], gy, fill=LINE)
            self.create_text(p["l"] - 6, gy, anchor="e", text=fmt(gv, "{:,.2f}"),
                             fill=MUTED, font=f_small)
        # x labels (time on the bars)
        step = max(1, n // 6)
        for i in range(0, n, step):
            t = self.times[i]
            self.create_text(x(i), h - 6, anchor="s", text=t.strftime("%H:%M"),
                             fill=MUTED, font=f_small)

        # fixed horizontal lines (thresholds etc.)
        for hv, color, dashed in self.hlines:
            if lo <= hv <= hi:
                self.create_line(p["l"], y(hv), w - p["r"], y(hv),
                                 fill=color, dash=(4, 3) if dashed else ())

        # series: fills first, then lines
        for name, vals, color, width, dashed, fill in self.series:
            pts = []
            seg = []
            for i, v in enumerate(vals):
                if v is None or not math.isfinite(v):
                    if len(seg) > 1:
                        pts.append(seg)
                    seg = []
                else:
                    seg.append((x(i), y(v)))
            if len(seg) > 1:
                pts.append(seg)
            if fill and self.fill_below_zero and lo < 0 < hi:
                y0 = y(0.0)
                for seg in pts:
                    poly = [seg[0][0], y0] + [c for pt in seg for c in pt] + [seg[-1][0], y0]
                    self.create_polygon(*poly, fill=color, outline="")
            for seg in pts:
                flat = [c for pt in seg for c in pt]
                self.create_line(*flat, fill=color, width=width,
                                 smooth=False, dash=(5, 3) if dashed else ())


class Candles(tk.Canvas):
    """Candlestick chart (red down / green up) with optional overlay lines."""

    def __init__(self, master, height=400, bg=PANEL, **kw):
        super().__init__(master, height=height, bg=bg, highlightthickness=0, **kw)
        self.ohlc = []        # [(o, h, l, c), ...]
        self.times = []
        self.overlays = []    # [(name, values, color, width, dashed)]
        self.price = None     # current (live) price for the moving line + badge
        self.price_dir = 1    # +1 up / -1 down vs previous tick (badge color)
        self.padding = {"l": 62, "r": 74, "t": 10, "b": 20}
        self.bind("<Configure>", lambda e: self.draw())

    def set_data(self, times, ohlc, overlays, price=None, price_dir=1):
        self.times = times
        self.ohlc = ohlc
        self.overlays = overlays
        if price is not None:
            self.price = price
            self.price_dir = price_dir
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        n = len(self.ohlc)
        if w < 100 or h < 60 or n < 2:
            return
        p = self.padding
        plot_w = w - p["l"] - p["r"]
        plot_h = h - p["t"] - p["b"]

        vals = [v for oh in self.ohlc for v in oh]
        for _, s, *_ in self.overlays:
            vals.extend(v for v in s if v is not None and math.isfinite(v))
        if self.price is not None and math.isfinite(self.price):
            vals.append(self.price)
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        lo -= span * 0.05
        hi += span * 0.05

        def x(i):
            return p["l"] + (i + 0.5) / n * plot_w

        def y(v):
            return p["t"] + (1 - (v - lo) / (hi - lo)) * plot_h

        # grid + y labels
        f_small = tkfont.Font(family=FONT, size=8)
        for gi in range(5):
            gy = p["t"] + gi / 4 * plot_h
            gv = hi - gi / 4 * (hi - lo)
            self.create_line(p["l"], gy, w - p["r"], gy, fill=LINE)
            self.create_text(p["l"] - 6, gy, anchor="e", text=fmt(gv, "{:,.2f}"),
                             fill=MUTED, font=f_small)
        step = max(1, n // 6)
        for i in range(0, n, step):
            self.create_text(x(i), h - 6, anchor="s",
                             text=self.times[i].strftime("%H:%M"), fill=MUTED, font=f_small)

        # candles
        body_w = max(1.0, plot_w / n * 0.65)
        live = self.price is not None
        for i, (o, hi_c, lo_c, c) in enumerate(self.ohlc):
            is_forming = live and i == n - 1
            color = GREEN if c >= o else RED
            cx = x(i)
            self.create_line(cx, y(hi_c), cx, y(lo_c), fill=color,
                             width=2 if is_forming else 1)
            y1, y2 = y(max(o, c)), y(min(o, c))
            if y2 - y1 < 1.5:
                y2 = y1 + 1.5
            self.create_rectangle(cx - body_w / 2, y1, cx + body_w / 2, y2,
                                  fill=color, outline=color)
            if is_forming:
                # subtle bright halo around the live candle so it stands out
                self.create_rectangle(cx - body_w / 2 - 2, y1 - 2,
                                      cx + body_w / 2 + 2, y2 + 2,
                                      outline="#e9edf5", width=1)

        # overlays
        for name, vals_s, color, width, dashed in self.overlays:
            seg = []
            for i, v in enumerate(vals_s):
                if v is None or not math.isfinite(v):
                    if len(seg) > 1:
                        self.create_line(*[c for pt in seg for c in pt],
                                         fill=color, width=width, dash=(5, 3) if dashed else ())
                    seg = []
                else:
                    seg.append((x(i), y(v)))
            if len(seg) > 1:
                self.create_line(*[c for pt in seg for c in pt],
                                 fill=color, width=width, dash=(5, 3) if dashed else ())

        # current-price line + right-edge badge (the always-moving "live" marker)
        if live:
            pyc = y(self.price)
            pcolor = GREEN if self.price_dir >= 0 else RED
            self.create_line(p["l"], pyc, w - p["r"], pyc, fill=pcolor,
                             width=1, dash=(2, 2))
            tag = f"{self.price:,.2f}"
            f_tag = tkfont.Font(family=FONT, size=8, weight="bold")
            tw = f_tag.measure(tag) + 8
            bx0, bx1 = w - p["r"] + 2, w - p["r"] + 2 + tw
            by0, by1 = pyc - 9, pyc + 9
            self.create_rectangle(bx0, by0, bx1, by1, fill=pcolor, outline=pcolor)
            self.create_text((bx0 + bx1) / 2, (by0 + by1) / 2, text=tag,
                             fill=BG, font=f_tag)


class Gauge(tk.Canvas):
    """Horizontal [-1, +1] gauge with a center pivot and a value marker."""

    def __init__(self, master, height=26, **kw):
        super().__init__(master, height=height, bg=PANEL, highlightthickness=0, **kw)
        self.value = 0.0
        self.bind("<Configure>", lambda e: self.draw())

    def set_value(self, v):
        self.value = 0.0 if (v is None or not math.isfinite(v)) else max(-1, min(1, v))
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40:
            return
        mid, r = w / 2, h / 2 - 3
        self.create_line(4, r, w - 4, r, fill=LINE, width=3)
        # green/red half tints
        self.create_rectangle(4, r - 5, mid, r + 5, fill="#1c2b26", outline="")
        self.create_rectangle(mid, r - 5, w - 4, r + 5, fill="#2b1c20", outline="")
        self.create_line(mid, r - 7, mid, r + 7, fill=MUTED)
        px = 4 + (self.value + 1) / 2 * (w - 8)
        self.create_line(px, r - 8, px, r + 8, fill=sig_color(self.value), width=3)
        self.create_oval(px - 4, r - 4, px + 4, r + 4, fill=sig_color(self.value), outline="")


class App:
    MAX_SKEW = 240          # rolling skew samples to keep (one per refresh)

    def __init__(self, root, weights, interval, bars):
        self.root = root
        self.weights = weights
        self.interval = interval        # REST snapshot poll (klines history)
        self.bars = min(bars, KLINE_LIMIT)
        self.last = None                # latest REST snapshot (chart history)
        self.live = None                # latest live (WebSocket-derived) dict
        self.live_state = LiveState(weights)
        self._pending = deque(maxlen=64)
        self.ws_active = False
        self._last_rendered = None
        self.last_price = None
        self.last_price_txt = ""
        self.error = None
        self._last_update_ts = None
        self._last_tick_ts = None
        self._last_update_str = ""
        self._pulse = True
        self._last_skew_minute = None
        self.skew_hist = deque(maxlen=self.MAX_SKEW)
        self.running = True

        root.title(f"{SYMBOL} 1m Composite Signal")
        root.geometry("1180x900")
        root.minsize(960, 720)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=(FONT, 10))
        self.build()
        self._sync_zoom()
        threading.Thread(target=self.worker_ws, daemon=True).start()
        threading.Thread(target=self.worker, daemon=True).start()
        self.heartbeat()
        self.tick()

    def _sync_zoom(self):
        """Point the zoom selector at the option matching self.bars (if any)."""
        match = [k for k, v in self.ZOOM_BARS.items() if v == self.bars]
        if match:
            self.cmb_zoom.set(match[0])

    # ---- layout -----------------------------------------------------------
    def build(self):
        f_title = tkfont.Font(family=FONT, size=15, weight="bold")
        f_big = tkfont.Font(family=FONT, size=22, weight="bold")
        f_med = tkfont.Font(family=FONT, size=11)
        f_small = tkfont.Font(family=FONT, size=9)
        f_tiny = tkfont.Font(family=FONT, size=8)

        # header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=14, pady=(12, 4))
        self.lbl_title = tk.Label(hdr, text=f"{SYMBOL}  1-minute composite signal",
                                  bg=BG, fg=TEXT, font=f_title, anchor="w")
        self.lbl_title.pack(side="left")
        self.lbl_clock = tk.Label(hdr, text="", bg=BG, fg=MUTED, font=f_small, anchor="e")
        self.lbl_clock.pack(side="right")
        # pulsing LIVE indicator
        self.lbl_live = tk.Label(hdr, text="● LIVE", bg=BG, fg=GREEN, font=f_med, anchor="e")
        self.lbl_live.pack(side="right", padx=(0, 12))
        # zoom selector: fewer bars = bigger candles, more obvious movement
        self.cmb_zoom = ttk.Combobox(hdr, values=["30m", "60m", "120m", "3h", "5h"],
                                     state="readonly", width=6)
        self.cmb_zoom.set("3h")
        self.cmb_zoom.pack(side="right", padx=(0, 10))
        self.cmb_zoom.bind("<<ComboboxSelected>>", self.on_zoom)
        tk.Label(hdr, text="zoom", bg=BG, fg=MUTED, font=f_small).pack(side="right")

        # status row: price / ATR / VWAP / SMAs / verdict
        status = tk.Frame(self.root, bg=BG)
        status.pack(fill="x", padx=14, pady=(0, 8))
        self.lbl_price = tk.Label(status, text="--", bg=BG, fg=TEXT, font=f_big)
        self.lbl_price.pack(side="left")
        self.lbl_verdict = tk.Label(status, text="--", bg=BG, fg=MUTED, font=f_med)
        self.lbl_verdict.pack(side="left", padx=(18, 0), pady=(6, 0))
        self.lbl_meta = tk.Label(status, text="", bg=BG, fg=MUTED, font=f_small, anchor="e")
        self.lbl_meta.pack(side="right")
        self.lbl_error = tk.Label(self.root, text="", bg=BG, fg=RED, font=f_tiny, anchor="w")
        self.lbl_error.pack(fill="x", padx=14)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=5)
        body.rowconfigure(1, weight=3)

        def panel(parent, r, c, title):
            fr = tk.Frame(parent, bg=PANEL, bd=1, relief="flat")
            fr.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            tk.Label(fr, text=title, bg=PANEL, fg=MUTED, font=f_small,
                     anchor="w").pack(fill="x", padx=10, pady=(6, 0))
            return fr

        # price chart (left column, spans both rows)
        price_fr = panel(body, 0, 0, "price 1m  —  candles | SMA9 | SMA21 | VWAP")
        self.price_chart = Candles(price_fr, height=420)
        self.price_chart.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        price_fr.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)

        # composite chart
        comp_fr = panel(body, 0, 1, "composite signal  —  green band BUY / red band SELL")
        self.comp_chart = LineChart(comp_fr, height=220, y_range=(-1.1, 1.1),
                                    fill_below_zero=True,
                                    hlines=[(0.2, GREEN, True), (-0.2, RED, True),
                                            (0.6, GREEN, True), (-0.6, RED, True)])
        self.comp_chart.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # bottom row
        rsi_fr = panel(body, 1, 0, "RSI(14)")
        self.rsi_chart = LineChart(rsi_fr, height=200, y_range=(0, 100),
                                   hlines=[(70, RED, True), (30, GREEN, True)])
        self.rsi_chart.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # gauges panel
        gauge_fr = panel(body, 1, 1, "component signals  (raw value — weight — normalized)")
        self.gauges = {}
        self.gauge_labels = {}
        for idx, name in enumerate(COMPONENTS):
            row = tk.Frame(gauge_fr, bg=PANEL)
            row.pack(fill="x", padx=10, pady=3)
            lbl = tk.Label(row, text=f"{COMPONENT_LABELS[name]:<18}", bg=PANEL,
                           fg=TEXT, font=f_med, width=18, anchor="w")
            lbl.pack(side="left")
            val = tk.Label(row, text="", bg=PANEL, fg=MUTED, font=f_small, width=14, anchor="w")
            val.pack(side="left")
            g = Gauge(row)
            g.pack(side="left", fill="x", expand=True, padx=(6, 2))
            wlab = tk.Label(row, text=f"w={self.weights[name]:.2f}", bg=PANEL, fg=MUTED,
                            font=f_tiny, width=7, anchor="e")
            wlab.pack(side="right")
            self.gauges[name] = g
            self.gauge_labels[name] = val
        # composite gauge row
        c_row = tk.Frame(gauge_fr, bg=PANEL)
        c_row.pack(fill="x", padx=10, pady=(8, 8))
        tk.Label(c_row, text=f"{'COMPOSITE':<18}", bg=PANEL, fg=TEXT, font=f_med,
                 width=18, anchor="w").pack(side="left")
        self.gauge_label_comp = tk.Label(c_row, text="", bg=PANEL, fg=MUTED, font=f_small,
                                         width=14, anchor="w")
        self.gauge_label_comp.pack(side="left")
        self.gauge_comp = Gauge(c_row)
        self.gauge_comp.pack(side="left", fill="x", expand=True, padx=(6, 2))
        tk.Label(c_row, text="w=1.00", bg=PANEL, fg=MUTED, font=f_tiny, width=7,
                 anchor="e").pack(side="right")

    # ---- data loop ---------------------------------------------------------
    def worker_ws(self):
        """Binance WebSocket trade stream -> live_state. Auto-reconnects.

        Trades are pushed to _pending; the GUI thread drains them every 50ms
        (tkinter is single-threaded). REST worker re-seeds the state on every
        snapshot so the two can never drift apart.
        """
        while self.running:
            try:
                import websockets
                loop = __import__("asyncio").new_event_loop()

                async def run():
                    url = (f"wss://stream.binance.com:9443/ws/"
                           f"{SYMBOL.lower()}@aggTrade")
                    while self.running:
                        try:
                            async with websockets.connect(url, ping_interval=15,
                                                          close_timeout=2) as ws:
                                self.ws_active = True
                                async for msg in ws:
                                    if not self.running:
                                        break
                                    try:
                                        d = json.loads(msg)
                                        self._pending.append((
                                            float(d["p"]),
                                            float(d["q"]),
                                            int(d["T"]),
                                            bool(d["m"]),
                                        ))
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        self.ws_active = False
                        if self.running:
                            await __import__("asyncio").sleep(1.0)

                loop.run_until_complete(run())
                loop.close()
            except Exception:
                pass
            self.ws_active = False
            time.sleep(1.0)

    def worker(self):
        """REST worker: full kline history + indicator re-sync (drift anchor)."""
        import httpx
        while self.running:
            try:
                with httpx.Client(timeout=10) as client:
                    self.last = snapshot(client, self.weights)
                self.error = None
                self.live_state.seed(self.last)
                # sample the REST skew into the rolling history on a per-minute
                # cadence (used to backfill the composite chart)
                m = int(time.time() // 60)
                if m != self._last_skew_minute:
                    self._last_skew_minute = m
                    self.skew_hist.append(self.last["signals"]["skew"])
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            time.sleep(self.interval)

    def flush_live(self):
        """Drain pending WebSocket trades and compute the latest live state."""
        try:
            n0 = len(self._pending)
            if not n0:
                return
            while self._pending:
                p, q, t, m = self._pending.popleft()
                self.live = self.live_state.on_trade(p, q, t, m)
            self._last_tick_ts = time.time()
        except Exception:
            pass

    def tick(self):
        if self.error:
            self.lbl_error.config(text=f"feed error: {self.error}")
        else:
            self.lbl_error.config(text="")
        self.flush_live()
        if self.last is not None:
            new = self.last is not self._last_rendered
            if new:
                self._last_rendered = self.last
                self._last_update_ts = time.time()
                self.render(new_rest=True)
            elif self.live is not None:
                self.render(new_rest=False)
        self.root.after(50, self.tick)

    # fast heartbeat: pulses the LIVE dot and updates "N ms ago" every 100ms
    def heartbeat(self):
        now = time.time()
        if self.error:
            self.lbl_live.config(text="● STALLED", fg=RED)
        elif self._last_update_ts is None:
            self.lbl_live.config(text="● CONNECTING…", fg=AMBER)
        else:
            age_ms = int((now - self._last_update_ts) * 1000)
            self._pulse = not self._pulse
            on = self._pulse
            ws_on = self.ws_active or self.live is not None
            tick_age = (self._last_tick_ts is not None
                        and int((now - self._last_tick_ts) * 1000))
            if ws_on:
                self.lbl_live.config(text="● WS LIVE", fg=GREEN if on else "#2f6b4f")
                self.lbl_clock.config(
                    text=f"updated {self._last_update_str}  ·  "
                         f"{tick_age if tick_age is not None else '--'} ms since trade  [ws]")
            else:
                self.lbl_live.config(text="● REST", fg=AMBER if on else "#8a6d3b")
                self.lbl_clock.config(
                    text=f"updated {self._last_update_str}  ·  {age_ms} ms ago  [rest]")
        self.root.after(100, self.heartbeat)

    # change the displayed bar count (zoom) without restarting
    ZOOM_BARS = {"30m": 30, "60m": 60, "120m": 120, "3h": 180, "5h": 300}

    def on_zoom(self, _event=None):
        self.bars = self.ZOOM_BARS.get(self.cmb_zoom.get(), self.bars)
        if self.last is not None:
            self.render(new_rest=True)

    # ---- rendering ----------------------------------------------------------
    def render(self, new_rest):
        s = self.last
        h = s["history"]

        if new_rest:
            times = h["times"]
            n = len(times)
            # trim to requested bar count
            if n > self.bars:
                times = times[-self.bars:]
                sl = slice(-self.bars, None)
            else:
                sl = slice(None)
            ohlc = list(h["ohlc"][sl])
            times = list(times)

            # composite history: backfill from per-bar series; skew history
            # from live samples overlays the most recent part of the tail
            comp = list(h["composite"][sl])
            live_skew = list(self.skew_hist)
            m = min(len(live_skew), len(comp))
            if m:
                base = comp[: len(comp) - m]
                for i in range(m):
                    raw = {}
                    for name in COMPONENTS:
                        v = h["components"][name][len(h["closes"]) - m + i]
                        raw[name] = 0.0 if v is None else v
                    raw["skew"] = live_skew[len(live_skew) - m + i]
                    wsum = sum(self.weights[x] for x in COMPONENTS)
                    base.append(max(-1, min(1, sum(self.weights[x] * raw[x] for x in COMPONENTS) / wsum)))
                comp = base
            self.comp_chart.set_data(times, [("composite", comp, AMBER, 2, False, True)])
            rsi_vals = list(h["rsi"][sl])
            self.rsi_chart.set_data(times, [("rsi", rsi_vals, PURPLE, 2, False, False)])
            self._rest_chart = (
                ohlc, times,
                [("sma9", h["ema9"][sl], BLUE, 1, False),
                 ("sma21", h["ema21"][sl], AMBER, 1, False),
                 ("vwap", h["vwap"][sl], GREEN, 1, True)],
            )
            self._last_update_str = f"{s['time']} · bar {s['bar']}"

        if not hasattr(self, "_rest_chart"):
            return
        ohlc, times, overlays = self._rest_chart

        # ---- merge live values (WebSocket) over the REST snapshot ----------
        lv = self.live
        if lv is not None:
            s = {**s, **lv}
            # advance the forming candle + extend the forming-bar overlays
            price = s["price"]
            o, hi_c, lo_c, _c = ohlc[-1]
            ohlc[-1] = (o, max(hi_c, price), min(lo_c, price), price)
            overlays = [
                (nm, list(vals), col, w, dash) for nm, vals, col, w, dash in overlays
            ]
            overlays[0][1][-1] = s["sma9"]
            overlays[1][1][-1] = s["sma21"]
            overlays[2][1][-1] = s["vwap"]
        self._last_chart_price = s["price"]

        up = self.last_price is None or s["price"] >= self.last_price
        self.price_chart.set_data(times, ohlc, overlays,
                                  price=s["price"], price_dir=1 if up else -1)

        # header text
        txt = f"{s['price']:,.2f}"
        self.lbl_price.config(text=txt, fg=GREEN if up else RED)
        # flash on tick so small moves are visible
        if txt != self.last_price_txt:
            self.lbl_price.config(bg="#1c2b26" if up else "#2b1c20")
            self.root.after(700, lambda: self.lbl_price.config(bg=BG))
            self.last_price_txt = txt
        self.last_price = s["price"]
        meta = (f"ATR {fmt(s['atr'], '{:,.2f}')}   VWAP {fmt(s['vwap'], '{:,.2f}')}   "
                f"SMA9 {fmt(s['sma9'], '{:,.2f}')} / SMA21 {fmt(s['sma21'], '{:,.2f}')}")
        self.lbl_meta.config(text=meta)
        vtext = verdict(s["composite"])
        vcolor = sig_color(s["composite"])
        self.lbl_verdict.config(text=f"{vtext}  ({s['composite']:+.2f})", fg=vcolor)

        # gauges
        rawdisp = {
            "rsi": f"{fmt(s['rsi'], '{:.1f}')}",
            "momentum": f"{(math.exp(s['ret10']) - 1) if s['ret10'] is not None else 0:+.4%}",
            "vwap": f"{(s['price'] - s['vwap']) / s['atr']:+.2f}x ATR" if s["vwap"] and s["atr"] else "--",
            "sma": f"{(s['sma9'] - s['sma21']) / s['price']:+.4%}",
            "skew": f"{s['trades']} trades",
        }
        for name in COMPONENTS:
            self.gauges[name].set_value(s["signals"][name])
            self.gauge_labels[name].config(
                text=f"{rawdisp[name]}  {s['signals'][name]:+.2f}",
                fg=COMPONENT_COLORS[name])
        self.gauge_comp.set_value(s["composite"])
        self.gauge_label_comp.config(text=f"{s['composite']:+.2f}  {vtext}", fg=vcolor)

    def close(self):
        self.running = False
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="BTCUSD 1m composite signal GUI")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    ap.add_argument("--bars", type=int, default=180, help="bars to display (default 180, max 300)")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma list rsi,momentum,vwap,sma,skew (default .25,.2,.25,.2,.1)")
    args = ap.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            vals = [float(x) for x in args.weights.split(",")]
            assert len(vals) == 5 and all(v >= 0 for v in vals) and sum(vals) > 0
        except Exception:
            raise SystemExit("--weights must be 5 comma-separated non-negative numbers")
        weights = dict(zip(COMPONENTS, vals))

    root = tk.Tk()
    app = App(root, weights, max(2.0, args.interval), args.bars)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
