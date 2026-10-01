#!/usr/bin/env python3
"""
btc_multi_source.py
===================
Fetch the BTC/USD (BTCUSD) price from many public exchange APIs at a high
refresh rate (default 100 ms / 10 Hz) and print a live, side-by-side table.

All sources are polled *concurrently* with async HTTP so the loop can hold a
100 ms cadence even though each request has its own network latency. A failed
or rate-limited source is shown inline (its last good price is kept) instead of
crashing the run.

Usage:
    python btc_multi_source.py
    python btc_multi_source.py --interval 250 --window 5
    python btc_multi_source.py --once
    python btc_multi_source.py --sources binance,kraken,okx
    python btc_multi_source.py --list
    python btc_multi_source.py --no-color

Requires: httpx   (pip install httpx)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import httpx
except ImportError:
    sys.exit("This script needs the 'httpx' package.  Install it with:  pip install httpx")


# --------------------------------------------------------------------------- #
# Per-request value coercion + per-exchange JSON parsers
# --------------------------------------------------------------------------- #
def _f(x: Any) -> Optional[float]:
    """Best-effort float coercion; returns None if not parseable."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_binance(d):            return _f(d.get("price"))
def _parse_coinbase(d):
    data = d.get("data") if isinstance(d, dict) else None
    return _f(data.get("amount")) if isinstance(data, dict) else None
def _parse_cboe(d):               return _f(d.get("price"))
def _parse_kraken(d):
    res = d.get("result") if isinstance(d, dict) else None
    pair = next(iter(res.values()), None) if isinstance(res, dict) else None
    c = pair.get("c") if isinstance(pair, dict) else None
    return _f(c[0]) if isinstance(c, (list, tuple)) and c else None
def _parse_kucoin(d):
    data = d.get("data") if isinstance(d, dict) else None
    return _f(data.get("price")) if isinstance(data, dict) else None
def _parse_okx(d):
    data = d.get("data") if isinstance(d, dict) else None
    return _f(data[0].get("last")) if isinstance(data, list) and data else None
def _parse_bybit(d):
    res = d.get("result") if isinstance(d, dict) else None
    lst = res.get("list") if isinstance(res, dict) else None
    return _f(lst[0].get("lastPrice")) if isinstance(lst, list) and lst else None
def _parse_bitstamp(d):           return _f(d.get("last"))
def _parse_bitfinex(d):
    try:
        return _f(d[6])           # v2 flat array, index 6 == "last" trade price
    except Exception:
        return None
def _parse_gemini(d):             return _f(d.get("last"))


# --------------------------------------------------------------------------- #
# Source definitions (easy to add / remove / reorder)
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    key: str
    name: str
    url: str
    parse: Callable[[Any], Optional[float]]
    params: Optional[dict] = None
    note: str = ""   # which pair is actually quoted (shown in the PAIR column)


# NOTE: USDT pairs are used where an exchange has no direct USD pair; USDT is
# pegged ~1.00 to USD, so the values are directly comparable across sources.
SOURCES: list[Source] = [
    Source("binance",       "Binance",
           "https://api.binance.com/api/v3/ticker/price",
           _parse_binance, {"symbol": "BTCUSDT"}, "BTC/USDT"),
    Source("coinbase",      "Coinbase",
           "https://api.coinbase.com/v2/prices/BTC-USD/spot",
           _parse_coinbase, None, "BTC/USD"),
    Source("cboe-xchange",  "CB Exchange",
           "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
           _parse_cboe, None, "BTC/USD"),
    Source("kraken",        "Kraken",
           "https://api.kraken.com/0/public/Ticker",
           _parse_kraken, {"pair": "XBTUSD"}, "XBT/USD"),
    Source("kucoin",        "KuCoin",
           "https://api.kucoin.com/api/v1/market/orderbook/level1",
           _parse_kucoin, {"symbol": "BTC-USDT"}, "BTC/USDT"),
    Source("okx",           "OKX",
           "https://www.okx.com/api/v5/market/ticker",
           _parse_okx, {"instId": "BTC-USDT"}, "BTC/USDT"),
    Source("bybit",         "Bybit",
           "https://api.bybit.com/v5/market/tickers",
           _parse_bybit, {"category": "spot", "symbol": "BTCUSDT"}, "BTC/USDT"),
    Source("bitstamp",      "Bitstamp",
           "https://www.bitstamp.net/api/v2/ticker/btcusd/",
           _parse_bitstamp, None, "BTC/USD"),
    Source("bitfinex",      "Bitfinex",
           "https://api.bitfinex.com/v2/ticker/tBTCUSD",
           _parse_bitfinex, None, "BTC/USD"),
    Source("gemini",        "Gemini",
           "https://api.gemini.com/v1/pubticker/BTCUSD",
           _parse_gemini, None, "BTC/USD"),
]


# --------------------------------------------------------------------------- #
# Live state per source
# --------------------------------------------------------------------------- #
class SrcState:
    __slots__ = ("price", "ts", "latency", "error", "hist", "lat_hist", "age_hist")

    def __init__(self) -> None:
        self.price: Optional[float] = None
        self.ts: float = 0.0                 # epoch time of last good price
        self.latency: float = 0.0            # ms, last request
        self.error: Optional[str] = "..."    # last error (None == ok)
        self.hist: deque = deque(maxlen=8192)  # (epoch, price) ring buffer
        self.lat_hist: deque = deque(maxlen=100)  # recent request latencies (ms)
        self.age_hist: deque = deque(maxlen=100)  # sampled tick ages (ms)

    def record_age(self, now: float) -> None:
        """Sample the current tick age (called once per render, ~10 Hz)."""
        if self.price is not None and self.ts:
            self.age_hist.append((now - self.ts) * 1000.0)

    @property
    def price_avg(self) -> Optional[float]:
        """Arithmetic mean of the last up-to-50 good prices (per-source average)."""
        if not self.hist:
            return None
        return statistics.mean(p for _, p in list(self.hist)[-50:])

    @property
    def lat_avg(self) -> Optional[float]:
        """Running mean request latency over the last 100 successful requests (ms)."""
        return statistics.mean(self.lat_hist) if self.lat_hist else None

    @property
    def age_avg(self) -> Optional[float]:
        """Running mean tick age over the last ~100 renders (ms)."""
        return statistics.mean(self.age_hist) if self.age_hist else None


# --------------------------------------------------------------------------- #
# Colours / terminal helpers
# --------------------------------------------------------------------------- #
UP, DOWN, WARN, OKG = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[32m"
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"
_COLOR = False


def C(s: str, code: str = "") -> str:
    """Wrap text in an ANSI colour code (no-op if colour is disabled)."""
    return f"{code}{s}{RESET}" if (_COLOR and code) else s


def enable_ansi_windows() -> None:
    """Turn on VT/ANSI processing on Windows consoles (Win10+)."""
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            for h in (-11, -12):                      # stdout, stderr
                handle = k32.GetStdHandle(h)
                mode = ctypes.c_uint32()
                if k32.GetConsoleMode(handle, ctypes.byref(mode)):
                    k32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
async def do_fetch(src: Source, client: "httpx.AsyncClient", state: dict) -> None:
    """One non-blocking fetch for a single source; updates state (never raises)."""
    st = state[src.key]
    t0 = time.perf_counter()
    try:
        resp = await client.get(src.url, params=src.params)
        latency = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            st.error = f"HTTP {resp.status_code}"
            return
        payload = resp.json()
    except httpx.TimeoutException:
        st.error = "timeout"
        return
    except httpx.HTTPError as e:
        st.error = type(e).__name__.lower()
        return
    except Exception as e:                        # DNS, TLS, JSON decode, ...
        st.error = type(e).__name__.lower()
        return

    try:
        price = src.parse(payload)
    except Exception:
        price = None
    if price is None or price <= 0:
        st.error = "no-price"
        return

    st.price = price
    st.ts = time.time()
    st.latency = latency
    st.lat_hist.append(latency)
    st.error = None
    st.hist.append((time.time(), price))


async def source_worker(src: Source, client: "httpx.AsyncClient", state: dict,
                        interval_s: float, stop: asyncio.Event) -> None:
    """Keep one source on a fixed cadence without blocking the others."""
    while not stop.is_set():
        t0 = time.perf_counter()
        await do_fetch(src, client, state)
        delay = interval_s - (time.perf_counter() - t0)
        if delay <= 0:
            await asyncio.sleep(0.001)            # fell behind; avoid busy-spin
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def cell(text: str, width: int, align: str = "right", code: str = "") -> str:
    if align == "left":
        body = text.ljust(width)
    elif align == "right":
        body = text.rjust(width)
    else:
        body = text.center(width)
    return C(body, code)


def _delta_over_window(st: SrcState, now: float, window: float) -> Optional[float]:
    if st.price is None or len(st.hist) < 2:
        return None
    target = now - window
    ref = None
    for ts, p in st.hist:
        if ts >= target:
            ref = p
            break
    if ref is None:
        ref = st.hist[0][1]
    if not ref:
        return None
    return (st.price - ref) / ref * 100.0


def render(state: dict, sources: list, window: float, interval_ms: int,
           now: float, started: float, clear: bool) -> None:
    prices = [state[s.key].price for s in sources if state[s.key].price is not None]
    median = statistics.median(prices) if prices else None
    spread = (max(prices) - min(prices)) if prices else 0.0
    spread_pct = (spread / median * 100.0) if (median and prices) else None
    live = len(prices)

    med_str = f"${median:,.2f}" if median is not None else "--"
    spread_str = (f"${spread:,.2f} ({spread_pct:.3f}%)"
                  if spread_pct is not None else "--")
    elapsed = now - started
    hms = time.strftime("%H:%M:%S", time.localtime(now)) + f"{int((now % 1) * 1000):03d}"

    lines = [
        C("BTC/USD", BOLD) + "  "
        + C(f"{len(sources)} sources @ {interval_ms}ms", DIM) + "   "
        + "median " + C(med_str, BOLD) + "   "
        + "spread " + C(spread_str, DIM) + "   "
        + C(f"elapsed {elapsed:.1f}s", DIM),
        C(f"window {window:g}s", DIM) + "   "
        + C(f"live {live}/{len(sources)}", OKG if live else WARN) + "   "
        + C(f"updated {hms}", DIM),
        "",
        cell("NAME", 12, "left", BOLD) + " "
        + cell("PAIR", 9, "left", BOLD) + " "
        + cell("PRICE", 13, "right", BOLD) + " "
        + cell(f"D{window:g}s", 9, "right", BOLD) + " "
        + cell("AGE", 8, "right", BOLD) + " "
        + cell("AGE AVG", 8, "right", BOLD) + " "
        + cell("LAT", 7, "right", BOLD) + " "
        + cell("LAT AVG", 8, "right", BOLD) + " "
        + cell("STATUS", 12, "left", BOLD),
        "-" * 94,
    ]

    for src in sources:
        st = state[src.key]
        st.record_age(now)
        pct = _delta_over_window(st, now, window)

        if st.price is None:
            p_txt, p_code = "--", DIM
        else:
            p_txt, p_code = f"{st.price:,.2f}", ""

        if pct is None:
            d_txt, d_code = "--", DIM
        else:
            sign = "+" if pct >= 0 else "-"
            d_txt = f"{sign}{abs(pct):.3f}%"
            d_code = UP if pct > 0 else (DOWN if pct < 0 else "")

        if not st.ts:
            a_txt, a_code = "--", DIM
        else:
            a_txt = f"{(now - st.ts) * 1000.0:.0f}ms"
            a_code = WARN if (now - st.ts) > 3 * interval_ms / 1000.0 else ""

        if st.price is None:
            s_txt, s_code = (st.error or "no-data"), WARN
        elif (now - st.ts) > 3 * interval_ms / 1000.0:
            s_txt, s_code = "stale", WARN
        elif st.error is not None:
            s_txt, s_code = st.error, DOWN
        else:
            s_txt, s_code = "ok", OKG

        lines.append(
            cell(src.name, 12, "left") + " "
            + cell(src.note, 9, "left", DIM) + " "
            + cell(p_txt, 13, "right", p_code) + " "
            + cell(d_txt, 9, "right", d_code) + " "
            + cell(a_txt, 8, "right", a_code) + " "
            + cell(f"{st.age_avg:.0f}ms" if st.age_avg is not None else "--", 8, "right", DIM) + " "
            + cell(f"{st.latency:.0f}ms", 7, "right") + " "
            + cell(f"{st.lat_avg:.0f}ms" if st.lat_avg is not None else "--", 8, "right", DIM) + " "
            + cell(s_txt, 12, "left", s_code)
        )

    # Global AVG row: mean of each source's running AGE AVG and LAT AVG.
    age_avgs = [state[s.key].age_avg for s in sources if state[s.key].age_avg is not None]
    lat_avgs = [state[s.key].lat_avg for s in sources if state[s.key].lat_avg is not None]
    g_age = statistics.mean(age_avgs) if age_avgs else None
    g_lat = statistics.mean(lat_avgs) if lat_avgs else None
    live = len(age_avgs)
    lines.append(
        cell("AVG", 12, "left", BOLD)
        + " " + cell("", 9, "left")
        + " " + cell("", 13, "right")
        + " " + cell("", 9, "right")
        + " " + cell("--", 8, "right")
        + " " + cell(f"{g_age:.0f}ms" if g_age is not None else "--", 8, "right", BOLD)
        + " " + cell("--", 7, "right")
        + " " + cell(f"{g_lat:.0f}ms" if g_lat is not None else "--", 8, "right", BOLD)
        + " " + cell(f"{live} live", 12, "left", DIM)
    )

    lines.append("")
    lines.append(C("green=up  red=down  yellow=warn/stale  -- = no data yet", DIM))

    text = ("\x1b[2J\x1b[H" if clear else "") + "\n".join(lines) + "\n"
    sys.stdout.write(text)
    sys.stdout.flush()


async def render_loop(state: dict, sources: list, window: float, interval_ms: int,
                      started: float, stop: asyncio.Event, interval_s: float) -> None:
    while not stop.is_set():
        render(state, sources, window, interval_ms, time.time(), started, clear=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace, sources: list) -> None:
    global _COLOR
    _COLOR = (not args.no_color) and sys.stdout.isatty()
    interval_s = max(args.interval, 1) / 1000.0
    started = time.time()
    state = {s.key: SrcState() for s in sources}

    timeout = httpx.Timeout(connect=1.0, read=args.timeout, write=1.0, pool=args.timeout)
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers,
                                 follow_redirects=True) as client:
        if args.once:
            await asyncio.gather(*[do_fetch(s, client, state) for s in sources])
            render(state, sources, args.window, args.interval, time.time(), started,
                   clear=False)
            return

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass                                # Windows: rely on KeyboardInterrupt

        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")   # hide cursor + clear
        sys.stdout.flush()
        tasks = [asyncio.create_task(source_worker(s, client, state, interval_s, stop))
                 for s in sources]
        tasks.append(asyncio.create_task(render_loop(state, sources, args.window,
                                                     args.interval, started, stop,
                                                     interval_s)))
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch BTC/USD from many exchanges at a high refresh rate.")
    p.add_argument("--interval", type=int, default=100,
                   help="refresh interval in milliseconds (default: 100)")
    p.add_argument("--window", type=float, default=10.0,
                   help="seconds used for the per-source delta column (default: 10)")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="per-request read timeout in seconds (default: 2.0)")
    p.add_argument("--sources", type=str, default="",
                   help="comma-separated source keys to use (default: all)")
    p.add_argument("--once", action="store_true",
                   help="fetch once, print a single table, and exit")
    p.add_argument("--list", action="store_true", help="list available sources and exit")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print(f"{'KEY':<14} {'NAME':<14} PAIR")
        for s in SOURCES:
            print(f"{s.key:<14} {s.name:<14} {s.note}")
        return

    sources = SOURCES
    if args.sources.strip():
        wanted = {k.strip() for k in args.sources.split(",") if k.strip()}
        unknown = wanted - {s.key for s in SOURCES}
        if unknown:
            sys.exit(f"Unknown source key(s): {', '.join(sorted(unknown))}. "
                     f"Use --list to see valid keys.")
        sources = [s for s in SOURCES if s.key in wanted]

    enable_ansi_windows()
    started = time.time()
    try:
        asyncio.run(run(args, sources))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")            # restore cursor
        sys.stdout.flush()
        print(f"Captured {time.time() - started:.1f}s across {len(sources)} sources.")


if __name__ == "__main__":
    main()
