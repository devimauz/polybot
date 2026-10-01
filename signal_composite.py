"""Weighted composite technical signal for BTCUSD 1-minute candles.

Combines five per-component signals, each normalized to [-1, +1], into a
single composite:

  RSI(14)          (rsi - 50) / 50
  Momentum         tanh(z-score of 10-bar log return vs 1-min ATR)
  VWAP deviation   tanh((price - session VWAP) / (4 * ATR))
  SMA crossover    tanh((SMA9 - SMA21) / (2 * ATR))
  Market skew      taker-buy vs taker-sell volume over the last 5 minutes

Data: Binance spot REST (BTCUSDT 1m klines + recent trades), same source the
other bots in this folder use. Run live (updates in place) or one-shot:

  python signal_composite.py              # live panel, refresh every 5s
  python signal_composite.py --once       # single snapshot
  python signal_composite.py --interval 10 --csv signal_log.csv
  python signal_composite.py --weights 0.25,0.2,0.25,0.2,0.1
"""
import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import httpx

SYMBOL = "BTCUSDT"
BASE = "https://api.binance.com/api/v3"
KLINE_LIMIT = 300          # 1m bars -> 5h history, plenty for all indicators
TRADE_LOOKBACK_S = 300     # market skew window: last 5 minutes
TRADE_LIMIT = 1000

DEFAULT_WEIGHTS = {"rsi": 0.25, "momentum": 0.20, "vwap": 0.25, "sma": 0.20, "skew": 0.10}
COMPONENTS = ["rsi", "momentum", "vwap", "sma", "skew"]


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def ema(values, period):
    k = 2.0 / (period + 1)
    out = []
    e = None
    for v in values:
        e = v if e is None else v * k + e * (1 - k)
        out.append(e)
    return out


def rsi_series(closes, period=14):
    """Wilder RSI. Returns list aligned to closes (NaN until warm)."""
    n = len(closes)
    rsi = [float("nan")] * n
    if n <= period:
        return rsi
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / period, losses / period
    rsi[period] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)
    for i in range(period + 1, n):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
        rsi[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)
    return rsi


def atr(closes, period=14):
    """Simple ATR from 1m bars using |close-to-close| (high/low optional)."""
    if len(closes) < period + 1:
        return None
    returns = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    # Wilder smoothing
    a = sum(returns[:period]) / period
    for r in returns[period:]:
        a = (a * (period - 1) + r) / period
    return a


def atr_series(closes, period=14):
    """Per-bar Wilder ATR aligned to closes (None until warmed up)."""
    n = len(closes)
    out: list = [None] * n
    if n <= period:
        return out
    prev = sum(abs(closes[i] - closes[i - 1]) for i in range(1, period + 1)) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + abs(closes[i] - closes[i - 1])) / period
        out[i] = prev
    return out


def vwap_deviation(closes, volumes, period=0):
    """VWAP for the current UTC day plus z-score of price vs VWAP (ATR-scaled later)."""
    if not closes:
        return None, None
    # find the start of the current UTC day
    start = 0
    last_day = None
    for i in range(len(closes) - 1, -1, -1):
        day = i % 1440  # 1m bars: UTC day boundary is a multiple of 1440 bars
        if last_day is None:
            last_day = day
        if day != last_day:
            start = i
            break
    pv = [closes[i] * volumes[i] for i in range(start, len(closes))]
    vv = [volumes[i] for i in range(start, len(closes))]
    vol_sum = sum(vv)
    if vol_sum <= 0:
        return None, None
    v = sum(pv) / vol_sum
    return v, (closes[-1] - v)


def market_skew(client, symbol=SYMBOL, lookback_s=TRADE_LOOKBACK_S):
    """Taker buy vs sell volume imbalance over the recent trades window.

    Returns (skew in [-1,1], n_trades). +1 = all taker-buy flow (aggressive
    buying), -1 = all taker-sell flow.
    """
    now_ms = time.time() * 1000
    r = client.get(f"{BASE}/trades", params={"symbol": symbol, "limit": TRADE_LIMIT})
    r.raise_for_status()
    trades = r.json()
    buy = sell = 0.0
    n = 0
    for t in trades:
        if t["time"] >= now_ms - lookback_s * 1000:
            n += 1
            q = float(t["qty"])
            if t["isBuyerMaker"]:  # aggressive SELL hit the bid
                sell += q
            else:                  # aggressive BUY lifted the ask
                buy += q
    total = buy + sell
    if total <= 0:
        return 0.0, 0
    return (buy - sell) / total, n


def compute(components_raw, weights):
    """Raw component dict -> (normalized signals, composite)."""
    signals = {}
    for name in COMPONENTS:
        v = components_raw.get(name)
        signals[name] = 0.0 if v is None or not math.isfinite(v) else clamp(v)
    wsum = sum(weights[n] for n in COMPONENTS)
    composite = sum(weights[n] * signals[n] for n in COMPONENTS) / wsum
    return signals, clamp(composite)


def series(closes, volumes, weights, look=10):
    """Per-bar indicator arrays for charting (all aligned to `closes`)."""
    n = len(closes)
    rsi = rsi_series(closes, 14)
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    atrs = atr_series(closes, 14)

    # session VWAP (UTC-day anchored) per bar
    vwap: list = [None] * n
    pv = vv = 0.0
    for i in range(n):
        if i % 1440 == 0 and i > 0:
            pv = vv = 0.0
        pv += closes[i] * volumes[i]
        vv += volumes[i]
        vwap[i] = pv / vv if vv > 0 else None

    # per-bar 10-bar log return
    ret10: list = [None] * n
    for i in range(look, n):
        ret10[i] = math.log(closes[i] / closes[i - look])

    # per-bar normalized component signals + composite
    comp = {name: list([None] * n) for name in COMPONENTS}
    composite: list = [None] * n
    for i in range(n):
        a = atrs[i]
        if a is None or a <= 0:
            continue
        raw = {}
        raw["rsi"] = None if math.isnan(rsi[i]) else (rsi[i] - 50.0) / 50.0
        if ret10[i] is not None:
            expected = (a / closes[i]) * math.sqrt(look)
            raw["momentum"] = math.tanh((ret10[i] / expected) * 0.5) if expected > 0 else None
        else:
            raw["momentum"] = None
        raw["vwap"] = math.tanh((closes[i] - vwap[i]) / (4.0 * a)) if vwap[i] is not None else None
        raw["sma"] = math.tanh((ema9[i] - ema21[i]) / (2.0 * a))
        raw["skew"] = None  # skew is live-only; filled in for the latest bar
        for name in COMPONENTS:
            v = raw[name]
            comp[name][i] = 0.0 if (v is None or not math.isfinite(v)) else clamp(v)
        wsum = sum(weights[name] for name in COMPONENTS)
        composite[i] = clamp(sum(weights[name] * comp[name][i] for name in COMPONENTS) / wsum)

    return {
        "closes": closes, "volumes": volumes, "rsi": rsi,
        "ema9": ema9, "ema21": ema21, "vwap": vwap, "atr": atrs,
        "ret10": ret10, "components": comp, "composite": composite,
    }


def verdict(s):
    if s >= 0.60:
        return "STRONG BUY"
    if s >= 0.20:
        return "BUY"
    if s <= -0.60:
        return "STRONG SELL"
    if s <= -0.20:
        return "SELL"
    return "NEUTRAL"


class LiveState:
    """Incremental live indicator state, advanced trade-by-trade.

    Mirrors the exact formulas in snapshot() but updates per WebSocket
    trade instead of per REST poll: Wilder RSI, EMA9/21, Wilder ATR,
    UTC-day-anchored VWAP, and the rolling 5-min taker-buy/sell skew.
    Seed from a REST snapshot, then call on_trade() for each trade.
    """

    PERIOD = 14

    def __init__(self, weights, look=10, skew_window_s=TRADE_LOOKBACK_S):
        self.weights = weights
        self.look = look
        self.skew_window_s = skew_window_s
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.price = None
        self.prev = None
        self.ema9 = None
        self.ema21 = None
        self.atr = None
        self.avg_g = 0.0
        self.avg_l = 0.0
        self.rsi = float("nan")
        self.pv = 0.0
        self.vv = 0.0
        self.vwap = None
        self.day = None
        self.minute = None
        self.closed = deque(maxlen=self.look)   # closed 1m closes (momentum base)
        self.skew = deque()                     # (ts_ms, signed qty: +buy / -sell)

    def seed(self, snap):
        """Resync from a REST snapshot (called when a fresh snapshot lands)."""
        with self.lock:
            closes = snap["history"]["closes"]
            vols = snap["history"]["volumes"]
            times = snap["history"]["times"]
            period = self.PERIOD
            gains = losses = 0.0
            for i in range(1, period + 1):
                d = closes[i] - closes[i - 1]
                gains += max(d, 0.0)
                losses += max(-d, 0.0)
            self.avg_g, self.avg_l = gains / period, losses / period
            self.rsi = 100.0 if self.avg_l == 0 else 100.0 - 100.0 / (1 + self.avg_g / self.avg_l)
            for i in range(period + 1, len(closes)):
                d = closes[i] - closes[i - 1]
                self.avg_g = (self.avg_g * (period - 1) + max(d, 0.0)) / period
                self.avg_l = (self.avg_l * (period - 1) + max(-d, 0.0)) / period
                self.rsi = 100.0 if self.avg_l == 0 else 100.0 - 100.0 / (1 + self.avg_g / self.avg_l)
            self.ema9 = snap["history"]["ema9"][-1]
            self.ema21 = snap["history"]["ema21"][-1]
            a = snap["history"]["atr"][-1]
            self.atr = a if a else None
            # day-anchored VWAP accumulators (same boundary logic as vwap_deviation)
            last_day = (len(closes) - 1) % 1440
            start = len(closes) - 1
            for i in range(len(closes) - 1, -1, -1):
                if i % 1440 != last_day:
                    start = i + 1
                    break
            self.pv = sum(closes[i] * vols[i] for i in range(start, len(closes)))
            self.vv = sum(vols[i] for i in range(start, len(closes)))
            self.vwap = self.pv / self.vv if self.vv > 0 else None
            self.price = closes[-1]
            self.prev = closes[-1]
            self.closed = deque(closes[-(self.look + 1):-1], maxlen=self.look)
            last = times[-1]
            self.minute = int(last.timestamp() // 60)
            self.day = self.minute // 1440

    def on_trade(self, price, qty, ts_ms, is_buyer_maker):
        """Advance state with one trade. Returns a fresh live snapshot dict."""
        with self.lock:
            if self.price is None:          # not seeded yet
                return None
            minute = int(ts_ms // 60000)
            if minute != self.minute:       # a 1m bar just closed
                if self.prev is not None:
                    self.closed.append(self.prev)
                day = minute // 1440
                if day != self.day:         # UTC day rollover -> VWAP reset
                    self.day = day
                    self.pv = self.vv = 0.0
                    self.vwap = None
                self.minute = minute
            if price != self.prev:
                d = price - self.prev
                period = self.PERIOD
                self.avg_g = (self.avg_g * (period - 1) + max(d, 0.0)) / period
                self.avg_l = (self.avg_l * (period - 1) + max(-d, 0.0)) / period
                self.rsi = 100.0 if self.avg_l == 0 else 100.0 - 100.0 / (1 + self.avg_g / self.avg_l)
                if self.atr:
                    self.atr = (self.atr * (period - 1) + abs(d)) / period
                e9 = self.ema9
                e21 = self.ema21
                if e9 is not None and e21 is not None:
                    k9, k21 = 2.0 / 10, 2.0 / 22
                    self.ema9 = price * k9 + e9 * (1 - k9)
                    self.ema21 = price * k21 + e21 * (1 - k21)
                self.prev = price
            self.price = price
            self.pv += price * qty
            self.vv += qty
            if self.vv > 0:
                self.vwap = self.pv / self.vv
            self.skew.append((ts_ms, -qty if is_buyer_maker else qty))
            cutoff = ts_ms - self.skew_window_s * 1000
            while self.skew and self.skew[0][0] < cutoff:
                self.skew.popleft()
            return self.compute()

    def compute(self):
        """Same component formulas + composite as snapshot(). Caller holds lock."""
        p = self.price
        a = self.atr
        if p is None or a is None or a <= 0:
            return None
        raw = {}
        raw["rsi"] = (self.rsi - 50.0) / 50.0 if math.isfinite(self.rsi) else None
        ret10 = None
        a_ok = a is not None and a > 0
        if len(self.closed) == self.look and p is not None:
            ret10 = math.log(p / self.closed[0])
        raw["momentum"] = None
        if ret10 is not None and a_ok:
            expected = (a / p) * math.sqrt(self.look)
            if expected > 0:
                raw["momentum"] = math.tanh((ret10 / expected) * 0.5)
        vwap = self.vwap
        raw["vwap"] = math.tanh((p - vwap) / (4.0 * a)) if (vwap is not None and a_ok) else None
        e9 = self.ema9
        e21 = self.ema21
        raw["sma"] = math.tanh((e9 - e21) / (2.0 * a)) if (a_ok and e9 is not None and e21 is not None) else None
        buy = sum(q for _, q in self.skew if q > 0)
        sell = -sum(q for _, q in self.skew if q < 0)
        raw["skew"] = (buy - sell) / (buy + sell) if (buy + sell) > 0 else 0.0
        signals, composite = compute(raw, self.weights)
        return {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "price": p,
            "rsi": self.rsi if math.isfinite(self.rsi) else None,
            "vwap": self.vwap,
            "sma9": self.ema9,
            "sma21": self.ema21,
            "atr": a,
            "ret10": ret10,
            "trades": len(self.skew),
            "raw": raw,
            "signals": signals,
            "composite": composite,
        }


def bar(s, width=20):
    """Text bar for a [-1,1] value, pivot at zero."""
    mid = width // 2
    pos = int(round((s + 1) / 2 * (width - 1)))
    return " " * min(pos, mid) + "|" + " " * max(0, width - 1 - max(pos, mid))


def snapshot(client, weights):
    k = client.get(f"{BASE}/klines", params={"symbol": SYMBOL, "interval": "1m", "limit": KLINE_LIMIT})
    k.raise_for_status()
    rows = k.json()
    if len(rows) < 30:
        raise ValueError(f"not enough klines ({len(rows)})")
    closes = [float(r[4]) for r in rows]
    volumes = [float(r[5]) for r in rows]
    ohlc = [(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
    ts = [int(r[0]) for r in rows]
    times = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in ts]
    price = closes[-1]
    bar_time = times[-1]

    a = atr(closes) or (price * 1e-4)
    r = rsi_series(closes, 14)
    rsi_now = r[-1]
    rsi_sig = None if math.isnan(rsi_now) else (rsi_now - 50.0) / 50.0

    # momentum: z-score of 10-bar log return vs expected vol (ATR%)
    look = 10
    mom_sig = None
    ret10 = None
    if len(closes) > look and a > 0:
        logret = math.log(closes[-1] / closes[-1 - look])
        ret10 = logret
        expected = (a / price) * math.sqrt(look)
        if expected > 0:
            mom_sig = math.tanh((logret / expected) * 0.5)

    # VWAP deviation
    vwap, dev = vwap_deviation(closes, volumes)
    vwap_sig = None
    if vwap and a > 0:
        vwap_sig = math.tanh(dev / (4.0 * a))

    # SMA 9/21 crossover
    ema9 = ema(closes, 9)[-1]
    ema21 = ema(closes, 21)[-1]
    sma_sig = None
    if a > 0:
        sma_sig = math.tanh((ema9 - ema21) / (2.0 * a))

    skew, n_trades = market_skew(client)

    raw = {"rsi": rsi_sig, "momentum": mom_sig, "vwap": vwap_sig, "sma": sma_sig, "skew": skew}
    signals, composite = compute(raw, weights)

    # per-bar history for the GUI charts (skew history is not available
    # from klines; the GUI carries its own rolling skew history live-only)
    hist = series(closes, volumes, weights, look=look)
    hist["times"] = times
    hist["ohlc"] = ohlc

    return {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "bar": bar_time.strftime("%H:%M UTC"),
        "price": price,
        "rsi": rsi_now if math.isfinite(rsi_now) else None,
        "vwap": vwap,
        "sma9": ema9,
        "sma21": ema21,
        "atr": a,
        "ret10": ret10,
        "trades": n_trades,
        "signals": signals,
        "raw": raw,
        "composite": composite,
        "history": hist,
    }


def print_panel(s, weights):
    lines = []
    lines.append(f"BTCUSD 1m  composite signal   {s['time']}  (bar {s['bar']})")
    lines.append(f"price {s['price']:,.2f}   ATR {s['atr']:,.2f}   VWAP {s['vwap']:,.2f}   "
                 f"SMA9 {s['sma9']:,.2f} / SMA21 {s['sma21']:,.2f}")
    lines.append("")
    labels = {
        "rsi": f"RSI(14)          {s['rsi']:5.1f}",
        "momentum": f"Momentum(10m)    {(math.exp(s['ret10']) - 1) if s['ret10'] is not None else 0:+.6%}",
        "vwap": f"VWAP dev         {(s['price'] - s['vwap']) / s['atr']:+.2f} xATR",
        "sma": f"SMA 9/21         {(s['sma9'] - s['sma21']) / s['price']:+.6%}",
        "skew": f"Market skew      {s['trades']} trades",
    }
    for name in COMPONENTS:
        sig = s["signals"][name]
        w = weights[name]
        lines.append(f"{labels[name]} w={w:4.2f}  {sig:+5.2f}  {bar(sig)}")
    c = s["composite"]
    lines.append(f"{'COMPOSITE':<19} w=1.00  {c:+5.2f}  {bar(c)}  ->  {verdict(c)}")
    text = "\n".join(lines)
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[2J\x1b[H" + text + "\n")
    else:
        print(text)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="Weighted composite signal on BTCUSD 1m")
    ap.add_argument("--once", action="store_true", help="single snapshot, no loop")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    ap.add_argument("--weights", type=str, default=None,
                    help="comma list rsi,momentum,vwap,sma,skew (default .25,.2,.25,.2,.1)")
    ap.add_argument("--csv", type=str, default=None, help="also log each snapshot to CSV")
    args = ap.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            vals = [float(x) for x in args.weights.split(",")]
            assert len(vals) == 5 and all(v >= 0 for v in vals) and sum(vals) > 0
        except Exception:
            sys.exit("--weights must be 5 comma-separated non-negative numbers, e.g. 0.25,0.2,0.25,0.2,0.1")
        weights = dict(zip(COMPONENTS, vals))

    fields = ["time", "bar", "price", "rsi", "ret10", "vwap", "sma9", "sma21", "atr"] \
        + [f"{n}_sig" for n in COMPONENTS] + ["composite", "verdict"]
    csv_file = open(args.csv, "a", newline="") if args.csv else None
    if csv_file and csv_file.tell() == 0:
        csv.writer(csv_file).writerow(fields)

    try:
        with httpx.Client(timeout=10) as client:
            while True:
                try:
                    s = snapshot(client, weights)
                    print_panel(s, weights)
                    if csv_file:
                        row = [s["time"], s["bar"], f'{s["price"]:.2f}']
                        row += [f'{s["rsi"]:.4f}' if s["rsi"] is not None else ""]
                        row += [f'{(math.exp(s["ret10"]) - 1):.6f}' if s["ret10"] is not None else ""]
                        row += [f'{s[k]:.4f}' if s[k] is not None else "" for k in ("vwap", "sma9", "sma21")]
                        row += [f'{s["atr"]:.4f}']
                        row += [f'{s["signals"][n]:.4f}' for n in COMPONENTS]
                        row += [f'{s["composite"]:.4f}', verdict(s["composite"])]
                        csv.writer(csv_file).writerow(row)
                        csv_file.flush()
                except Exception as e:
                    print(f"\n[snapshot error] {type(e).__name__}: {e}")
                if args.once:
                    break
                # sleep to just after the next second tick so readings line up
                time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
