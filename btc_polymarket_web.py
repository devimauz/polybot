"""Web dashboard for the BTC Polymarket jump trader.

Dry run: python btc_polymarket_web.py
Live:    python btc_polymarket_web.py --live
"""
import argparse
import os
import re
import sqlite3
import threading
import time
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from bot import ROOT
import btc_price_chart_polymarket as strategy
from btc_price_chart_polymarket import (
    ENTRY_SLIPPAGE, FRAME_MS, JUMP_NOISE_USD, MAX_ENTRY_PRICE,
    MAX_TRADES_PER_WINDOW,
    MIN_ENTRY_PRICE,
    SELL_RETRY_SECONDS, SELL_RETRY_WINDOW_SECONDS, STAKE, TAKE_PROFIT,
    TRADE_START_DELAY_SECONDS,
    CoinbasePriceFeed, Discovery, EventDrivenJumpDetector, MarketFeed,
    PolymarketBTCFeed,
    PolymarketJumpTrader, directional_jump, price_trend,
)

# Web-only strategy profile copied from btc_price_chart.py. The shared GUI and
# terminal trader retain their original live configuration in other processes.
ENTRY_SHARES = Decimal("5.10")
# Maximum possible budget at the 0.70 entry ceiling plus 0.03 slippage.
STAKE = Decimal("3.73")
TAKE_PROFIT = Decimal("0.05")
ENTRY_SLIPPAGE = Decimal("0.03")
MIN_ENTRY_PRICE = Decimal("0.30")
MAX_ENTRY_PRICE = Decimal("0.70")
TRADE_START_DELAY_SECONDS = 0.0
TRADE_COOLDOWN_SECONDS = 1.0
MAX_TRADES_PER_WINDOW = 300  # one-second cooldown is the effective limit
WEB_JUMP_THRESHOLD_USD = 20.0
# Official CLOB endpoint limits (per IP): UPDATE 50/10s, GET balance 200/10s,
# and ledger/order reads 900/10s.
BALANCE_UPDATE_INTERVAL_SECONDS = 0.20
BALANCE_GET_INTERVAL_SECONDS = 0.05
TP_ORDER_POLL_INTERVAL_SECONDS = 10.0 / 900.0
FRAME_MS = 1000.0 / 90.0
strategy.STAKE = STAKE
strategy.TAKE_PROFIT = TAKE_PROFIT
strategy.ENTRY_SLIPPAGE = ENTRY_SLIPPAGE
strategy.MIN_ENTRY_PRICE = MIN_ENTRY_PRICE
strategy.MAX_ENTRY_PRICE = MAX_ENTRY_PRICE
strategy.TRADE_START_DELAY_SECONDS = TRADE_START_DELAY_SECONDS
strategy.TRADE_COOLDOWN_SECONDS = TRADE_COOLDOWN_SECONDS
strategy.MAX_TRADES_PER_WINDOW = MAX_TRADES_PER_WINDOW

HISTORY_SECONDS = 300


def summarize_market_activity(activity):
    """Build dashboard metrics from public depth and trade events."""
    depth = activity.get("depth", {})
    trades = activity.get("trades", [])
    outcome_depth = {}
    for outcome in ("Up", "Down"):
        book = depth.get(outcome, {})
        bid_shares = sum(level[1] for level in book.get("bids", []))
        ask_shares = sum(level[1] for level in book.get("asks", []))
        outcome_depth[outcome] = {
            "bid_shares": bid_shares, "ask_shares": ask_shares,
            "bids": book.get("bids", []), "asks": book.get("asks", []),
        }
    up_interest = outcome_depth["Up"]["bid_shares"]
    down_interest = outcome_depth["Down"]["bid_shares"]
    interest_total = up_interest + down_interest

    # Express trade flow directionally: buying UP or selling DOWN is UP flow;
    # buying DOWN or selling UP is DOWN flow. Values are shares, not people.
    up_flow = down_flow = unknown_flow = 0.0
    for trade in trades:
        size = float(trade.get("size", 0) or 0)
        outcome = trade.get("outcome")
        side = trade.get("side")
        if (outcome == "Up" and side == "BUY") or (
                outcome == "Down" and side == "SELL"):
            up_flow += size
        elif (outcome == "Down" and side == "BUY") or (
                outcome == "Up" and side == "SELL"):
            down_flow += size
        else:
            unknown_flow += size
    flow_total = up_flow + down_flow
    return {
        "depth": outcome_depth,
        "up_pressure_pct": (50.0 if not interest_total else
                            up_interest / interest_total * 100),
        "down_pressure_pct": (50.0 if not interest_total else
                              down_interest / interest_total * 100),
        "up_flow": up_flow, "down_flow": down_flow,
        "unknown_flow": unknown_flow,
        "up_flow_pct": 50.0 if not flow_total else up_flow / flow_total * 100,
        "down_flow_pct": 50.0 if not flow_total else down_flow / flow_total * 100,
        "trade_count": len(trades), "trades": trades[-50:],
        "window_seconds": 60,
    }


def mark_position_from_polymarket(position, quotes):
    """Return bid-marked P&L using the live Polymarket outcome quote."""
    if not position:
        return None
    quote = quotes.get(position.get("outcome"))
    if quote is None:
        return dict(position)
    up_axis_entry = Decimal(str(position["entry"]))
    entry = (up_axis_entry if position["outcome"] == "Up"
             else Decimal("1") - up_axis_entry)
    shares = Decimal(str(position["shares"]))
    bid = Decimal(str(quote.bid))
    cost = entry * shares
    value = bid * shares
    pnl = value - cost
    result = dict(position)
    result.update({
        "polymarket_bid": float(bid),
        "cost": float(cost),
        "value": float(value),
        "profit": float(pnl),
        "profit_pct": None if cost == 0 else float(pnl / cost * 100),
    })
    return result


class WebPolymarketJumpTrader(PolymarketJumpTrader):
    """Web-only trader whose 5% take-profit rests on Polymarket.

    Market quotes already arrive over WebSocket. Only the balance/allowance
    endpoints are polled, at their documented per-endpoint ceilings, until the
    newly purchased outcome shares become available for a GTC sell order.
    Once accepted, Polymarket owns the TP lifecycle even if this process stops.
    """

    def _entry_order_terms(self, ask, tick):
        """FOK limit for exactly 5.1 shares within 3c slippage."""
        limit = (min(Decimal("0.99"), ask + ENTRY_SLIPPAGE) /
                 tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        budget = (limit * ENTRY_SHARES).quantize(
            Decimal("0.01"), rounding=ROUND_CEILING)
        return limit, budget, "FOK"

    def _create_entry_order(self, token, limit, _buy_amount,
                            entry_order_type, options):
        """Create an exact-share entry; never spend a budget into extra shares."""
        from py_clob_client_v2 import OrderArgs, OrderType, Side

        order_type = getattr(OrderType, entry_order_type)
        signed = self.client.create_order(
            OrderArgs(token_id=token, price=float(limit),
                      size=float(ENTRY_SHARES), side=Side.BUY), options)
        return signed, order_type

    def _recover_position(self):
        """Restore the persisted GTC order id so status sync survives restart."""
        super()._recover_position()
        with self.lock:
            position = self.open_position
            trade_id = position.get("id") if position else None
        if not trade_id:
            return
        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT detail FROM trades WHERE request_id=?", (trade_id,)
            ).fetchone()
        detail = row[0] if row and row[0] else ""
        match = re.search(r"Polymarket GTC TP order=([^\s]+)", detail)
        if not match:
            return
        with self.lock:
            position = self.open_position
            if position and position.get("id") == trade_id:
                position["resting_order_id"] = match.group(1)
                position["resting_placed"] = True
                position["tp_resting_since"] = time.monotonic()
                position["exit_mode"] = "Polymarket GTC"
                self.status = "recovered Polymarket GTC; synchronizing order status"

    def _reindex_balance(self, position):
        # _arm_take_profit owns the rate-limited UPDATE/GET schedule. Keeping
        # the inherited second worker would double the UPDATE request rate.
        return

    def observe(self, slug, quotes):
        """Only synchronize the exchange-managed GTC take-profit status."""
        if not self.live:
            return
        now = time.monotonic()
        with self.lock:
            position = self.open_position
            if not position or position.get("market") != slug:
                return
            order_id = position.get("resting_order_id")
            if (order_id and
                    not position.get("tp_checking") and
                    now >= position.get("next_tp_check", 0.0)):
                position["tp_checking"] = True
                position["next_tp_check"] = (
                    now + TP_ORDER_POLL_INTERVAL_SECONDS)
                threading.Thread(
                    target=self._check_tp_order,
                    args=(position["id"], order_id, dict(position)),
                    name="web-polymarket-tp-status", daemon=True).start()

    def _check_tp_order(self, trade_id, order_id, position):
        """Fetch the real GTC state so a filled TP disappears immediately."""
        try:
            order = self.client.get_order(order_id)
            status = str(order.get("status", "")).lower()
            matched_text = order.get(
                "size_matched", order.get("sizeMatched", "0")) or "0"
            matched = Decimal(str(matched_text))
            shares = Decimal(str(position["shares"]))
            fully_filled = matched >= shares - Decimal("0.001")
            if status in {"matched", "filled"} and matched == 0:
                # Some client versions omit size_matched after a full match.
                fully_filled = True
            if fully_filled:
                self._finalize_sold(
                    trade_id, position, position["target"], by="Polymarket GTC")
                return

            terminal = {"canceled", "cancelled", "expired", "failed", "rejected"}
            terminal_state = None
            with self.lock:
                current = self.open_position
                if not current or current.get("id") != trade_id:
                    return
                if matched > 0:
                    current["remaining_shares"] = max(
                        Decimal("0"), shares - matched)
                if status in terminal:
                    current["resting_order_id"] = None
                    current["resting_placed"] = False
                    current["exit_mode"] = "TP no longer active"
                    self.status = f"Polymarket TP {status}; position still open"
                    terminal_state = f"tp_{status}"
            if terminal_state:
                self._update(trade_id, tp_state=terminal_state)
        except Exception as exc:
            with self.lock:
                current = self.open_position
                if current and current.get("id") == trade_id:
                    self.status = (f"TP status fetch {type(exc).__name__}: "
                                   f"{str(exc)[:140]}")
        finally:
            with self.lock:
                current = self.open_position
                if current and current.get("id") == trade_id:
                    current["tp_checking"] = False

    def position(self, slug):
        position = super().position(slug)
        if not position:
            return position
        result = dict(position)
        with self.lock:
            current = self.open_position
            if current and current.get("market") == slug:
                result["shares"] = float(current.get(
                    "remaining_shares", current["shares"]))
        return result

    def _arm_take_profit(self, position):
        trade_id = position["id"]
        token = position["token"]
        shares = position["shares"]
        tick = position["tick"]
        target = position["target"]
        if target >= Decimal("1"):
            self._update(trade_id, tp_state="unreachable")
            with self.lock:
                self.status = f"{position['outcome']} TP is above the valid price range"
            return

        try:
            from py_clob_client_v2 import (
                AssetType, BalanceAllowanceParams, OrderArgs, OrderType,
                PartialCreateOrderOptions, Side,
            )

            balance_params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token)
            options = PartialCreateOrderOptions(
                tick_size=str(tick), neg_risk=False)
            settle_at = self._settle_at(position["market"])
            next_update = 0.0

            while not self.stop.is_set() and time.time() < settle_at - 5:
                with self.lock:
                    current = self.open_position
                    if not current or current.get("id") != trade_id:
                        return

                now = time.monotonic()
                if now >= next_update:
                    self.client.update_balance_allowance(balance_params)
                    next_update = now + BALANCE_UPDATE_INTERVAL_SECONDS

                balance = self.client.get_balance_allowance(balance_params)
                available = (Decimal(str(balance.get("balance", "0"))) /
                             Decimal("1000000"))
                if available + Decimal("0.001") < shares:
                    with self.lock:
                        self.status = (f"{position['outcome']} TP waiting for "
                                       f"balance ({available:.3f}/{shares:.3f} shares)")
                    self.stop.wait(BALANCE_GET_INTERVAL_SECONDS)
                    continue

                with self.lock:
                    current = self.open_position
                    if (not current or current.get("id") != trade_id or
                            current.get("selling")):
                        return
                    current["tp_posting"] = True
                signed = self.client.create_order(
                    OrderArgs(token_id=token, price=float(target),
                              size=float(shares), side=Side.SELL), options)
                response = self.client.post_order(
                    signed, OrderType.GTC, defer_exec=True)
                status = str(response.get("status", "")).lower()
                order_id = self._order_id_from(response)

                if status == "matched" and not order_id:
                    self._finalize_sold(
                        trade_id, position, target, by="Polymarket GTC")
                    return
                if not response.get("success") or not order_id or status not in {
                        "live", "matched", "delayed", "unmatched"}:
                    raise RuntimeError(
                        f"Polymarket rejected GTC TP: {str(response)[:300]}")

                with self.lock:
                    current = self.open_position
                    if not current or current.get("id") != trade_id:
                        self._cancel_order(order_id)
                        return
                    current["tp_posting"] = False
                    current["resting_order_id"] = order_id
                    current["resting_placed"] = True
                    current["tp_resting_since"] = time.monotonic()
                    current["exit_mode"] = "Polymarket GTC"
                    self.status = (f"{position['outcome']} 5% TP resting on "
                                   f"Polymarket @ {target:.4f} ({shares:.3f} shares)")
                self._update(
                    trade_id, tp_state="resting",
                    detail=(f"Polymarket GTC TP order={order_id} "
                            f"target={target} response={str(response)[:1000]}"))
                return

            raise TimeoutError("market ended before the GTC TP could be submitted")
        except Exception as exc:
            with self.lock:
                current = self.open_position
                if current and current.get("id") == trade_id:
                    current["tp_posting"] = False
            self._update(
                trade_id, tp_state="tp_rejected",
                detail=f"{type(exc).__name__}: {str(exc)[:1000]}")
            with self.lock:
                current = self.open_position
                if current and current.get("id") == trade_id:
                    self.status = f"TP order failed: {type(exc).__name__}: {str(exc)[:180]}"


class WebTradingEngine:
    def __init__(self, real_money=False):
        self.real_money = real_money
        self.feed = MarketFeed()
        self.btc_feed = PolymarketBTCFeed()
        self.coinbase_feed = CoinbasePriceFeed()
        self.discovery = Discovery(self.feed)
        self.trader = WebPolymarketJumpTrader(live=real_money)
        self.threshold = float(os.environ.get(
            "WEB_JUMP_THRESHOLD_USD", str(WEB_JUMP_THRESHOLD_USD)))
        if self.threshold <= 0:
            raise ValueError("JUMP_THRESHOLD_USD must be positive")
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = {"ready": False, "mode": "LIVE" if real_money else "DRY RUN"}
        self.thread = threading.Thread(target=self._run, name="web-trading-engine", daemon=True)
        self.jump_detector = EventDrivenJumpDetector(
            self.discovery, self.feed, self.trader, self.threshold)
        self.coinbase_feed.set_callback(self.jump_detector.on_price)

    def start(self):
        self.discovery.start()
        self.btc_feed.start()
        self.coinbase_feed.start()
        self.thread.start()

    def close(self):
        self.stop.set()
        self.discovery.stop.set()
        self.btc_feed.stop.set()
        self.coinbase_feed.stop.set()
        self.trader.close()
        self.feed.generation += 1
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    @staticmethod
    def _quote(quote):
        if quote is None:
            return None
        return {"bid": float(quote.bid), "ask": float(quote.ask),
                "mid": float(quote.probability), "spread": float(quote.ask - quote.bid),
                "tick": float(quote.tick)}

    def _run(self):
        current_slug = offset = None
        trend_samples = deque()
        btc_history = deque()
        up_history = deque()
        current_trade_count = 0
        next_count_check = 0.0
        while not self.stop.is_set():
            loop_started = time.monotonic()
            try:
                wall_now = time.time()
                market, market_error = self.discovery.snapshot()
                btc, beat, btc_error = self.btc_feed.snapshot()
                coinbase, coinbase_error = self.coinbase_feed.snapshot()
                quotes = self.feed.quotes()
                market_activity = summarize_market_activity(
                    self.feed.market_activity())

                if market and market[0] != current_slug:
                    current_slug, offset = market[0], None
                    trend_samples.clear(); btc_history.clear(); up_history.clear()
                    self.trader.prepare(current_slug, market[2])
                    self.jump_detector.reset(current_slug, market[2])
                if btc is not None and coinbase is not None and offset is None:
                    offset = btc - coinbase

                monotonic_now = time.monotonic()
                _, last_move, moves = self.jump_detector.snapshot()
                if coinbase is not None:
                    trend_samples.append((monotonic_now, coinbase))
                    synced_value = coinbase if offset is None else coinbase + offset
                    btc_history.append((wall_now, synced_value))
                up_quote = quotes.get("Up")
                if up_quote is not None:
                    up_history.append((wall_now, float(up_quote.probability)))
                while trend_samples and trend_samples[0][0] < monotonic_now - 10:
                    trend_samples.popleft()
                for history in (btc_history, up_history):
                    while history and history[0][0] < wall_now - HISTORY_SECONDS:
                        history.popleft()
                if market:
                    self.trader.observe(market[0], quotes)

                trend = price_trend(list(trend_samples))
                latest = self.trader.latest_trade(market[0] if market else None)
                if monotonic_now >= next_count_check:
                    current_trade_count = (
                        self.trader.trade_count(market[0]) if market else 0)
                    next_count_check = monotonic_now + 0.25
                position = self.trader.position(market[0]) if market else None
                if position and position.get("tp_state") == "sold":
                    position = None
                position = mark_position_from_polymarket(position, quotes)
                errors = list(filter(None, (market_error, btc_error, coinbase_error)))
                market_elapsed = max(0, wall_now - (market[1] - 300)) if market else 0
                state = {
                    "ready": True, "mode": "LIVE" if self.real_money else "DRY RUN",
                    "timestamp": wall_now, "market": market[0] if market else None,
                    "remaining": max(0, int(market[1] - wall_now)) if market else 0,
                    "market_elapsed": market_elapsed,
                    "market_progress": min(100, market_elapsed / 300 * 100),
                    "btc": btc, "price_to_beat": beat, "coinbase": coinbase,
                    "synced_coinbase": None if coinbase is None or offset is None else coinbase + offset,
                    "offset": offset, "delta": None if btc is None or beat is None else btc - beat,
                    "last_move": last_move,
                    "trend": None if not trend else {"direction": trend[0], "change": trend[1]},
                    "moves": list(moves), "threshold": self.threshold,
                    "entry_min": float(MIN_ENTRY_PRICE), "entry_max": float(MAX_ENTRY_PRICE),
                    "config": {"entry_shares": float(ENTRY_SHARES),
                               "take_profit_pct": float(TAKE_PROFIT * 100),
                               "entry_slippage": float(ENTRY_SLIPPAGE),
                               "exit_mode": "Polymarket-managed 5% GTC sell only",
                               "balance_update_poll_ms": int(
                                   BALANCE_UPDATE_INTERVAL_SECONDS * 1000),
                               "balance_get_poll_ms": int(
                                   BALANCE_GET_INTERVAL_SECONDS * 1000),
                               "order_status_poll_ms": round(
                                   TP_ORDER_POLL_INTERVAL_SECONDS * 1000, 2),
                               "trade_start_delay_seconds": float(TRADE_START_DELAY_SECONDS),
                               "trade_cooldown_seconds": TRADE_COOLDOWN_SECONDS,
                               "history_seconds": HISTORY_SECONDS},
                    "jump_noise_filter_usd": JUMP_NOISE_USD,
                    "max_trades_per_window": MAX_TRADES_PER_WINDOW,
                    "trade_count": current_trade_count,
                    "up": self._quote(quotes.get("Up")),
                    "down": self._quote(quotes.get("Down")),
                    "trader_status": self.trader.snapshot(), "trade": latest,
                    "warmup_status": self.trader.warmup_snapshot(),
                    "position": position,
                    "jump_history": self.jump_detector.jump_history(),
                    "execution_timings": self.trader.timing_history(),
                    "market_activity": market_activity,
                    "errors": errors, "btc_history": list(btc_history),
                    "up_history": list(up_history),
                }
                with self.lock:
                    self.state = state
            except Exception as exc:
                with self.lock:
                    self.state["engine_error"] = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - loop_started
            self.stop.wait(max(0, FRAME_MS / 1000 - elapsed))


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Polymarket Trader</title>
<style>
:root{color-scheme:dark;--bg:#070b14;--card:#111827;--line:#243247;--text:#e5edf8;--muted:#8290a6;--green:#22e49a;--red:#ff5d73;--cyan:#39c6f4;--yellow:#ffca4b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif}.wrap{max-width:1500px;margin:auto;padding:18px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.top h1{margin:0;font-size:22px}.badge{padding:7px 12px;border-radius:20px;font-weight:800}.dry{background:#604b13;color:#ffe083}.live{background:#681c2c;color:#ff9aaa}.progress{height:6px;background:#1d2839;border-radius:8px;overflow:hidden;margin-bottom:14px}.progress div{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .2s}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}.metrics{grid-column:span 12;display:grid;grid-template-columns:repeat(6,1fr);gap:14px}.pnl-top{grid-column:span 12;display:grid;grid-template-columns:repeat(4,1fr);gap:18px;border-color:#35506b}.metric .label,.label{color:var(--muted);font-size:12px}.metric .value{font-size:21px;font-weight:750;margin-top:3px}.pnl-top .value{font-size:28px}.half{grid-column:span 6}.third{grid-column:span 4}canvas{width:100%;height:260px;display:block}h2{font-size:14px;margin:0 0 12px;color:#b7c4d8}table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.up{color:var(--green)}.down,.error{color:var(--red)}.status{font-size:16px;line-height:1.5}.rows{display:grid;grid-template-columns:auto 1fr;gap:8px 16px}.rows div:nth-child(odd){color:var(--muted)}.events{max-height:260px;overflow:auto;font:12px ui-monospace,monospace}.event{display:grid;grid-template-columns:75px 85px 1fr;gap:10px;padding:6px;border-bottom:1px solid var(--line)}.event-type{color:var(--yellow);font-weight:700}.pressure{height:18px;display:flex;overflow:hidden;border-radius:9px;background:#1d2839;margin:8px 0}.pressure .p-up{background:var(--green)}.pressure .p-down{background:var(--red)}.pressure-labels{display:flex;justify-content:space-between;font-weight:750}.depth-wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px}.depth-table td,.depth-table th{padding:5px}.note{color:var(--muted);font-size:11px;margin-top:8px}details pre{white-space:pre-wrap;word-break:break-word;color:#a9b8cc;max-height:350px;overflow:auto}@media(max-width:900px){.metrics,.pnl-top{grid-template-columns:repeat(2,1fr)}.half,.third{grid-column:span 12}.depth-wrap{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="top"><h1>BTC Polymarket 5-Minute Trader</h1><span id="mode" class="badge dry">CONNECTING</span></div><div class="progress"><div id="progress"></div></div>
<div class="grid">
 <section class="card metrics">
  <div class="metric"><div class="label">Coinbase BTC</div><div class="value" id="coinbase">--</div></div>
  <div class="metric"><div class="label">Polymarket BTC</div><div class="value" id="btc">--</div></div>
  <div class="metric"><div class="label">Price to beat</div><div class="value" id="beat">--</div></div>
  <div class="metric"><div class="label">Delta</div><div class="value" id="delta">--</div></div>
  <div class="metric"><div class="label">Last jump</div><div class="value" id="jump">--</div></div>
  <div class="metric"><div class="label">Time remaining</div><div class="value" id="remaining">--</div></div>
 </section>
 <section class="card pnl-top">
  <div class="metric"><div class="label">LIVE POSITION P&amp;L</div><div class="value" id="topProfit">--</div></div>
  <div class="metric"><div class="label">RETURN</div><div class="value" id="topProfitPct">--</div></div>
  <div class="metric"><div class="label">POLYMARKET BID</div><div class="value" id="topBid">--</div></div>
  <div class="metric"><div class="label">LIQUIDATION VALUE</div><div class="value" id="topValue">--</div></div>
 </section>
 <section class="card" style="grid-column:span 12"><h2>Synced Coinbase BTC + Polymarket UP + open trade</h2><canvas id="sharedChart" style="height:390px"></canvas></section>
 <section class="card third"><h2>Order book</h2><table><thead><tr><th>Outcome</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Spread</th></tr></thead><tbody><tr class="up"><td>UP</td><td id="ub">--</td><td id="ua">--</td><td id="um">--</td><td id="us">--</td></tr><tr class="down"><td>DOWN</td><td id="db">--</td><td id="da">--</td><td id="dm">--</td><td id="ds">--</td></tr></tbody></table></section>
 <section class="card third"><h2>Signal</h2><div class="rows"><div>Trend</div><div id="trend">--</div><div>Sequence</div><div id="sequence">--</div><div>Threshold</div><div id="threshold">--</div><div>Noise filter</div><div id="noiseFilter">--</div><div>Entry filter</div><div id="filter">--</div></div></section>
 <section class="card third"><h2>Market</h2><div class="rows"><div>Slug</div><div id="market">--</div><div>Elapsed</div><div id="elapsed">--</div><div>Synced price</div><div id="synced">--</div><div>Offset</div><div id="offset">--</div><div>Trades this window</div><div id="tradeCount">--</div><div>Order warmup</div><div id="warmup">--</div></div></section>
 <section class="card half"><h2>Live directional pressure</h2><div class="label">RESTING BID DEPTH (TOP 12 LEVELS)</div><div class="pressure-labels"><span class="up" id="upPressureLabel">UP --</span><span class="down" id="downPressureLabel">DOWN --</span></div><div class="pressure"><div class="p-up" id="upPressure"></div><div class="p-down" id="downPressure"></div></div><div class="label">EXECUTED FLOW (LAST 60 SECONDS)</div><div class="pressure-labels"><span class="up" id="upFlowLabel">UP --</span><span class="down" id="downFlowLabel">DOWN --</span></div><div class="pressure"><div class="p-up" id="upFlow"></div><div class="p-down" id="downFlow"></div></div><div class="note">Shares and orders, not unique people. BUY UP / SELL DOWN count as UP flow; BUY DOWN / SELL UP count as DOWN flow.</div></section>
 <section class="card half"><h2>Liquidity summary</h2><div class="rows"><div>UP bids</div><div id="upBidDepth">--</div><div>UP asks</div><div id="upAskDepth">--</div><div>DOWN bids</div><div id="downBidDepth">--</div><div>DOWN asks</div><div id="downAskDepth">--</div><div>Trades observed (60s)</div><div id="observedTrades">--</div><div>Directional flow</div><div id="netFlow">--</div></div></section>
 <section class="card half"><h2>UP depth</h2><div class="depth-wrap"><table class="depth-table"><thead><tr><th>Bid price</th><th>Shares</th></tr></thead><tbody id="upBidRows"></tbody></table><table class="depth-table"><thead><tr><th>Ask price</th><th>Shares</th></tr></thead><tbody id="upAskRows"></tbody></table></div></section>
 <section class="card half"><h2>DOWN depth</h2><div class="depth-wrap"><table class="depth-table"><thead><tr><th>Bid price</th><th>Shares</th></tr></thead><tbody id="downBidRows"></tbody></table><table class="depth-table"><thead><tr><th>Ask price</th><th>Shares</th></tr></thead><tbody id="downAskRows"></tbody></table></div></section>
 <section class="card" style="grid-column:span 12"><h2>Recent Polymarket trades (last 60 seconds)</h2><div class="events"><table><thead><tr><th>Time</th><th>Outcome</th><th>Side</th><th>Price</th><th>Shares</th><th>Value</th></tr></thead><tbody id="marketTradeRows"></tbody></table></div></section>
 <section class="card half"><h2>Trader status</h2><div class="status" id="status">Starting…</div></section>
 <section class="card half"><h2>Position / Polymarket TP</h2><div class="status" id="trade">No trade in this process.</div></section>
 <section class="card" style="grid-column:span 12"><h2>Feed status</h2><div id="errors" class="status up">Connecting…</div></section>
 <section class="card half"><h2>Strategy configuration</h2><div class="rows"><div>Entry size</div><div id="cfgStake">--</div><div>Polymarket TP</div><div id="cfgTp">--</div><div>Software sell</div><div>Disabled</div><div>Entry slippage</div><div id="cfgEntrySlip">--</div><div>Exit mode</div><div id="cfgExitMode">--</div><div>Balance UPDATE poll</div><div id="cfgUpdatePoll">--</div><div>Balance GET poll</div><div id="cfgGetPoll">--</div><div>Order status poll</div><div id="cfgOrderPoll">--</div></div><span id="cfgStop" hidden></span><span id="cfgTpFallback" hidden></span></section>
 <section class="card half"><h2>Open position details</h2><div class="rows"><div>Outcome</div><div id="posOutcome">--</div><div>Entry on UP axis</div><div id="posEntry">--</div><div>Polymarket bid</div><div id="posBid">--</div><div>Position cost</div><div id="posCost">--</div><div>Liquidation value</div><div id="posValue">--</div><div>Live profit</div><div id="posProfit">--</div><div>Profit %</div><div id="posProfitPct">--</div><div>5% TP on UP axis</div><div id="posTarget">--</div><div>Shares</div><div id="posShares">--</div><div>Sell state</div><div id="posState">--</div></div></section>
 <section class="card" style="grid-column:span 12"><h2>Jump history - latest 200 Coinbase moves</h2><div class="events"><table><thead><tr><th>Time</th><th>Direction</th><th>BTC</th><th>Move</th><th>Streak</th><th>Total</th><th>UP ask</th><th>DOWN ask</th><th>Action</th></tr></thead><tbody id="jumpRows"></tbody></table></div></section>
 <section class="card" style="grid-column:span 12"><h2>Order execution timing</h2><div class="events"><table><thead><tr><th>Time</th><th>Outcome</th><th>Jump</th><th>Detect to queue</th><th>Queue wait</th><th>Sign + match response</th><th>Total</th><th>Result</th></tr></thead><tbody id="timingRows"></tbody></table></div></section>
 <section class="card" style="grid-column:span 12"><h2>Activity log</h2><div class="events" id="activity"></div></section>
 <section class="card" style="grid-column:span 12"><details><summary>All safe API data</summary><pre id="raw">Waiting...</pre></details></section>
</div></div><script>
const $=id=>document.getElementById(id), money=v=>v==null?'--':'$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}), num=(v,n=3)=>v==null?'--':Number(v).toFixed(n);
function sharedPlot(s){const c=$('sharedChart'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);const L=72,R=w-58,T=42,B=h-28,btc=s.btc_history||[],up=s.up_history||[];if(btc.length<2&&up.length<2){x.fillStyle='#8290a6';x.fillText('Waiting for live data…',L,T);return}let prices=btc.map(p=>p[1]);if(s.price_to_beat!=null)prices.push(s.price_to_beat);if(!prices.length)prices=[0,1];let lo=Math.min(...prices),hi=Math.max(...prices),pad=Math.max(1,(hi-lo)*.12);lo-=pad;hi+=pad;const start=Date.now()/1000-300,xp=t=>L+(t-start)/300*(R-L),yb=v=>B-(v-lo)/(hi-lo)*(B-T),yp=v=>B-v*(B-T);x.font='11px system-ui';for(let i=0;i<5;i++){const y=T+(B-T)*i/4;x.strokeStyle='#243247';x.setLineDash([]);x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.fillStyle='#39c6f4';x.textAlign='right';x.fillText('$'+(hi-(hi-lo)*i/4).toLocaleString(undefined,{maximumFractionDigits:0}),L-8,y+4);x.fillStyle='#8290a6';x.textAlign='left';x.fillText((1-i/4).toFixed(2),R+8,y+4)}function line(points,color,map){if(points.length<2)return;x.strokeStyle=color;x.lineWidth=2;x.setLineDash([]);x.beginPath();points.forEach((p,i)=>{const xx=xp(p[0]),yy=map(p[1]);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke()}line(btc,'#39c6f4',yb);line(up,'#22e49a',yp);function level(value,color,dash,label,map){if(value==null)return;const y=map(value);x.strokeStyle=color;x.lineWidth=1.5;x.setLineDash(dash);x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.setLineDash([]);x.fillStyle=color;x.textAlign='right';x.fillText(label,R-5,y-5)}level(s.price_to_beat,'#ffffff',[8,6],'TO BEAT '+money(s.price_to_beat),yb);if(s.position){level(s.position.entry,'#ff9f43',[9,5],`OPEN ${s.position.outcome} · UP ${num(s.position.entry)} · ${num(s.position.shares,3)} shares`,yp);if(s.position.target>=0&&s.position.target<=1)level(s.position.target,'#ffca4b',[3,5],`10% SELL TARGET · UP ${num(s.position.target)} · ${s.position.tp_state}`,yp);if(s.position.stop>=0&&s.position.stop<=1)level(s.position.stop,'#ff6577',[4,4],`20% STOP · UP ${num(s.position.stop)}`,yp)}x.textAlign='left';x.fillStyle='#39c6f4';x.fillText('● Synced Coinbase BTC',L,18);x.fillStyle='#22e49a';x.fillText('● Polymarket UP',L+155,18);x.fillStyle='#ff9f43';x.fillText('— Open trade',L+285,18);x.fillStyle='#ffca4b';x.fillText('— 10% TP',L+385,18);x.fillStyle='#ff6577';x.fillText('— 20% SL',L+455,18);x.fillStyle='#39c6f4';x.textAlign='right';x.fillText('BTC $',L-8,T-10);x.fillStyle='#8290a6';x.textAlign='left';x.fillText('UP probability',R+8,T-10)}
function quote(prefix,q){$(prefix+'b').textContent=num(q?.bid);$(prefix+'a').textContent=num(q?.ask);$(prefix+'m').textContent=num(q?.mid);$(prefix+'s').textContent=num(q?.spread)}
function update(s){const live=s.mode==='LIVE';$('mode').textContent=live?'LIVE REAL-MONEY':'DRY RUN';$('mode').className='badge '+(live?'live':'dry');$('coinbase').textContent=money(s.coinbase);$('btc').textContent=money(s.btc);$('beat').textContent=money(s.price_to_beat);$('delta').textContent=s.delta==null?'--':(s.delta>=0?'+':'')+s.delta.toFixed(2);$('jump').textContent=s.last_move==null?'--':(s.last_move>=0?'+':'')+s.last_move.toFixed(2);$('remaining').textContent=s.remaining+'s';$('market').textContent=s.market||'--';$('synced').textContent=money(s.synced_coinbase);$('offset').textContent=s.offset==null?'--':s.offset.toFixed(2);quote('u',s.up);quote('d',s.down);$('trend').textContent=s.trend?s.trend.direction+' '+(s.trend.change>=0?'+':'')+s.trend.change.toFixed(2):'--';$('sequence').textContent=s.moves?.length?s.moves.map(v=>(v>=0?'+':'')+v.toFixed(2)).join(', '):'--';$('threshold').textContent=num(s.threshold,2)+' USD';$('filter').textContent=num(s.entry_min,2)+' – '+num(s.entry_max,2);$('status').textContent=s.trader_status||'--';const t=s.trade;$('trade').textContent=t?`${t.outcome} / ${t.entry_state} · entry ${num(t.price)} · shares ${num(t.shares,6)} · target ${num(t.target)} · sell ${t.tp_state||'--'}`:'No trade in this process.';const errors=[...(s.errors||[]),s.engine_error].filter(Boolean);$('errors').textContent=errors.length?errors.join(' | '):'All feeds connected';$('errors').className='status '+(errors.length?'error':'up');sharedPlot(s)}
const activity=[],seen={};function addEvent(type,message){activity.unshift({time:new Date().toLocaleTimeString(),type,message});if(activity.length>200)activity.pop();const box=$('activity');box.replaceChildren(...activity.map(e=>{const row=document.createElement('div');row.className='event';for(const [value,cls] of [[e.time,''],[e.type,'event-type'],[e.message,'']]){const cell=document.createElement('div');cell.className=cls;cell.textContent=value;row.appendChild(cell)}return row}))}function changed(key,value,type,message){if(seen[key]!==value){if(seen[key]!==undefined)addEvent(type,message);seen[key]=value}}
function enrich(s){$('progress').style.width=(s.market_progress||0)+'%';$('elapsed').textContent=num(s.market_elapsed,1)+'s / 300s';$('tradeCount').textContent=`${s.trade_count??'--'} / ${s.max_trades_per_window??'--'}`;const c=s.config||{};$('cfgStake').textContent=money(c.stake);$('cfgTp').textContent=num(c.take_profit_pct,1)+'%';$('cfgStop').textContent=num(c.stop_loss_pct,1)+'%';$('cfgEntrySlip').textContent=num(c.entry_slippage,2);$('cfgExitMode').textContent=c.exit_mode||'--';$('cfgUpdatePoll').textContent=(c.balance_update_poll_ms??'--')+'ms';$('cfgGetPoll').textContent=(c.balance_get_poll_ms??'--')+'ms';const p=s.position||{};$('posOutcome').textContent=p.outcome||'--';$('posEntry').textContent=num(p.entry);$('posBid').textContent=num(p.polymarket_bid);$('posCost').textContent=money(p.cost);$('posValue').textContent=money(p.value);const profit=$('posProfit'),profitPct=$('posProfitPct');profit.textContent=p.profit==null?'--':(p.profit>=0?'+':'')+money(p.profit);profitPct.textContent=p.profit_pct==null?'--':(p.profit_pct>=0?'+':'')+num(p.profit_pct,2)+'%';profit.className=p.profit==null?'':(p.profit>=0?'up':'down');profitPct.className=p.profit_pct==null?'':(p.profit_pct>=0?'up':'down');$('posTarget').textContent=num(p.target);$('posShares').textContent=num(p.shares,6);$('posState').textContent=p.tp_state||'--';if(s.trade?.detail)$('trade').textContent+=' | '+s.trade.detail;changed('market',s.market,'MARKET','Switched to '+(s.market||'none'));changed('status',s.trader_status,'TRADER',s.trader_status||'--');const tradeKey=s.trade?JSON.stringify([s.trade.market,s.trade.entry_state,s.trade.tp_state,s.trade.price,s.trade.shares,s.trade.detail]):'none';changed('trade',tradeKey,'POSITION',s.trade?`${s.trade.outcome} entry=${num(s.trade.price)} shares=${num(s.trade.shares,6)} target=${num(s.trade.target)} state=${s.trade.tp_state||s.trade.entry_state}${s.trade.detail?' detail='+s.trade.detail:''}`:'No active trade');const errorKey=JSON.stringify([...(s.errors||[]),s.engine_error].filter(Boolean));changed('errors',errorKey,'FEEDS',errorKey==='[]'?'All feeds connected':errorKey);const safe={...s,btc_history:`${s.btc_history?.length||0} samples`,up_history:`${s.up_history?.length||0} samples`};$('raw').textContent=JSON.stringify(safe,null,2)}
function updateTopPnl(s){const p=s.position||{},profit=$('topProfit'),pct=$('topProfitPct');profit.textContent=p.profit==null?'--':(p.profit>=0?'+':'')+money(p.profit);pct.textContent=p.profit_pct==null?'--':(p.profit_pct>=0?'+':'')+num(p.profit_pct,2)+'%';profit.className='value '+(p.profit==null?'':p.profit>=0?'up':'down');pct.className='value '+(p.profit_pct==null?'':p.profit_pct>=0?'up':'down');$('topBid').textContent=num(p.polymarket_bid);$('topValue').textContent=money(p.value)}
function updateTpFallbackConfig(s){const seconds=s.config?.tp_fallback_seconds;$('cfgTpFallback').textContent=seconds==null?'--':num(seconds,1)+'s then software sell'}
function updateOpenPositionSummary(s){const p=s.position,box=$('trade');if(!p){box.textContent='No open position on Polymarket.';box.className='status up';return}box.textContent=`${p.outcome} OPEN · entry ${num(p.entry)} · bid ${num(p.polymarket_bid)} · ${num(p.shares,6)} shares · P&L ${p.profit>=0?'+':''}${money(p.profit)} (${p.profit_pct>=0?'+':''}${num(p.profit_pct,2)}%) · TP ${num(p.target)} · ${p.tp_state||'--'}`;box.className='status '+(p.profit==null?'':p.profit>=0?'up':'down')}
function renderRows(id,rows,cells,rowClass){const body=$(id),fragment=document.createDocumentFragment();for(const row of rows){const tr=document.createElement('tr');if(rowClass)tr.className=rowClass(row);for(const value of cells(row)){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}fragment.appendChild(tr)}body.replaceChildren(fragment)}
function renderMonitoring(s){const jumps=[...(s.jump_history||[])].reverse();renderRows('jumpRows',jumps,j=>[new Date(j.timestamp*1000).toLocaleTimeString(),j.direction,money(j.price),(j.delta>=0?'+':'')+num(j.delta,2),j.streak_count,(j.streak_total>=0?'+':'')+num(j.streak_total,2),num(j.up_ask),num(j.down_ask),j.action],j=>j.action.includes('queued')?'up':j.action.includes('skipped')?'down':'');const timings=[...(s.execution_timings||[])].reverse();const ms=v=>v==null?'--':num(v,2)+' ms';renderRows('timingRows',timings,t=>[new Date(t.timestamp*1000).toLocaleTimeString(),t.outcome,(t.move>=0?'+':'')+num(t.move,2),ms(t.detected_to_queue_ms),ms(t.queue_wait_ms),ms(t.request_ms),ms(t.total_ms),t.result],t=>t.result==='filled'?'up':t.result==='rejected'||String(t.result).includes('error')?'down':'')}
function renderMarketActivity(s){const a=s.market_activity||{},d=a.depth||{},up=d.Up||{},down=d.Down||{},pct=v=>Math.max(0,Math.min(100,Number(v)||0)),shares=v=>(Number(v)||0).toLocaleString(undefined,{maximumFractionDigits:2})+' sh';const upP=pct(a.up_pressure_pct),downP=pct(a.down_pressure_pct),upF=pct(a.up_flow_pct),downF=pct(a.down_flow_pct);$('upPressure').style.width=upP+'%';$('downPressure').style.width=downP+'%';$('upPressureLabel').textContent='UP '+num(upP,1)+'%';$('downPressureLabel').textContent='DOWN '+num(downP,1)+'%';$('upFlow').style.width=upF+'%';$('downFlow').style.width=downF+'%';$('upFlowLabel').textContent='UP '+shares(a.up_flow)+' ('+num(upF,1)+'%)';$('downFlowLabel').textContent='DOWN '+shares(a.down_flow)+' ('+num(downF,1)+'%)';$('upBidDepth').textContent=shares(up.bid_shares);$('upAskDepth').textContent=shares(up.ask_shares);$('downBidDepth').textContent=shares(down.bid_shares);$('downAskDepth').textContent=shares(down.ask_shares);$('observedTrades').textContent=a.trade_count??0;const net=(Number(a.up_flow)||0)-(Number(a.down_flow)||0);$('netFlow').textContent=(net>=0?'+':'')+shares(net)+' '+(net>0?'UP':net<0?'DOWN':'balanced');const levels=(id,rows,cls)=>renderRows(id,rows||[],r=>[num(r[0]),shares(r[1])],()=>cls);levels('upBidRows',up.bids,'up');levels('upAskRows',up.asks,'down');levels('downBidRows',down.bids,'down');levels('downAskRows',down.asks,'up');const trades=[...(a.trades||[])].reverse();renderRows('marketTradeRows',trades,t=>[new Date(t.timestamp*1000).toLocaleTimeString(),String(t.outcome||'--').toUpperCase(),t.side||'--',num(t.price),shares(t.size),money((Number(t.price)||0)*(Number(t.size)||0))],t=>t.outcome==='Up'?'up':'down')}
const enrichBase=enrich;enrich=s=>{enrichBase(s);updateTopPnl(s);updateTpFallbackConfig(s);$('warmup').textContent=s.warmup_status||'--';$('noiseFilter').textContent='ignore <= '+num(s.jump_noise_filter_usd,2)+' USD'};
const basicUpdate=update;update=s=>{basicUpdate(s);enrich(s);updateOpenPositionSummary(s);renderMonitoring(s);renderMarketActivity(s)};
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});update(await r.json())}catch(e){$('errors').textContent='Dashboard connection lost: '+e;$('errors').className='status error'}}setInterval(poll,100);poll();addEventListener('resize',poll);
</script></body></html>'''

# Keep the dense embedded dashboard script readable above while updating its
# generated labels and configuration bindings for this web-only profile.
HTML = HTML.replace("10% SELL TARGET", "5% SELL TARGET")
HTML = HTML.replace("10% TP", "5% TP").replace("20% SL", "")
HTML = HTML.replace(
    "$('cfgStake').textContent=money(c.stake)",
    "$('cfgStake').textContent=num(c.entry_shares,2)+' shares'")
HTML = HTML.replace(
    "$('cfgGetPoll').textContent=(c.balance_get_poll_ms??'--')+'ms'",
    "$('cfgGetPoll').textContent=(c.balance_get_poll_ms??'--')+'ms';"
    "$('cfgOrderPoll').textContent=(c.order_status_poll_ms??'--')+'ms'")


def create_app(engine):
    @asynccontextmanager
    async def lifespan(_app):
        yield
        engine.close()
    app = FastAPI(title="BTC Polymarket Dashboard", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.get("/api/state")
    async def state():
        return engine.snapshot()
    return app


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Submit real-money orders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    args = parser.parse_args()
    engine = WebTradingEngine(real_money=args.live)
    engine.start()
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url} ({'LIVE REAL-MONEY' if args.live else 'DRY RUN'})")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
