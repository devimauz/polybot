"""Web dashboard for the BTC Polymarket jump trader.

Dry run: python btc_polymarket_qwen.py
Live:    python btc_polymarket_qwen.py --live
"""
import argparse
import os
import threading
import time
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from decimal import Decimal

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
    RESTING_TIMEOUT_SECONDS,
    SELL_RETRY_SECONDS, SELL_RETRY_WINDOW_SECONDS, STAKE, TAKE_PROFIT,
    TRADE_START_DELAY_SECONDS,
    CoinbasePriceFeed, Discovery, EventDrivenJumpDetector, MarketFeed,
    PolymarketBTCFeed,
    PolymarketJumpTrader, directional_jump, price_trend,
)

# Web-only strategy profile copied from btc_price_chart.py. The shared GUI and
# terminal trader retain their original live configuration in other processes.
STAKE = Decimal("1.10")
TAKE_PROFIT = Decimal("0.07")
MIN_ENTRY_PRICE = Decimal("0.20")
MAX_ENTRY_PRICE = Decimal("0.80")
TRADE_START_DELAY_SECONDS = 0.0
TRADE_COOLDOWN_SECONDS = 1.0
MAX_TRADES_PER_WINDOW = 300  # one-second cooldown is the effective limit
strategy.STAKE = STAKE
strategy.TAKE_PROFIT = TAKE_PROFIT
strategy.MIN_ENTRY_PRICE = MIN_ENTRY_PRICE
strategy.MAX_ENTRY_PRICE = MAX_ENTRY_PRICE
strategy.TRADE_START_DELAY_SECONDS = TRADE_START_DELAY_SECONDS
strategy.TRADE_COOLDOWN_SECONDS = TRADE_COOLDOWN_SECONDS
strategy.MAX_TRADES_PER_WINDOW = MAX_TRADES_PER_WINDOW

HISTORY_SECONDS = 300


class WebTradingEngine:
    def __init__(self, real_money=False):
        self.real_money = real_money
        self.feed = MarketFeed()
        self.btc_feed = PolymarketBTCFeed()
        self.coinbase_feed = CoinbasePriceFeed()
        self.discovery = Discovery(self.feed)
        self.trader = PolymarketJumpTrader(live=real_money)
        self.threshold = float(os.environ.get("JUMP_THRESHOLD_USD", "10"))
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
                latest = self.trader.latest_trade(market[0]) if market else None
                if monotonic_now >= next_count_check:
                    current_trade_count = (
                        self.trader.trade_count(market[0]) if market else 0)
                    next_count_check = monotonic_now + 0.25
                position = self.trader.position(market[0]) if market else None
                if position and position.get("tp_state") == "sold":
                    position = None
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
                    "config": {"stake": float(STAKE), "take_profit_pct": float(TAKE_PROFIT * 100),
                               "entry_slippage": float(ENTRY_SLIPPAGE),
                               "exit_mode": "resting limit TP at target, forced market sell if not filled in 0.5s",
                               "resting_timeout_seconds": float(RESTING_TIMEOUT_SECONDS),
                               "sell_retry_ms": int(SELL_RETRY_SECONDS * 1000),
                               "sell_window_seconds": float(SELL_RETRY_WINDOW_SECONDS),
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
                    "jump_history": self.jump_detector.jump_history(),
                    "position": position,
                    "execution_timings": self.trader.timing_history(),
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
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif}.wrap{max-width:1500px;margin:auto;padding:18px}.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.top h1{margin:0;font-size:22px}.badge{padding:7px 12px;border-radius:20px;font-weight:800}.dry{background:#604b13;color:#ffe083}.live{background:#681c2c;color:#ff9aaa}.progress{height:6px;background:#1d2839;border-radius:8px;overflow:hidden;margin-bottom:14px}.progress div{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .2s}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}.metrics{grid-column:span 12;display:grid;grid-template-columns:repeat(6,1fr);gap:14px}.metric .label,.label{color:var(--muted);font-size:12px}.metric .value{font-size:21px;font-weight:750;margin-top:3px}.half{grid-column:span 6}.third{grid-column:span 4}canvas{width:100%;height:260px;display:block}h2{font-size:14px;margin:0 0 12px;color:#b7c4d8}table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.up{color:var(--green)}.down,.error{color:var(--red)}.status{font-size:16px;line-height:1.5}.rows{display:grid;grid-template-columns:auto 1fr;gap:8px 16px}.rows div:nth-child(odd){color:var(--muted)}.events{max-height:260px;overflow:auto;font:12px ui-monospace,monospace}.event{display:grid;grid-template-columns:75px 85px 1fr;gap:10px;padding:6px;border-bottom:1px solid var(--line)}.event-type{color:var(--yellow);font-weight:700}details pre{white-space:pre-wrap;word-break:break-word;color:#a9b8cc;max-height:350px;overflow:auto}@media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr)}.half,.third{grid-column:span 12}}
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
 <section class="card" style="grid-column:span 12"><h2>Synced Coinbase BTC + Polymarket UP + open trade</h2><canvas id="sharedChart" style="height:390px"></canvas></section>
 <section class="card third"><h2>Order book</h2><table><thead><tr><th>Outcome</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Spread</th></tr></thead><tbody><tr class="up"><td>UP</td><td id="ub">--</td><td id="ua">--</td><td id="um">--</td><td id="us">--</td></tr><tr class="down"><td>DOWN</td><td id="db">--</td><td id="da">--</td><td id="dm">--</td><td id="ds">--</td></tr></tbody></table></section>
 <section class="card third"><h2>Signal</h2><div class="rows"><div>Trend</div><div id="trend">--</div><div>Sequence</div><div id="sequence">--</div><div>Threshold</div><div id="threshold">--</div><div>Noise filter</div><div id="noiseFilter">--</div><div>Entry filter</div><div id="filter">--</div></div></section>
 <section class="card third"><h2>Market</h2><div class="rows"><div>Slug</div><div id="market">--</div><div>Elapsed</div><div id="elapsed">--</div><div>Synced price</div><div id="synced">--</div><div>Offset</div><div id="offset">--</div><div>Trades this window</div><div id="tradeCount">--</div><div>Order warmup</div><div id="warmup">--</div></div></section>
 <section class="card half"><h2>Trader status</h2><div class="status" id="status">Starting…</div></section>
 <section class="card half"><h2>Position / software sell</h2><div class="status" id="trade">No trade in this process.</div></section>
 <section class="card" style="grid-column:span 12"><h2>Feed status</h2><div id="errors" class="status up">Connecting…</div></section>
 <section class="card half"><h2>Strategy configuration</h2><div class="rows"><div>Stake</div><div id="cfgStake">--</div><div>Software TP</div><div id="cfgTp">--</div><div>Entry slippage</div><div id="cfgEntrySlip">--</div><div>Exit mode</div><div id="cfgExitMode">--</div><div>Sell retry</div><div id="cfgRetry">--</div><div>Sell retry window</div><div id="cfgWindow">--</div></div></section>
 <section class="card half"><h2>Open position details</h2><div class="rows"><div>Outcome</div><div id="posOutcome">--</div><div>Entry on UP axis</div><div id="posEntry">--</div><div id="posTpLabel">TP on UP axis</div><div id="posTarget">--</div><div>Shares</div><div id="posShares">--</div><div>Sell state</div><div id="posState">--</div></div></section>
 <section class="card" style="grid-column:span 12"><h2>Jump history - latest 200 Coinbase moves</h2><div class="events"><table><thead><tr><th>Time</th><th>Direction</th><th>BTC</th><th>Move</th><th>Streak</th><th>Total</th><th>UP ask</th><th>DOWN ask</th><th>Action</th></tr></thead><tbody id="jumpRows"></tbody></table></div></section>
 <section class="card" style="grid-column:span 12"><h2>Order execution timing</h2><div class="events"><table><thead><tr><th>Time</th><th>Outcome</th><th>Jump</th><th>Detect to queue</th><th>Queue wait</th><th>Sign + match response</th><th>Total</th><th>Result</th></tr></thead><tbody id="timingRows"></tbody></table></div></section>
 <section class="card" style="grid-column:span 12"><h2>Activity log</h2><div class="events" id="activity"></div></section>
 <section class="card" style="grid-column:span 12"><details><summary>All safe API data</summary><pre id="raw">Waiting...</pre></details></section>
</div></div><script>
const $=id=>document.getElementById(id), money=v=>v==null?'--':'$'+Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}), num=(v,n=3)=>v==null?'--':Number(v).toFixed(n);
function sharedPlot(s){const c=$('sharedChart'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);const L=72,R=w-58,T=42,B=h-28,btc=s.btc_history||[],up=s.up_history||[];if(btc.length<2&&up.length<2){x.fillStyle='#8290a6';x.fillText('Waiting for live data…',L,T);return}let prices=btc.map(p=>p[1]);if(s.price_to_beat!=null)prices.push(s.price_to_beat);if(!prices.length)prices=[0,1];let lo=Math.min(...prices),hi=Math.max(...prices),pad=Math.max(1,(hi-lo)*.12);lo-=pad;hi+=pad;const start=Date.now()/1000-300,xp=t=>L+(t-start)/300*(R-L),yb=v=>B-(v-lo)/(hi-lo)*(B-T),yp=v=>B-v*(B-T);x.font='11px system-ui';for(let i=0;i<5;i++){const y=T+(B-T)*i/4;x.strokeStyle='#243247';x.setLineDash([]);x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.fillStyle='#39c6f4';x.textAlign='right';x.fillText('$'+(hi-(hi-lo)*i/4).toLocaleString(undefined,{maximumFractionDigits:0}),L-8,y+4);x.fillStyle='#8290a6';x.textAlign='left';x.fillText((1-i/4).toFixed(2),R+8,y+4)}function line(points,color,map){if(points.length<2)return;x.strokeStyle=color;x.lineWidth=2;x.setLineDash([]);x.beginPath();points.forEach((p,i)=>{const xx=xp(p[0]),yy=map(p[1]);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke()}line(btc,'#39c6f4',yb);line(up,'#22e49a',yp);function level(value,color,dash,label,map){if(value==null)return;const y=map(value);x.strokeStyle=color;x.lineWidth=1.5;x.setLineDash(dash);x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.setLineDash([]);x.fillStyle=color;x.textAlign='right';x.fillText(label,R-5,y-5)}level(s.price_to_beat,'#ffffff',[8,6],'TO BEAT '+money(s.price_to_beat),yb);if(s.position){level(s.position.entry,'#ff9f43',[9,5],`OPEN ${s.position.outcome} · UP ${num(s.position.entry)} · ${num(s.position.shares,3)} shares`,yp);if(s.position.target>=0&&s.position.target<=1)level(s.position.target,'#ffca4b',[3,5],`${num((s.config||{}).take_profit_pct??5.0,1)}% SELL TARGET · UP ${num(s.position.target)} · ${s.position.tp_state}`,yp)}x.textAlign='left';x.fillStyle='#39c6f4';x.fillText('● Synced Coinbase BTC',L,18);x.fillStyle='#22e49a';x.fillText('● Polymarket UP',L+155,18);x.fillStyle='#ff9f43';x.fillText('— Open trade',L+285,18);x.fillStyle='#ffca4b';x.fillText('— '+num((s.config||{}).take_profit_pct??5.0,1)+'% TP',L+385,18);x.fillStyle='#39c6f4';x.textAlign='right';x.fillText('BTC $',L-8,T-10);x.fillStyle='#8290a6';x.textAlign='left';x.fillText('UP probability',R+8,T-10)}
function quote(prefix,q){$(prefix+'b').textContent=num(q?.bid);$(prefix+'a').textContent=num(q?.ask);$(prefix+'m').textContent=num(q?.mid);$(prefix+'s').textContent=num(q?.spread)}
function update(s){const live=s.mode==='LIVE';$('mode').textContent=live?'LIVE REAL-MONEY':'DRY RUN';$('mode').className='badge '+(live?'live':'dry');$('coinbase').textContent=money(s.coinbase);$('btc').textContent=money(s.btc);$('beat').textContent=money(s.price_to_beat);$('delta').textContent=s.delta==null?'--':(s.delta>=0?'+':'')+s.delta.toFixed(2);$('jump').textContent=s.last_move==null?'--':(s.last_move>=0?'+':'')+s.last_move.toFixed(2);$('remaining').textContent=s.remaining+'s';$('market').textContent=s.market||'--';$('synced').textContent=money(s.synced_coinbase);$('offset').textContent=s.offset==null?'--':s.offset.toFixed(2);quote('u',s.up);quote('d',s.down);$('trend').textContent=s.trend?s.trend.direction+' '+(s.trend.change>=0?'+':'')+s.trend.change.toFixed(2):'--';$('sequence').textContent=s.moves?.length?s.moves.map(v=>(v>=0?'+':'')+v.toFixed(2)).join(', '):'--';$('threshold').textContent=num(s.threshold,2)+' USD';$('filter').textContent=num(s.entry_min,2)+' – '+num(s.entry_max,2);$('status').textContent=s.trader_status||'--';const t=s.trade;$('trade').textContent=t?`${t.outcome} / ${t.entry_state} · entry ${num(t.price)} · shares ${num(t.shares,6)} · target ${num(t.target)} · sell ${t.tp_state||'--'}`:'No trade in this process.';const errors=[...(s.errors||[]),s.engine_error].filter(Boolean);$('errors').textContent=errors.length?errors.join(' | '):'All feeds connected';$('errors').className='status '+(errors.length?'error':'up');sharedPlot(s)}
const activity=[],seen={};function addEvent(type,message){activity.unshift({time:new Date().toLocaleTimeString(),type,message});if(activity.length>200)activity.pop();const box=$('activity');box.replaceChildren(...activity.map(e=>{const row=document.createElement('div');row.className='event';for(const [value,cls] of [[e.time,''],[e.type,'event-type'],[e.message,'']]){const cell=document.createElement('div');cell.className=cls;cell.textContent=value;row.appendChild(cell)}return row}))}function changed(key,value,type,message){if(seen[key]!==value){if(seen[key]!==undefined)addEvent(type,message);seen[key]=value}}
function enrich(s){$('progress').style.width=(s.market_progress||0)+'%';$('elapsed').textContent=num(s.market_elapsed,1)+'s / 300s';$('tradeCount').textContent=`${s.trade_count??'--'} / ${s.max_trades_per_window??'--'}`;const c=s.config||{};$('cfgStake').textContent=money(c.stake);$('cfgTp').textContent=num(c.take_profit_pct,1)+'%';$('cfgEntrySlip').textContent=num(c.entry_slippage,2);$('cfgExitMode').textContent=c.exit_mode||'--';$('cfgRetry').textContent=(c.sell_retry_ms??'--')+'ms';$('cfgWindow').textContent=num(c.sell_window_seconds,1)+'s';const p=s.position||{};$('posOutcome').textContent=p.outcome||'--';$('posEntry').textContent=num(p.entry);$('posTpLabel').textContent=num(c.take_profit_pct,1)+'% TP on UP axis';$('posTarget').textContent=num(p.target);$('posShares').textContent=num(p.shares,6);$('posState').textContent=p.tp_state||'--';if(s.trade?.detail)$('trade').textContent+=' | '+s.trade.detail;changed('market',s.market,'MARKET','Switched to '+(s.market||'none'));changed('status',s.trader_status,'TRADER',s.trader_status||'--');const tradeKey=s.trade?JSON.stringify([s.trade.market,s.trade.entry_state,s.trade.tp_state,s.trade.price,s.trade.shares,s.trade.detail]):'none';changed('trade',tradeKey,'POSITION',s.trade?`${s.trade.outcome} entry=${num(s.trade.price)} shares=${num(s.trade.shares,6)} target=${num(s.trade.target)} state=${s.trade.tp_state||s.trade.entry_state}${s.trade.detail?' detail='+s.trade.detail:''}`:'No active trade');const errorKey=JSON.stringify([...(s.errors||[]),s.engine_error].filter(Boolean));changed('errors',errorKey,'FEEDS',errorKey==='[]'?'All feeds connected':errorKey);const safe={...s,btc_history:`${s.btc_history?.length||0} samples`,up_history:`${s.up_history?.length||0} samples`};$('raw').textContent=JSON.stringify(safe,null,2)}
function renderRows(id,rows,cells,rowClass){const body=$(id),fragment=document.createDocumentFragment();for(const row of rows){const tr=document.createElement('tr');if(rowClass)tr.className=rowClass(row);for(const value of cells(row)){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}fragment.appendChild(tr)}body.replaceChildren(fragment)}
function renderMonitoring(s){const jumps=[...(s.jump_history||[])].reverse();renderRows('jumpRows',jumps,j=>[new Date(j.timestamp*1000).toLocaleTimeString(),j.direction,money(j.price),(j.delta>=0?'+':'')+num(j.delta,2),j.streak_count,(j.streak_total>=0?'+':'')+num(j.streak_total,2),num(j.up_ask),num(j.down_ask),j.action],j=>j.action.includes('queued')?'up':j.action.includes('skipped')?'down':'');const timings=[...(s.execution_timings||[])].reverse();const ms=v=>v==null?'--':num(v,2)+' ms';renderRows('timingRows',timings,t=>[new Date(t.timestamp*1000).toLocaleTimeString(),t.outcome,(t.move>=0?'+':'')+num(t.move,2),ms(t.detected_to_queue_ms),ms(t.queue_wait_ms),ms(t.request_ms),ms(t.total_ms),t.result],t=>t.result==='filled'?'up':t.result==='rejected'||String(t.result).includes('error')?'down':'')}
const enrichBase=enrich;enrich=s=>{enrichBase(s);$('warmup').textContent=s.warmup_status||'--';$('noiseFilter').textContent='ignore <= '+num(s.jump_noise_filter_usd,2)+' USD'};
const basicUpdate=update;update=s=>{basicUpdate(s);enrich(s);renderMonitoring(s)};
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});update(await r.json())}catch(e){$('errors').textContent='Dashboard connection lost: '+e;$('errors').className='status error'}}setInterval(poll,100);poll();addEventListener('resize',poll);
</script></body></html>'''


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
