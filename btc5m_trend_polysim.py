"""Trend-confirmed BTC 5m Polysimulator bot (paper orders only).

At 150s, Coinbase must be >=2 bps from its window open, its last 3s move must
agree, and the selected Polymarket bid must be non-decreasing over 3s. The bot
buys only at <=$0.80, then holds to resolution. One entry per window.
"""
import json,logging,os,time,uuid
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from btc5m_momentum_strategy import fee_for,momentum_signal
from btc5m_momentum_polysim import API,Journal,discover,get_book,polysim_winner
from polymarket_btc5m_logger import CoinbaseFeed

ROOT=Path(__file__).resolve().parent;LOG=logging.getLogger("btc5m-trend-polysim")

def main():
 load_dotenv(ROOT/".env");key=os.getenv("POLYSIM_API_KEY","").strip()
 if not key:raise SystemExit("Set POLYSIM_API_KEY in .env")
 logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
 feed=CoinbaseFeed();feed.start();journal=Journal(ROOT/"price_logs"/"trend-polysim"/"trades.jsonl")
 state={};pending=[];session=0.;last_settle=0
 try:
  with httpx.Client(timeout=8,headers={"X-API-Key":key})as client:
   while True:
    now=time.time();start=int(now)//300*300;t=now-start;cb=feed.snapshot(now)
    if state.get("start")!=start:
     state={"start":start,"market":None,"open":None,"done":False,"cb":deque(maxlen=100),
            "bids":{"Up":deque(maxlen=30),"Down":deque(maxlen=30)},"last_book":0,"last_discover":0}
     LOG.info("new window btc-updown-5m-%d",start)
    if cb.get("price"):
     state["cb"].append((now,cb["price"]))
     if state["open"]is None and t<=5:state["open"]=cb["price"]
    if state["market"]is None and now-state["last_discover"]>=1:
     state["last_discover"]=now
     try:state["market"]=discover(client,start)
     except httpx.HTTPError:pass
    market=state["market"]
    # Warm both bid histories shortly before the decision.
    if market and 145<=t<=152 and now-state["last_book"]>=.4:
     state["last_book"]=now
     for side in("Up","Down"):
      try:
       book=get_book(client,market["tokens"][side]);state["bids"][side].append((now,book.get("bid"),book))
      except(httpx.HTTPError,ValueError,KeyError):pass
    if t>=150 and not state["done"]:
     state["done"]=True;reason=None;signal=None;order=None
     try:
      if not market or not state["open"]or not cb.get("price"):reason="missing_data"
      else:
       move=(cb["price"]/state["open"]-1)*10000;side="Up"if move>0 else"Down"
       old_cb=next((p for stamp,p in state["cb"]if stamp>=now-3),None)
       recent=(cb["price"]/old_cb-1)*10000 if old_cb else 0
       history=state["bids"][side];current=history[-1]if history else None
       old_bid=next((bid for stamp,bid,_ in history if stamp>=now-3 and bid is not None),None)
       agrees=(side=="Up"and recent>=0)or(side=="Down"and recent<=0)
       if abs(move)<2:reason="open_move_below_2bps"
       elif not agrees:reason="coinbase_3s_trend_disagrees"
       elif not current or old_bid is None or current[1]is None or current[1]<old_bid:reason="bid_not_rising"
       else:
        signal=momentum_signal(state["open"],cb["price"],{side:current[2]},2,2,.45,.80)
        if not signal:reason="price_or_depth_filter"
        else:
         body={"market_id":market["condition_id"],"side":"BUY","outcome":side,"amount":"2",
               "price":f"{signal['worst_price']:.2f}","order_type":"market","time_in_force":"FOK"}
         response=client.post(API+"/orders",json=body,headers={"Idempotency-Key":str(uuid.uuid4())});order=response.json()
         if response.status_code not in(200,201)or order.get("status")!="FILLED":reason=order.get("error")or order.get("status");signal=None
     except(httpx.HTTPError,ValueError,KeyError)as exc:reason=type(exc).__name__;signal=None
     rec={"ts":datetime.now(timezone.utc).isoformat(),"event":"fill"if signal else"skip","slug":f"btc-updown-5m-{start}","reason":reason,"signal":signal,"order":order}
     if signal:
      fill=float(order.get("fill_price")or signal["avg_price"]);shares=float(order.get("quantity")or signal["shares"]);cost=2+fee_for(fill,shares)
      pending.append({"slug":market["slug"],"condition_id":market["condition_id"],"side":signal["side"],"shares":shares,"cost":cost})
      rec.update({"fill":fill,"shares":shares,"cost":cost});LOG.info("BUY %s fill=%.3f",signal["side"],fill)
     else:LOG.info("SKIP %s",reason)
     journal.write(rec)
    if pending and now-last_settle>=30:
     last_settle=now;still=[]
     for pos in pending:
      try:winner=polysim_winner(client,pos["condition_id"])
      except Exception:winner=None
      if not winner:still.append(pos);continue
      pnl=(pos["shares"]if winner==pos["side"]else 0)-pos["cost"];session+=pnl
      journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"settle","slug":pos["slug"],"side":pos["side"],"winner":winner,"pnl":pnl,"session_pnl":session})
      LOG.info("SETTLE winner=%s pnl=%+.2f session=%+.2f",winner,pnl,session)
     pending=still
    time.sleep(.1)
 except KeyboardInterrupt:LOG.info("stopped")
 finally:feed.close();journal.close()
if __name__=="__main__":main()
