"""Experimental multi-entry take-profit strategy using Polysimulator paper orders."""

import argparse, json, logging, math, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from btc5m_momentum_strategy import fee_for, momentum_signal
from btc5m_momentum_polysim import API, Journal, discover, get_book, polysim_winner
from polymarket_btc5m_logger import CoinbaseFeed

ROOT=Path(__file__).resolve().parent; LOG=logging.getLogger("btc5m-takeprofit-polysim")

def post(client, market, side, amount=None, quantity=None, price=None):
    body={"market_id":market["condition_id"],"side":"BUY" if amount is not None else "SELL",
          "outcome":side,"price":f"{price:.2f}","order_type":"market","time_in_force":"FOK"}
    body["amount" if amount is not None else "quantity"]=str(amount if amount is not None else quantity)
    r=client.post(API+"/orders",json=body,headers={"Idempotency-Key":str(uuid.uuid4())})
    try:data=r.json()
    except ValueError:data={"error":r.text[:300]}
    return r,data

def main():
    load_dotenv(ROOT/".env"); p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stake",type=float,default=2);p.add_argument("--entry-second",type=float,default=90)
    p.add_argument("--minimum-move-bps",type=float,default=6);p.add_argument("--take-profit",type=float,default=10,
                   help="minimum net position return after entry and exit fees")
    p.add_argument("--cooldown",type=float,default=10);p.add_argument("--max-trades",type=int,default=2)
    p.add_argument("--ask-min",type=float,default=.45);p.add_argument("--ask-max",type=float,default=.90)
    p.add_argument("--poll",type=float,default=.25);args=p.parse_args()
    key=os.getenv("POLYSIM_API_KEY","").strip()
    if not key:raise SystemExit("Set POLYSIM_API_KEY in .env")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    feed=CoinbaseFeed();feed.start();journal=Journal(ROOT/"price_logs"/"takeprofit-polysim"/"trades.jsonl")
    state={};pending=[];session_pnl=0;last_settle=0
    try:
      with httpx.Client(timeout=8,headers={"X-API-Key":key}) as client:
       while True:
        now=time.time();start=int(now)//300*300;t=now-start;cb=feed.snapshot(now)
        if state.get("start")!=start:
            if state.get("position") and state.get("market"):
                pending.append({**state["position"],"slug":state["market"]["slug"],
                                "condition_id":state["market"]["condition_id"]})
            state={"start":start,"market":None,"open":None,"entries":0,"position":None,"next":args.entry_second,"last_discover":0}
            LOG.info("new window btc-updown-5m-%d",start)
        if state["open"] is None and t<=5 and cb.get("price"):state["open"]=cb["price"]
        if state["market"] is None and now-state["last_discover"]>=1:
            state["last_discover"]=now
            try:state["market"]=discover(client,start)
            except httpx.HTTPError:pass
        m=state["market"];pos=state["position"]
        try:
         if pos and m:
            book=get_book(client,m["tokens"][pos["side"]]);bid=book.get("bid")
            net_at_bid=(pos["shares"]*bid-fee_for(bid,pos["shares"])) if bid is not None else None
            if net_at_bid is not None and net_at_bid>=pos["cost"]*(1+args.take_profit/100):
                r,data=post(client,m,pos["side"],quantity=pos["shares"],price=bid)
                if r.status_code in (200,201) and data.get("status")=="FILLED":
                    fill=float(data.get("fill_price") or bid);proceeds=pos["shares"]*fill
                    pnl=proceeds-fee_for(fill,pos["shares"])-pos["cost"];session_pnl+=pnl
                    journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"take_profit","slug":m["slug"],"pnl":pnl,"session_pnl":session_pnl,"order":data})
                    LOG.info("TAKE PROFIT %s pnl=%+.2f",pos["side"],pnl);state["position"]=None;state["next"]=t+args.cooldown
         elif (m and state["open"] and cb.get("price") and t>=state["next"] and t<290
               and state["entries"]<args.max_trades):
            move=(cb["price"]/state["open"]-1)*10000
            if abs(move)>=args.minimum_move_bps:
                side="Up" if move>0 else "Down";book=get_book(client,m["tokens"][side])
                sig=momentum_signal(state["open"],cb["price"],{side:book},args.stake,args.minimum_move_bps,args.ask_min,args.ask_max)
                if sig:
                    r,data=post(client,m,side,amount=args.stake,price=sig["worst_price"])
                    if r.status_code in (200,201) and data.get("status")=="FILLED":
                        fill=float(data.get("fill_price") or sig["avg_price"]);shares=float(data.get("quantity") or sig["shares"]);cost=args.stake+fee_for(fill,shares)
                        state["position"]={"side":side,"fill":fill,"shares":shares,"cost":cost};state["entries"]+=1
                        journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"fill","slug":m["slug"],"side":side,"fill":fill,"shares":shares,"cost":cost,"order":data})
                        LOG.info("BUY %s fill=%.3f entry=%d/%d",side,fill,state["entries"],args.max_trades)
        except (httpx.HTTPError,ValueError,KeyError) as exc:LOG.warning("cycle failed: %s",type(exc).__name__)
        if pending and now-last_settle>=30:
            last_settle=now;still=[]
            for pos in pending:
                try:winner=polysim_winner(client,pos["condition_id"])
                except Exception:winner=None
                if not winner:still.append(pos);continue
                pnl=(pos["shares"] if winner==pos["side"] else 0)-pos["cost"];session_pnl+=pnl
                journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"settle","slug":pos["slug"],"side":pos["side"],"winner":winner,"pnl":pnl,"session_pnl":session_pnl})
                LOG.info("SETTLE %s winner=%s pnl=%+.2f",pos["slug"],winner,pnl)
            pending=still
        time.sleep(args.poll)
    except KeyboardInterrupt:LOG.info("stopped")
    finally:feed.close();journal.close()

if __name__=="__main__":main()

