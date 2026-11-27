"""Direct-Polymarket BTC 5m value-capped momentum bot.

Dry-run unless --live is supplied. Defaults mirror the frozen backtest rule.
"""
from __future__ import annotations
import argparse,json,logging,math,os,time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from btc5m_momentum_strategy import (DEFAULT_ASK_MAX,DEFAULT_ASK_MIN,
 DEFAULT_ENTRY_SECOND,DEFAULT_FOK_ALLOWANCE,DEFAULT_MINIMUM_MOVE_BPS,DEFAULT_STAKE,
 fee_for,momentum_guard,momentum_signal,sweep_asks)
from polymarket_btc5m_logger import CoinbaseFeed,decode_list

ROOT=Path(__file__).resolve().parent;GAMMA="https://gamma-api.polymarket.com/markets";CLOB="https://clob.polymarket.com"
LOG=logging.getLogger("btc5m-momentum-polymarket")

class Journal:
 def __init__(self,path):
  path.parent.mkdir(parents=True,exist_ok=True);self.path=path;self.stream=path.open("a",encoding="utf-8",buffering=1);self.attempted=set()
  try:
   for line in path.read_text(encoding="utf-8").splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if r.get("event")in("order_attempt","order_unknown","fill","dry_run"):self.attempted.add(r.get("slug"))
  except OSError:pass
 def write(self,row):self.stream.write(json.dumps(row,separators=(",",":"))+"\n")
 def close(self):self.stream.close()

def discover(client,start):
 slug=f"btc-updown-5m-{start}";r=client.get(GAMMA,params={"slug":slug});r.raise_for_status()
 for m in r.json():
  if m.get("slug")!=slug or m.get("closed")or not m.get("active")or not m.get("acceptingOrders"):continue
  outcomes,tokens=decode_list(m.get("outcomes")),decode_list(m.get("clobTokenIds"))
  if len(outcomes)==len(tokens)==2 and set(outcomes)=={"Up","Down"}:
   return {"slug":slug,"condition_id":m.get("conditionId"),"tokens":dict(zip(outcomes,map(str,tokens))),"neg_risk":bool(m.get("negRisk",False))}
 return None

def book(client,token,timeout=None):
 kwargs={"params":{"token_id":token}}
 if timeout is not None:kwargs["timeout"]=timeout
 r=client.get(CLOB+"/book",**kwargs);r.raise_for_status();raw=r.json()
 asks=[{"price":float(x["price"]),"size":float(x["size"])}for x in raw.get("asks",[])if float(x["size"])>0]
 bids=[{"price":float(x["price"]),"size":float(x["size"])}for x in raw.get("bids",[])if float(x["size"])>0]
 return {"asks":asks,"bids":bids,"ask":min((x["price"]for x in asks),default=None),
         "bid":max((x["price"]for x in bids),default=None),"tick":str(raw.get("tick_size")or"0.01")}

def at_or_before(history,target):
 for stamp,value in reversed(history):
  if stamp<=target:return value
 return None

def is_confirmed_fok_kill(value):
 """True only when Polymarket explicitly says an FOK was killed unfilled."""
 payload=value
 if isinstance(value,Exception):
  try:
   from py_clob_client_v2.exceptions import PolyApiException
  except ImportError:return False
  if not isinstance(value,PolyApiException)or value.status_code!=400:return False
  payload=value.error_msg
 if not isinstance(payload,dict)or payload.get("success")is True or payload.get("status")=="matched":return False
 message=payload.get("error")or payload.get("errorMsg")or""
 expected="order couldn't be fully filled. fok orders are fully filled or killed."
 return isinstance(message,str)and message.strip().lower()==expected

def fill_within_limit(order_book,stake,limit):
 """Return an executable recorded fill without allowing price-cap fallback."""
 limited={"asks":[level for level in(order_book.get("asks")or[])
                  if float(level["price"])<=limit+1e-12]}
 return sweep_asks(limited,stake)if limited["asks"]else None

def actual_fill_from_response(response,submit_to_response_ms):
 """Build actual BUY accounting from a matched CLOB response, or return None."""
 if not isinstance(response,dict):return None
 try:
  paid=float(response["makingAmount"]);shares=float(response["takingAmount"])
  latency_ms=float(submit_to_response_ms)
 except(TypeError,ValueError,KeyError):return None
 if(not all(math.isfinite(value)for value in(paid,shares,latency_ms))
    or paid<=0 or shares<=0 or latency_ms<0):return None
 effective_price=paid/shares
 if not 0<effective_price<1:return None
 fee=fee_for(effective_price,shares)
 return {"paid":paid,"shares":shares,"effective_price":effective_price,
         "fee":fee,"all_in_cost":paid+fee,
         "submit_to_response_ms":latency_ms}

def live_client(http):
 blocked=http.get("https://polymarket.com/api/geoblock");blocked.raise_for_status()
 if blocked.json().get("blocked")is not False:raise RuntimeError("Polymarket trading is unavailable from this location")
 from py_clob_client_v2 import ClobClient
 key=os.getenv("POLY_PRIVATE_KEY","").strip();signature=int(os.getenv("POLY_SIGNATURE_TYPE","0"));funder=os.getenv("POLY_FUNDER","").strip()or None
 if not key:raise ValueError("Set POLY_PRIVATE_KEY in .env before --live")
 if signature not in(0,1,2,3)or(signature!=0 and not funder):raise ValueError("Set valid POLY_SIGNATURE_TYPE and POLY_FUNDER")
 result=ClobClient(CLOB,chain_id=137,key=key,signature_type=signature,funder=funder,retry_on_error=False)
 result.set_api_creds(result.create_or_derive_api_key());return result

def build_parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--live",action="store_true",help="submit a real-money Polymarket FOK order")
 p.add_argument("--stake",type=float,default=DEFAULT_STAKE,
                help="fixed USDC amount per trade in dry-run and live mode (default: 2)")
 p.add_argument("--entry-second",type=float,default=DEFAULT_ENTRY_SECOND)
 p.add_argument("--minimum-move-bps",type=float,default=DEFAULT_MINIMUM_MOVE_BPS);p.add_argument("--ask-min",type=float,default=DEFAULT_ASK_MIN);p.add_argument("--ask-max",type=float,default=DEFAULT_ASK_MAX)
 p.add_argument("--trend-guard",action="store_true",
                help="experimental: require confirmation; differs from the tested default")
 p.add_argument("--slippage",type=float,default=DEFAULT_FOK_ALLOWANCE,
                help="extra dollars/share added to the FOK buy limit (default: 0.05)")
 p.add_argument("--fok-retry-seconds",type=float,default=0,
                help="opt in to retries after a confirmed FOK kill (default: 0)")
 p.add_argument("--fok-max-attempts",type=int,default=1,
                help="maximum total FOK submissions per signal (default: 1)")
 p.add_argument("--poll",type=float,default=.1);p.add_argument("--once",action="store_true")
 return p

def main():
 load_dotenv(ROOT/".env");a=build_parser().parse_args()
 numeric=(a.stake,a.entry_second,a.minimum_move_bps,
          a.ask_min,a.ask_max,a.slippage,a.fok_retry_seconds,a.poll)
 if(not all(math.isfinite(x)for x in numeric)or a.stake<=0
    or not 0<=a.entry_second<300 or a.minimum_move_bps<0
    or not 0<a.ask_min<=a.ask_max<1 or a.poll<=0
    or not 0<=a.slippage<1 or a.fok_retry_seconds<0
    or a.fok_max_attempts<1):raise SystemExit("invalid settings")
 logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
 LOG.info("strategy=value-capped-momentum-v1 stake=$%.2f entry=%.1fs move>=%.2fbps ask=%.2f-%.2f FOK allowance=$%.2f attempts=%d retry_window=%.1fs",
          a.stake,a.entry_second,a.minimum_move_bps,a.ask_min,a.ask_max,
          a.slippage,a.fok_max_attempts,a.fok_retry_seconds)
 if a.trend_guard:LOG.warning("trend guard enabled: this differs from the tested strategy")
 journal_name="live.jsonl"if a.live else("guarded-dry-run.jsonl"if a.trend_guard else"dry-run.jsonl")
 journal=Journal(ROOT/"price_logs"/"momentum-polymarket"/journal_name);feed=CoinbaseFeed();feed.start();state={}
 try:
  with httpx.Client(timeout=8,headers={"User-Agent":"btc5m-momentum-polymarket/1.0"})as http:
   trader=live_client(http)if a.live else None
   while True:
    now=time.time();start=int(now)//300*300;t=now-start;cb=feed.snapshot(now)
    if state.get("start")!=start:
     state={"start":start,"market":None,"open":None,"evaluated":False,"last_discovery":0,
            "last_guard_book":0,"cb_history":deque(maxlen=400),
            "guard_bids":{"Up":deque(maxlen=100),"Down":deque(maxlen=100)},
            "window_stake":a.stake};LOG.info("new window btc-updown-5m-%d fixed stake=$%.2f",start,a.stake)
    if cb.get("price"):state["cb_history"].append((now,cb["price"]))
    if state["open"]is None and t<=5 and cb.get("price"):state["open"]=cb["price"]
    if state["market"]is None and now-state["last_discovery"]>=1:
     state["last_discovery"]=now
     try:state["market"]=discover(http,start)
     except httpx.HTTPError as exc:LOG.warning("market discovery: %s",type(exc).__name__)
    if(a.trend_guard and state["market"] and a.entry_second-21<=t<a.entry_second
       and now-state["last_guard_book"]>=.4):
     state["last_guard_book"]=now
     for guard_side in("Up","Down"):
      try:
       guard_book=book(http,state["market"]["tokens"][guard_side])
       state["guard_bids"][guard_side].append((now,guard_book.get("bid")))
      except(httpx.HTTPError,ValueError,KeyError):pass
    if a.once:
     LOG.info("diagnostic market=%s cb=%s open=%s elapsed=%.1f live=%s",bool(state["market"]),cb.get("price"),state["open"],t,a.live);break
    if t>=a.entry_second and not state["evaluated"]:
     state["evaluated"]=True;m=state["market"];slug=f"btc-updown-5m-{start}";reason=None;signal=None;response=None;actual_fill=None;guard=None;order_attempts=0;move=None;side=None;q=None
     try:
      if slug in journal.attempted:reason="already_attempted"
      elif not m or not state["open"]or not cb.get("price"):reason="missing_market_or_window_open"
      else:
       trade_stake=a.stake
       move=(cb["price"]/state["open"]-1)*10000;side="Up"if move>0 else"Down";q=book(http,m["tokens"][side])
       if a.trend_guard:
        state["guard_bids"][side].append((now,q.get("bid")))
        guard=momentum_guard(side,cb["price"],at_or_before(state["cb_history"],now-10),
                             q.get("bid"),at_or_before(state["guard_bids"][side],now-20))
        if guard is None:reason="missing_trend_guard_history"
        elif not guard["passed"]:reason="trend_guard_reversal"
       if reason is None:
        signal=momentum_signal(state["open"],cb["price"],{side:q},trade_stake,a.minimum_move_bps,a.ask_min,a.ask_max)
        if not signal:
         best_ask=q.get("ask")
         if abs(move)<a.minimum_move_bps:reason="move_below_minimum"
         elif best_ask is None:reason="missing_ask"
         elif not a.ask_min<=best_ask<=a.ask_max:reason="ask_outside_value_cap"
         else:reason="insufficient_depth"
        else:
         tick=float(q["tick"]);limit=min(1-tick,math.ceil((signal["worst_price"]+a.slippage-1e-12)/tick)*tick)
       if signal and not a.live:
        journal.attempted.add(slug);reason="dry_run"
       elif signal:
        journal.attempted.add(slug)
        from py_clob_client_v2 import MarketOrderArgs,OrderType,PartialCreateOrderOptions,Side as OrderSide
        original_tick=q["tick"]
        retry_budget=min(a.fok_retry_seconds,max(0,start+300-time.time()))
        retry_deadline=time.monotonic()+retry_budget
        while order_attempts<a.fok_max_attempts:
         order_attempts+=1
         journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"order_attempt",
                        "slug":slug,"side":side,"stake":trade_stake,
                        "book_limit":signal["worst_price"],"slippage":a.slippage,
                        "limit":limit,"attempt":order_attempts})
         try:
          signed=trader.create_market_order(MarketOrderArgs(token_id=m["tokens"][side],amount=trade_stake,side=OrderSide.BUY,price=limit,order_type=OrderType.FOK),PartialCreateOrderOptions(tick_size=original_tick,neg_risk=m["neg_risk"]))
         except Exception as exc:
          reason=f"retry_sign_error: {type(exc).__name__}: {str(exc)[:200]}";signal=None;break
         submitted_at=time.monotonic()
         try:response=trader.post_order(signed,OrderType.FOK)
         except Exception as exc:
          if not is_confirmed_fok_kill(exc):raise
          detail=getattr(exc,"error_msg",str(exc));response=None
          journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"fok_killed",
                         "slug":slug,"side":side,"attempt":order_attempts,
                         "limit":limit,"detail":detail})
         else:
          if isinstance(response,dict)and response.get("success")and response.get("status")=="matched":
           response_ms=(time.monotonic()-submitted_at)*1000
           actual_fill=actual_fill_from_response(response,response_ms)
           if actual_fill:
            LOG.info("actual fill %s %s paid=$%.6f shares=%.6f price=$%.6f fee=$%.5f all-in=$%.5f response=%.1fms",
                     slug,side,actual_fill["paid"],actual_fill["shares"],
                     actual_fill["effective_price"],actual_fill["fee"],
                     actual_fill["all_in_cost"],actual_fill["submit_to_response_ms"])
           else:LOG.warning("matched response missing valid fill accounting for %s %s",slug,side)
           break
          if not is_confirmed_fok_kill(response):raise RuntimeError(f"ambiguous order response: {str(response)[:250]}")
          journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"fok_killed",
                         "slug":slug,"side":side,"attempt":order_attempts,
                         "limit":limit,"detail":response});response=None
         if order_attempts>=a.fok_max_attempts or time.monotonic()>=retry_deadline:
          reason="fok_retry_exhausted";signal=None;break
         retry_ready=False
         while time.monotonic()<retry_deadline:
          time.sleep(min(.2,max(0,retry_deadline-time.monotonic())))
          if time.monotonic()>=retry_deadline:break
          retry_now=time.time();retry_cb=feed.snapshot(retry_now);retry_price=retry_cb.get("price")
          if not retry_price:
           reason="retry_missing_coinbase";signal=None;break
          retry_move=(retry_price/state["open"]-1)*10000
          if(side=="Up"and retry_move<a.minimum_move_bps)or(side=="Down"and retry_move>-a.minimum_move_bps):
           reason="retry_signal_reversed";signal=None;break
          try:
           remaining=retry_deadline-time.monotonic()
           retry_book=book(http,m["tokens"][side],timeout=max(.05,min(8,remaining)))
          except Exception as exc:
           reason=f"retry_book_error: {type(exc).__name__}";signal=None;break
          if time.monotonic()>=retry_deadline:break
          if retry_book["tick"]!=original_tick:
           reason="retry_tick_changed";signal=None;break
          retry_fill=fill_within_limit(retry_book,trade_stake,limit)
          if retry_fill:
           q=retry_book;signal={**signal,**retry_fill,"best_ask":retry_book.get("ask")};retry_ready=True;break
         if not retry_ready:
          if signal is not None:reason="fok_retry_expired";signal=None
          break
     except Exception as exc:
      reason=f"{type(exc).__name__}: {str(exc)[:300]}";journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":"order_unknown"if a.live else"error","slug":slug,"detail":reason});signal=None
     event="fill"if signal and a.live else("dry_run"if signal else"skip")
     journal.write({"ts":datetime.now(timezone.utc).isoformat(),"event":event,"slug":slug,
                    "reason":reason,"guard":guard,"signal":signal,"response":response,
                    "actual_fill":actual_fill,
                    "decision":{"move_bps":move,"side":side,
                                "best_ask":q.get("ask")if q else None},
                    "order_attempts":order_attempts,"stake":a.stake,
                    "window_stake":state.get("window_stake")})
     LOG.info("%s %s",event.upper(),reason or signal.get("side"))
    time.sleep(a.poll)
 except KeyboardInterrupt:LOG.info("stopped")
 finally:feed.close();journal.close()
if __name__=="__main__":main()
