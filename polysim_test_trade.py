"""Place one small FOK paper order on the current BTC 5-minute Polysimulator market."""
import argparse,json,os,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from btc5m_momentum_polysim import API,discover,get_book
from btc5m_momentum_strategy import sweep_asks
from polymarket_btc5m_logger import CoinbaseFeed

ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--amount",type=float,default=5);p.add_argument("--sample-seconds",type=float,default=3);p.add_argument("--side",choices=("auto","Up","Down"),default="auto");a=p.parse_args()
 load_dotenv(ROOT/".env");key=os.getenv("POLYSIM_API_KEY","").strip()
 if not key:raise SystemExit("Set POLYSIM_API_KEY in .env")
 feed=CoinbaseFeed();feed.start()
 try:
  deadline=time.time()+10
  while feed.snapshot().get("price")is None and time.time()<deadline:time.sleep(.1)
  first=feed.snapshot().get("price")
  if first is None:raise SystemExit("No Coinbase price received")
  time.sleep(a.sample_seconds);last=feed.snapshot().get("price")
  side=("Up"if last>=first else"Down") if a.side=="auto" else a.side
  start=int(time.time())//300*300
  with httpx.Client(timeout=10,headers={"X-API-Key":key})as client:
   market=discover(client,start)
   if not market:raise SystemExit("Current Polysimulator BTC 5m market not found")
   book=get_book(client,market["tokens"][side]);fill=sweep_asks(book,a.amount)
   if not fill:
    fallback="Down"if side=="Up"else"Up"
    book=get_book(client,market["tokens"][fallback]);fill=sweep_asks(book,a.amount)
    if not fill:raise SystemExit(f"Insufficient ask depth on both outcomes for ${a.amount:g}")
    side=fallback
   body={"market_id":market["condition_id"],"side":"BUY","outcome":side,"amount":str(a.amount),
         "price":f"{fill['worst_price']:.2f}","order_type":"market","time_in_force":"FOK"}
   response=client.post(API+"/orders",json=body,headers={"Idempotency-Key":str(uuid.uuid4())})
   try:data=response.json()
   except ValueError:data={"error":response.text[:500]}
   record={"ts":datetime.now(timezone.utc).isoformat(),"event":"test_order","slug":market["slug"],
           "side":side,"coinbase_first":first,"coinbase_last":last,"amount":a.amount,
           "limit":fill["worst_price"],"http_status":response.status_code,"response":data}
   out=ROOT/"price_logs"/"polysim-test-orders.jsonl";out.parent.mkdir(parents=True,exist_ok=True)
   with out.open("a",encoding="utf-8")as f:f.write(json.dumps(record)+"\n")
   print(json.dumps(record,indent=2))
 finally:feed.close()
if __name__=="__main__":main()
