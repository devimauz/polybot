import bisect,json,random
from pathlib import Path
from btc5m_momentum_strategy import fee_for
R=Path(__file__).parent/'price_logs'/'polymarket';W=[]
for p in sorted(R.glob('btc-updown-5m-*.jsonl')):
 a=[];win=None
 for l in p.open(encoding='utf8'):
  try:r=json.loads(l)
  except:continue
  if r.get('event')=='resolution':win=r.get('winner');continue
  if r.get('event')!='snapshot':continue
  t=r.get('seconds_into_window');c=(r.get('coinbase')or{}).get('price');u=r.get('Up')or{};d=r.get('Down')or{}
  if t is not None and c is not None:a.append((float(t),float(c),u.get('bid'),u.get('ask'),u.get('ask_size'),d.get('bid'),d.get('ask'),d.get('ask_size')))
 if win in('Up','Down')and a and a[0][0]<=5:W.append((p.stem,win,a))
def sim(data,cfg,seed=42):
 ent,om,look,bd,rcb,amax=cfg;out=[]
 for slug,win,a in data:
  ts=[x[0]for x in a];i=bisect.bisect_left(ts,ent)
  if i>=len(a):continue
  r=a[i];oi=bisect.bisect_left(ts,r[0]-look,0,i+1);old=a[min(oi,i)];move=(r[1]/a[0][1]-1)*10000;recent=(r[1]/old[1]-1)*10000
  if abs(move)<om:continue
  side='Up'if move>0 else'Down';bi,ai,si=(2,3,4)if side=='Up'else(5,6,7)
  if r[bi] is None or old[bi] is None or r[bi]-old[bi]<bd:continue
  if (side=='Up'and recent<rcb)or(side=='Down'and recent>-rcb):continue
  ask,size=r[ai],r[si]
  if ask is None or not .45<=ask<=amax or size is None or ask*size<2:continue
  rng=random.Random(f'{seed}:{slug}');j=bisect.bisect_left(ts,r[0]+rng.uniform(.3,.8),i+1)
  if j>=len(a):continue
  da,ds=a[j][ai],a[j][si]
  if da is None or da>ask or ds is None or da*ds<2:continue
  sh=2/da;cost=2+fee_for(da,sh);pnl=(sh if side==win else 0)-cost
  out.append(pnl)
 return sum(out),out
cut=int(len(W)*.6);tr=W[:cut];te=W[cut:];chunks=[tr[i*len(tr)//3:(i+1)*len(tr)//3]for i in range(3)];C=[]
for ent in(90,120,150,180):
 for om in(2,4,6):
  for look in(3,5,10,20):
   for bd in(0,.01,.02):
    for cb in(0,1,2):
     for am in(.8,.85,.88):
      cfg=(ent,om,look,bd,cb,am);rr=[sim(x,cfg)for x in chunks];ps=[x[0]for x in rr];n=sum(len(x[1])for x in rr)
      if n>=12:C.append((min(ps),sum(ps),n,cfg,ps))
C.sort(reverse=True)
for _,tot,n,cfg,ps in C[:20]:
 p,x=sim(te,cfg);print(round(tot,2),n,[round(q,2)for q in ps],'TEST',round(p,2),len(x),cfg)
import statistics
chosen=(150,2,3,0,0,.8)
for name,data in(('train',tr),('test',te),('all',W)):
 vals=[];ns=[]
 for seed in range(100):p,x=sim(data,chosen,seed);vals.append(p);ns.append(len(x))
 print('STRESS',name,round(statistics.median(vals),2),round(min(vals),2),round(max(vals),2),round(statistics.mean(ns),1))
