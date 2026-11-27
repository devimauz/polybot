#!/usr/bin/env python3
import json,glob,os
LOGDIR='/root/poly/latencylog'; now=1794267623; cut=now-86400
sett={}
for fp in glob.glob(LOGDIR+'/bot-*.jsonl'):
    w=int(os.path.basename(fp)[4:-6])
    if w<cut or w<1793801658: continue
    for line in open(fp):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        if r.get('src')=='settle' and isinstance(r.get('pnl'),(int,float)):
            ww=r.get('ws',w)
            if ww>=cut: sett[ww]=r['pnl']
pn=list(sett.values()); n=len(pn); m=sum(pn)/n
var=sum((p-m)**2 for p in pn)/(n-1); std=var**0.5
se=std/(n**0.5)
print(f'n={n} mean={m:.3f} std={std:.2f} se={se:.3f}')
print(f'z_vs0.70={(m-0.70)/se:.2f} z_vs0.77={(m-0.77)/se:.2f}')
print(f'p95_lo={m-1.96*se:.2f} p95_hi={m+1.96*se:.2f}')