import json, glob
s = set()
F = 1793801658
for p in glob.glob('/root/poly/latencylog/bot-*.jsonl'):
    if '/archive' in p:
        continue
    for l in open(p):
        if '"src":"settle"' in l:
            try:
                o = json.loads(l)
                if o.get('ws', 0) >= F:
                    s.add(o['ws'])
            except Exception:
                pass
print('CLEAN_ERA_TRADES', len(s))
