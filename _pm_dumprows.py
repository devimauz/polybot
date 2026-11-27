#!/usr/bin/env python3
"""Dump per-window (ws, settle_pnl, order_entry_ask) for last-24h for local analysis."""
import json, glob, os, sys

LOGDIR = "/root/poly/latencylog"
now = int(sys.argv[1])
start_cut = now - 86400
clean_start = 1793801658

orders = {}  # ws -> entry ask (from paper_order, prefer last)
settles = {} # ws -> pnl
for fp in glob.glob(os.path.join(LOGDIR, "bot-*.jsonl")):
    ws = int(os.path.basename(fp)[4:-6])
    if ws < start_cut or ws < clean_start:
        continue
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            src = r.get("src")
            w = r.get("ws", ws)
            if w < start_cut:
                continue
            if src == "paper_order":
                a = r.get("ask")
                if isinstance(a, (int, float)):
                    orders[w] = a
            elif src == "settle":
                p = r.get("pnl")
                if isinstance(p, (int, float)):
                    settles[w] = p

allws = sorted(set(orders) | set(settles))
out = []
for w in allws:
    out.append((w, settles.get(w), orders.get(w), round((settles.get(w) or 0)/5*100,1) if settles.get(w) else None))
print("WS,PNL,ENTRY_ASK")
for w,pn,ep,pct in out:
    print(w, pn, ep, sep=",")
import sys as _s
print("TOTAL_WITH_EP", sum(1 for _ in out if _[2] is not None), file=_s.stderr)
print("TOTAL_WITH_PNL", sum(1 for _ in out if _[1] is not None), file=_s.stderr)