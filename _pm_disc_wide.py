#!/usr/bin/env python3
"""_pm_disc_wide.py — WIDE automated sweep of orthogonal feature families.

Phase 1 (design slice): ~40 features x 4 horizons x 4 bands -> tercile analysis.
  A feature "shows" if the extreme terciles differ in win-rate by >=5pp with
  n>=30 each side, OR if a top-quintile cell has mean >= +$0.5 with n>=20.
Phase 2: survivors -> both-halves + placebo -> frozen set.
Phase 3: ONE holdout shot on the frozen set.

Features are orthogonal: price, book, tape, time, path, volatility, liquidity,
cross-window, horizon-dynamics. No re-run of falsified families.
"""
import os, pickle, math, datetime, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_pm_cache")
FEE = 0.07
STAKE = 5.0
HORIZONS = [60, 120, 180, 240]
BANDS = [(0.10,0.30),(0.30,0.50),(0.50,0.70),(0.70,0.90)]

def load_all():
    man = pickle.load(open(os.path.join(CACHE, "manifest.pkl"), "rb"))
    out = []
    for m in man:
        if not m["_winner"]: continue
        z = np.load(os.path.join(CACHE, m["_slug"] + ".npz"), allow_pickle=True)
        d = {k: z[k] for k in z.files if k != "_winner"}
        d["_winner"] = m["_winner"]; d["_slug"] = m["_slug"]
        d["ws"] = int(d["ws"][0]); d["t"] = d["t_src"] if "t_src" in d else d["t"]
        out.append(d)
    out.sort(key=lambda d: d["ws"])
    return out

def load_trades():
    z = np.load(os.path.join(CACHE, "trades.npz"), allow_pickle=True)
    slugs=list(z["slugs"]); starts=z["starts"]; lens=z["lens"]; flat=z["flat"]
    TR={}
    for k,s0,ln in zip(slugs,starts,lens):
        a=flat[:,s0:s0+ln]; o=np.argsort(a[0],kind="stable"); TR[str(k)]=a[:,o]
    return TR

def fee(p, sh): return sh * FEE * p * (1.0 - p)

def frame_idx(d, T):
    i = int(np.searchsorted(d["t"], T, side="right")) - 1
    return i if i >= 0 else None

def safe(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default

# ---------------- feature extractors (all return float or None) ----------------
def f_cb_return_30(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-30)&(t<=T)&np.isfinite(cb)
    if m.sum()<2: return None
    v=cb[m]; return safe((v[-1]-v[0])/(v[0]+1e-9)*100)

def f_cb_return_60(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-60)&(t<=T)&np.isfinite(cb)
    if m.sum()<2: return None
    v=cb[m]; return safe((v[-1]-v[0])/(v[0]+1e-9)*100)

def f_cb_vol_30(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-30)&(t<=T)&np.isfinite(cb)
    if m.sum()<3: return None
    return safe(np.std(np.diff(cb[m])))

def f_cb_vol_win(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<10: return None
    return safe(np.std(np.diff(cb[m])))

def f_up_mid(d, T):
    i=frame_idx(d,T)
    return safe(d["Up_mid"][i]) if i is not None else None

def f_up_spread(d, T):
    i=frame_idx(d,T)
    return safe(d["Up_spread"][i]) if i is not None else None

def f_dn_spread(d, T):
    i=frame_idx(d,T)
    return safe(d["Down_spread"][i]) if i is not None else None

def f_spread_sum(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    return safe(d["Up_spread"][i]+d["Down_spread"][i])

def f_overround(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    return safe(d["Up_ask"][i]+d["Down_ask"][i])

def f_up_bidsz(d, T):
    i=frame_idx(d,T)
    return safe(d["Up_bidsz"][i]) if i is not None else None

def f_up_asksz(d, T):
    i=frame_idx(d,T)
    return safe(d["Up_asksz"][i]) if i is not None else None

def f_dn_bidsz(d, T):
    i=frame_idx(d,T)
    return safe(d["Down_bidsz"][i]) if i is not None else None

def f_dn_asksz(d, T):
    i=frame_idx(d,T)
    return safe(d["Down_asksz"][i]) if i is not None else None

def f_up_liq(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    return safe(d["Up_bidsz"][i]+d["Up_asksz"][i] if np.isfinite(d["Up_asksz"][i]) else d["Up_bidsz"][i])

def f_up_asklad2(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    return safe(np.nansum(d["Up_asks_sz"][i][:3]))

def f_up_bidadlad2(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    return safe(np.nansum(d["Up_bids_sz"][i][:3]))

def f_liq_ratio(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    ub=d["Up_bidsz"][i]; nl=d["Down_asksz"][i]
    if not(np.isfinite(ub) and np.isfinite(nl)) or ub+nl<=0: return None
    return safe((ub-nl)/(ub+nl))

def f_top_imb(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    ub=d["Up_bidsz"][i]; db=d["Down_bidsz"][i]
    if not np.isfinite(ub) or not np.isfinite(db) or ub+db<=0: return None
    return safe((ub-db)/(ub+db))

def f_dist_from_open(d, T):
    cb=d["cb_px"]; m=np.isfinite(cb)
    if m.sum()<2: return None
    return safe((cb[m][-1]-cb[m][0])/(cb[m][0]+1e-9)*100)

def f_dist_from_mean(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; sd=np.std(v)
    return safe((v[-1]-np.mean(v))/(sd+1e-9))

def f_uptime_frac(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; o=v[0]
    return safe(np.mean(v>o))

def f_trend_eff(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]
    path=np.sum(np.abs(np.diff(v))); net=abs(v[-1]-v[0])
    return safe(net/(path+1e-9))

def f_cb_range_pos(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; lo=np.min(v); hi=np.max(v)
    if hi-lo<1e-9: return np.nan
    return safe((v[-1]-lo)/(hi-lo))

def f_last_20_slope(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-20)&(t<=T)&np.isfinite(cb)
    if m.sum()<3: return None
    v=cb[m]; x=t[m]-t[m][0]
    if len(x)<2: return None
    return safe(np.polyfit(x,v,1)[0])

def f_ask_slope_up(d, T):
    a=d["Up_ask"]; t=d["t"]
    m=(t>=T-60)&(t<=T)&np.isfinite(a)
    if m.sum()<3: return None
    v=a[m]; x=t[m]-t[m][0]
    return safe(np.polyfit(x,v,1)[0] if len(x)>=2 else np.nan)

def f_ask_slope_dn(d, T):
    a=d["Down_ask"]; t=d["t"]
    m=(t>=T-60)&(t<=T)&np.isfinite(a)
    if m.sum()<3: return None
    v=a[m]; x=t[m]-t[m][0]
    return safe(np.polyfit(x,v,1)[0] if len(x)>=2 else np.nan)

def f_mid_chg_up(d, T):
    m=d["Up_mid"]; t=d["t"]; mm=(t>=T-60)&(t<=T)&np.isfinite(m)
    if mm.sum()<2: return None
    return safe(m[mm][-1]-m[mm][0])

def f_mid_chg_dn(d, T):
    m=d["Down_mid"]; t=d["t"]; mm=(t>=T-60)&(t<=T)&np.isfinite(m)
    if mm.sum()<2: return None
    return safe(m[mm][-1]-m[mm][0])

def f_prev_win(d, T, ctx):
    pw=ctx["pmap"].get(d["ws"]-300)
    if pw is None: return None
    return 1.0 if pw=="Up" else 0.0

def f_prev_vol(d, T, ctx):
    pv=ctx["vmap"].get(d["ws"]-300)
    return safe(pv) if pv is not None else None

def f_run_len(d, T, ctx):
    return safe(float(ctx["runmap"].get(d["ws"], 0)))

def f_minute_of_day(d, T, ctx):
    return safe(float(datetime.datetime.utcfromtimestamp(d["ws"]).minute))

def f_hour_of_day(d, T, ctx):
    return safe(float(datetime.datetime.utcfromtimestamp(d["ws"]).hour))

def f_trade_count(d, T, ctx):
    a=ctx["TR"].get(d["_slug"])
    if a is None: return None
    t=a[0]; m=(t>=d["ws"])&(t<d["ws"]+T)
    return safe(float(m.sum()))

def f_trade_imbalance(d, T, ctx):
    a=ctx["TR"].get(d["_slug"])
    if a is None: return None
    t=a[0]; side=a[3]; oc=a[4]; sz=a[2]
    m=(t>=d["ws"])&(t<d["ws"]+T)
    if not m.any(): return None
    side=side[m]; oc=oc[m]; sz=sz[m]
    ub=sz[(oc==0)&(side==0)].sum(); us=sz[(oc==0)&(side==1)].sum()
    db=sz[(oc==1)&(side==0)].sum(); ds=sz[(oc==1)&(side==1)].sum()
    tot=ub+us+db+ds
    if tot<=0: return None
    return safe((ub-us-db+ds)/tot)

def f_trade_burst(d, T, ctx):
    a=ctx["TR"].get(d["_slug"])
    if a is None: return None
    t=a[0]; m=(t>=d["ws"])&(t<d["ws"]+T)
    if m.sum()<10: return None
    tt=np.sort(t[m]); dt=np.diff(tt)
    return safe(1.0/(np.median(dt)+1e-6))  # prints/sec

def f_trade_max_sz(d, T, ctx):
    a=ctx["TR"].get(d["_slug"])
    if a is None: return None
    t=a[0]; sz=a[2]; m=(t>=d["ws"])&(t<d["ws"]+T)
    if not m.any(): return None
    return safe(np.max(sz[m]))

def f_cb_to_strike(d, T, ctx):
    cb=d["cb_px"]; m=np.isfinite(cb)
    if m.sum()<2: return None
    return safe((cb[m][-1]-ctx["strikemap"].get(d["ws"], np.nan)))

def f_up_ask_x_dn_bid(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    ua=d["Up_ask"][i]; db=d["Down_bid"][i]
    if not np.isfinite(ua) or not np.isfinite(db): return None
    return safe(ua+db)  # should be ~1 if no arb

def f_up_bid_x_dn_ask(d, T):
    i=frame_idx(d,T)
    if i is None: return None
    ub=d["Up_bid"][i]; da=d["Down_ask"][i]
    if not np.isfinite(ub) or not np.isfinite(da): return None
    return safe(ub+da)

def f_mid_max_dist(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; mu=np.mean(v)
    return safe(np.max(np.abs(v-mu))/(np.std(v)+1e-9))

def f_time_to_settle(d, T):
    return safe(float(300-T))

def f_cb_accel(d, T):
    cb=d["cb_px"]; t=d["t"]; m=(t>=T-40)&(t<=T)&np.isfinite(cb)
    if m.sum()<5: return None
    v=cb[m]; x=t[m]-t[m][0]
    if len(x)<3: return None
    c=np.polyfit(x,v,2)
    return safe(c[0])

# (name, fn, needs_ctx)
FEATURES = [
    ("cb_ret_30", f_cb_return_30, False),
    ("cb_ret_60", f_cb_return_60, False),
    ("cb_vol_30", f_cb_vol_30, False),
    ("cb_vol_win", f_cb_vol_win, False),
    ("up_mid", f_up_mid, False),
    ("up_spread", f_up_spread, False),
    ("dn_spread", f_dn_spread, False),
    ("spread_sum", f_spread_sum, False),
    ("overround", f_overround, False),
    ("up_bidsz", f_up_bidsz, False),
    ("up_asksz", f_up_asksz, False),
    ("dn_bidsz", f_dn_bidsz, False),
    ("dn_asksz", f_dn_asksz, False),
    ("up_liq", f_up_liq, False),
    ("up_asklad2", f_up_asklad2, False),
    ("up_bidlad2", f_up_bidadlad2, False),
    ("liq_ratio", f_liq_ratio, False),
    ("top_imb", f_top_imb, False),
    ("dist_from_open", f_dist_from_open, False),
    ("dist_from_mean", f_dist_from_mean, False),
    ("uptime_frac", f_uptime_frac, False),
    ("trend_eff", f_trend_eff, False),
    ("range_pos", f_cb_range_pos, False),
    ("last20_slope", f_last_20_slope, False),
    ("ask_slope_up", f_ask_slope_up, False),
    ("ask_slope_dn", f_ask_slope_dn, False),
    ("mid_chg_up", f_mid_chg_up, False),
    ("mid_chg_dn", f_mid_chg_dn, False),
    ("prev_win", f_prev_win, True),
    ("prev_vol", f_prev_vol, True),
    ("run_len", f_run_len, True),
    ("minute_of_day", f_minute_of_day, True),
    ("hour_of_day", f_hour_of_day, True),
    ("trade_count", f_trade_count, True),
    ("trade_imb", f_trade_imbalance, True),
    ("trade_burst", f_trade_burst, True),
    ("trade_max_sz", f_trade_max_sz, True),
    ("cb_to_strike", f_cb_to_strike, True),
    ("upask_plus_dnbid", f_up_ask_x_dn_bid, False),
    ("upbid_plus_dnask", f_up_bid_x_dn_ask, False),
    ("mid_max_dist", f_mid_max_dist, False),
    ("time_to_settle", f_time_to_settle, False),
    ("cb_accel", f_cb_accel, False),
]

def build_ctx(W):
    pmap={d["ws"]:d["_winner"] for d in W}
    def win_vol(d):
        cb=d["cb_px"]; f=np.isfinite(cb)
        if f.sum()<10: return None
        return float(np.abs(cb[f][-1]-cb[f][0]))
    vmap={d["ws"]:win_vol(d) for d in W}
    strikemap={}
    for d in W:
        cb=d["cb_px"]; f=np.where(np.isfinite(cb))[0]
        if len(f): strikemap[d["ws"]]=float(cb[f[0]])
    # run lengths
    seq=sorted([(d["ws"],d["_winner"]) for d in W])
    run={}; cnt=0; cur=None
    for ws,w in seq:
        if w==cur: cnt+=1
        else: cur=w; cnt=1
        run[ws]=cnt
    TR=load_trades()
    return dict(pmap=pmap, vmap=vmap, strikemap=strikemap, runmap=run, TR=TR)

def trade_for(d, T, band):
    t=d["t"]; i0=int(np.searchsorted(t,T,side="right"))-1
    if i0<0: return None
    for i in range(i0,len(t)):
        for side in ("Up","Down"):
            ask=d[f"{side}_ask"][i]; asz=d[f"{side}_asksz"][i]
            if not np.isfinite(ask) or not (band[0]<=ask<=band[1]): continue
            sh=int(STAKE/ask)
            if sh<5 or not np.isfinite(asz) or asz<sh: continue
            won=(d["_winner"]==side)
            pnl=sh*(1.0 if won else 0.0)-sh*ask-fee(ask,sh)
            return dict(ws=d["ws"],side=side,ask=ask,won=won,pnl=pnl,sh=sh)
    return None

def stats(rows):
    if not rows: return None
    pnl=np.array([r["pnl"] for r in rows])
    return dict(n=len(rows), wr=float(np.mean([r["won"] for r in rows])),
                ask=float(np.mean([r["ask"] for r in rows])),
                be=float(np.mean([r["ask"]*(1+FEE*(1-r["ask"])) for r in rows])),
                net=float(pnl.sum()), mean=float(pnl.mean()), med=float(np.median(pnl)))

def cluster_t(rows):
    if not rows: return float('nan')
    pnl=np.array([r["pnl"] for r in rows]); ws=np.array([r["ws"] for r in rows])
    uq=np.unique(ws); s=np.array([pnl[ws==w].sum() for w in uq])
    if len(s)<2: return float('nan')
    m=s.mean(); se=s.std(ddof=1)/math.sqrt(len(s))
    return m/se if se>0 else float('nan')

def placebo(rows, draws=200, seed=11):
    rng=np.random.default_rng(seed)
    real=np.array([r["pnl"] for r in rows]).sum()
    edges=[0.10,0.30,0.50,0.70,0.90]
    bi=np.digitize([r["ask"] for r in rows],edges)-1
    wins=np.array([r["won"] for r in rows],dtype=float)
    sh=np.array([r["sh"] for r in rows],dtype=float)
    ask=np.array([r["ask"] for r in rows],dtype=float)
    ge=0
    for _ in range(draws):
        w=wins.copy()
        for b in np.unique(bi):
            m=bi==b; w[m]=rng.permutation(wins[m])
        pnl=sh*w - sh*ask - sh*FEE*ask*(1-ask)
        if pnl.sum()>=real: ge+=1
    return ge/draws

def main():
    t0=time.time()
    W=load_all(); n=len(W); split=int(n*0.70)
    design, holdout = W[:split], W[split:]
    print(f"Loaded {n} | design={len(design)} holdout={len(holdout)}")
    ctx=build_ctx(W)
    print(f"ctx built in {time.time()-t0:.1f}s, {len(FEATURES)} features, {len(HORIZONS)} horizons, {len(BANDS)} bands")

    # Phase 1: design tercile sweep
    survivors=[]
    for fi,(fname,fn,needs_ctx) in enumerate(FEATURES):
        for T in HORIZONS:
            for blo,bhi in BANDS:
                rows=[]
                for d in design:
                    tr=trade_for(d,T,(blo,bhi))
                    if tr is None: continue
                    val=fn(d,T,ctx) if needs_ctx else fn(d,T)
                    if val is None or not np.isfinite(val): continue
                    r=dict(tr); r["feat"]=val; rows.append(r)
                if len(rows)<40: continue
                vals=np.array([r["feat"] for r in rows])
                q1,q2=np.quantile(vals,[0.2,0.8])
                lo=[r for r in rows if r["feat"]<=q1]
                hi=[r for r in rows if r["feat"]>=q2]
                if len(lo)<15 or len(hi)<15: continue
                slo=stats(lo); shi=stats(hi)
                if slo is None or shi is None: continue
                wr_gap=abs(slo["wr"]-shi["wr"])
                # survivor conditions
                conds=[]
                if wr_gap>=0.08 and min(slo["n"],shi["n"])>=15: conds.append("wrgap")
                if shi["mean"]>=0.8 and shi["n"]>=15: conds.append("him")
                if slo["mean"]>=0.8 and slo["n"]>=15: conds.append("lom")
                if conds:
                    survivors.append(dict(f=fname,T=T,band=(blo,bhi),
                        q1=float(q1),q2=float(q2),
                        lo=slo,hi=shi,
                        lo_n=slo["n"],hi_n=shi["n"],
                        lo_mean=slo["mean"],hi_mean=shi["mean"],
                        lo_wr=slo["wr"],hi_wr=shi["wr"],
                        conds=conds,rows_lo=lo,rows_hi=hi))
    print(f"\n=== PHASE 1: {len(survivors)} design survivors (of {len(FEATURES)}x{len(HORIZONS)}x{len(BANDS)}) ===")
    for s in sorted(survivors, key=lambda x: -max(abs(x["lo_mean"]),abs(x["hi_mean"])))[:30]:
        print(f"  {s['f']:16s} T={s['T']:3d} band={s['band']} "
              f"lo(mean={s['lo_mean']:+.2f},wr={s['lo_wr']:.2f},n={s['lo_n']}) "
              f"hi(mean={s['hi_mean']:+.2f},wr={s['hi_wr']:.2f},n={s['hi_n']}) {s['conds']}")

    # Phase 2: filter survivors on both-halves + placebo
    final=[]
    for s in survivors:
        for arm,rows in [("lo",s["rows_lo"]),("hi",s["rows_hi"])]:
            rows=sorted(rows,key=lambda r:r["ws"]); m=len(rows)//2
            h1=stats(rows[:m]); h2=stats(rows[m:])
            if h1 is None or h2 is None: continue
            if not (h1["mean"]>0 and h2["mean"]>0): continue
            p=placebo(rows)
            if p>=0.3: continue
            st=stats(rows)
            final.append(dict(f=s["f"],T=s["T"],band=s["band"],arm=arm,
                cut=(s["q1"] if arm=="lo" else s["q2"]),
                n=st["n"],mean=st["mean"],wr=st["wr"],t=cluster_t(rows),
                h1=h1["mean"],h2=h2["mean"],placebo=p,med=st["med"],
                rows=rows))
    print(f"\n=== PHASE 2: {len(final)} survivors after both-halves + placebo ===")
    for s in sorted(final, key=lambda x: -x["mean"])[:20]:
        print(f"  {s['f']:16s} T={s['T']:3d} {s['band']} arm={s['arm']} cut={s['cut']:+.2f} "
              f"n={s['n']} mean={s['mean']:+.2f} wr={s['wr']:.2f} t={s['t']:+.2f} "
              f"halves={s['h1']:+.2f}/{s['h2']:+.2f} plac={s['placebo']:.2f} med={s['med']:+.2f}")

    # Phase 3: one holdout shot on top survivors
    print(f"\n=== PHASE 3: HOLDOUT (one shot) on phase-2 survivors ===")
    for s in sorted(final, key=lambda x: -x["mean"])[:15]:
        # rebuild on holdout
        hrows=[]
        for d in holdout:
            tr=trade_for(d,s["T"],s["band"])
            if tr is None: continue
            fn=dict((f[0],f) for f in FEATURES)[s["f"]]
            val=fn[1](d,s["T"],ctx) if fn[2] else fn[1](d,s["T"])
            if val is None or not np.isfinite(val): continue
            r=dict(tr); r["feat"]=val; hrows.append(r)
        if s["arm"]=="lo": hrows=[r for r in hrows if r["feat"]<=s["cut"]]
        else: hrows=[r for r in hrows if r["feat"]>=s["cut"]]
        st=stats(hrows)
        if st is None: print(f"  {s['f']} T={s['T']} {s['band']} {s['arm']}: N=0"); continue
        print(f"  {s['f']:16s} T={s['T']:3d} {s['band']} {s['arm']} cut={s['cut']:+.2f} "
              f"| HOLDOUT n={st['n']} mean={st['mean']:+.2f} wr={st['wr']:.2f} med={st['med']:+.2f}")
    print(f"\ndone in {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()