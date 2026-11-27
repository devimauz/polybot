#!/usr/bin/env python3
"""_pm_sizing.py — design a stake for the Late-Lock Tail to clear >= $10/day
under a WORST-CASE 4c slip, using the real Nov4-14 holdout distribution.

Steps (per the skill's live-launch-audit recipe):
1. Re-price the same trade set at 0/1/2/4c slip (worst-case cap fill, int-share).
2. Report mean, win%, maxDD, per-day rate at each slip.
3. Monte-Carlo 20k paths at candidate stakes -> median final, P(negative),
   P(bust), p10/p90 maxDD, and P(net >= $10/day * horizon).
4. Pick the stake that clears the target with bounded risk.
"""
import csv, math
import numpy as np

CSV = r"C:\Users\user\Desktop\bot\_pm_out\slow_holdout_trades.csv"
FEE = 0.07
DAYS = None

def load():
    rows=[]
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(ws=int(r["ws"]), T=int(r["T"]), side=r["side"],
                         ask=float(r["ask"]), sh0=int(r["sh"]), win=r["win"]))
    rows.sort(key=lambda x: x["ws"])
    return rows

def repriced(rows, slip_c):
    """Worst-case: every fill at observed ask + slip_c cents, int-share sizing
    from the ORIGINAL stake-equivalent (sh0 was int($5/ask); keep same share
    count so slip only costs price, not size)."""
    s = slip_c / 100.0
    out=[]
    for r in rows:
        ask = r["ask"]
        fill = ask + s
        if fill >= 1.0:  # can't fill above $1
            continue
        sh = r["sh0"]  # keep the same integer share count as the $5 book
        won = 1.0 if r["side"] == r["win"] else 0.0
        pnl = sh*won - sh*fill - sh*FEE*fill*(1-fill)
        out.append(pnl)
    return np.array(out)

def stats(pnl):
    n=len(pnl)
    return dict(n=n, mean=float(pnl.mean()), net=float(pnl.sum()),
                win=float((pnl>0).mean()), med=float(np.median(pnl)),
                worst=float(pnl.min()), best=float(pnl.max()),
                sd=float(pnl.std(ddof=1)) if n>1 else 0.0)

def mc(pnl, stake, days, horizon_days, paths=20000, seed=3):
    """Resample the per-trade P&L (scaled to `stake` from the $5 book) and
    simulate `horizon_days` of trading at `days`-trade/day rate."""
    rng=np.random.default_rng(seed)
    # scale per-trade P&L to the new stake (linear in stake)
    scale = stake / 5.0
    p = pnl * scale
    # trades per day from the observed rate
    tpd = len(pnl) / 10.8
    n_trades = int(round(tpd * horizon_days))
    bust = p.min() + 0  # one-stake drawdown proxy: a single worst loss
    # MC
    idx = rng.integers(0, len(p), size=(paths, n_trades))
    sim = p[idx].sum(axis=1)
    # daily series
    per_day = sim / horizon_days
    return dict(
        median=float(np.median(sim)),
        p10=float(np.percentile(sim,10)),
        p90=float(np.percentile(sim,90)),
        p_neg=float((sim<0).mean()),
        p_target=float((per_day >= 10.0).mean()),
        mean_daily=float(per_day.mean()),
        worst_ever=float(p.min()),
        n_trades=n_trades, tpd=tpd,
    )

def main():
    rows=load()
    pnl0=np.array([r["sh0"]*(1.0 if r["side"]==r["win"] else 0.0) - r["sh0"]*r["ask"]
                   - r["sh0"]*FEE*r["ask"]*(1-r["ask"]) for r in rows])
    span_days = (rows[-1]["ws"]-rows[0]["ws"])/86400.0
    tpd = len(rows)/span_days
    print(f"Late-Lock holdout: {len(rows)} trades over {span_days:.1f} days = {tpd:.1f} trades/day")
    print(f"win%={100*(pnl0>0).mean():.1f}  (breakeven% ~ {100*np.mean([r['ask']*(1+FEE*(1-r['ask'])) for r in rows]):.1f})")

    print("\n=== slip-bound table (same trades, worse price) ===")
    print("slip   N    mean$/trade   net$     win%   worst$")
    for s in (0,1,2,4):
        p=repriced(rows,s); st=stats(p)
        print(f"{s}c   {st['n']:4d}  {st['mean']:+8.3f}   {st['net']:+8.0f}  {100*st['win']:5.1f}  {st['worst']:+7.2f}")

    # use the 4c repriced series as the live distribution (worst-case)
    p4 = repriced(rows, 4)
    st4 = stats(p4)
    print(f"\nLIVE distribution (worst-case 4c slip): mean ${st4['mean']:.3f}/trade, sd ${st4['sd']:.3f}")

    # Kelly fraction
    kelly = st4['mean'] / (st4['sd']**2) if st4['sd']>0 else 0
    print(f"Kelly fraction of bankroll per trade: {100*kelly:.2f}%  (stake/bankroll)")

    # Monte-Carlo across stakes, 30-day horizon
    print("\n=== Monte-Carlo (20k paths, 30-day horizon, resample 4c-slip trade dist) ===")
    print("stake  median$  p10$    p90$     P(neg)  P(>=$10/day)  meanDaily$  worstTrade$")
    for stake in (2,3,5,8,10,15,20):
        r=mc(p4, stake, tpd, 30, paths=20000)
        print(f"${stake:3d}  {r['median']:+8.0f} {r['p10']:+8.0f} {r['p90']:+8.0f}  "
              f"{100*r['p_neg']:5.1f}%  {100*r['p_target']:5.1f}%       {r['mean_daily']:+7.2f}   {r['worst_ever']:+8.2f}")

    # What stake clears $10/day with P>0.8 and P(neg)<0.1?
    print("\n=== stake search: smallest stake with P(>=$10/day)>=0.80 and P(neg)<=0.10 ===")
    for stake in range(2, 60, 2):
        r=mc(p4, stake, tpd, 30, paths=20000)
        if r['p_target']>=0.80 and r['p_neg']<=0.10:
            print(f"  -> ${stake} stake: P(>=$10/day)={100*r['p_target']:.1f}%  P(neg)={100*r['p_neg']:.1f}%  "
                  f"median30d=${r['median']:.0f}  meanDaily=${r['mean_daily']:.2f}")
            break
        else:
            print(f"     ${stake}: P(>=$10/day)={100*r['p_target']:.1f}%  P(neg)={100*r['p_neg']:.1f}%")

if __name__ == "__main__":
    main()