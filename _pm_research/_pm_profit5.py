"""Profit summary at the $5 stake, plus the derived per-period figures.

Everything here is computed from the shipping rule's 38 backtest trades.
The per-period numbers are simple arithmetic on the measured net, scaled by
the observed firing rate - they are NOT extra evidence, just an
extrapolation of the measured rate.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import btc5m_endgame_backtest as B
import btc5m_endgame_bot as E

W = B.load_all()
rows = B.build_rows(W)
sel = B.first_per_window([r for r in rows
                          if r["T"] >= B.SHIP_T_MIN and r["mkt"] >= 0.5
                          and B.SHIP_BAND[0] <= r["ask"] < B.SHIP_BAND[1]])
ask = np.array([r["ask"] for r in sel])
won = np.array([r["won"] for r in sel])
wi = [r["wi"] for r in sel]

# the covered trading time
span_h = (max(d["ws"] for d in W) - min(d["ws"] for d in W)) / 3600.0
# BTC 5-minute markets run 24/7, so a daily rate is the per-hour rate over
# 24h of MARKET time.  The sample covers 22.42 h of market time spread over
# 2 UTC days; dividing by "2 days" would understate the rate because the
# sample does not cover 48 h of trading.
per_day = len(sel) / span_h * 24.0

print(f"trades {len(sel)}  wins {int(won.sum())}  losses {int((1-won).sum())}  "
      f"win rate {won.mean():.4f}")
print(f"covered market time {span_h:.2f} h  ->  {len(sel)/span_h:.2f} trades/hour  "
      f"= {per_day:.0f} trades per 24h of market time")
print()

for nm, stake_fn in (("flat  $5      ", lambda a: 5.0),
                     ("sized $5/$15  ", lambda a: E.stake_for_ask(a))):
    st = np.array([stake_fn(a) for a in ask], float)
    sh = st / ask
    pnl = sh * won - st - B.fee(ask, sh)
    _, t, _ = B.cluster_t(wi, pnl)
    lo, hi = B.boot_ci(wi, pnl)
    net = pnl.sum()
    staked = st.sum()
    print(f"{nm} staked ${staked:,.2f}   NET ${net:+.2f}   ROI {net/staked*100:+.2f}%")
    print(f"{'':<13}per trade ${pnl.mean():+.3f}   t {t:+.2f}   "
          f"CI95 [{lo:+.3f}, {hi:+.3f}]   worst ${pnl.min():+.2f}")
    print(f"{'':<13}at {per_day:.0f} trades/day -> ${net/span_h*24:+,.2f}/day, "
          f"${net/span_h*24*30:+,.0f}/30d, ${net/span_h*24*365:+,.0f}/yr (extrapolated)")
    print()

# what the two loss/expense lines look like
st5 = np.full(len(sel), 5.0)
sh5 = st5 / ask
fees = B.fee(ask, sh5).sum()
print(f"fees paid on the flat-$5 portfolio: ${fees:.2f}  "
      f"({fees/st5.sum()*100:.2f}% of ${st5.sum():.0f} staked)")
loss = -pnl.min()
print(f"the single loss: ${loss:.2f}  =  {loss/pnl[pnl > 0].mean():.1f} average winners")
