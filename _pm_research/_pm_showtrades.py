"""Print the full trade list for the shipping strategy.

Same rule as the backtest, with the conditional sizing overlay applied:
  T >= 262s, favourite side, executable ask in [0.80, 0.96)
  stake = $30 when the favourite's ask < 0.85, else $10
  hold to settlement
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import btc5m_endgame_backtest as B
import btc5m_endgame_bot as E

W = B.load_all()
rows = B.build_rows(W)
sel = B.first_per_window([r for r in rows if r["T"] >= 262 and r["mkt"] >= 0.5
                          and 0.80 <= r["ask"] < 0.96])
sel.sort(key=lambda r: r["ws"])

FEE = 0.07
print("=" * 118)
print("THE 38 SHIPPING-RULE TRADES   (T>=262s, favourite ask [0.80,0.96), "
      f"stake ${E.STAKE_CHEAP} if ask<{E.CHEAP_ASK} else ${E.STAKE}, hold to settlement)")
print("=" * 118)
print(f"{'#':>3} {'entry UTC':>16} {'T':>4} {'side':>5} {'ask':>6} {'stake':>6} {'shares':>7} "
      f"{'fee':>6} {'res':>5} {'PnL flat':>9} {'PnL sized':>10} {'window start UTC':>19}")

tot10 = 0.0
tots = 0.0
staked10 = 0.0
stakeds = 0.0
nwin = 0
for i, r in enumerate(sel, 1):
    a = r["ask"]
    won = r["won"]
    st_flat = float(E.STAKE)
    st_sized = E.stake_for_ask(a)
    sh_f = st_flat / a
    sh_s = st_sized / a
    fee_f = sh_f * FEE * a * (1 - a)
    fee_s = sh_s * FEE * a * (1 - a)
    pnl_f = sh_f * won - st_flat - fee_f
    pnl_s = sh_s * won - st_sized - fee_s
    tot10 += pnl_f
    tots += pnl_s
    staked10 += st_flat
    stakeds += st_sized
    nwin += won
    ws = datetime.fromtimestamp(r["ws"], tz=timezone.utc)
    ent = ws + __import__("datetime").timedelta(seconds=r["T"])
    print(f"{i:>3} {ent.strftime('%m-%d %H:%M:%S'):>16} {r['T']:>4.0f} {r['side']:>5} "
          f"{a:>6.3f} {st_sized:>6.0f} {sh_s:>7.2f} {fee_s:>6.3f} "
          f"{'WON' if won else 'LOST':>5} {pnl_f:>+8.2f} {pnl_s:>+10.2f} "
          f"{ws.strftime('%m-%d %H:%M:%S'):>19}")

print("-" * 118)
print(f"{'':>3} {'':>16} {'':>4} {'':>5} {'mean':>6} {'tot':>6} {'':>7} {'':>6} {'':>5} "
      f"{tot10:>+8.2f} {tots:>+10.2f}")
print()
print(f"  trades {len(sel)}   wins {nwin}   losses {len(sel)-nwin}   "
      f"win rate {nwin/len(sel):.4f}")
print(f"  flat ${E.STAKE} : staked ${staked10:.0f}  net ${tot10:+.2f}  ROI {tot10/staked10*100:+.2f}%  "
      f"per trade ${tot10/len(sel):+.3f}")
print(f"  sized    : staked ${stakeds:.0f}  net ${tots:+.2f}  ROI {tots/stakeds*100:+.2f}%  "
      f"per trade ${tots/len(sel):+.3f}")
print(f"  difference: sized nets ${tots-tot10:+.2f} more on ${stakeds-staked10:+.0f} "
      f"capital ({(tots/stakeds)/(tot10/staked10)-1:+.1%} ROI)")

# stats
for nm, stake_fn in (("flat", lambda r: float(E.STAKE)),
                     ("sized", lambda r: E.stake_for_ask(r["ask"]))):
    p = []
    for r in sel:
        a = r["ask"]
        st = stake_fn(r)
        sh = st / a
        p.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
    p = np.array(p)
    m, t, _ = B.cluster_t([r["wi"] for r in sel], p)
    lo, hi = B.boot_ci([r["wi"] for r in sel], p)
    o = np.argsort(p)
    print(f"  {nm:<9} cluster-t {t:+.2f}   CI95 [{lo:+.2f},{hi:+.2f}]   "
          f"worst ${p.min():+.2f}   best ${p.max():+.2f}   top5 P&L share {p[o[-5:]].sum()/p.sum()*100:.0f}%")

# side split
print()
for side in ("Up", "Down"):
    sub = [r for r in sel if r["side"] == side]
    p = []
    for r in sub:
        a = r["ask"]
        st = E.stake_for_ask(a)
        sh = st / a
        p.append(sh * r["won"] - st - sh * FEE * a * (1 - a))
    print(f"  {side:<5} N={len(sub):>2}  wins {sum(r['won'] for r in sub):>2}  "
          f"mean ask {np.mean([r['ask'] for r in sub]):.4f}  net ${sum(p):+.2f}")

# the one loss
for i, r in enumerate(sel, 1):
    if not r["won"]:
        ws = datetime.fromtimestamp(r["ws"], tz=timezone.utc)
        ent = ws + __import__("datetime").timedelta(seconds=r["T"])
        a = r["ask"]
        st = E.stake_for_ask(a)
        sh = st / a
        print(f"\n  THE ONLY LOSS: trade #{i}  {ent.strftime('%m-%d %H:%M:%S')} UTC  T={r['T']:.0f}  "
              f"{r['side']} @ {a:.3f}  stake ${st:.0f}  ->  LOST ${sh*a + sh*FEE*a*(1-a):.2f}")
        print(f"    window resolved as: {W[r['wi']]['_winner']}  "
              f"(the 60s settlement TWAP moved against the {r['side']} side)")
