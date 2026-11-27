#!/usr/bin/env python3
"""_pm_rate.py — what trade RATE is actually available, and what $/hour it yields.

We have:
  - Late-Lock 5m: 1.19/hour (28.7/day) at 35% win, +$1.64/trade at $5.
  - 15m book: separate windows (the skill's _pm15 work). 15m windows are 3x
    longer, so ~1/3 the window rate, but the skill found 15m taker cells dead.

The honest question: can ANY combination of non-latency engines reach $10/hour?
We compute the max achievable $/hour from Late-Lock at various stakes, and from
running Late-Lock on BOTH the 5m grid and a hypothetical 2nd independent book.

$10/hour = $240/day. At $5/trade and +$1.64/trade, that needs 146 trades/day.
Late-Lock gives 28.7/day. Gap = 5.1x. Only more independent windows fill it.
"""
import math

# Late-Lock measured
MEAN_TRADE_5 = 1.644   # $5 stake, 4c slip
TPD = 28.7             # trades/day on 5m
T截 = TPD/24

print("=== Late-Lock 5m: what stake clears $10/hour? ===")
for stake in (5,10,15,20,25,30):
    mean = MEAN_TRADE_5 * stake/5.0
    per_h = mean * T截
    print(f"  ${stake:2d} stake: ${mean:.2f}/trade -> ${per_h:.2f}/hour  (${per_h*24:.0f}/day)")

print("\n=== trade-rate ceiling ===")
print(f"  Late-Lock 5m: {TPD:.1f}/day = {T截:.2f}/hour")
print(f"  need for $10/hr at $5 stake: {10/MEAN_TRADE_5:.1f} trades/hour = {10/MEAN_TRADE_5*24:.0f}/day")
print(f"  gap: {10/MEAN_TRADE_5*24/TPD:.1f}x more trades needed")

print("\n=== independent books available ===")
print("  5m grid: 288 windows/day (one every 5 min)")
print("  15m grid: 96 windows/day (one every 15 min)")
print("  Late-Lock fires on ~10% of 5m windows -> ~29/day")
print("  Even firing on EVERY 5m window (288/day) at +$1.64 = $%.0f/day = $%.1f/hour" % (288*1.644, 288*1.644/24))

print("\n=== the only way to $10/hour non-latency ===")
print("  A) $25 stake on Late-Lock (needs ~$2,400 bankroll, 1.06% Kelly)")
print("  B) a 2nd independent engine firing on the other windows")
print("  C) both 5m + 15m books (15m taker cells measured dead)")