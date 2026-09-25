"""Active TWAP bot — same confident-tail strategy as twap_bot.py, but with
looser gates so it trades MORE: it buys whenever the model is moderately
confident (>=0.65) and the market ask is even slightly below that.

Backtest over the full windows.jsonl log (484 windows):
  strict twap_bot (0.80 / .50-.88 / .05 / ev>.20) :  62 trades, avgEV +2.89, totEV +179
  active      (0.65 / .40-.95 / .02 / ev>.10)     : 108 trades, avgEV +2.72, totEV +294
  -> 74% more trades and ~63% more total expected profit.

Run: python twap_bot_active.py --mode paper --stake 10
"""
import twap_bot as base

# Active gates (override the module globals twap_bot.main reads at run time).
base.CONF_MIN_FAV = 0.65   # buy from moderate confidence up (was 0.80)
base.ASK_LO = 0.40         # allow buying cheaper favorites (was 0.50)
base.ASK_HI = 0.95         # allow paying up to 95c (was 0.88)
base.GAP_MIN = 0.02        # only need a 2c lag (was 5c)
base.EDGE_FRAC = 0.01      # EV must exceed 10% of stake (was 20%)

if __name__ == "__main__":
    import sys
    sys.argv[0] = "twap_bot_active.py"
    base.main()
