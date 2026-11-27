# `_pm_research/` — the research trail

These are the working scripts behind `ENDGAME_EDGE.md`. They are kept for
provenance, not for reuse: **the deliverables are `btc5m_endgame_backtest.py`
and `btc5m_endgame_bot.py` in the parent folder.** Anything here may carry a
bug that was later found and fixed elsewhere.

## If you are looking for the result

Read `../ENDGAME_EDGE.md`. The shipped rule and all its numbers are there,
reproducible with:

```
python ../_pm_extract.py                # rebuild the compact cache (~31 s)
python ../btc5m_endgame_backtest.py     # the backtest
python ../_pm_replay_bot.py             # proves the bot == the backtest
```

## Script inventory

| group | scripts | what they establish |
|---|---|---|
| **extraction** | `_pm_extract.py` (parent folder) | builds `_pm_cache/`; stores the logger stamp `t`, the **causal** stamp `t_src`, and the skew `t_lag` |
| **calibration / mechanism** | `_pm_analyze1/2`, `_pm_cond_ev`, `_pm_resolution`, `_pm_res_weight`, `_pm_zscale`, `_pm_rule`, `_pm_volscale` | the market is calibrated at the price paid; settlement is the Chainlink 60-s TWAP (263/269 agreement vs 91.1% for final spot); the favourite is underpriced only in the 0.80–0.93 band |
| **rejected strategies** | `_pm_early`, `_pm_decisive`, `_pm_stability`, `_pm_wick`, `_pm_robust`, `_pm_martingale`, `_pm_late_fav`, `_pm_can_model_win`, `_pm_latency`, `_pm_arb_test`, `_pm_backtest`, `_pm_walkforward` | every one of these LOST — see the "what failed" table in `ENDGAME_EDGE.md` |
| **endgame iterations** | `_pm_endgame`, `_pm_endgame_rig`, `_pm_endgame_exits`, `_pm_zdiff`, `_pm_final`, `_pm_endo`, `_pm_twap_edge`, `_pm_honest` | the endgame candidate, in order of increasing rigour. `_pm_endo` is the placebo audit that killed the model-based version |
| **final gate** | `_pm_final_honest`, `_pm_selected_rule`, `_pm_oos_sweep`, `_pm_leakcheck`, `_pm_ship_rule` | the source-aligned reassessment, the train→test split, the OOS sweep of all 40 rules, the leak audit, and the shipping gate |
| **alternative-strategy search** (all rejected) | `_pm_altsearch` (1,023-rule grid), `_pm_altjudge` (placebo + concentration), `_pm_altband` (band plateau), `_pm_altgate` (the shipped gate on the band grid), `_pm_altwalk` (walk-forward on the lower bound), `_pm_altbattery` (finalists), `_pm_altfamilies` (early / dip / momentum / TWAP-dislocation), `_pm_altsummary` (one protocol for everything) | no alternative RULE beats the shipped one: the wide band dies under 1 s latency, the longshots die under the outcome placebo (p=0.956) |
| **conditional sizing** (adopted) | `_pm_altsize`, `_pm_altsize_oos`, `_pm_altsize_stress`, `_pm_altsize2`, `_pm_altship`, `_pm_altrisk`, `_pm_altrisk2`, `_pm_altfinal`, `_pm_altfinal2` | the sizing overlay ($30 when the favourite's ask < 0.85): mechanism, threshold plateau, bootstrap of the sizing decision, outcome placebo, walk-forward, break-even win rate, drawdown |
| **superseded** | `_pm_volscale` (units bug: absolute distance vs log-return variance, no transport term), `_pm_resolution_weight` (superseded by `_pm_res_weight`) | marked in their own docstrings |

## Traps these scripts fell into — worth knowing

- **The logger's `seconds_into_window` is ~0.96 s optimistic.** It is written
  before the REST calls complete. Using it inflated the first headline rule
  from t=+1.83 (not significant) to t=+3.32. Use `t_src`.
- **A family-wise permutation must reassign OUTCOMES across windows.**
  Shuffling window labels preserves the win count per price band and
  produces a null identical to the observed.
- **A 100% win rate in one half of the data is a leak until proven
  otherwise.** `_pm_leakcheck.py` is the audit: balanced winners, resolution
  agreeing with the price direction in both halves, and control price bands
  the strategy never trades.
- **The "48/48 model-confirmed" rule was equivalent to a random-z placebo**
  (`_pm_endo.py`). Model z explodes near expiry; the market's does not.
- **A rule that looks better on every headline metric can still be a
  selection artefact.** The wide band [0.70,0.96) beat the shipped band on
  net$, t, CI and train-only selection — and still failed under 1 s
  execution latency. Always run the latency stress before believing a
  better-looking rule.
- **A permutation placebo that produces a null BETTER than the observed
  result kills the strategy outright.** The longshot rule's p = 0.956 means
  random outcome reassignment beats it.

## Reproducing the cache

`_pm_extract.py` reads 1.9 GB of JSONL and writes one `.npz` per window plus
`manifest.pkl`. It takes ~31 s. Deleting `_pm_cache/` and re-running is
always safe — it is a pure function of the raw logs.
