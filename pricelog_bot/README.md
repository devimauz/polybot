# Polymarket 5m BTC up/down — backtest study (price_logs/polymarket)

Data: 522 window files (2026-10-29 onward), 519 scored (resolution line), 276,953 valid-book
frames, ~366h span. Logger stamp ~0.72s OPTIMISTIC vs causal clock
(max(book_ts_up, book_ts_dn, cb_ts) - window_start) — all studies use the causal clock.

Files (shared loader = loader.py, cache snapshots.pkl):
- calibrate2.py  — (time band x ask band) edge surface, pooled sides, n>=300
- backtest.py    — dog-side grid + train/test split, latency stress, plateau, outcome-permutation placebo
- backtest_lead.py — leader-side grid
- model.py       — model-driven P(Up) (partial TWAP projection vs open, sigma from realized 1s returns),
                   entry when p_model - ask >= EDGE
- scalp2.py      — same model, enter/exit round trips (exit at bid when gap closes)
- pair.py        — pair arb (both asks sum < TH), latency + FOK depth both legs
- focus.py       — window-level test of best surface cells

## Results (stake $10/trade, taker fee 0.05*min(p,1-p) both legs, FOK depth-checked, min 5 shares)

| Strategy | Train (n) | Test (n) | OOS verdict |
|---|---|---|---|
| Dog ask [0.05,0.30) t>=20, settle | +$379 (256) | -$98 (245) | sign flip; placebo p=1.00 — DEAD |
| Leader ask cells (all) | negative | mixed | negative window-level — DEAD |
| Model gap >=0.02-0.08, t[30,285) | +$143 (248) | +$14 (250) | t=+0.10 OOS — noise |
| Model scalp (36 cells) | negative | 0/36 positive | DEAD |
| Pair sum<=0.97 | +$242 (177) | -$25 (5) | dies at 0.5s latency — DEAD |

Conclusion: no taker edge survives fees (~3-5c friction on a 90c contract) in this dataset.
Surface cells (+2-4%/sh) were frame-level autocorrelation artifacts; they do not survive
window-level first-entry replay. Winner sanity: 267/519 Up (51.4%), late-leader wr vs ask consistent.

Re-run `loader.py --force && backtest.py` as new windows accrue (logs keep growing).