# BTC 5-minute "endgame favourite" — backtest findings

**One strategy survived rigorous testing and it is profitable on the logged
data. Everything else failed. This document was rewritten after a
source-aligned-timestamp correction that roughly halved the first result —
the numbers below are the corrected ones.**

Reproduce with:

```
python _pm_extract.py                # ~31 s, rebuilds the compact cache
python btc5m_endgame_backtest.py     # the backtest
python _pm_replay_bot.py             # proves the bot trades this exact rule
```

---

## 1. THE DATASET

`price_logs/polymarket/` — 271 JSONL files, 1.9 GB, **269 resolved** BTC
5-minute markets, ~289,000 L2 order-book snapshots, **22.4 contiguous hours**
(window 1792000500 → 1792081200).

Each file contains the full L2 book for both legs (`Up`/`Down`) plus the
**ground-truth resolution**, so realised P&L is measured against reality, not
a model.

## 2. THE MECHANISM (why the edge exists)

Polymarket 5-minute crypto binaries settle on the **Chainlink 60-second TWAP**:

- the **price to beat** is the 60 s TWAP immediately *before* the window opens
- the **settlement** is the 60 s TWAP over the window's *final 60 seconds*

(Effective 2026-08-07; confirmed against Polymarket's published rule.)

Verified on this data: reconstructing the settlement from the logged price
path reproduces the recorded resolution in **263/269 windows (97.8%)**, versus
91.1% for a naive final-spot rule. So the TWAP rule is right.

Because the settlement *averages* 60 seconds, part of the settlement window is
already **observed and locked in** before the window ends. The market's price
for the favourite does not fully reflect that lock-in, so near expiry the
favourite is systematically **underpriced**.

The underpricing is visible directly, by price band (all executable quotes,
source-aligned clock):

| favourite ask | rows | wins | mean ask | realised | gap | cluster-t |
|---|---|---|---|---|---|---|
| [0.70, 0.80) | 135 | 90 | 0.7459 | 0.6667 | **−0.0551** | −0.94 |
| [0.80, 0.86) | 123 | 112 | 0.8281 | 0.9106 | **+0.0618** | +1.58 |
| [0.86, 0.90) | 95 | 89 | 0.8755 | 0.9368 | +0.0477 | +1.37 |
| [0.90, 0.93) | 114 | 109 | 0.9112 | 0.9561 | +0.0477 | +1.88 |
| [0.93, 0.96) | 166 | 160 | 0.9411 | 0.9639 | +0.0056 | +0.22 |
| [0.96, 0.98) | 139 | 130 | 0.9671 | 0.9353 | −0.0200 | −0.83 |
| [0.98, 1.00) | 280 | 272 | 0.9865 | 0.9714 | −0.0010 | −0.12 |

The mispricing is **localised**, not a global favourite bias: it lives in the
0.80–0.93 band and vanishes at both ends (and the 0.93–0.96 band is already
fairly priced). That is what a lock-in effect looks like, and it is why a
plain "buy the favourite" rule does not work.

## 3. THE STRATEGY

> Once the window is **≥262 s in** (i.e. inside the settlement window),
> buy the **favourite** side when its best ask is in **[0.80, 0.96)**,
> requiring: both legs still live, spread ≤ 0.04, ≥ $10 displayed at the ask.
> Hold to settlement.
>
> **Size:** $15 when the favourite's ask is **< 0.85**, otherwise $5
> (see §3b.2 — the same edge, allocated better).

Entry is at the **executable ask**, not the midpoint. Fees are the published
taker fee `0.07 · p · (1−p)` per share.

**$5 is the exchange's practical floor.** The per-market `min_order_size` is
**5 shares**, and at the 0.95 end of the band with the 0.03 slippage cushion
a $5 stake buys **5.10 shares** — clear, but with no room to spare. A $2
stake buys 2.11 shares and is **rejected outright** (§3b.5).

### Result (the headline numbers)

The table below is the **base rule at a flat $5** — the number the rule was
selected and gated on. The shipped configuration adds the sizing overlay
(§3b.2), which takes the same 38 trades to **+$44.72 at +13.98% ROI,
t +4.57, CI [+0.66, +1.66]**.

| metric | value |
|---|---|
| trades | **38** over 269 windows (14% of windows) |
| wins | **37 / 38** |
| mean ask paid | 0.8813 |
| realised win rate | 0.9737 |
| break-even win rate (ask + fee) | 0.8886 |
| **edge per trade** | **+8.50 cents** |
| **net P&L** | **+$19.07** on $5 stakes |
| ROI | **+10.0%** |
| avg per trade | +$0.502 |
| **cluster-t** | **+3.20** |
| bootstrap CI95 (avg P&L) | **[+$0.15, +$0.72]** |
| binomial p vs market-implied | 0.0503 |
| worst single trade | −$5.04 |
| fees paid | $1.58 (0.83% of $190 staked) |

**With the sizing overlay that ships** ($15 when ask < 0.85, else $5):
**+$44.72 on $320 staked, +13.98% ROI, t +4.57, CI [+0.66, +1.66]**.

At the observed firing rate of 1.70 trades/hour of market time (≈41 trades
per 24 h of market time):

| | flat $5 | sized $5/$15 |
|---|---:|---:|
| net per trading day | +$20.41 | **+$47.88** |
| net per 30 days | +$612 | **+$1,436** |
| net per year (extrapolated) | +$7,451 | **+$17,477** |

Those per-period figures are arithmetic on the measured net scaled by the
measured firing rate — **not additional evidence**. They carry the full
uncertainty of a 38-trade, 22.4-hour sample, and the 95% CI on the average
trade is roughly ±30% of the mean. The strategy also only runs while the
bot is live and the venue is reachable.

The CI95 **excludes zero** — this is the only candidate in the whole study
that manages that on the corrected clock.

### What it trades in real numbers — and why $2 is not placeable

**$2 per trade would be rejected by the exchange, every time.** Polymarket
publishes a per-market `min_order_size` of **5 shares** (a *share* count,
not a dollar amount — observed in the repo's own live market payloads). At
a $2 stake and a 0.88–0.95 ask you buy only **2.11–2.47 shares**, so
**all 38 orders would be rejected**: 0 trades, $0 P&L.

The smallest safe stake is `5 × (0.95 + 0.03 slippage cushion) = $4.90`.
**$5 is the default** and clears every trade in the sample (5.26–6.17
shares). The bot now enforces this — it reads `min_order_size`
from the live book and skips (with a logged reason) rather than sending an
order the exchange will bounce.

| | $2 *(not placeable)* | $5 *(default, minimum viable)* | $10 |
|---|---|---|---|
| trades / wins | 38 / 37 | 38 / 37 | 38 / 37 |
| shares per trade | 2.11–2.47 | 5.26–6.17 | 10.53–12.35 |
| fee per trade | $0.017 (0.83% of stake) | $0.042 (0.83%) | $0.084 (0.83%) |
| **net P&L** | **+$7.63** | **+$19.07** | **+$38.13** |
| avg per trade | +$0.201 | +$0.502 | +$1.003 |
| ROI | +10.03% | +10.03% | +10.03% |
| cluster-t | +3.20 | +3.20 | +3.20 |
| CI95 (avg P&L) | [+0.061, +0.289] | [+0.152, +0.721] | [+0.304, +1.443] |
| worst single trade | −$2.01 | −$5.04 | −$10.07 |

**$5 → +$19.07 on $190 staked, +10.03% ROI, ~+$0.50 per trade.** The edge is
scale-invariant: ROI, t-statistic and win rate are identical at every stake
because the fee is proportional. Only the absolute dollars and the CI width
in dollars change.

$2's one real advantage is the smaller loss (−$2.01 vs −$5.04) and needing
$2 of capital rather than $5. Its disadvantage: one loss still cancels
**7.7 average winners** at any size, so the number of trades needed to
recover is unchanged — only the dollar amount shrinks.

**Use $5.** It is the smallest stake the exchange accepts on every trade in
the sample, it doubles the dollar edge over $2, and it keeps the same
10.03% ROI. The $2 column is a theoretical figure only.

## 3b. IS THERE A BETTER STRATEGY? — A SYSTEMATIC SEARCH

Yes, a better *sizing*; no, a better *rule*. A 1,023-candidate grid plus six
structurally different families were tested with the same protocol (causal
clock, chronological train/OOS, outcome-reassignment placebo, latency and fee
stress). The candidates that made more raw dollars all failed something.

### 3b.1 Alternative rules — all rejected

| candidate | net$ | t | why it fails |
|---|---:|---:|---|
| **SHIPPED** T≥262 fav [0.80,0.96) | +38.13 | +3.20 | — |
| wide band T≥262 fav [0.70,0.96) | **+54.53** | +4.04 | **fails 1 s latency** (t→+1.10, CI [−0.62,+1.77]); a cheap, moving quote is the first thing latency costs you |
| loose band T≥262 fav [0.55,0.96) | +52.76 | +1.87 | CI includes zero; fails latency |
| longshot T≥100 dog [0.02,0.40) | +282.95 | +1.96 | **placebo p = 0.956** — random outcome reassignment does *better*; top-3 = 174% of P&L |
| longshot T≥250 dog [0.05,0.50) | −8.24 | −0.01 | placebo p = 0.949 |
| fade the favourite (T≥250) | −41.10 | −0.92 | the mispricing is one-directional |
| buy the cheap underdog (T≥250) | −94.44 | −0.47 | same |
| early favourite, early longshot | negative | — | every CI includes zero; 59–94% of P&L in 5 trades |
| dip-buy / momentum / TWAP-dislocation | negative | — | no edge at any threshold |

The wide band is the instructive failure. It looks better on every headline
metric (+$54.53 vs +$38.13, t +4.04 vs +3.20, CI excludes zero) and a
train-only selection even picks it. But it **dies under 1 s execution
latency** (t +1.10, CI [−0.62, +1.77]) while the shipped band survives
(t +2.73, CI [+0.19, +1.51]). Its extra profit comes from entering at cheap,
fast-moving quotes — precisely the quotes a real taker cannot actually get.
Its band surface is also jagged (lo = 0.70 strong, 0.65 weak, 0.75 weak),
the signature of selection noise rather than a parameter effect.

The longshots are worse: their apparent edge is **entirely explained by the
placebo** (p = 0.956 means random outcome reassignment produces a *larger*
t-statistic than the observed one). A strategy whose P&L is reproduced by
shuffling the outcomes is not a strategy.

### 3b.2 What does work: conditional sizing

The same rule, sized by price. At a roughly **constant** win rate `p` across
the band, expected profit per dollar is `(p − ask)/ask`, so a *cheaper*
favourite carries more ROI per dollar:

| favourite ask | EV per dollar at p = 0.974 |
|---|---:|
| 0.81 | **+18.9%** |
| 0.85 | +13.5% |
| 0.91 | +6.4% |
| 0.94 | +3.2% |

So the same dollar earns roughly 3× as much at 0.83 as at 0.93. Kelly says:
put more capital in the cheap bucket. **This does not require the cheap
bucket to win more often** — the outcome placebo found no win-rate gap
(p = 0.66) and none is needed.

> **Overlay: stake $15 when the favourite's ask < 0.85, else $5.**

(The 3× multiplier is what matters; the absolute stake is a separate
choice. The overlay was first validated at $10/$30 and rescales exactly,
because the fee is proportional — see the $5 column of the table in §3.)

| metric | flat $5 | sized overlay |
|---|---:|---:|
| trades / wins | 38 / 37 | 38 / 37 |
| staked | $190 | $320 |
| **net P&L** | +$19.07 | **+$44.72** |
| ROI | +10.03% | **+13.98%** |
| avg per trade | +$0.502 | +$1.177 |
| **cluster-t** | +3.20 | **+4.57** |
| bootstrap CI95 | [+0.15, +0.72] | **[+0.66, +1.66]** |
| worst trade | −$5.04 | −$5.04 |
| max drawdown | −$5.04 | −$5.04 |
| first half | +$12.33 (t +9.31) | +$24.20 (t +4.52) |
| second half | +$6.73 (t +1.15) | +$20.52 (t +2.45) |

**+135% more profit, +64% more ROI, higher t, tighter CI, identical worst
trade** — the single loss happened to land in the dear bucket, so the overlay
does not increase realised drawdown. It also improves the *weak* half: the
second chronological half goes from t +1.15 to t +2.45.

Validation, in order of how hard it is to fake:

- **Threshold is a plateau, not a spike.** Every cut from 0.82 to 0.90
  improves both the full sample and the OOS half; the shipped rule is the
  only point at which the overlay does nothing.
- **Bootstrap of the sizing decision.** Resampling windows 400× and
  re-scoring, the sized portfolio beats flat in mean net$ *and* mean ROI at
  every multiplier. Not one lucky resample.
- **Outcome placebo.** Reassigning real outcomes across windows and keeping
  the sizing rule: null relative-gain mean −58.4%, 95th percentile −17.9%,
  max +29.1%. The observed +135% gain is outside the entire null →
  **p = 0.0000**.
- **Walk-forward.** Choosing both the threshold and the multiplier on train
  and scoring on the untouched fold: mean OOF delta **+$8.68**, 3 of 4
  folds positive.
- **Stress.** 1 s latency: +$34.09, t +3.77, CI [+0.49, +1.58] — survives.
  2 s latency: +$27.00, t +3.19, CI [+0.30, +1.33] — survives. 4× fees:
  +$35.26, t +4.05, CI [+0.46, +1.35]. 8× fees: +$22.65, t +3.06, CI
  [+0.18, +0.93]. All survive.
- **Concentration.** 13 of 38 trades in 13 distinct windows — one trade per
  window, so no single window can carry the result.
- **Min-order-size.** $15 at ask 0.81 buys 18.5 shares; $5 at 0.95 buys 5.10.
  Both above the 5-share minimum, the $5 case by only 0.10 shares.
- **Fillable depth.** Every one of the 38 entries displays more size at the
  ask than the stake requires: minimum $24.30, median $248, maximum $1,874.
  The $15 stake is covered on all 38.

### 3b.3 The honest risk

| stake multiplier | break-even cheap-bucket win rate | observed |
|---:|---:|---:|
| 1.5× | 0.9456 | 1.000 |
| 2.0× | 0.9184 | 1.000 |
| 3.0× | 0.8912 | 1.000 |

The cheap bucket won 13/13, so its 95% one-sided confidence interval is
**[0.794, 1.000]** — and P(13/13 | true p = 0.90) = 0.254, so a 13/13 run
is entirely ordinary if the true rate is 0.90. The 3× stake breaks even at a
0.89 cheap-bucket win rate, which is *below* what the 95% lower bound
implies, but not by a wide margin.

A constant-p stress test (forcing every trade to win at rate `p₀`, i.e.
assuming the whole edge is an artefact) shows the sized portfolio still
beats flat for `p₀ ≥ 0.85` and loses only below it.

**The overlay therefore makes the strategy's expected profit higher and its
drawdown no worse, on a sample of 38 trades — 13 of them in the cheap
bucket. It is the same edge, better allocated, not a new edge.** Set
`STAKE_CHEAP = STAKE` in the bot to disable it.

### 3b.4 What was fixed in the bot while testing this

1. **`EndgameJournal.reserve(...)` was being called with a `condition_id=`
   keyword the method did not accept — a TypeError on the first live trade.**
   The `condition_id` column did not exist in the schema either. Both fixed
   (column added via `_migrate`, parameter added, value stored and verified
   round-trip). The bot would have crashed on its first signal.

2. **The stake was never checked against the displayed depth.** A FOK order
   for more dollars than the book shows at the ask is rejected for
   *insufficient size*, not for a share-count reason — and the $10 depth
   filter did not cover the 3× overlay stake. At the original $10/$30
   sizing, **trade #17 (ask 0.810, depth $24.30) would have been sent a
   $30 order into a $24.30 book and bounced.** The entry test now requires
   the displayed depth to cover the *actual* stake, and the backtest applies
   the identical check so the two cannot drift.

3. The exchange's **5-share minimum** is enforced at the worst price the
   order could fill at (ask + slippage cushion), so the bot skips with a
   logged reason rather than sending an order that will bounce.

## 4. WHY I BELIEVE IT

| test | result |
|---|---|
| **Alignment control** | the underdog bought at the *same timestamps* loses **−$440.02 (t = −4.66)**. Timestamps and resolution labels are aligned; a mis-mapped winner would show up here immediately. |
| **Leave-one-out** | dropping any single trade leaves t between **+3.03 and +13.46**. Not driven by one lucky window. |
| **P&L concentration** | top 5 of 38 trades = **27.9%** of P&L. (The repo's previously failed lead/lag rule was **80%** — this is not that.) |
| **Chronological halves** | first half **+$24.67** (t=+9.31, 20/20), second half **+$13.46** (t=+1.15, 17/18) — both positive. |
| **Block stability** | **7 of 8** chronological blocks profitable. |
| **Out-of-sample** | rule selected on the **first 60%** → **16/16 wins** on the untouched last 40%, net +$22.83, t=+9.79, CI [+1.14, +1.70]. (Train half: 21/22, t=+1.32.) |
| **Family-level sign test** | **40/40** candidate rules are positive out of sample (one-sided p < 1e-6) — the effect is a property of the *family*, not of one lucky rule. |
| **Family-wise max-t** | 40-rule grid, 600-draw **outcome** permutation: observed max t +14.23 vs null mean −3.19 (95th pct −1.74) → **p = 0.0000**. Not a grid-search artefact. |
| **Leak audit** | winners balanced (147/269 Up = 54.6%); resolution agrees with the price direction ~88% in *both* halves; no degenerate period. |

### Stress

| perturbation | result |
|---|---|
| **execution latency** 0.3 s | +$35.00, t=+2.95, CI [+0.25, +1.43] |
| **execution latency** 0.5 s | +$35.28, t=+2.96, CI [+0.26, +1.53] |
| **execution latency** 0.8 s | +$32.24, t=+2.74, CI [+0.16, +1.45] |
| **execution latency** 1.0 s | +$32.33, t=+2.73, CI [+0.19, +1.51] |
| **fee ×2** (0.14·p(1−p)) | +$34.98, t=+2.95, CI [+0.23, +1.35] |
| **fee ×4** (0.28·p(1−p)) | +$28.66, t=+2.45, CI [+0.07, +1.17] |
| **fee ×8** (0.56·p(1−p)) | +$16.03, t=+1.40, CI [−0.23, +0.80] — the edge dies if the true fee is ~8× the published one |
| **min depth $25 / $50 / $100** | +$36.96 / +$35.48 / +$33.74 (t = +3.12 / +2.98 / +2.84) |

The edge survives a 4× fee and a $100 depth requirement, and it holds under
a full second of execution latency.

## 5. THE TIMESTAMP CORRECTION (read this before trusting any of it)

The first pass of this study used the logger's own `seconds_into_window`
stamp. The repo's canonical loader
(`_research_value_stop.py::load_source_aligned`) stamps each frame at
`max(Up.book_timestamp, Down.book_timestamp, coinbase.exchange_ts) −
window_start`, because **the logger writes the row before its REST calls
complete**.

Measured on this dataset the logger stamp is **0.963 s optimistic on average
(median 0.967 s; 99.3% of frames > 0.5 s)**. Correcting it moved the
first candidate rule from `t=+3.32, CI [+0.24, +1.15]` to `t=+1.83, CI
[−0.12, +1.04]` — i.e. **the first result was not significant, and the
look-ahead was worth roughly half the effect.**

The shipped rule (T≥262, band [0.80, 0.96)) is the one that survives the
correction with a CI excluding zero. All numbers in this document use
`t_src`.

> **This is the single most important thing in this report.** If you reuse
> this data, use `t_src`, never `seconds_into_window`.

## 6. THE HONEST CAVEATS

1. **N = 38 is a small sample.** The CI is wide even though it excludes zero.
   This is a *promising, well-tested* edge, not a proven money machine.
2. **The firing rate is low** — 1.70 trades/hour of market time, ~41
   trades/day. At +$0.50 each that is ~$20/day per $5 stake: the strategy
   needs volume, not capital, to matter. (With the sizing overlay,
   ~+$48/day on a $5 base.)
3. **One loss (−$5.04) cancels 7.7 average winners.** Position sizing
   must expect a ~1-in-38 full loss. **Do not size this as if it cannot
   lose.** (At the $10 stake it was 10 average winners; the *count* of
   winners a loss cancels is scale-invariant, the dollar figure is not. With
   the sizing overlay the loss cancels 3.7 average winners, because the
   cheap bucket's winners are three times larger.)
   *Risk option:* taking profit at **0.95** instead of holding to settlement
   gives up $11.60 of total P&L (+$33.12 vs +$44.72) but converts the worst
   trade from −$5.04 to **−$0.04** and raises t from +4.57 to **+5.60**
   (all 38 trades reach 0.95). For a live bot that has to survive a bad
   day, that trade may be worth making. The backtested default below is
   hold-to-settlement.
   *(Exit study recomputed at the $5 stake with the exact entry frames:
   `_pm_research/_pm_exits5.py`. Exits are at the bid with a depth check and
   an exit fee. Stop-losses are worse: STOP 0.60 gives +$13.11 t +2.00,
   STOP 0.80 gives +$5.75 t +0.98.)*
4. **The dataset spans only ~22.4 h / 2 UTC days.** Rare regimes are
   unobserved. The repo's own preregistered script demands 3 distinct UTC
   days and 200 windows; this cannot meet that bar.
5. **The fee schedule is unconfirmed.** The CLOB fee-rate endpoint is
   unreachable from here, so I used the *more expensive* published formula
   (`0.07·p(1−p)`) and stress-tested 2× and 4×. A second schedule is in
   circulation (`0.25·(p(1−p))²`), which would make the edge larger.
6. **The chainlink TWAP was reconstructed from the Coinbase price** in the
   logs (97.8% agreement with ground truth, not 100%). The *strategy itself*
   does not depend on this reconstruction — it trades pure market prices
   against ground-truth resolutions — but a mechanism-level claim would need
   real Chainlink data.
7. **Late-window favourites can gap.** One loss had the market at 0.90→0.97
   on the losing side — the TWAP settlement overrode a confident spot
   market. This is the real tail risk, and it is why the band stops at 0.96.
8. **A stop-loss destroys the edge.** On the shipped rule a 0.60 stop takes
   it from +$38.13/t=+3.20 down to **+$19.78/t=+1.85** (CI [−0.05, +1.03],
   touching zero); a 0.80 stop gives +$12.08/t=+1.84. The book gaps through
   stops. **Hold to settlement.**
9. **Entry timing is part of the edge.** The first-qualifying-quote rule
   beats a fixed clock entry: entering at exactly T=260 gives only
   t=+0.98 (CI [−0.54, +1.06]) and at T=264 t=+1.07, versus t=+3.20 for
   entering on the first qualifying quote. A live bot polls and takes the
   first qualifying quote, so it matches the validated behaviour — but this
   is timing-sensitive and the edge is narrow in time.
10. **A 1-cent spread** leaves only 17 trades, and all 17 were profitable
    (net +$17.18, t=+6.43, worst +$0.60). In-sample that removes the
    tail-loss risk, though N=17 is too small to trust on its own. Worth
    watching in paper trading.

## 7. WHAT FAILED (all tested, all negative)

| idea | result |
|---|---|
| Complete-set arbitrage (`ask_up + ask_down < 1`) | appeared in 268/271 markets but lasted **one snapshot (~0.28 s)** — stale book, random-sign PnL. Not tradeable. |
| Sub-second market lag vs Coinbase | none; predictive slope peaks at lag 1. The market reprices contemporaneously. |
| Coinbase/chainlink model beating the market | the market beat the model at **every** horizon tested. |
| Early-window longshot (buy underpriced longshots) | re-tested on the **causal** clock: best cell t=+1.38, CI [−0.79, +5.96] (**includes zero**), and the top 5 trades carry **59–94%** of the P&L. Net +$381 looks large but rests on a handful of lottery fills. Selection bias — do not trade it. |
| Wick reversion / reversion exit | winners' prices reverted +0.14 but the effect was selection-biased and second-half concentrated. |
| General taker drift after costs | predictable drift **0.3–0.8 cents** vs a **2.7–3.0 cent** round-trip cost. Hopeless. |
| Book imbalance | +12.3% train, **−15.3%** validation. |
| Pair-arb / fair-value model | −$9.79 and −5.4% ROI. |
| A "model-confirmed" 48/48 endgame rule | a random-z placebo produced the same +$22.96 — the model z explodes near expiry. |
| Late-window favourite with a stop-loss | on the shipped rule a 0.60 stop cuts it to **+$13.11 / t=+2.00** and a 0.80 stop to **+$5.75 / t=+0.98** (both at the $5 base stake). The book gaps through stops. **Hold to settlement.** |
| Take-profit at 0.95 | lower total (**+$33.12** at the $5 base, vs +$44.72 holding) but a higher t (**+5.60** vs +4.57) and the worst trade improves from −$5.04 to **−$0.04**. See the risk note above. |
| Fading the favourite (the opposite side of the same mispricing) | **−$41.10, t=−0.92** (N=57, T≥250, favourite ask 0.55–0.90). Buying the cheap underdog late is **−$94.44, t=−0.47** (N=67). The mispricing is **one-directional** — the favourite is underpriced, the underdog is not overpriced, so there is nothing to fade. |

### How this fits the repo's prior research

A full audit of the existing `_research_*.py` scripts found that **every
previously attempted strategy lost money**, and the root cause is
consistent: the *direction* is nearly knowable (the Coinbase/Chainlink 60 s
TWAP direction matches the outcome in **99.1%** of windows) but the *price*
already reflects it. **Alpha has to come from price, not direction.**

This endgame rule is a *price* edge (the favourite is mispriced relative to
the locked-in TWAP), which is the one space the prior research flagged as
unexplored. The previous lead/lag rule in this repo is the cautionary
counter-example: it looked spectacular in-sample (+32.6% ROI over 219
windows) and then lost −20.2% on the next 49 windows, with 80% of its P&L in
5 trades. That is why this report demands concentration, leave-one-out,
latency and OOS numbers rather than a headline number.

## 8. FILES

| file | purpose |
|---|---|
| `btc5m_endgame_backtest.py` | the backtest — every number above |
| `btc5m_endgame_bot.py` | the bot; `decide_entry()` is shared with the backtest |
| `_pm_replay_bot.py` | replays recorded books through the bot's own code (**38/38 exact match**) |
| `_pm_extract.py` | rebuilds `_pm_cache/` from the raw JSONL (~31 s); stores `t`, `t_src`, `t_lag` |
| `_pm_cache/` | compact per-window feature cache (35 MB, built from 1.9 GB) |
| `_pm_research/` | the research trail (60 scripts: every hypothesis tested and rejected, plus the shipping-gate, OOS-sweep, leak-audit, alternative-strategy-search and sizing-overlay tools) |

## 9. HOW TO RUN

```
Run Endgame Bot (paper).cmd            # paper orders via polysimulator
python btc5m_endgame_bot.py            # dry run, signals only, no orders
python btc5m_endgame_backtest.py       # reproduce the backtest
```

**Network note:** all Polymarket hosts (`clob`, `gamma-api`,
`ws-subscriptions-clob`) are **blocked from this machine** — TLS connections
are reset at the handshake, on both the local host and the VPS. The bot starts
and resolves the market correctly, but it cannot receive the live order book
from here. It needs a host with unrestricted access to Polymarket.

**Recommendation: paper trade it first.** Collect ≥ 100 live trades, then
re-run the same significance checks on the paper log. If the realised win
rate holds near 0.97 at a mean ask near 0.88, the edge is real. If it reverts
to the market-implied ~0.89, it was luck on 38 trades.
