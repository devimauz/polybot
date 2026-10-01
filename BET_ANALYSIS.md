# Bot Bet Analysis — 2026-09-15 → 2026-09-25

Full machine-readable logs: **`bet_log_all.csv`** (3,817 orders from every
sqlite journal) and **`twap_paper_bets.csv`** (47 settled paper trades).

## 1. Where bets are logged

| Source | Contents |
|---|---|
| `live.sqlite3` → `orders`, `exits` | bot.py live BTC attempts + stop-loss/take-profit sells |
| `multi-live.sqlite3` → `orders`, `exits` | multi_bot.py (SOL/DOGE/XRP/ETH/BNB) live |
| `polysimulator.sqlite3` → `orders`, `profit_orders`, `reversal_watch` | live trader on api.polysimulator.com; `profit_orders` = take-profit limit sells |
| `edge-polysimulator.sqlite3`, `rebound-polysimulator.sqlite3` | edge/rebound bot fills |
| `price_logs/polysimulator/*.jsonl`, `price_logs/rebound/*.jsonl` | 526k quotes, 2,120 `buy_matched` events |
| `price_logs/twap/trades.jsonl` + `twap_bot.log` | twap_bot paper entries, settlements, session P&L |
| `dry-run.sqlite3`, `multi-dry-run.sqlite3`, `*-dry-run.sqlite3` | simulated signals only (no fills) |
| `window_outcomes.json` / `resolved_windows.json` | mostly `null` winners — local resolution barely captured outcomes |

## 2. Real-money bots (Polymarket wallet, Sep 15–18)

| Bot | Attempts | Filled | USDC spent | Rejected | Unknown/timeout |
|---|---|---|---|---|---|
| bot.py (BTC live) | 428 | 343 | $1,660.30 | 72 | 6 |
| multi_bot.py (5 assets) | 950 | 779 | $3,824.60 | 135 | 10 |
| **Total** | **1,378** | **1,122** | **$5,484.90** | 207 | 16 |

- Fills: 343 of 364 BTC fills went through bot.py; multi_bot filled all five
  assets evenly (e.g. SOL 194, DOGE 174, XRP 162, ETH 169, BNB 80).
- Both directions bought in the same window (reversals) — e.g. Sep 15
  `1789510200`: ETH Up, ETH Down, SOL Up, XRP Down, DOGE Down all in one window.
- Order sizes evolved: $1 → $1.5 → $2/$2.10 → $3 → $5 → $10 per fill.
- Exits: 20 sells in live.sqlite3 (80.73 shares, take-profit/stop-loss,
  Sep 17), 62 sells in multi-live.sqlite3 (246.39 shares). ~80% of filled
  positions were never sold — they ride to settlement.
- **Final P&L for these ~$5.5k of bets is NOT in the local logs**: winners
  weren't recorded (`bottest_markets.winner` is null; outcome JSONs are null).
  Reconcile via the Polymarket data API / wallet `realizedPnl` (bot_gui.py does
  this) — local records only prove what was bought/sold, not the result.

## 3. Live trader on Polysimulator (Sep 19–21)

- 2,321 orders, 2,101 fills, **$18,728.52** notional across BTC/SOL/DOGE/XRP/ETH/BNB.
- Fees: **$194.26**.
- 909 take-profit limit sells queued in `profit_orders` (mostly PENDING at
  snapshot time); 150 markets put on `reversal_watch`.
- Same caveat: settlement results not stored locally.

## 4. Edge & rebound bots (Sep 19–20)

- edge_trader: 29 attempts, 12 fills, $18.70 (17 rejections, incl. a burst of
  `HTTPStatusError`).
- rebound_bot: 20 attempts, 19 fills, $153.55 (avg ~$5/bet, entries like
  Down @ 0.48, Down @ 0.37, Up @ 0.64, Up @ 0.29).

## 5. twap_bot — the only fully settled book (Sep 21 → Sep 25, paper)

| Metric | Value |
|---|---|
| Settled bets | 156 (18 in Sep 21 log + 138 in current log) |
| Wins / losses | 124 / 32 (**79.5% win rate**) |
| Total settled P&L | +$45.92 (Sep 21: +$12.44; Sep 23–25: +$33.48) |
| Current session P&L | **+$25.25** (running since last reset, 595 windows watched) |
| By side | Up 85/113 (75%), Down 39/43 (91%) |
| Per asset (47-bet settled jsonl subset) | ETH +$220, BNB +$19, BTC −$103, SOL −$18, XRP −$60 |
| Avg entry | ~$10.50 cost at 0.29–0.93 ask; model p_up 0.55–0.93 |

Pattern: buys the *leader* side right after a sharp spot move (model p ≈ 0.88–0.93
vs ask 0.66–0.83), betting the TWAP reversion doesn't catch up. Big losers are
when the move keeps going (e.g. −$10.18 on an xrp Up bet).

## 6. Dry runs (no money)

- bot.py dry: 3 signals; multi dry: 4; polysim dry: 58 (51,451 quotes logged);
  opening_bot paper: 6 trades (opening-paper.sqlite3).
- 92,552 lines of BTC quotes in polysimulator/BTC.jsonl alone (9.25k BTC +
  others) — the bulk of `price_logs/` is market data, not bets.

## 7. Key takeaways

1. **~$24.4k total bought across all bots** ($5.48k real wallet + $18.7k
   polysimulator + $0.17k edge/rebound), 3,235 filled bets, plus 156 settled
   paper bets at +$45.92.
2. The one book with known outcomes (twap paper) is profitable: 79.5% win,
   but it's paper and only 156 bets — small sample, one bad session would
   erase it (min session_pnl hit −$204 intraday).
3. **Local logs can't answer "did the real-money bots win"**: settlement
   winners were never persisted (nulls in every outcome file). To close the
   loop: `bot_gui.py` (reads wallet realizedPnl), or data-api.polymarket.com
   trades for the wallet, or re-resolve each slug via the Gamma API
   (`markets?slug=…` → `outcomePrices`) — network to Polymarket endpoints was
   being reset from this machine, so do it from a host that can reach them.
4. 16 "unknown" attempts (7.4% rejected, 16 unresolved submissions) are worth
   manually checking on Polymarket before trusting the totals.
