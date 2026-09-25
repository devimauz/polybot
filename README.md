# BTC five-minute Polymarket bot

Price logs append to `price_logs/BTC.jsonl`, `SOL.jsonl`, `DOGE.jsonl`,
`XRP.jsonl`, `ETH.jsonl`, and `BNB.jsonl`. Each line is a JSON record with UTC
timestamp, market, Up/Down outcome, live/dry-run mode, and event type.
`buy_price` is the best ask; `bid_price` and `sell_price` are the best bid
(a quoted selling price, not an executed sale). Records also include midpoint
and buy limit. Confirmed buys include `fill_price` when fill quantities are
available. Dry-run buys are labeled `buy_simulated` and have no fill price.
Quotes are logged when evaluated for entry, after the initial session delay
and before an outcome has been attempted. No automatic selling is enabled.

Both bots wait until 30 seconds after each five-minute market starts before
placing new buys. Neither bot performs automatic stop-loss selling. Both directions
require a midpoint from 0.86 through 0.93 inclusive; buy limits never exceed 0.93.

## Windows GUI

Double-click `Start BTC GUI.cmd`, or run:

```powershell
python bot_gui.py
```

This opens a native desktop window with cash balance, realized profit/loss,
open-position value, closed-position win rate, a performance chart, searchable
positions, bot orders and CSV export. Account data refreshes every 15 seconds.
The GUI monitors only; continue running `python bot.py --live` separately to trade.
Closing the GUI does not stop a separately running bot.

"Bot markets" matches wallet positions to market/outcome pairs in `live.sqlite3`.
"All BTC positions" includes every retrieved BTC five-minute position. Polymarket
combines trades into positions, so manual trades in the same market/outcome can
affect bot-market totals. Cash is for the entire shared wallet, including funds
used by the multi-asset bot. Unknown attempts are not counted as confirmed fills.

Realized P&L comes from Polymarket's `realizedPnl`; current unrealized P&L is
`currentValue - initialValue`. Claimable holdings remain separate from closed
profits until reflected as closed by the API. The chart and win rate use closed
positions only; the realized total can also include fees or realized components
of current positions. API lag and fee accounting can affect the figures. History
is limited to 10,000 current / 1,000 closed wallet positions, with a warning when
that limit is reached. Refresh failures retain the last snapshot with a warning.
Credentials stay in `.env` and are never displayed or served through a web page.

## Multi-asset bot

`multi_bot.py` follows the current SOL, DOGE, XRP, ETH and BNB five-minute markets,
automatically advancing to each new window. The timestamps in the supplied links
identify example windows, not permanently fixed markets.

```powershell
python multi_bot.py
python multi_bot.py --live
```

The first command is dry run; the second places real orders. It uses the same
local `.env` credentials. Every order is $3 of shares plus fees. It checks all five
assets sequentially, then waits `POLL_SECONDS` (default 0.5); network time
is additional. Each direction can be attempted once per market, so a window
can contain up to ten $3 orders if both directions trigger across all five assets.
Assets without an active matching five-minute market are skipped for that cycle.
There is no daily cap, volume check, share-depth check or profit model.
Separate `multi-live.sqlite3` and `multi-dry-run.sqlite3` journals prevent repeats.
A market-data failure skips that asset for the cycle; confirmed FOK no-fill
rejections continue, while unresolved disconnects remain reserved for review and other trading continues.
Running both scripts can spend funds from the same wallet.

## BTC Strategy

API bot for the requested strategy:

- UP midpoint from 86% through 93% inclusive: buy $2 of UP shares, plus fees.
- DOWN midpoint from 86% through 93% inclusive: buy $2 of DOWN shares, plus fees.
- At most one attempt per outcome per five-minute market. Both outcomes can
  trigger if the market reverses. It holds shares; no automatic selling or redemption.
- Probability means the current best bid/ask midpoint, not a prediction model.
- No volume, expected-profit, or available-share quantity filter. Polymarket
  decides whether the $10 fill-or-kill order can fill at the allowed price.

## $10 market buys

The bot submits dollar-denominated fill-or-kill market buys. It does not reject
them locally based on the order book's `min_order_size` share field. Polymarket
validates each submitted order; the bot never increases the $10 share purchase to meet
a reported share minimum. Actual live acceptance has not been tested here.

## Run on Windows

Dependencies are installed in `.venv` in this workspace. Start without credentials:

```powershell
.\.venv\Scripts\python.exe bot.py
```

One public-data check:

```powershell
.\.venv\Scripts\python.exe bot.py --once
```

For a fresh installation with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dry run records qualifying signals, not simulated fills or profit. Stop with Ctrl+C.

## Live configuration

Create a local `.env` using the fields in `.env.example`. Enter the wallet private
key locally, never in chat or source control. `POLY_FUNDER` is the trading wallet
holding collateral and shares; `POLY_SIGNATURE_TYPE` must match that account.
Supported account types are EOA (0), proxy (1), Gnosis Safe (2), and Deposit Wallet (3).
For type 3, use the Deposit Wallet address as `POLY_FUNDER` and its authorized
owner's private key as `POLY_PRIVATE_KEY`. The SDK handles ERC-1271 signing.

The account must already be funded with the collateral required by Polymarket
(currently pUSD) and have trading approvals configured. The bot derives its API
credentials through the official V2 SDK. It does not fund or approve your wallet.

```powershell
.\.venv\Scripts\python.exe bot.py --live
```

This submits real orders. Live trading has not been tested with a funded account.

## Execution rules

- No daily spending cap. Any old `DAILY_BUDGET_USD` setting in `.env` is ignored.
- Wait 0.5 seconds between cycles; change `POLL_SECONDS` (0.5-60). API response
  time is additional, so this is not a guaranteed two checks per second.
- $10 is the share purchase amount, with fees extra. Keep enough collateral for both.
  Do not set the SDK's `user_usdc_balance` to $1: that shrinks the purchase below
  Polymarket's $1 marketable-buy minimum. SDK rounding still applies.
- Fill-or-kill orders: fill immediately in full or reject; no resting orders.
- Maximum buy price: best ask plus one cent, capped at 0.93 and rounded down to the valid market tick.
- Skip empty books, books over 30 seconds old, spreads above five cents,
  and the last ten seconds of a market. Quotes can move before execution.
- Public data failures retry on the next poll. Confirmed FOK no-fill rejections continue; unresolved disconnects remain reserved for reconciliation while other trading continues.

- `live.sqlite3` records attempts before submission. Restarts and multiple processes
  using the same database cannot repeat the same market/outcome attempt.
- Rejected or uncertain attempts stay recorded to prevent duplicates. Check your Polymarket
  orders/trades before resolving an uncertain attempt. Do not delete the live database
  to retry an order whose result is unknown. Separate copies of this project do not
  share duplicate protection.
- Dry run uses a separate `dry-run.sqlite3` database.

Inspect recorded attempts:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('live.sqlite3'); print(c.execute('SELECT * FROM orders ORDER BY rowid DESC LIMIT 20').fetchall())"
```

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

## API references

- [Official V2 SDK](https://github.com/Polymarket/py-clob-client-v2)
- [V2 migration and collateral](https://docs.polymarket.com/v2-migration)
- [Market discovery](https://docs.polymarket.com/market-data/discover-markets)

Market prices are implied probabilities and do not guarantee winning trades.

### Price log viewer

Double-click **Start Price Log Viewer.cmd** to browse `price_logs/*.jsonl`.
The read-only desktop viewer includes file selection, market/outcome/event/mode
filters, text search (press Enter or Apply), a price chart, and automatic refresh
every two seconds. The chart shows the latest matching market and outcome;
select a market and outcome to inspect an older contract. Prices are dollars
per share, and chart times are UTC. Rows appear newest first in pages of 500.
Double-click a row for its complete JSON, or export all matching rows to a new
JSONL file. Malformed lines are counted and skipped; incomplete writes are retried.
No additional packages are required beyond Python with Tkinter.

To open a different log directory:

```powershell
.\.venv\Scripts\python.exe price_log_viewer.py "C:\path\to\logs"
```

### Disconnect retries

Both bots retry transient submission failures up to three total attempts, waiting
1 then 2 seconds and reusing the exact signed order. Retries stop before the final
10 seconds of a market. An unresolved attempt stays marked unknown and blocks
further buys for that market/direction, while the bot continues other trading.
No new order is created to replace an uncertain submission, including after restart.
Transient failures before submission are retried with fresh quotes on the next cycle.

### Experimental bot

`bottest.py` adds a default one-direction-per-market rule, executable-depth and
freshness checks, continued price logging, bounded rejection retries, background
reconciliation, and settlement tracking. It defaults to dry run. See
[BOTTEST.md](BOTTEST.md) for settings, the read-only performance report, and
validation limits.
