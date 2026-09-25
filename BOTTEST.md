# Experimental BTC bot

`bottest.py` defaults to dry run. It does not start or change running bot processes.

```powershell
.\.venv\Scripts\python.exe bottest.py
```

The default is $10 per order, a buy signal of **0.88 <= midpoint < 0.89**,
entry after 30 seconds, and no new submissions during the last 10 seconds.
`--min-midpoint` is inclusive; `--max-midpoint` is exclusive. The separate
`--max-price` purchase-price cap remains 0.93, with at most 0.01 above the observed
ask. The selected midpoint band comes from the last-quote heuristic, not confirmed
settlement performance, and is not a guarantee of higher profit.
Positions are held to resolution; this version does not submit exits.

## Execution changes

- One direction per market by default, enforced inside the journal transaction.
  Confirmed no-fill rejections do not count as a position. `--allow-both-directions`
  explicitly restores the ability to buy the other direction.
- Maximum two fresh attempts per direction, five seconds apart, only following
  a confirmed no-fill rejection. Attempt history and the limit survive restarts.
  Legacy attempts without the new audit records remain blocked.
- A maximum five-second book age, at most five seconds of future clock skew,
  and a maximum two-second quote request duration. Invalid and nonfinite
  book values are rejected.
- The ask must be within the signed limit and enough visible ask liquidity must
  exist within that limit to cover the dollar amount. This check cannot guarantee
  liquidity remains available at matching time.
- Read the book again after order preparation. Abandon an unsubmitted order if
  the signal, spread, liquidity, tick size, risk flag, or entry window is no
  longer acceptable.
- Up to three submissions of the **same signed order** on transport failures,
  with a fresh signal check before a retry. An uncertain submission never results
  in a newly signed replacement. Save the SDK-derived order hash before posting.
- Background account reads reconcile new uncertain orders by that exact hash.
  A missing order/404 is not proof of no fill. A partial fill still blocks entry.
  Only an explicitly canceled order with zero matched shares can become retryable.
- Cache market discovery for its five-minute window. Continue recording both
  outcomes before entry and after a fill or rejected attempt. Quote logging still
  requires a valid, fresh two-sided book; unavailable quotes get explicit events.

## Journals, logs, and settlement

`multi_bottest.py` runs the same experimental strategy for SOL, DOGE, XRP, ETH,
and BNB. It accepts the same command-line settings as `bottest.py`:

```powershell
.\.venv\Scripts\python.exe multi_bottest.py
.\.venv\Scripts\python.exe multi_bottest.py --once --amount 3 --max-spread 0.02
```

Both commands above are dry runs. Live mode requires `--live` explicitly.
The multi-asset runner uses `multi-live.sqlite3` for live attempts and
`multi-bottest-dry-run.sqlite3` for dry runs. Use it in place of `multi_bot.py`
when running the experimental strategy. Each asset retains its own market cache
and confirmation timers, and all events go to its own `price_logs/ASSET.jsonl`.
It processes the five assets sequentially, followed by the configured polling delay.

Live mode uses `live.sqlite3` so it respects pre-existing live attempts. Dry run
uses `bottest-dry-run.sqlite3`. New audit and market-resolution tables are prefixed
`bottest_`. Source log files remain `price_logs/BTC.jsonl`; records carry
`strategy_version: bottest-v3`, settings, configured size, book time, executable
depth, skip reasons, and actual paid/share amounts when returned by the exchange.

Use this program as a replacement for the original BTC process, rather than
running both against the same wallet. The original program does not implement
this version's direction lock. `multi_bot.py` continues to import `bot.py`, so it
does not acquire these changes.

During a continuous run, background monitoring checks expired markets for the
CLOB's explicit winning token. Closure or a price close to $1 alone is insufficient.
Resolution is recorded in SQLite and JSONL. Gross P&L assumes holding the logged
fill through resolution. Actual fees, external sells, and redemption cash flows
are not available from these order acknowledgements, so **net P&L stays unknown**.
Reconciled fills without a verified spend also do not receive an invented P&L.
`--once` performs one observation/entry cycle without starting background monitoring.

## Compare strategy variants

Run `--help` for all settings. Examples for **dry-run comparisons**, not validated
profitable settings:

```powershell
.\.venv\Scripts\python.exe bottest.py --max-spread 0.02 --confirmation-seconds 2
.\.venv\Scripts\python.exe bottest.py --entry-delay 120 --max-price 0.90
```

Use separate five-minute windows or fresh dry-run journals for repeated experiments;
the journal intentionally prevents re-entering a window already attempted.
The confirmation timer resets after a broken signal or a gap exceeding the book-age limit.
No narrower price band, later entry, or stop-loss is claimed to improve returns.

View actual logged fills against settlement events:

```powershell
.\.venv\Scripts\python.exe bottest_report.py
.\.venv\Scripts\python.exe bottest_report.py --group-by entry-time
.\.venv\Scripts\python.exe bottest_report.py --group-by strategy
```

The report is read-only and makes no network requests. It deduplicates order IDs,
excludes conflicting resolution labels, and leaves unresolved trades out of the
win-rate denominator. P&L includes only records with sufficient amount data;
the displayed counts identify that subset. Legacy logs lack settlement labels
and post-entry quote coverage, so they cannot establish a winning parameter set.

To evaluate an improvement, collect several days of complete quotes and outcomes,
compare profit after verified fees as well as win rate and drawdown, and reserve
a later period for evaluation. Replay timing must use only data available at the
decision time and account for executable depth and latency. Higher-priced entries
can increase win rate while reducing profitability.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_bottest
```

These tests use synthetic books, temporary journals, a public SDK signing fixture,
and mocked exchange calls. They neither access wallet credentials nor place orders.

API behavior was checked against the installed `py-clob-client-v2==1.1.0` and
the official [order management](https://docs.polymarket.com/trading/manage-orders)
and [fee](https://docs.polymarket.com/trading/fees) documentation.
