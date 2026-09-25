# Rebound bot

`rebound_bot.py` is a separate Polysimulator strategy. It uses the existing
`POLYSIM_API_KEY` in `.env`, with its own journal and logs. It does not change
the entry rules in `bot.py` or `edge_trader.py`.

## Rules

- Monitor each outcome's bid/ask midpoint in the current five-minute market.
- Within a rolling 60-second history, require a decline of at least 15 cents
  from an earlier peak to a later low.
- Require a recovery of at least 8 cents from that low, sustained across at
  least three distinct snapshots and three seconds.
- Buy the recovering outcome only when its ask is between 15 and 65 cents,
  the spread is at most 3 cents, and its book is no more than 5 seconds old.
- Only enter between 30 and 240 seconds after the window starts. A restart
  rebuilds the price history; long data gaps reset confirmation.
- Default stake: $5. Respect the market's minimum share size without raising
  the stake. Maximum buy price: ask plus 2 cents, capped at 65 cents.
- One entry attempt per asset/session total, across both outcomes and restarts.
- After a confirmed fill, place a resting GTC sell for all bought shares at
  20% above the actual fill price, rounded up to the allowed price increment.
  This is a gross target, before fees; it is not guaranteed profit.

There is no stop-loss. A position whose sell never fills remains exposed until
resolution and may lose its stake. This is an experimental pattern rule, not
a calibrated probability model or a strategy validated by the screenshots.
The final-minute exclusion deliberately skips the late spikes in the examples.

## Run

Start in dry-run mode (records signals but submits no orders):

```powershell
uv run --with httpx --with python-dotenv python rebound_bot.py --asset btc
```

Submit simulated orders:

```powershell
uv run --with httpx --with python-dotenv python rebound_bot.py --paper --asset btc --amount 5 --tp-percent 20
```

Use `--asset eth` or another asset supported by `bot.py` to select a market.
`--poll 0.5` is the default pause. API pacing detects the account tier.
`--once` performs one scan; several fresh observations are needed to form a
signal, so one scan will not validate the full signal lifecycle.

The paper journal is `rebound-polysimulator.sqlite3`; dry runs use
`rebound-polysimulator-dry-run.sqlite3`. Quotes go to `price_logs/rebound/`,
and progress messages to `rebound_bot.log`. Keep these files to compare signals
and outcomes before changing parameters. The rate limiter is shared within
one process, not with other bots running on the same API key.

Ctrl+C stops monitoring. Resting sells already submitted to Polysimulator remain
active. Submission errors are logged and uncertain buys are never replayed.
