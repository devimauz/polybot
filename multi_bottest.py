"""SOL, DOGE, ETH and BNB runner using bottest.py. Dry run by default."""

import logging
import os
import time

import httpx
from dotenv import load_dotenv

from bottest import AMOUNT, Bot, LOG, ROOT, STRATEGY_VERSION, argument_parser, settings_from_args


class MultiBot(Bot):
    amount = AMOUNT
    assets = ("sol", "doge", "eth", "bnb")
    asset = "sol"
    journal_prefix = "multi-"

    def __init__(self, live, settings=None, amount=AMOUNT):
        super().__init__(live, settings, amount)
        self.asset_states = {}

    def cycle(self):
        succeeded = True
        for asset in self.assets:
            self.asset = asset
            # Each asset has its own cached market and signal-confirmation timers.
            self.market, self.signals = self.asset_states.get(asset, (None, {}))
            try:
                self.step()
            except httpx.HTTPError as exc:
                LOG.warning("%s market data failed (%s)", asset.upper(), type(exc).__name__)
                succeeded = False
            finally:
                self.asset_states[asset] = (self.market, self.signals)
        return succeeded

    def log_prices(self, slug, outcome, quote, event="quote", response=None, **extra):
        # Background reconciliation can process any asset in the shared journal.
        # Route by the market being recorded, not the foreground cycle's asset.
        asset, separator, _ = slug.partition("-updown-5m-")
        previous = self.asset
        if separator and asset in self.assets:
            self.asset = asset
        try:
            return super().log_prices(slug, outcome, quote, event, response, **extra)
        finally:
            self.asset = previous


def main():
    load_dotenv(ROOT / ".env")
    args = argument_parser(__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = None
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.5 <= poll <= 60:
            raise ValueError("Polling must be 0.5-60 seconds")
        settings = settings_from_args(args)
        bot = MultiBot(args.live, settings, args.amount)
        if not args.once:
            bot.start_maintenance()
        LOG.info("%s %s %s; $%s per order plus fees; one direction=%s; cycle delay=%ss",
                 STRATEGY_VERSION, "LIVE" if args.live else "DRY RUN",
                 "/".join(bot.assets).upper(), bot.amount, settings.one_direction, poll)
        LOG.info("Buy signal: %s <= midpoint < %s; maximum purchase price=%s",
                 settings.min_midpoint, settings.max_midpoint, settings.max_price)
        while True:
            succeeded = bot.cycle()
            if args.once:
                return 0 if succeeded else 1
            time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s). Check journal and account before retrying.", type(exc).__name__)
        return 1
    finally:
        if bot:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
