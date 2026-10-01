"""NineFive Polymarket variant with a $1 stake and no stop loss.

This keeps positions until market settlement. The default loop delay is 0.1
seconds. POLYMARKET_POLL_SECONDS can still override that delay when explicitly
set in the environment.

Dry run (default):
  python ninefive_polymarket_bot_no_stoploss.py

Real-money trading:
  python ninefive_polymarket_bot_no_stoploss.py --live
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv

import ninefive_bot as strategy
import ninefive_polymarket_bot as base
from bot import Bot, D, LOG, ROOT


# The base implementation consults these module globals for both dry-run and
# live sizing, so set them before any bot instances are created.
base.STAKE = D("1")
base.DEFAULT_POLL_SECONDS = 0.1
# The shared signal uses a strict lower comparison. With one-cent market ticks,
# > $0.50 means the first eligible price is $0.51.
base.FAV_ASK_MIN = D("0.50")
base.FAV_ASK_MAX = D("0.98")
strategy.FAV_ASK_MIN = D("0.50")
strategy.FAV_ASK_MAX = D("0.98")
base.__doc__ = __doc__
_BasePolymarketBot = base.NineFivePolymarketBot


class NineFivePolymarketBotNoStopLoss(_BasePolymarketBot):
    """$1 NineFive bot that holds every filled position to settlement."""

    amount = D("1")
    journal_prefix = "ninefive-polymarket-no-stoploss-"
    price_log_dir = ROOT / "price_logs" / "ninefive-polymarket-no-stoploss"

    def _buy_leg(self, slug, end, tokens, outcome, max_price, slippage=None):
        """Submit either Up or Down without bidding above the $0.98 cap."""
        return super()._buy_leg(slug, end, tokens, outcome,
                                D("0.98"), D("0"))

    def step(self):
        try:
            self.audit_quotes()
            # Skip _BasePolymarketBot.step(), which is where stop-loss
            # monitoring is invoked, and call NineFiveBot.step() directly.
            return super(_BasePolymarketBot, self).step()
        finally:
            self.check_settlements()


# Reuse the original command-line runner and dashboard with this variant.
base.NineFivePolymarketBot = NineFivePolymarketBotNoStopLoss


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Place REAL orders using a fixed $1 stake (default: dry run)")
    parser.add_argument("--asset", choices=Bot.assets,
                        help="Trade one asset only (default: all assets)")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    parser.add_argument("--gui", action="store_true",
                        help="Show the direct Polymarket terminal dashboard")
    args = parser.parse_args()
    assets = (args.asset,) if args.asset else Bot.assets
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(
            ROOT / "ninefive_polymarket_bot_no_stoploss.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots = []
    try:
        poll = float(os.environ.get("POLYMARKET_POLL_SECONDS",
                                    str(base.DEFAULT_POLL_SECONDS)))
        if not 0.01 <= poll <= 60:
            raise ValueError("POLYMARKET_POLL_SECONDS must be between 0.01 and 60")
        args.poll_seconds = poll
        if args.gui:
            base.run_dashboard(args, assets)
            return 0
        bots = [NineFivePolymarketBotNoStopLoss(args.live, asset)
                for asset in assets]
        LOG.warning(
            "%s: fixed $%s direct Polymarket orders; no stop loss; polling %.3fs",
            "REAL MONEY" if args.live else "DRY RUN", base.STAKE, poll)
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            while True:
                futures = [(bot, workers.submit(bot.step)) for bot in bots]
                for bot, future in futures:
                    try:
                        future.result()
                    except httpx.HTTPError as exc:
                        LOG.warning("%s request failed (%s)", bot.asset.upper(),
                                    type(exc).__name__)
                if args.once:
                    break
                time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        for bot in bots:
            bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
