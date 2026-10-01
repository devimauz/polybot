"""SOL, DOGE, ETH and BNB bot with per-asset price logs in price_logs/.

For real runs, one MultiBot instance per asset runs on its own worker thread so
the whole basket is polled at the full POLL_SECONDS cadence instead of a long
sequential cycle. All instances share the same journal file (SQLite, WAL) with
immediate transactions, so restart-deduplication and the daily budget still hold
across assets. The synchronous ``cycle()`` keeps the single-instance, per-asset
loop used by the tests.
"""

import argparse
import logging
import os
import threading
import time

import httpx
from dotenv import load_dotenv

from bot import AMOUNT, Bot, D, LOG, ROOT


class MultiBot(Bot):
    assets = ("sol", "doge", "eth", "bnb")
    asset = "sol"
    journal_prefix = "multi-"

    def __init__(self, live, asset="sol"):
        self.asset = asset
        super().__init__(live)
        self.amount = AMOUNT

    def cycle(self):
        """Run one synchronous check of every asset on this instance (tests)."""
        succeeded = True
        for asset in self.assets:
            self.asset = asset
            try:
                self.step()
            except httpx.HTTPError as exc:
                LOG.warning("%s market data failed (%s)", asset.upper(), type(exc).__name__)
                succeeded = False
        return succeeded


def _run_one(bot, poll):
    while not bot._stop.is_set():
        try:
            bot.step()
        except httpx.HTTPError as exc:
            LOG.warning("%s market data failed (%s)", bot.asset.upper(), type(exc).__name__)
        bot._stop.wait(poll)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Submit real orders plus fees")
    parser.add_argument("--once", action="store_true", help="Check all configured markets once")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots, threads = [], []
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.5 <= poll <= 60:
            raise ValueError("Polling must be 0.5-60 seconds")
        for asset in MultiBot.assets:
            bot = MultiBot(args.live, asset)
            bot._stop = threading.Event()
            bots.append(bot)
            if not args.once:
                thread = threading.Thread(target=_run_one, args=(bot, poll), daemon=True, name=f"bot-{asset}")
                thread.start()
                threads.append(thread)
        LOG.info("%s %s; $%s per direction per market plus fees; daily budget $%s; per-asset poll %ss",
                 "LIVE" if args.live else "DRY RUN", "/".join(MultiBot.assets).upper(),
                 AMOUNT, bots[0].daily_budget, poll)
        if args.once:
            ok = True
            for bot in bots:
                try:
                    bot.step()
                except httpx.HTTPError as exc:
                    LOG.warning("%s market data failed (%s)", bot.asset.upper(), type(exc).__name__)
                    ok = False
            return 0 if ok else 1
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s). Check journal and account before retrying.", type(exc).__name__)
        return 1
    finally:
        for bot in bots:
            bot._stop.set()
        for t in threads:
            t.join()
        for bot in bots:
            try:
                bot.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
