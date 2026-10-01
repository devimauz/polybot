import unittest
import time
from decimal import Decimal
from unittest.mock import Mock

from bot import Quote
from ninefive_bot import (ABOVE_HOLD_SECONDS, FAV_ASK_MAX, FAV_ASK_MIN,
                          TRADE_WINDOW_SECONDS, NineFiveBot)
from polymarket_feed import MarketFeed


class NineFivePriceSourceTests(unittest.TestCase):
    def test_favorite_trigger_is_above_90c_through_98c(self):
        self.assertFalse(FAV_ASK_MIN < Decimal("0.90") <= FAV_ASK_MAX)
        self.assertTrue(FAV_ASK_MIN < Decimal("0.91") <= FAV_ASK_MAX)
        self.assertTrue(FAV_ASK_MIN < Decimal("0.98") <= FAV_ASK_MAX)
        self.assertFalse(FAV_ASK_MIN < Decimal("0.99") <= FAV_ASK_MAX)

    def test_buys_qualifying_side_only_in_final_minute(self):
        def quote(bid, ask):
            bid, ask = Decimal(bid), Decimal(ask)
            return Quote((bid + ask) / 2, ask, ask, "0.01", False, bid)

        for triggered in ("Up", "Down"):
            with self.subTest(triggered=triggered):
                bot = NineFiveBot.__new__(NineFiveBot)
                bot.asset = "btc"
                bot.condition_id = "condition"
                bot.journal = Mock()
                bot.journal.seen_market.return_value = False
                bot.feed = Mock()
                bot.above_since = {"Up": None, "Down": None}
                bot.timer_market = None
                other = "Down" if triggered == "Up" else "Up"
                bot.feed.quotes.return_value = {
                    triggered: quote("0.94", "0.95"),
                    other: quote("0.04", "0.05"),
                }
                bot.discover = Mock(return_value=(
                    "btc-updown-5m-1000", 1000.0 + TRADE_WINDOW_SECONDS + 1,
                    {"Up": "up-token", "Down": "down-token"}))
                bot.log_prices = Mock()
                bot._buy_leg = Mock(return_value=True)

                with unittest.mock.patch("ninefive_bot.time.time", return_value=1000.0):
                    bot.step()
                bot._buy_leg.assert_not_called()
                bot.discover.return_value = (
                    "btc-updown-5m-1000", 1000.0 + TRADE_WINDOW_SECONDS,
                    {"Up": "up-token", "Down": "down-token"})
                with unittest.mock.patch("ninefive_bot.time.time", return_value=1000.0):
                    bot.step()
                bot._buy_leg.assert_not_called()
                with unittest.mock.patch(
                        "ninefive_bot.time.time",
                        return_value=1000.0 + ABOVE_HOLD_SECONDS + 0.01):
                    bot.step()

                bot._buy_leg.assert_called_once()
                self.assertEqual(bot._buy_leg.call_args.args[3], triggered)

    def test_quotes_come_from_polymarket_feed(self):
        bot = NineFiveBot.__new__(NineFiveBot)
        bot.feed = Mock()
        expected = {"Up": object(), "Down": object()}
        bot.feed.quotes.return_value = expected
        tokens = {"Up": "up-token", "Down": "down-token"}

        result = bot.polymarket_quotes(tokens)

        bot.feed.subscribe.assert_called_once_with(tokens)
        bot.feed.quotes.assert_called_once_with()
        self.assertIs(result, expected)

    def test_initial_snapshot_without_event_type(self):
        feed = MarketFeed()
        tokens = {"up": "Up", "down": "Down"}
        feed.tokens = tokens
        feed.generation = 1
        feed._handle({"asset_id": "up", "bids": [{"price": "0.94", "size": "1"}],
                      "asks": [{"price": "0.95", "size": "1"}]}, tokens, 1)
        self.assertEqual(str(feed.quotes()["Up"].ask), "0.95")


if __name__ == "__main__":
    unittest.main()
