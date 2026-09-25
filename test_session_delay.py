import unittest
from unittest.mock import Mock, patch

from botqwenlocal import Bot, Journal
from multi_bot import MultiBot
from test_bot import book


class SessionDelayTests(unittest.TestCase):
    def test_each_session_waits_sixty_seconds(self):
        for bot_type in (Bot, MultiBot):
            for live in (False, True):
                with self.subTest(bot=bot_type.__name__, live=live):
                    bot = bot_type.__new__(bot_type)
                    bot.live = live
                    bot.journal = Journal(":memory:")
                    self.addCleanup(bot.journal.db.close)
                    bot.log_prices = Mock()
                    bot.client = Mock()
                    bot.client.post_order.return_value = {"success": True, "status": "matched"}
                    bot.check_stop_losses = Mock()
                    bot.get = Mock()
                    for start in (3000, 3300):
                        slug = f"market-{start}"
                        bot.discover = Mock(return_value=(slug, start + 300, {"Up": "123"}))
                        bot.get.reset_mock()
                        bot.client.reset_mock()
                        bot.check_stop_losses.reset_mock()
                        with patch("bot.time.time", return_value=start + 59.999), patch("bot.triggered", return_value=True):
                            bot.step()
                        bot.get.assert_not_called()
                        bot.client.create_market_order.assert_not_called()
                        bot.client.post_order.assert_not_called()
                        self.assertFalse(bot.journal.seen(slug, "Up"))
                        bot.check_stop_losses.assert_not_called()
                        with patch("bot.time.time", return_value=start + 60), patch("bot.triggered", return_value=True):
                            bot.get.return_value = book()
                            bot.step()
                        self.assertTrue(bot.journal.seen(slug, "Up"))
                        self.assertEqual(bot.client.post_order.call_count, int(live))


if __name__ == "__main__":
    unittest.main()
