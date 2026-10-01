import time
import unittest
from unittest.mock import Mock

import httpx

from botqwenlocal import Bot, D, Journal
from multi_bot import MultiBot
from test_bot import book


class MultiBotTests(unittest.TestCase):
    def setUp(self):
        self.bot = MultiBot.__new__(MultiBot)
        self.bot.live = True
        self.bot.journal = Journal(":memory:")
        self.bot.log_prices = Mock()
        self.bot.client = Mock()
        self.bot.client.post_order.return_value = {"success": True, "status": "matched"}
        self.bot.get = Mock(return_value=book())
        self.trigger = unittest.mock.patch("bot.triggered", return_value=True)
        self.trigger.start()
        self.addCleanup(self.trigger.stop)
        self.bot.discover = lambda now: (f"{self.bot.asset}-market", time.time() + 100, {"Up": self.bot.asset})

    def tearDown(self):
        self.bot.journal.db.close()

    def test_four_asset_orders_and_restart_protection(self):
        self.assertEqual(self.bot.assets, ("sol", "doge", "eth", "bnb"))
        self.assertTrue(self.bot.cycle())
        self.assertEqual(self.bot.client.post_order.call_count, 4)
        self.assertEqual([call.args[0].token_id for call in self.bot.client.create_market_order.call_args_list], ["sol", "doge", "eth", "bnb"])
        for call in self.bot.client.create_market_order.call_args_list:
            self.assertEqual(call.args[0].amount, float(MultiBot.amount))
            self.assertEqual(call.args[0].user_usdc_balance, 0)
            self.assertEqual(call.args[0].order_type, "FOK")
        self.assertEqual(self.bot.journal.db.execute("SELECT SUM(amount) FROM orders").fetchone()[0], float(MultiBot.amount * 4))
        self.bot.cycle()
        self.assertEqual(self.bot.client.post_order.call_count, 4)

    def test_data_failure_does_not_skip_other_assets(self):
        self.bot.step = Mock(side_effect=[httpx.ConnectError("offline")] + [None] * 3)
        self.assertFalse(self.bot.cycle())
        self.assertEqual(self.bot.step.call_count, 4)

    def test_unknown_order_stops_cycle(self):
        self.bot.client.post_order.side_effect = ValueError("invalid submission")
        with self.assertRaises(RuntimeError):
            self.bot.cycle()
        self.assertEqual(self.bot.client.post_order.call_count, 1)

    def test_fok_rejection_continues_to_other_assets(self):
        from py_clob_client_v2.exceptions import PolyApiException
        error = PolyApiException(httpx.Response(400, json={"error": "order couldn't be fully filled. FOK orders are fully filled or killed."}))
        self.bot.client.post_order.side_effect = [error] + [{"success": True, "status": "matched"}] * 3
        self.assertTrue(self.bot.cycle())
        self.assertEqual(self.bot.client.post_order.call_count, 4)
        self.bot.cycle()
        self.assertEqual(self.bot.client.post_order.call_count, 4)
        self.assertEqual(self.bot.journal.db.execute("SELECT status FROM orders WHERE market='sol-market'").fetchone()[0], "rejected")


if __name__ == "__main__":
    unittest.main()
