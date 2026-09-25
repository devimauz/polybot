import tempfile
import time
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock

from botqwenlocal import AMOUNT, Bot, D, Journal, quote_book, triggered


def book(minimum="1"):
    return {"timestamp": str(int(time.time() * 1000)),
            "bids": [{"price": "0.81", "size": "10"}],
            "asks": [{"price": "0.84", "size": "1000"}, {"price": "0.83", "size": "1000"}],
            "tick_size": "0.01", "min_order_size": minimum, "neg_risk": False}


class StrategyTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        for outcome in ("Up", "Down"):
            for price, expected in (("0.8599", False), ("0.86", True),
                                    ("0.90", True), ("0.93", True), ("0.9301", False)):
                with self.subTest(outcome=outcome, price=price):
                    self.assertEqual(triggered(outcome, D(price)), expected)

    def test_buy_limit_capped_at_upper_bound(self):
        b = book()
        b["bids"] = [{"price": "0.92", "size": "100"}]
        b["asks"] = [{"price": "0.94", "size": "100"}]
        self.assertEqual(quote_book(b, time.time()).limit, D("0.93"))


    def test_unsorted_asks_and_price_limit(self):
        q = quote_book(book(), time.time())
        self.assertEqual(q.probability, D("0.82"))
        self.assertEqual(q.limit, D("0.84"))

    def test_stale_and_empty_books(self):
        b = book()
        b["timestamp"] = "0"
        with self.assertRaises(ValueError):
            quote_book(b, time.time())
        b = book()
        b["asks"] = []
        with self.assertRaises(ValueError):
            quote_book(b, time.time())

    def test_wide_spread(self):
        b = book()
        b["bids"][0]["price"] = "0.5"
        with self.assertRaises(ValueError):
            quote_book(b, time.time())


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.temp.name) / "test.sqlite3"
        self.bot = Bot.__new__(Bot)
        self.bot.live = False
        self.bot.journal = Journal(self.path)
        self.bot.discover = Mock(return_value=("market", time.time() + 100, {"Up": "123"}))
        self.bot.get = Mock(return_value=book())
        self.trigger = unittest.mock.patch("bot.triggered", return_value=True)
        self.trigger.start()
        self.addCleanup(self.trigger.stop)
        self.bot.log_prices = Mock()
        self.bot.client = Mock()

    def tearDown(self):
        self.bot.journal.db.close()
        self.temp.cleanup()

    def test_dry_run_never_posts_and_deduplicates(self):
        self.bot.step()
        self.bot.step()
        self.bot.client.post_order.assert_not_called()
        self.assertEqual(self.bot.get.call_count, 1)

    def test_market_buy_not_blocked_by_reported_share_minimum(self):
        self.bot.live = True
        self.bot.client.post_order.return_value = {"success": True, "status": "matched"}
        self.bot.get.return_value = book("5")
        self.bot.step()
        self.assertTrue(self.bot.journal.seen("market", "Up"))
        self.bot.client.post_order.assert_called_once()
        args = self.bot.client.create_market_order.call_args.args[0]
        self.assertEqual(args.amount, float(AMOUNT))
        self.assertEqual(args.user_usdc_balance, 0)
        self.assertEqual(args.order_type, "FOK")

    def test_closing_market_skips(self):
        self.bot.discover.return_value = ("market", time.time() + 5, {"Up": "123"})
        self.bot.step()
        self.assertFalse(self.bot.journal.seen("market", "Up"))

    def test_no_daily_cap_and_restart_deduplication(self):
        j = self.bot.journal
        self.assertTrue(j.reserve("market", "Up"))
        self.assertTrue(j.reserve("market", "Down"))
        self.assertTrue(j.reserve("next-market", "Up"))
        self.assertAlmostEqual(j.db.execute("SELECT SUM(amount) FROM orders").fetchone()[0], float(AMOUNT * 3))
        j.db.close()
        self.bot.journal = Journal(self.path)
        self.assertFalse(self.bot.journal.reserve("market", "Up"))

    @unittest.mock.patch("bot.time.sleep")
    def test_timeout_retries_same_order_then_keeps_reservation(self, sleep):
        self.bot.live = True
        self.bot.client.post_order.side_effect = TimeoutError()
        self.bot.step()
        self.bot.step()
        self.assertEqual(self.bot.client.post_order.call_count, 3)
        self.bot.client.create_market_order.assert_called_once()
        status = self.bot.journal.db.execute("SELECT status FROM orders").fetchone()[0]
        self.assertEqual(status, "unknown")

    def test_confirmed_fok_rejection_continues_without_repeating(self):
        import httpx
        from py_clob_client_v2.exceptions import PolyApiException
        self.bot.live = True
        error = {"error": "order couldn't be fully filled. FOK orders are fully filled or killed.", "orderID": "test-id"}
        self.bot.client.post_order.side_effect = PolyApiException(httpx.Response(400, json=error))
        self.bot.step()
        self.bot.step()
        self.assertEqual(self.bot.client.post_order.call_count, 1)
        self.assertEqual(self.bot.journal.db.execute("SELECT status FROM orders").fetchone()[0], "rejected")

    def test_fok_rejection_in_response_continues(self):
        self.bot.live = True
        self.bot.client.post_order.return_value = {"success": False, "errorMsg": "order couldn't be fully filled. FOK orders are fully filled or killed."}
        self.bot.step()
        self.assertEqual(self.bot.journal.db.execute("SELECT status FROM orders").fetchone()[0], "rejected")

    def test_other_api_errors_still_stop(self):
        import httpx
        from py_clob_client_v2.exceptions import PolyApiException
        self.bot.live = True
        self.bot.client.post_order.side_effect = PolyApiException(httpx.Response(400, json={"error": "not enough balance / allowance"}))
        with self.assertRaises(RuntimeError):
            self.bot.step()

    def test_live_order_amount_and_fok(self):
        self.bot.live = True
        self.bot.client.post_order.return_value = {"success": True, "status": "matched"}
        self.bot.step()
        args, options = self.bot.client.create_market_order.call_args.args
        self.assertEqual(args.amount, float(AMOUNT))
        self.assertEqual(args.user_usdc_balance, 0)
        self.assertEqual(args.price, 0.84)
        self.assertEqual(args.order_type, "FOK")
        self.assertEqual(self.bot.client.post_order.call_args.args[1], "FOK")

    def test_insufficient_visible_shares_does_not_block_order(self):
        self.bot.get.return_value["asks"] = [{"price": "0.83", "size": "1"}]
        self.bot.live = True
        self.bot.client.post_order.return_value = {"success": True, "status": "matched"}
        self.bot.step()
        self.bot.client.create_market_order.assert_called_once()
        self.bot.client.post_order.assert_called_once()
        self.assertEqual(self.bot.client.create_market_order.call_args.args[0].amount, float(AMOUNT))
        self.assertEqual(self.bot.client.post_order.call_args.args[1], "FOK")

    def test_discovery_does_not_require_volume_or_fetch_trade_data(self):
        start = int(time.time()) // 300 * 300
        row = {"slug": f"btc-updown-5m-{start}", "active": True,
               "acceptingOrders": True, "closed": False,
               "endDate": datetime.fromtimestamp(start + 300, timezone.utc).isoformat(),
               "outcomes": '["Up", "Down"]', "clobTokenIds": '["123", "456"]'}
        self.bot.discover = Bot.discover.__get__(self.bot)
        self.bot.get.return_value = [row]
        for volume in (None, "0", "17.96"):
            with self.subTest(volume=volume):
                row["volumeNum"] = volume
                self.bot.get.reset_mock()
                self.assertIsNotNone(self.bot.discover(start + 50))
                self.bot.get.assert_called_once_with("https://gamma-api.polymarket.com/markets", slug=row["slug"])


    def test_does_not_require_1000_dollars_of_asks(self):
        self.bot.get.return_value["asks"] = [{"price": "0.83", "size": "10"}]
        self.bot.step()
        self.assertTrue(self.bot.journal.seen("market", "Up"))


    def test_price_limit_ignores_large_distant_asks(self):
        b = book()
        b["asks"] = [{"price": "0.83", "size": "10"}, {"price": "0.90", "size": "100000"}]
        b["bids"] = [{"price": "0.81", "size": "100000"}]
        self.assertEqual(quote_book(b, time.time()).limit, D("0.84"))


if __name__ == "__main__":
    unittest.main()
