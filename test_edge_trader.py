import json
import time
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

import httpx

from bot import Journal
from edge_trader import EdgeTrader, book_quote, meets_minimum_size


class EdgeTraderTests(unittest.TestCase):
    def test_minimum_shares_with_fixed_stake(self):
        for price, expected in (("0.33", False), ("0.22", True), ("0.2201", False), ("0.20", True)):
            with self.subTest(price=price):
                enough, minimum, shares = meets_minimum_size(
                    {"min_order_size": "5"}, 1.10, Decimal(price))
                self.assertEqual(enough, expected)
                self.assertEqual(minimum, Decimal("5"))
        for minimum in ("NaN", "0", "-5"):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                meets_minimum_size({"min_order_size": minimum}, 1.10, Decimal("0.33"))

    def setUp(self):
        self.trader = EdgeTrader.__new__(EdgeTrader)
        self.trader.api_key = "test-key"
        self.trader.dry_run = False
        self.trader.spent = 0
        self.trader.journal = Journal(":memory:")
        self.addCleanup(self.trader.journal.db.close)
        patch("edge_trader.pace_simulator").start()
        self.addCleanup(patch.stopall)

    def buy(self):
        return self.trader.buy("market", "condition", "Down", Decimal("0.60"), time.time() + 180)

    def test_paper_buy_and_one_entry_per_window(self):
        calls = []

        def handle(request):
            calls.append(request)
            self.assertEqual(str(request.url), "https://api.polysimulator.com/v1/orders")
            self.assertEqual(request.headers["X-API-Key"], "test-key")
            self.assertTrue(request.headers["Idempotency-Key"])
            self.assertEqual(json.loads(request.content), {
                "market_id": "condition", "side": "BUY", "outcome": "Down",
                "amount": "1.65", "price": "0.60", "order_type": "market", "time_in_force": "FOK"})
            return httpx.Response(200, json={"status": "FILLED", "price": "0.60", "quantity": "1.8333"})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            self.trader.http = client
            self.assertTrue(self.buy())
            self.assertFalse(self.trader.buy("market", "condition", "Up", Decimal("0.6"), time.time() + 180))
        self.assertEqual(len(calls), 1)

    def test_timeout_remains_reserved(self):
        self.trader.http = Mock()
        self.trader.http.post.side_effect = httpx.ReadTimeout("lost response")
        with self.assertRaises(RuntimeError):
            self.buy()
        self.assertFalse(self.buy())
        self.assertEqual(self.trader.http.post.call_count, 1)
        self.assertEqual(self.trader.journal.db.execute("SELECT status FROM orders").fetchone()[0], "unknown")

    def test_dry_run_has_no_order(self):
        self.trader.dry_run = True
        self.trader.http = Mock()
        self.assertTrue(self.buy())
        self.trader.http.post.assert_not_called()

    def test_rejection_records_server_reason_and_redacts_key(self):
        request = httpx.Request("POST", "https://api.polysimulator.com/v1/orders")
        response = httpx.Response(400, request=request, json={
            "code": "INSUFFICIENT_LIQUIDITY", "message": "FOK cannot fill",
            "echo": "test-key"})
        self.trader.http = Mock()
        self.trader.http.post.return_value = response
        with self.assertLogs("edge-trader", level="WARNING") as logs:
            self.assertTrue(self.buy())
        status, raw = self.trader.journal.db.execute("SELECT status, detail FROM orders").fetchone()
        detail = json.loads(raw)
        self.assertEqual(status, "rejected")
        self.assertEqual(detail["http_status"], 400)
        self.assertIn("INSUFFICIENT_LIQUIDITY", detail["response"])
        self.assertNotIn("test-key", raw)
        self.assertNotIn("test-key", "".join(logs.output))
        self.assertIn("FOK cannot fill", "".join(logs.output))
        self.assertFalse(self.buy())

    def test_balance_uses_dollars_and_floor_stops(self):
        self.trader.api_get = Mock(return_value={"balance": "69.50"})
        self.assertEqual(self.trader.balance(), 69.5)
        self.trader.discover = Mock()
        with patch("edge_trader.WALLET_FLOOR", 70.00):
            self.assertFalse(self.trader.cycle())
        self.trader.discover.assert_not_called()

    def test_iso_book_and_stale_rejection(self):
        from datetime import datetime
        stamp = "2026-09-19T02:02:42.771325+00:00"
        now = datetime.fromisoformat(stamp).timestamp()
        book = {"timestamp": stamp, "tick_size": "0.01", "asks": [{"price": "0.6", "size": "10"}]}
        self.assertEqual(book_quote(book, now), ((0.6, 10.0), Decimal("0.01")))
        with self.assertRaises(ValueError):
            book_quote(book, now + 31)


if __name__ == "__main__":
    unittest.main()
