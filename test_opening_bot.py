import json
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

import httpx

from opening_bot import OpeningBot, profit_target


class OpeningBotTests(unittest.TestCase):
    def setUp(self):
        patch.dict("os.environ", {"POLYSIM_API_KEY": "test-key"}).start()
        patch("opening_bot.pace_simulator").start()
        patch("bot.pace_simulator").start()
        patch("opening_bot.time.time", return_value=3001).start()
        self.addCleanup(patch.stopall)
        self.bot = OpeningBot(paper=True, db_path=":memory:")
        self.addCleanup(self.bot.close)
        self.bot.market = ("btc-updown-5m-3000", 3300, {"Up": "up", "Down": "down"})
        self.bot.condition_id = "condition"
        self.bot.http.close()
        self.orders = []
        self.bot.http = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        self.assertEqual(request.url.host, "api.polysimulator.com")
        self.assertEqual(request.headers["X-API-Key"], "test-key")
        if request.method == "GET":
            return httpx.Response(200, json={"timestamp": "3001000", "tick_size": "0.01",
                                            "min_order_size": "5", "asks": [{"price": "0.50", "size": "100"}]})
        body = json.loads(request.content)
        self.orders.append(body)
        self.assertTrue(request.headers["Idempotency-Key"])
        if body["side"] == "BUY":
            return httpx.Response(200, json={"status": "FILLED", "price": "0.50", "quantity": "10", "order_id": "buy"})
        return httpx.Response(200, json={"status": "OPEN", "order_id": "sell"})

    def test_both_entries_then_server_side_profit_orders(self):
        self.bot.step()
        self.assertEqual([x["side"] for x in self.orders], ["BUY", "BUY", "SELL", "SELL"])
        self.assertEqual([x["outcome"] for x in self.orders], ["Up", "Down", "Up", "Down"])
        for order in self.orders[2:]:
            self.assertEqual(order["price"], "0.55")
            self.assertEqual(order["quantity"], "10")
            self.assertEqual(order["order_type"], "limit")
            self.assertEqual(order["time_in_force"], "GTC")
        self.bot.step()
        self.assertEqual(len(self.orders), 4)

    def test_profit_target_rounds_up_to_tick(self):
        self.assertEqual(profit_target(Decimal("0.51"), Decimal("0.01")), Decimal("0.57"))
        self.assertEqual(profit_target(Decimal("0.51"), Decimal("0.01"),
                                       Decimal("0.05")), Decimal("0.54"))

    def test_midwindow_does_not_buy(self):
        with patch("opening_bot.time.time", return_value=3031):
            self.bot.step()
        self.assertEqual(self.orders, [])

    def test_dry_run_never_submits(self):
        self.bot.paper = False
        self.bot.step()
        self.assertEqual(self.orders, [])
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM trades WHERE entry_state='dry-run'").fetchone()[0], 2)

    def test_lost_response_does_not_duplicate_entry(self):
        self.bot.http.close()
        calls = []
        def timeout(request):
            if request.method == "GET":
                return self.handle(request)
            calls.append(request)
            raise httpx.ReadTimeout("lost")
        self.bot.http = httpx.Client(transport=httpx.MockTransport(timeout))
        self.bot.step()
        self.bot.step()
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.bot.db.execute("SELECT COUNT(*) FROM trades WHERE entry_state='unknown'").fetchone()[0], 2)

    def test_pending_profit_work_survives_next_cycle(self):
        self.bot.enter(self.bot.market[0], 3300, "Up", "up")
        self.assertEqual(len(self.orders), 1)
        with patch("opening_bot.time.time", return_value=3050):
            self.bot.step()
        self.assertEqual([x["side"] for x in self.orders], ["BUY", "SELL"])

    def test_direct_entry_skips_book_and_uses_contract_ceiling(self):
        get_calls = 0
        original = self.bot.http

        def direct_handler(request):
            nonlocal get_calls
            if request.method == "GET":
                get_calls += 1
            return self.handle(request)

        original.close()
        self.bot.http = httpx.Client(transport=httpx.MockTransport(direct_handler))
        self.bot.enter(self.bot.market[0], 3300, "Up", "up", direct_tick="0.01")
        self.assertEqual(get_calls, 0)
        self.assertEqual(self.orders[0]["price"], "0.99")

    def test_direct_entry_honors_absolute_price_cap(self):
        self.bot.enter(self.bot.market[0], 3300, "Up", "up",
                       direct_tick="0.01", direct_limit=Decimal("0.90"))
        self.assertEqual(self.orders[0]["price"], "0.90")


if __name__ == "__main__":
    unittest.main()
