import json
import time
import unittest
from unittest.mock import patch

import httpx
from bot import Bot, Journal


class ProfitTests(unittest.TestCase):
    def test_fill_based_profit_persisted_and_submitted_once(self):
        bot = Bot.__new__(Bot)
        bot.asset, bot.api_key, bot.live = "sol", "test-key", True
        bot.journal = Journal(":memory:")
        self.addCleanup(bot.journal.db.close)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            calls.append(body)
            self.assertEqual(body["side"], "SELL")
            self.assertEqual(body["price"], "0.97")
            self.assertEqual(body["quantity"], "50")
            self.assertEqual(body["time_in_force"], "GTC")
            return httpx.Response(200, json={"status": "OPEN", "order_id": "tp"})

        market = "sol-updown-5m-test"
        bot.journal.reserve(market, "Up")
        bot.journal.record_profit(market, "Up", "cid", time.time() + 100, "0.01",
                                  {"takingAmount": "50", "makingAmount": "5"})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client, patch("bot.pace_simulator"):
            bot.http = client
            bot.place_profits()
            bot.place_profits()
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
