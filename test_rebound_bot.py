import json
import unittest
from unittest.mock import Mock, patch

import httpx
from bot import D, Journal, Quote
from rebound_bot import ReboundBot, ReboundSignal


class SignalTests(unittest.TestCase):
    def test_confirmed_rebound(self):
        signal = ReboundSignal()
        result = [signal.update(t, D(p)) for t, p in
                  ((0, "0.50"), (1, "0.40"), (2, "0.30"), (3, "0.20"),
                   (4, "0.29"), (5, "0.30"), (6, "0.29"), (7, "0.31"))]
        self.assertEqual(result, [False] * 7 + [True])

    def test_one_tick_spike_does_not_trigger(self):
        signal = ReboundSignal()
        for t, p in ((0, "0.50"), (1, "0.40"), (2, "0.30"), (3, "0.20"),
                     (4, "0.35"), (5, "0.21"), (6, "0.22"), (7, "0.23")):
            self.assertFalse(signal.update(t, D(p)))

    def test_straight_rise_is_not_rebound(self):
        signal = ReboundSignal()
        for t in range(20):
            self.assertFalse(signal.update(t, D("0.10") + D(t) / 100))

    def test_cached_quotes_and_gaps_do_not_confirm(self):
        signal = ReboundSignal()
        for t, p in ((0, "0.50"), (1, "0.40"), (2, "0.30"), (3, "0.20"), (4, "0.29")):
            self.assertFalse(signal.update(t, D(p)))
        for _ in range(20):
            self.assertFalse(signal.update(4, D("0.29")))
        self.assertFalse(signal.update(20, D("0.35")))
        self.assertEqual(len(signal.history), 1)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.bot = ReboundBot.__new__(ReboundBot)
        self.bot.asset, self.bot.api_key, self.bot.live = "btc", "test-key", True
        self.bot.amount, self.bot.tp_multiplier = D("5"), D("1.20")
        self.bot.condition_id = "cid"
        self.bot.journal = Journal(":memory:")
        self.addCleanup(self.bot.journal.db.close)
        self.bot.log_prices = Mock()
        patch("rebound_bot.time.time", return_value=3100).start()
        patch("bot.pace_simulator").start()
        self.addCleanup(patch.stopall)
        self.requests = []
        self.bot.http = httpx.Client(transport=httpx.MockTransport(self.handle))
        self.addCleanup(self.bot.http.close)
        self.quote = Quote(D("0.39"), D("0.40"), D("0.41"), "0.01", False, D("0.38"))
        self.book = {"timestamp": "3100000", "min_order_size": "5"}

    def handle(self, request):
        self.assertEqual(request.url.host, "api.polysimulator.com")
        body = json.loads(request.content)
        self.requests.append(body)
        if body["side"] == "BUY":
            self.assertEqual(body["amount"], "5")
            self.assertEqual(body["time_in_force"], "FOK")
            return httpx.Response(200, json={"status": "FILLED", "price": "0.40", "quantity": "12.5", "order_id": "buy"})
        self.assertEqual(body["quantity"], "12.5")
        self.assertEqual(body["price"], "0.48")
        self.assertEqual(body["time_in_force"], "GTC")
        return httpx.Response(200, json={"status": "PENDING", "order_id": "sell"})

    def enter(self, outcome="Down", end=3300):
        return self.bot.execute_entry("btc-updown-5m-3000", end, outcome, self.quote, self.book)

    def test_fill_and_actual_fill_based_tp_once(self):
        self.assertTrue(self.enter())
        self.assertFalse(self.enter("Up"))
        self.assertEqual([r["side"] for r in self.requests], ["BUY", "SELL"])

    def test_dry_run_never_submits(self):
        self.bot.live = False
        self.assertTrue(self.enter())
        self.assertEqual(self.requests, [])

    def test_final_minute_and_minimum_size(self):
        self.assertFalse(self.enter(end=3150))
        self.book["min_order_size"] = "100"
        self.assertFalse(self.enter())
        self.assertEqual(self.requests, [])

    def test_timeout_remains_reserved(self):
        self.bot.post_with_retry = Mock(return_value=None)
        self.assertTrue(self.enter())
        self.assertFalse(self.enter())
        self.bot.post_with_retry.assert_called_once()

    def test_skip_log_explains_invalid_snapshot(self):
        self.bot.signal_market = None
        self.bot.signals = {}
        self.bot.next_status = 0
        self.bot.discover = Mock(return_value=("btc-updown-5m-3000", 3300, {"Up": "token"}))
        self.bot.simulator_get = Mock(return_value={"timestamp": "3090000"})
        with self.assertLogs("rebound-bot", level="INFO") as logs:
            self.bot.step()
        self.assertTrue(any("snapshot age 10.00s" in line for line in logs.output))
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
