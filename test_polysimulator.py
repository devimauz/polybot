import json
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import httpx

from bot import Bot, Journal, quote_book


class PolysimulatorTests(unittest.TestCase):
    def test_simple_midpoint_threshold_both_sides(self):
        from bot import triggered, D
        for side in ("Up", "Down"):
            for price, expected in (("0.8199", False), ("0.82", True), ("0.95", True), ("0.97", False)):
                self.assertEqual(triggered(side, D(price)), expected)

    def test_pacing_respects_both_tier_limits(self):
        from bot import simulator_interval
        tiers = [{"tier": "pro", "requests_per_second": 30, "requests_per_minute": 1800},
                 {"tier": "free", "requests_per_second": 2, "requests_per_minute": 120}]
        self.assertAlmostEqual(simulator_interval({"tier": "pro"}, tiers), 1.1 / 30)
        self.assertAlmostEqual(simulator_interval({"tier": "free"}, tiers), 0.55)
        tiers[0]["requests_per_minute"] = 600
        self.assertAlmostEqual(simulator_interval({"tier": "pro"}, tiers), 0.11)
        tiers[0]["requests_per_second"] = 0
        with self.assertRaises(ValueError):
            simulator_interval({"tier": "pro"}, tiers)

    def test_simple_entries_both_directions_once(self):
        from decimal import Decimal
        self.bot.live = False
        self.bot.discover = Mock(return_value=("market", time.time() + 100, {"Up": "Up", "Down": "Down"}))
        self.bot.log_prices = Mock()
        prices = {"Up": Decimal("0.81"), "Down": Decimal("0.19")}
        def book(path, token_id):
            mid = prices[token_id]
            return {"timestamp": str(time.time() * 1000), "tick_size": "0.01",
                    "bids": [{"price": str(mid - Decimal("0.01")), "size": "100"}],
                    "asks": [{"price": str(mid + Decimal("0.01")), "size": "100"}]}
        self.bot.simulator_get = Mock(side_effect=book)
        self.bot.step()
        self.assertFalse(self.bot.journal.seen_market("market"))
        prices.update(Up=Decimal("0.82"), Down=Decimal("0.18"))
        self.bot.step()
        self.assertTrue(self.bot.journal.seen("market", "Up"))
        self.assertFalse(self.bot.journal.seen("market", "Down"))
        prices.update(Up=Decimal("0.18"), Down=Decimal("0.82"))
        self.bot.step()
        prices.update(Up=Decimal("0.40"), Down=Decimal("0.60"))
        self.bot.step()
        self.bot.step()
        self.assertEqual(self.bot.journal.db.execute("SELECT outcome,amount FROM orders ORDER BY rowid").fetchall(),
                         [("Up", 10.0), ("Down", 10.0)])

    def test_one_entry_per_asset_session(self):
        journal = self.bot.journal
        self.assertTrue(journal.reserve("btc-updown-5m-3000", "Up", once_per_market=True))
        journal.finish("btc-updown-5m-3000", "Up", "rejected", "test")
        self.assertFalse(journal.reserve("btc-updown-5m-3000", "Down", once_per_market=True))
        self.assertTrue(journal.reserve("sol-updown-5m-3000", "Down", once_per_market=True))
        self.assertTrue(journal.reserve("btc-updown-5m-3300", "Down", once_per_market=True))

    def setUp(self):
        self.bot = Bot.__new__(Bot)
        self.bot.api_key = "test-key"
        self.bot.asset = "sol"
        self.bot.condition_id = "condition"
        self.bot.journal = Journal(":memory:")
        self.addCleanup(self.bot.journal.db.close)
        self.pacing = patch("bot.pace_simulator").start()
        self.addCleanup(patch.stopall)

    def test_discovery_maps_nested_outcomes_without_reordering(self):
        start = int(time.time()) // 300 * 300
        slug = f"sol-updown-5m-{start}"
        row = {"slug": slug, "active": True, "condition_id": "condition",
               "end_date": datetime.fromtimestamp(start + 300, timezone.utc).isoformat(),
               "markets": [{"condition_id": "condition", "outcomes": '["Down", "Up"]',
                            "token_ids": '["down-token", "up-token"]'}]}
        self.bot.simulator_get = Mock(return_value={"markets": [row]})
        market = self.bot.discover(start + 60)
        self.assertEqual(market[2], {"Down": "down-token", "Up": "up-token"})
        self.bot.simulator_get.assert_called_once_with("markets/updown", asset="SOL", interval="5M")
        self.assertEqual(self.bot.discover(start + 61), market)
        self.assertEqual(self.bot.simulator_get.call_count, 1)
        self.assertIsNone(self.bot.discover(start + 301))
        self.assertEqual(self.bot.simulator_get.call_count, 2)

    def test_paper_order_payload_and_fill(self):
        body = {"market_id": "condition", "side": "BUY", "outcome": "Down",
                "amount": "5", "price": "0.93", "order_type": "market", "time_in_force": "FOK"}

        def handle(request):
            self.assertEqual(str(request.url), "https://api.polysimulator.com/v1/orders")
            self.assertEqual(request.headers["X-API-Key"], "test-key")
            self.assertTrue(request.headers["Idempotency-Key"])
            self.assertEqual(json.loads(request.content), body)
            return httpx.Response(200, json={"status": "FILLED", "price": "0.90",
                                             "quantity": "5.3763", "order_id": "paper-1"})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            self.bot.http = client
            response = self.bot.post_with_retry(body, time.time() + 100)
        self.assertEqual(response["status"], "matched")
        self.assertEqual(response["takingAmount"], "5.3763")
        self.assertEqual(response["orderID"], "paper-1")

    def test_sell_fill_may_improve_above_limit(self):
        body = {"market_id": "condition", "side": "SELL", "outcome": "Down",
                "quantity": "10", "price": "0.45", "order_type": "market",
                "time_in_force": "FOK"}

        def handle(request):
            return httpx.Response(200, json={"status": "FILLED", "price": "0.46",
                                             "quantity": "10", "order_id": "paper-sell"})

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            self.bot.http = client
            response = self.bot.post_with_retry(body, time.time() + 100)
        self.assertEqual(response["status"], "matched")
        self.assertEqual(response["makingAmount"], "4.60")

    def prepare_step(self, live):
        from decimal import Decimal
        self.bot.live = live
        self.bot.journal.observe_reversal("market", "Down", Decimal("0.80"))
        self.bot.discover = Mock(return_value=("market", time.time() + 100, {"Up": "up", "Down": "down"}))
        def book(path, token_id):
            bid, ask = ("0.81", "0.83") if token_id == "up" else ("0.17", "0.19")
            return {"timestamp": str(time.time() * 1000), "tick_size": "0.01",
                    "bids": [{"price": bid, "size": "100"}], "asks": [{"price": ask, "size": "100"}]}
        self.bot.simulator_get = Mock(side_effect=book)
        self.bot.log_prices = Mock()

    def test_dry_run_never_submits(self):
        self.prepare_step(False)
        self.bot.post_with_retry = Mock()
        self.bot.step()
        self.bot.step()
        self.bot.post_with_retry.assert_not_called()
        self.assertEqual(self.bot.simulator_get.call_count, 4)

    def test_timeout_never_replays_order(self):
        self.prepare_step(True)
        self.bot.http = Mock()
        self.bot.http.post.side_effect = httpx.ReadTimeout("lost response")
        self.bot.step()
        self.bot.step()
        self.assertEqual(self.bot.http.post.call_count, 1)
        self.assertEqual(self.bot.journal.db.execute("SELECT status FROM orders").fetchone()[0], "unknown")

    def test_server_stale_flag_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            quote_book({"stale": True}, time.time())

    def test_quote_timestamp_formats_and_freshness(self):
        iso = "2026-09-19T02:02:42.771325+00:00"
        now = datetime.fromisoformat(iso).timestamp()
        book = {"bids": [{"price": "0.89", "size": "100"}],
                "asks": [{"price": "0.90", "size": "100"}], "tick_size": "0.01"}
        for stamp in (iso, iso.replace("+00:00", "Z"),
                      "2026-09-19T04:02:42.771325+02:00",
                      now * 1000, str(now * 1000)):
            with self.subTest(timestamp=stamp):
                book["timestamp"] = stamp
                self.assertEqual(str(quote_book(book, now).ask), "0.90")
                for check_time in (now + 31, now - 6):
                    with self.assertRaisesRegex(ValueError, "stale"):
                        quote_book(book, check_time)
        for stamp in ("invalid", None, "2026-09-19T02:02:42", "NaN", "Infinity"):
            with self.subTest(invalid_timestamp=stamp):
                book["timestamp"] = stamp
                with self.assertRaises(ValueError):
                    quote_book(book, now)


if __name__ == "__main__":
    unittest.main()
