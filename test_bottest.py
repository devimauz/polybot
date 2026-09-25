"""Offline regression tests. No wallet credentials or real orders are used."""

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bottest import Bot, D, Journal, Settings, decimal, quote_book, resolved_winner, signed_order_id, triggered


def book(bid="0.88", ask="0.89", size="100", age=0):
    return {"timestamp": str((time.time() - age) * 1000),
            "bids": [{"price": bid, "size": "100"}],
            "asks": [{"price": ask, "size": size}], "tick_size": "0.01", "neg_risk": False}


class QuoteTests(unittest.TestCase):
    def test_default_midpoint_band_has_exclusive_upper_bound(self):
        for outcome in ("Up", "Down"):
            for midpoint, expected in (("0.87999", False), ("0.88", True),
                                       ("0.885", True), ("0.88999", True),
                                       ("0.89", False), ("0.90", False), ("0.93", False)):
                with self.subTest(outcome=outcome, midpoint=midpoint):
                    self.assertEqual(triggered(outcome, D(midpoint)), expected)

    def test_signal_upper_bound_is_independent_of_purchase_cap(self):
        self.assertTrue(triggered("Up", D("0.885"), Settings(max_price=D("0.90"))))
        self.assertFalse(triggered("Up", D("0.90"), Settings(max_price=D("0.95"))))
        self.assertTrue(triggered("Up", D("0.91"), Settings(min_midpoint=D("0.90"), max_midpoint=D("0.94"))))

    def test_invalid_numbers_and_settings(self):
        for value in ("NaN", "Infinity", "-Infinity", "broken"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decimal(value)
        for settings in ({"max_book_age": float("nan")}, {"min_midpoint": D("NaN")},
                         {"max_midpoint": D("NaN")}, {"max_midpoint": D("0.88")},
                         {"max_midpoint": D("0.87")},
                         {"entry_delay": 295}, {"max_attempts": 0}, {"slippage": D("-0.1")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Settings(**settings)

    def test_stale_future_crossed_and_nonfinite_books(self):
        for invalid in (book(age=6), book(age=-6), book(bid="0.90"), book(ask="NaN"),
                        book(size="-2"), book(size="Infinity")):
            with self.subTest(book=invalid), self.assertRaises(ValueError):
                quote_book(invalid, time.time())

    def test_small_exchange_clock_lead_is_not_stale(self):
        now = time.time()
        data = book()
        data["timestamp"] = str((now + 1.2) * 1000)
        quote = quote_book(data, now)
        bot = Bot.__new__(Bot)
        with patch("bottest.time.time", return_value=now):
            self.assertIsNone(bot.entry_reason(quote, now + 120))

    def test_timestamp_error_reports_age_or_clock_skew(self):
        with self.assertRaisesRegex(ValueError, "ahead of local clock"):
            quote_book(book(age=-6), time.time())
        with self.assertRaisesRegex(ValueError, "old.*maximum"):
            quote_book(book(age=6), time.time())

    def test_depth_only_counts_executable_levels(self):
        data = book(bid="0.87", ask="0.88", size="1")
        data["asks"] += [{"price": "0.95", "size": "10000"}, {"price": "0.89", "size": "2"}]
        quote = quote_book(data, time.time())
        self.assertEqual(quote.depth(), D("2.66"))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.journal = Journal(":memory:")
        self.addCleanup(self.journal.db.close)

    def test_opposite_direction_locked_by_default(self):
        self.assertTrue(self.journal.reserve("m", "Up"))
        self.assertFalse(self.journal.reserve("m", "Down"))
        self.assertTrue(self.journal.reserve("next", "Down"))

    def test_explicit_both_directions_option(self):
        settings = Settings(one_direction=False)
        self.assertTrue(self.journal.reserve("m", "Up", settings=settings))
        self.assertTrue(self.journal.reserve("m", "Down", settings=settings))
        self.assertFalse(self.journal.reserve("m", "Up", settings=settings))

    def test_rejection_cooldown_attempt_limit_and_history(self):
        self.journal.reserve("m", "Up")
        self.journal.finish("m", "Up", "rejected", '{"first":true}')
        self.assertFalse(self.journal.reserve("m", "Up"))
        self.journal.db.execute("UPDATE bottest_attempts SET updated=0")
        self.journal.db.commit()
        self.assertTrue(self.journal.reserve("m", "Up"))
        self.journal.finish("m", "Up", "rejected", '{}')
        self.journal.db.execute("UPDATE bottest_attempts SET updated=0")
        self.journal.db.commit()
        self.assertFalse(self.journal.reserve("m", "Up"))
        history = self.journal.db.execute("SELECT response FROM bottest_attempts ORDER BY id").fetchall()
        self.assertEqual(history, [('{"first":true}',), ('{}',)])

    def test_unknown_and_legacy_attempts_are_never_retried(self):
        self.journal.reserve("m", "Up")
        self.journal.finish("m", "Up", "unknown", '{}')
        self.assertFalse(self.journal.reserve("m", "Down"))
        self.assertFalse(self.journal.reserve("m", "Up"))
        self.journal.db.execute("INSERT INTO orders VALUES ('legacy','Up','day',10,'rejected','{}')")
        self.journal.db.commit()
        self.assertFalse(self.journal.reserve("legacy", "Up"))

    def test_direction_lock_survives_restart_and_other_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite3"
            first, second = Journal(path), Journal(path)
            try:
                self.assertTrue(first.reserve("m", "Up"))
                self.assertFalse(second.reserve("m", "Down"))
            finally:
                first.db.close()
                second.db.close()
            reopened = Journal(path)
            try:
                self.assertFalse(reopened.reserve("m", "Up"))
            finally:
                reopened.db.close()


class BotTests(unittest.TestCase):
    def setUp(self):
        self.bot = Bot.__new__(Bot)
        self.bot.live = True
        self.bot.amount = D("10")
        self.bot.settings = Settings()
        self.bot.signals = {}
        self.bot.market = None
        self.bot.journal = Journal(":memory:")
        self.addCleanup(self.bot.journal.db.close)
        self.bot.client = Mock()
        self.bot.client.post_order.return_value = {"success": True, "status": "matched", "orderID": "id",
                                                  "makingAmount": "10", "takingAmount": "11.4"}
        self.bot.get = Mock(side_effect=lambda *a, **kw: book())
        self.bot.log_prices = Mock()
        self.bot.discover = Mock(return_value=("market", time.time() + 120, {"Up": "up", "Down": "down"}))
        self.hash = patch("bottest.signed_order_id", return_value="id").start()
        self.addCleanup(patch.stopall)
        patch("bottest.time.sleep").start()

    def status(self):
        return self.bot.journal.db.execute("SELECT status FROM orders ORDER BY rowid LIMIT 1").fetchone()[0]

    def test_one_buy_but_continued_logging_both_directions(self):
        self.bot.step()
        self.bot.step()
        self.bot.client.post_order.assert_called_once()
        quote_calls = [c for c in self.bot.log_prices.call_args_list if len(c.args) == 3]
        self.assertEqual(len(quote_calls), 4)
        self.assertEqual(self.status(), "submitted")

    def test_dry_run_checks_depth_and_never_signs_or_submits(self):
        self.bot.live = False
        self.bot.step()
        self.bot.client.create_market_order.assert_not_called()
        self.bot.client.post_order.assert_not_called()
        self.assertEqual(self.status(), "simulated")

    def test_upper_midpoint_boundary_never_signs_order(self):
        self.bot.get.side_effect = lambda *a, **kw: book(bid="0.88", ask="0.90")
        self.bot.step()
        self.bot.client.create_market_order.assert_not_called()
        self.bot.client.post_order.assert_not_called()
        self.assertFalse(self.bot.journal.seen("market", "Up"))

    def test_nonexecutable_signals_do_not_consume_attempts(self):
        for snapshot in (book(bid="0.92", ask="0.94"), book(size="1"), book(bid="0.84", ask="0.90")):
            with self.subTest(snapshot=snapshot):
                self.bot.get.side_effect = lambda *a, **kw: snapshot
                self.bot.step()
                self.bot.client.create_market_order.assert_not_called()
                self.assertEqual(self.bot.journal.db.execute("SELECT count(*) FROM orders").fetchone()[0], 0)

    def test_signing_then_price_reversal_does_not_submit(self):
        self.bot.get.side_effect = [book(), book(bid="0.70", ask="0.71"), book(bid="0.70", ask="0.71")]
        self.bot.step()
        self.bot.client.post_order.assert_not_called()
        self.assertFalse(self.bot.journal.seen("market", "Up"))

    def test_before_entry_delay_and_near_close_still_log(self):
        for remaining in (285, 5):
            self.bot.discover.return_value = ("market", time.time() + remaining, {"Up": "up"})
            self.bot.step()
        self.bot.client.create_market_order.assert_not_called()
        self.assertEqual(self.bot.log_prices.call_count, 2)

    def test_prepare_timeout_does_not_reserve_or_stop_other_quotes(self):
        self.bot.client.create_market_order.side_effect = TimeoutError()
        self.bot.step()
        self.bot.client.post_order.assert_not_called()
        self.assertFalse(self.bot.journal.seen("market", "Up"))

    def test_submit_timeout_never_creates_fresh_order(self):
        self.bot.client.post_order.side_effect = TimeoutError()
        self.bot.step()
        self.bot.step()
        self.assertEqual(self.bot.client.post_order.call_count, 3)
        self.bot.client.create_market_order.assert_called_once()
        self.assertEqual(self.status(), "unknown")

    def test_disconnect_then_rejection_stays_unknown(self):
        self.bot.client.post_order.side_effect = [TimeoutError(), {"success": False, "error": "FOK order couldn't be fully filled"}]
        self.bot.step()
        self.assertEqual(self.status(), "unknown")

    def test_changed_quote_stops_resubmission_of_uncertain_order(self):
        self.bot.client.post_order.side_effect = TimeoutError()
        self.bot.still_executable = Mock(side_effect=[True, False])
        self.bot.step()
        self.bot.client.post_order.assert_called_once()
        self.assertEqual(self.status(), "unknown")

    def test_initial_submit_after_deadline_is_blocked(self):
        self.assertIsNone(self.bot.post_with_retry(object(), time.time() + 9))
        self.bot.client.post_order.assert_not_called()

    def test_matching_response_must_have_expected_order_id(self):
        self.bot.client.post_order.return_value["orderID"] = "unexpected"
        self.bot.step()
        self.assertEqual(self.status(), "unknown")

    def test_confirmation_resets_after_signal_break(self):
        self.bot.settings = Settings(confirmation_seconds=2)
        with patch("bottest.time.monotonic", return_value=10):
            self.assertFalse(self.bot.signal_confirmed("m", "Up"))
        with patch("bottest.time.monotonic", return_value=12):
            self.assertTrue(self.bot.signal_confirmed("m", "Up"))
        with patch("bottest.time.monotonic", return_value=20):
            self.assertFalse(self.bot.signal_confirmed("m", "Up"))

    def test_market_discovery_cached_until_rollover(self):
        start = int(time.time()) // 300 * 300
        slug = f"btc-updown-5m-{start}"
        from datetime import datetime, timezone
        self.bot.get.side_effect = None
        self.bot.get.return_value = [{"slug": slug, "active": True, "closed": False, "acceptingOrders": True,
            "endDate": datetime.fromtimestamp(start + 300, timezone.utc).isoformat(),
            "outcomes": '["Up","Down"]', "clobTokenIds": '["up","down"]', "conditionId": "condition"}]
        Bot.discover(self.bot, start + 60)
        Bot.discover(self.bot, start + 61)
        self.bot.get.assert_called_once()
        self.bot.get.return_value = []
        self.assertIsNone(Bot.discover(self.bot, start + 301))
        self.assertEqual(self.bot.get.call_count, 2)

    def uncertain(self):
        self.bot.journal.reserve("m", "Up", metadata={"order_id": "id", "token_id": "up"})
        self.bot.journal.finish("m", "Up", "unknown", '{}')
        self.bot.journal.db.execute("UPDATE bottest_attempts SET updated=0")
        self.bot.journal.db.commit()

    def test_reconciliation_missing_order_does_not_release(self):
        self.uncertain()
        self.bot.client.get_order.return_value = None
        self.bot.reconcile_one()
        self.assertEqual(self.status(), "unknown")

    def test_reconciliation_partial_fill_blocks_further_buying(self):
        self.uncertain()
        self.bot.client.get_order.return_value = {"id": "id", "asset_id": "up", "side": "BUY", "size_matched": "2", "status": "CANCELED"}
        self.bot.reconcile_one()
        self.assertEqual(self.status(), "reconciled")
        self.assertFalse(self.bot.journal.eligible("m", "Up", self.bot.settings))

    def test_reconciliation_canceled_zero_fill_allows_bounded_retry(self):
        self.uncertain()
        self.bot.client.get_order.return_value = {"id": "id", "asset_id": "up", "side": "BUY", "size_matched": "0", "status": "CANCELED"}
        self.bot.reconcile_one()
        self.assertEqual(self.status(), "rejected")
        self.assertFalse(self.bot.journal.eligible("m", "Up", self.bot.settings))
        self.assertTrue(self.bot.journal.eligible("m", "Up", self.bot.settings, now=time.time() + 6))

    def test_settlement_requires_authoritative_winner_and_logs_gross_only(self):
        self.bot.journal.db.execute("INSERT INTO bottest_markets(market,end,condition_id,tokens) VALUES ('m',0,'c',?)", (json.dumps({"Up": "up", "Down": "down"}),))
        self.bot.journal.db.commit()
        self.bot.journal.reserve("m", "Up")
        self.bot.journal.finish("m", "Up", "submitted", '{"makingAmount":"10","takingAmount":"11.4"}')
        self.bot.get.side_effect = None
        self.bot.get.return_value = {"condition_id": "c", "closed": True, "tokens": [{"token_id": "up", "outcome": "Up", "winner": True}]}
        self.bot.settle_one()
        self.assertEqual(self.bot.log_prices.call_args_list[0].kwargs["gross_pnl_before_fees"], "1.4")
        self.assertEqual(self.bot.journal.db.execute("SELECT winner FROM bottest_markets").fetchone()[0], "Up")
        self.bot.settle_one()
        self.assertEqual(self.bot.get.call_count, 1)

    def test_closed_market_without_winner_is_not_settlement(self):
        self.assertIsNone(resolved_winner({"closed": True, "tokens": [{"outcome": "Up", "price": 1}]}, {"Up": "up"}))
        self.assertIsNone(resolved_winner({"closed": True, "tokens": [{"outcome": "Up", "token_id": "wrong", "winner": True}]}, {"Up": "up"}))


class SDKIntegrationTests(unittest.TestCase):
    def test_order_hash_is_stable_and_does_not_depend_on_signature(self):
        from py_clob_client_v2 import MarketOrderArgs, PartialCreateOrderOptions, Side
        from py_clob_client_v2.order_builder.builder import OrderBuilder
        from py_clob_client_v2.signer import Signer
        # Public test fixture only; never loaded from the environment or used on a network.
        builder = OrderBuilder(Signer("0x" + "01" * 32, 137))
        client = SimpleNamespace(builder=builder)
        for version in (1, 2):
            signed = builder.build_market_order(MarketOrderArgs(token_id="123", amount=10, side=Side.BUY, price=.89),
                PartialCreateOrderOptions(tick_size="0.01", neg_risk=False), version=version)
            order_id = signed_order_id(client, signed, False)
            self.assertEqual(len(order_id), 66)
            self.assertEqual(order_id, signed_order_id(client, replace(signed, signature=""), False))
            self.assertNotEqual(order_id, signed_order_id(client, signed, True))

    def test_dry_run_background_worker_starts_and_stops_without_network(self):
        with tempfile.TemporaryDirectory() as directory, patch("bottest.ROOT", Path(directory)):
            with patch("httpx.Client.get", side_effect=AssertionError("Unexpected network call")):
                bot = Bot(False)
                try:
                    bot.start_maintenance()
                finally:
                    bot.close()
                self.assertFalse(bot.worker.is_alive())

    def test_log_includes_execution_fields_and_unknown_fees(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = Bot.__new__(Bot)
            bot.live = True
            bot.price_log_dir = Path(directory)
            bot.log_prices("market", "Up", quote_book(book(), time.time()), "buy_matched",
                           {"makingAmount": "10", "takingAmount": "11.4", "orderID": "id"})
            bot.log_prices("market", "Up", None, "market_resolved", winner="Up")
            rows = [json.loads(line) for line in (Path(directory) / "BTC.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["paid"], "10")
            self.assertEqual(rows[0]["shares"], "11.4")
            self.assertEqual(rows[0]["strategy_version"], "bottest-v3")
            self.assertIsNone(rows[0]["fees_paid"])
            self.assertIsNone(rows[1]["buy_price"])


class ReportTests(unittest.TestCase):
    def test_missing_resolution_is_not_a_loss_and_duplicate_fills_ignored(self):
        from bottest_report import summarize
        fill = {"event": "buy_matched", "mode": "live", "asset": "BTC", "market": "m",
                "outcome": "Up", "order_id": "id", "paid": "10", "shares": "11.4"}
        group = summarize([fill, fill])["BTC"]
        self.assertEqual(group["fills"], 1)
        self.assertEqual(group["resolved"], 0)
        settlement = {"event": "market_resolved", "market": "m", "winner": "Up"}
        group = summarize([fill, settlement])["BTC"]
        self.assertEqual(group["wins"], 1)
        self.assertEqual(group["gross"], D("1.4"))
        self.assertEqual(group["with_fees"], 0)

    def test_conflicting_resolutions_are_excluded(self):
        from bottest_report import summarize
        rows = [{"event": "market_resolved", "market": "m", "winner": w} for w in ("Up", "Down")]
        rows.append({"event": "buy_matched", "mode": "live", "asset": "BTC", "market": "m", "outcome": "Up"})
        self.assertEqual(summarize(rows)["BTC"]["resolved"], 0)


if __name__ == "__main__":
    unittest.main()
