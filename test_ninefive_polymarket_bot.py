import time
import unittest
import json
import sqlite3
import tempfile
from pathlib import Path
from decimal import Decimal
from unittest.mock import Mock, patch

from bot import Quote
from ninefive_polymarket_bot import NineFivePolymarketBot, STAKE, STOP_LOSS_PRICE


class NineFivePolymarketBotTests(unittest.TestCase):
    def test_audit_logs_each_bbo_change_without_duplicate_snapshots(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot._logged_bbo = {}
        bot.last_quotes = ("", {})
        tokens = {"Up": "up", "Down": "down"}
        bot.discover = Mock(return_value=("btc-updown-5m-1000", time.time() + 100,
                                          tokens))
        bot.feed = Mock()
        first = Quote(Decimal("0.95"), Decimal("0.96"), Decimal("0.96"),
                      "0.01", False, Decimal("0.94"))
        second = Quote(Decimal("0.96"), Decimal("0.97"), Decimal("0.97"),
                       "0.01", False, Decimal("0.95"))
        bot.feed.quotes.return_value = {"Up": first}
        bot.log_prices = Mock()

        bot.audit_quotes()
        bot.audit_quotes()
        bot.feed.quotes.return_value = {"Up": second}
        bot.audit_quotes()

        self.assertEqual(bot.log_prices.call_count, 2)
        self.assertEqual(bot.log_prices.call_args.args[3], "bbo_change")

    def test_stop_loss_is_eighty_cents_for_up_and_down(self):
        self.assertEqual(STOP_LOSS_PRICE, Decimal("0.80"))
        for outcome in ("Up", "Down"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory, patch(
                    "ninefive_polymarket_bot.ROOT", Path(directory)):
                bot = NineFivePolymarketBot(False, "btc")
                try:
                    slug = "btc-updown-5m-1000"
                    bot.market = (slug, time.time() + 30,
                                  {"Up": "up-token", "Down": "down-token"})
                    bot.journal.reserve(slug, outcome, Decimal("10"),
                                        condition_id="condition")
                    bot.journal.finish(
                        slug, outcome, "matched",
                        json.dumps({"makingAmount": "10", "takingAmount": "11"}))
                    quote = Quote(Decimal("0.805"), Decimal("0.81"), Decimal("0.81"),
                                  "0.01", False, Decimal("0.80"))
                    bot.feed = Mock()
                    bot.feed.quotes.return_value = {outcome: quote}
                    bot.log_prices = Mock()
                    bot.notifier = Mock()
                    bot.notifier.send.return_value = False

                    bot.monitor_stop_loss()

                    exit_row = bot.journal.db.execute(
                        "SELECT status,sold,proceeds FROM polymarket_exits "
                        "WHERE market=? AND outcome=?", (slug, outcome)).fetchone()
                    self.assertEqual(exit_row, ("sold", 11.0, 8.8))
                finally:
                    bot.close()

    def test_stop_loss_does_not_trigger_above_eighty_cents(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "ninefive_polymarket_bot.ROOT", Path(directory)):
            bot = NineFivePolymarketBot(False, "btc")
            try:
                slug = "btc-updown-5m-1000"
                bot.market = (slug, time.time() + 30,
                              {"Up": "up-token", "Down": "down-token"})
                bot.journal.reserve(slug, "Up", Decimal("10"), condition_id="condition")
                bot.journal.finish(
                    slug, "Up", "matched",
                    json.dumps({"makingAmount": "10", "takingAmount": "11"}))
                quote = Quote(Decimal("0.815"), Decimal("0.82"), Decimal("0.82"),
                              "0.01", False, Decimal("0.81"))
                bot.feed = Mock()
                bot.feed.quotes.return_value = {"Up": quote}
                bot.log_prices = Mock()

                bot.monitor_stop_loss()

                self.assertIsNone(bot.journal.db.execute(
                    "SELECT 1 FROM polymarket_exits").fetchone())
            finally:
                bot.close()

    def test_fresh_journal_has_condition_id_before_first_order(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "ninefive_polymarket_bot.ROOT", Path(directory)):
            bot = NineFivePolymarketBot(False, "btc")
            try:
                columns = {row[1] for row in bot.journal.db.execute(
                    "PRAGMA table_info(orders)")}
                self.assertIn("condition_id", columns)
                bot.check_settlements()
            finally:
                bot.close()

    def test_stake_is_ten_dollars(self):
        self.assertEqual(STAKE, Decimal("10"))

    def test_live_stake_is_fixed_when_balance_and_slot_are_available(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.client = Mock()
        bot.client.get_balance_allowance.return_value = {"balance": "100250000"}
        bot.open_trade_count = Mock(return_value=2)

        self.assertEqual(bot.available_stake(), Decimal("10"))
        self.assertEqual(bot.last_balance, Decimal("100.25"))
        self.assertEqual(bot.last_open_trade_count, 2)

    def test_live_stake_is_zero_when_all_twelve_slots_are_open(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.client = Mock()
        bot.client.get_balance_allowance.return_value = {"balance": "20000000"}
        bot.open_trade_count = Mock(return_value=12)

        self.assertEqual(bot.available_stake(), Decimal("0"))

    def test_live_stake_is_zero_when_balance_is_below_ten(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.client = Mock()
        bot.client.get_balance_allowance.return_value = {"balance": "9990000"}
        bot.open_trade_count = Mock(return_value=0)

        self.assertEqual(bot.available_stake(), Decimal("0"))

    def test_dry_run_reserves_ten_dollars_without_live_client(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.asset = "btc"
        bot.live = False
        bot.amount = STAKE
        bot.condition_id = "condition"
        bot.client = None
        bot.journal = Mock()
        bot.journal.seen.return_value = False
        bot.journal.reserve.return_value = True
        bot.feed = Mock()
        quote = Quote(Decimal("0.95"), Decimal("0.95"), Decimal("0.95"),
                      "0.01", False, Decimal("0.94"))
        bot.feed.quotes.return_value = {"Up": quote}
        bot.log_prices = Mock()

        result = bot._buy_leg("btc-updown-5m-1000", time.time() + 30,
                              {"Up": "up-token", "Down": "down-token"},
                              "Up", Decimal("0.98"))

        self.assertTrue(result)
        self.assertEqual(bot.journal.reserve.call_args.args[2], Decimal("10"))
        bot.journal.finish.assert_called_once()
        self.assertIsNone(bot.client)

    def test_final_fok_failure_is_rejected_not_uncertain(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.asset = "doge"
        bot.live = True
        bot.amount = STAKE
        bot.condition_id = "condition"
        bot.neg_risk = False
        bot.available_stake = Mock(return_value=Decimal("30.09"))
        bot.last_balance = Decimal("60.20")
        bot.last_open_trade_count = 0
        bot.journal = Mock()
        bot.journal.seen.return_value = False
        bot.journal.reserve.return_value = True
        bot.feed = Mock()
        quote = Quote(Decimal("0.97"), Decimal("0.98"), Decimal("0.98"),
                      "0.01", False, Decimal("0.96"))
        bot.feed.quotes.return_value = {"Down": quote}
        bot.client = Mock()
        bot.client.create_market_order.return_value = object()
        bot.client.post_order.side_effect = Exception(
            "order couldn't be fully filled. FOK orders are fully filled or killed.")
        bot._notify_once = Mock()

        result = bot._buy_leg_impl("doge-updown-5m-1000", time.time() + 30,
                                   {"Up": "up", "Down": "down"},
                                   "Down", Decimal("0.98"))

        self.assertFalse(result)
        self.assertEqual(bot.client.post_order.call_count, 3)
        self.assertEqual(bot.journal.finish.call_args.args[2], "rejected")
        self.assertEqual(bot._notify_once.call_args.args[2], "order_rejected")

    def test_discovery_uses_gamma_not_polysimulator(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.asset = "btc"
        bot.market = None
        bot.condition_id = None
        bot.neg_risk = False
        bot.get = Mock(return_value=[{
            "slug": "btc-updown-5m-900", "closed": False, "active": True,
            "acceptingOrders": True, "endDate": "1970-01-01T00:20:00Z",
            "outcomes": '["Up", "Down"]', "clobTokenIds": '["up", "down"]',
            "conditionId": "condition", "negRisk": True,
        }])

        market = bot.discover(1000)

        self.assertEqual(market, ("btc-updown-5m-900", 1200.0,
                                  {"Up": "up", "Down": "down"}))
        self.assertTrue(bot.neg_risk)
        self.assertIn("gamma-api.polymarket.com", bot.get.call_args.args[0])

    def test_settlement_sends_one_pnl_notification(self):
        bot = NineFivePolymarketBot.__new__(NineFivePolymarketBot)
        bot.live = True
        bot.asset = "btc"
        bot._next_settlement_check = 0
        bot.notifier = Mock()
        bot.notifier.send.return_value = True
        bot.get = Mock(return_value={
            "closed": True,
            "tokens": [{"outcome": "Up", "winner": True},
                       {"outcome": "Down", "winner": False}],
        })
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE orders (market TEXT, outcome TEXT, status TEXT, detail TEXT, condition_id TEXT)")
        db.execute("""CREATE TABLE telegram_notifications (
            market TEXT, outcome TEXT, event TEXT, sent REAL,
            PRIMARY KEY(market, outcome,event))""")
        db.execute("""CREATE TABLE polymarket_settlements (
            market TEXT, outcome TEXT, winner TEXT, pnl REAL, resolved REAL,
            PRIMARY KEY(market,outcome))""")
        db.execute("""CREATE TABLE polymarket_exits (
            market TEXT, outcome TEXT, status TEXT, detail TEXT,
            sold REAL, proceeds REAL, pnl REAL, updated REAL,
            PRIMARY KEY(market,outcome))""")
        slug = "btc-updown-5m-1000"
        detail = json.dumps({"makingAmount": "5", "takingAmount": "5.5"})
        db.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                   (slug, "Up", "matched", detail, "condition"))
        db.commit()
        bot.journal = Mock()
        bot.journal.db = db

        with patch("ninefive_polymarket_bot.time.time", return_value=1400):
            bot.check_settlements()
            bot._next_settlement_check = 0
            bot.check_settlements()

        bot.notifier.send.assert_called_once()
        message = bot.notifier.send.call_args.args[0]
        self.assertIn("WIN", message)
        self.assertIn("+$0.50", message)


if __name__ == "__main__":
    unittest.main()
