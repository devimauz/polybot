import unittest
import sqlite3
from decimal import Decimal
from types import SimpleNamespace

import btc_price_chart_90c_15m_polysim as strategy


class FifteenMinuteStrategyTests(unittest.TestCase):
    def test_take_profit_manager_only_disables_tp(self):
        bot = object.__new__(strategy.FifteenMinuteOpeningBot)
        bot.db = sqlite3.connect(":memory:")
        bot.db.execute("CREATE TABLE trades (entry_state TEXT, tp_state TEXT)")
        bot.db.execute("INSERT INTO trades VALUES ('filled', 'needed')")
        bot.place_profits()
        state = bot.db.execute("SELECT tp_state FROM trades").fetchone()[0]
        bot.db.close()
        self.assertEqual(state, "disabled")

    def test_market_parser_uses_15_minute_slug_and_end(self):
        now = 1_800.0
        payload = [{
            "slug": "btc-updown-15m-1800", "active": True,
            "closed": False, "conditionId": "condition",
            "outcomes": '["Up", "Down"]',
            "clobTokenIds": '["up-token", "down-token"]',
        }]
        self.assertEqual(
            strategy.market_from_payload(payload, now),
            ("btc-updown-15m-1800", 2700,
             {"Up": "up-token", "Down": "down-token"}))

    def test_symmetric_90c_trigger_selects_down(self):
        trader = object.__new__(strategy.NinetyCentPaperTrader)
        trader.status = ""
        trader.active_trade = lambda slug: None
        trader.trade_window_state = lambda slug: (0, 0)
        submitted = []
        trader.submit = lambda *args, **kwargs: submitted.append((args, kwargs)) or True
        quotes = {
            "Up": SimpleNamespace(ask=Decimal("0.10"), tick=Decimal("0.01")),
            "Down": SimpleNamespace(ask=Decimal("0.90"), tick=Decimal("0.01")),
        }
        self.assertTrue(trader.evaluate(
            "market", 0, {"Up": "u", "Down": "d"}, quotes, 0))
        self.assertEqual(submitted[0][0][1], "Down")

    def test_only_one_entry_per_market(self):
        trader = object.__new__(strategy.NinetyCentPaperTrader)
        trader.status = ""
        trader.active_trade = lambda slug: None
        trader.trade_window_state = lambda slug: (1, 1)
        trader.submit = lambda *args, **kwargs: self.fail("unexpected entry")
        quotes = {
            "Up": SimpleNamespace(ask=Decimal("0.91"), tick=Decimal("0.01"))}
        self.assertFalse(trader.evaluate("market", 0, {}, quotes, 0))

    def test_reversal_threshold_is_symmetric(self):
        self.assertTrue(strategy.NinetyCentPaperTrader.should_reverse(
            "Up", Decimal("0.60")))
        self.assertTrue(strategy.NinetyCentPaperTrader.should_reverse(
            "Down", Decimal("0.59")))
        self.assertFalse(strategy.NinetyCentPaperTrader.should_reverse(
            "Up", Decimal("0.61")))


if __name__ == "__main__":
    unittest.main()
