import unittest
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path

from btc_price_chart_trend_polysim import (
    JUMP_STAKE, MAX_ENTRY_PRICE, MIN_ENTRY_PRICE, JumpPaperTrader,
    RESTING_BID_PRESSURE_EXIT, SELL_RETRY_ATTEMPTS, TRADE_COOLDOWN_SECONDS,
    crossed_above, resting_pressure_crossing, should_exit_pressure,
)


class RestingBidPressureStrategyTests(unittest.TestCase):
    def test_only_fresh_crossing_above_80_percent_triggers(self):
        self.assertTrue(crossed_above(0.80, 0.801, 0.80))
        self.assertTrue(crossed_above(0.79, 0.81, 0.80))
        self.assertFalse(crossed_above(None, 0.90, 0.80))
        self.assertFalse(crossed_above(0.81, 0.90, 0.80))
        self.assertFalse(crossed_above(0.79, 0.80, 0.80))
        self.assertFalse(crossed_above(0.81, 0.79, 0.80))

    def test_stake_is_one_hundred_dollars(self):
        self.assertEqual(JUMP_STAKE, Decimal("100"))

    def test_entry_price_range(self):
        self.assertEqual(MIN_ENTRY_PRICE, Decimal("0.30"))
        self.assertEqual(MAX_ENTRY_PRICE, Decimal("0.70"))

    def test_trade_cooldown_is_five_seconds(self):
        self.assertEqual(TRADE_COOLDOWN_SECONDS, 5.0)

    def test_software_exit_configuration(self):
        self.assertEqual(RESTING_BID_PRESSURE_EXIT, 0.70)
        self.assertGreater(SELL_RETRY_ATTEMPTS, 1)
        self.assertTrue(should_exit_pressure("Up", 0.70))
        self.assertTrue(should_exit_pressure("Down", 0.30))
        self.assertFalse(should_exit_pressure("Up", 0.71))
        self.assertFalse(should_exit_pressure("Down", 0.29))

    def test_only_one_trade_can_remain_open(self):
        with tempfile.TemporaryDirectory() as folder:
            trader = JumpPaperTrader.__new__(JumpPaperTrader)
            trader.db_path = Path(folder) / "trades.sqlite3"
            db = sqlite3.connect(trader.db_path)
            try:
                db.execute("""CREATE TABLE trades (
                    market TEXT, outcome TEXT, entry_state TEXT, tp_state TEXT)""")
                db.execute("INSERT INTO trades VALUES (?,?,?,?)",
                           ("window#jump-1", "Up", "filled", "watching"))
                db.commit()
            finally:
                db.close()
            self.assertTrue(trader.has_open_trade("window"))
            db = sqlite3.connect(trader.db_path)
            try:
                db.execute("UPDATE trades SET tp_state='sold'")
                db.commit()
            finally:
                db.close()
            self.assertFalse(trader.has_open_trade("window"))

    def test_up_and_down_pressure_crossings(self):
        self.assertEqual(resting_pressure_crossing(0.79, 0.81), ("Up", 0.81))
        self.assertEqual(resting_pressure_crossing(0.21, 0.19), ("Down", 0.81))
        self.assertIsNone(resting_pressure_crossing(0.85, 0.82))
        self.assertIsNone(resting_pressure_crossing(None, 0.10))


if __name__ == "__main__":
    unittest.main()
