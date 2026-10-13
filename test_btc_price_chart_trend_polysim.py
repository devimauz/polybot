import unittest
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path

from btc_price_chart_trend_polysim import (
    JUMP_STAKE, KRAKEN_WS, MAX_ENTRY_PRICE, MIN_ENTRY_PRICE,
    DisplayOnlyTrader, JumpPaperTrader,
    TAKE_PROFIT_RATE, TRADE_COOLDOWN_SECONDS,
    crossed_above, kraken_move_signal, resting_pressure_crossing,
)


class RestingBidPressureStrategyTests(unittest.TestCase):
    def test_only_fresh_crossing_above_80_percent_triggers(self):
        self.assertTrue(crossed_above(0.80, 0.801, 0.80))
        self.assertTrue(crossed_above(0.79, 0.81, 0.80))
        self.assertFalse(crossed_above(None, 0.90, 0.80))
        self.assertFalse(crossed_above(0.81, 0.90, 0.80))
        self.assertFalse(crossed_above(0.79, 0.80, 0.80))
        self.assertFalse(crossed_above(0.81, 0.79, 0.80))

    def test_stake_is_ten_dollars(self):
        self.assertEqual(JUMP_STAKE, Decimal("10"))

    def test_entry_price_range(self):
        self.assertEqual(MIN_ENTRY_PRICE, Decimal("0.30"))
        self.assertEqual(MAX_ENTRY_PRICE, Decimal("0.70"))

    def test_trade_cooldown_is_five_seconds(self):
        self.assertEqual(TRADE_COOLDOWN_SECONDS, 5.0)

    def test_polysimulator_take_profit_is_five_percent(self):
        self.assertEqual(TAKE_PROFIT_RATE, Decimal("0.05"))

    def test_kraken_move_entry_signals(self):
        self.assertEqual(kraken_move_signal(100, 105), ("Up", 5))
        self.assertEqual(kraken_move_signal(100, 95), ("Down", -5))
        self.assertIsNone(kraken_move_signal(100, 104.99))
        self.assertIsNone(kraken_move_signal(None, 100))

    def test_display_only_trader_never_submits(self):
        trader = DisplayOnlyTrader()
        self.assertFalse(trader.submit("window", "Up", 81.0))
        self.assertIsNone(trader.position("window"))
        self.assertIsNone(trader.latest_trade())
        self.assertIn("disabled", trader.snapshot())

    def test_kraken_uses_public_websocket_v2(self):
        self.assertEqual(KRAKEN_WS, "wss://ws.kraken.com/v2")

    def test_entry_unlocks_after_polysimulator_accepts_tp(self):
        with tempfile.TemporaryDirectory() as folder:
            trader = JumpPaperTrader.__new__(JumpPaperTrader)
            trader.db_path = Path(folder) / "trades.sqlite3"
            db = sqlite3.connect(trader.db_path)
            try:
                db.execute("""CREATE TABLE trades (
                    market TEXT, outcome TEXT, entry_state TEXT, tp_state TEXT)""")
                db.execute("INSERT INTO trades VALUES (?,?,?,?)",
                           ("window#jump-1", "Up", "filled", "needed"))
                db.commit()
            finally:
                db.close()
            self.assertTrue(trader.has_open_trade("window"))
            db = sqlite3.connect(trader.db_path)
            try:
                db.execute("UPDATE trades SET tp_state='submitted'")
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
