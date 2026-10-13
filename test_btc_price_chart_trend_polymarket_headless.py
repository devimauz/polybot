import unittest
from decimal import Decimal

import btc_price_chart_trend_polymarket as live_strategy
from btc_price_chart_trend_polymarket_headless import (
    ENTRY_SHARES, MAX_ENTRY_PRICE, MIN_ENTRY_PRICE, TAKE_PROFIT,
    configure_live_strategy,
)


class HeadlessLiveStrategyTests(unittest.TestCase):
    def test_live_configuration(self):
        configure_live_strategy()
        self.assertEqual(ENTRY_SHARES, Decimal("5.10"))
        self.assertEqual(TAKE_PROFIT, Decimal("0.08"))
        self.assertEqual(MIN_ENTRY_PRICE, Decimal("0.20"))
        self.assertEqual(MAX_ENTRY_PRICE, Decimal("0.80"))
        self.assertEqual(live_strategy.ENTRY_SHARES, Decimal("5.10"))
        self.assertEqual(live_strategy.TAKE_PROFIT, Decimal("0.08"))
        self.assertEqual(live_strategy.MIN_ENTRY_PRICE, Decimal("0.20"))
        self.assertEqual(live_strategy.MAX_ENTRY_PRICE, Decimal("0.80"))


if __name__ == "__main__":
    unittest.main()
