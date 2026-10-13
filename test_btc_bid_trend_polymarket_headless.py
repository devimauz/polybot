import unittest
import threading
from collections import deque
from decimal import Decimal

from btc_bid_trend_polymarket_headless import (
    STAKE, LiveBidTrendTrader, buy_limit, remaining_after_sell, sell_limit,
)


class SlippageLimitTests(unittest.TestCase):
    def test_stake_is_one_dollar_ten(self):
        self.assertEqual(STAKE, Decimal("1.10"))

    def test_buy_limit_is_at_most_three_cents_above_ask(self):
        self.assertEqual(buy_limit("0.50", "0.01"), Decimal("0.53"))
        self.assertEqual(buy_limit("0.98", "0.01"), Decimal("0.99"))

    def test_sell_limit_is_at_most_three_cents_below_bid(self):
        self.assertEqual(sell_limit("0.50", "0.01"), Decimal("0.47"))
        self.assertEqual(sell_limit("0.02", "0.01"), Decimal("0.01"))

    def test_partial_fak_sell_tracks_remaining_shares(self):
        self.assertEqual(remaining_after_sell("5", "3.25"), Decimal("1.75"))

    def test_tiny_fak_rounding_remainder_is_treated_as_closed(self):
        self.assertEqual(remaining_after_sell("2.115385", "2.11"), Decimal("0"))

    def test_rounding_never_exceeds_slippage(self):
        self.assertEqual(buy_limit("0.501", "0.01"), Decimal("0.53"))
        self.assertEqual(sell_limit("0.509", "0.01"), Decimal("0.48"))

    def test_event_log_collapses_and_truncates_errors(self):
        trader = LiveBidTrendTrader.__new__(LiveBidTrendTrader)
        trader.lock = threading.RLock()
        trader.events = deque(maxlen=20)
        trader._log("error\n" + "x" * 300)
        event = trader.log_snapshot()[0]
        self.assertNotIn("\n", event)
        self.assertLessEqual(len(event), 250)

    def test_polymarket_exception_fields_are_logged(self):
        class Error(Exception):
            status_code = 400
            error_msg = {"error": "not enough balance"}

        detail = LiveBidTrendTrader._error_detail(Error())
        self.assertIn("HTTP 400", detail)
        self.assertIn("not enough balance", detail)


if __name__ == "__main__":
    unittest.main()
