import unittest
from decimal import Decimal

from btc_price_chart_hedge_polysim import (
    MAX_ENTRY_PRICE, MIN_ENTRY_PRICE, TRADE_COOLDOWN_SECONDS,
    ask_depth_signal, entry_price_allowed, hedge_terms, kraken_entry_signal,
    marked_position_pnl, up_bid_trend,
)


class HedgeTermsTests(unittest.TestCase):
    def test_profitable_equal_share_pair(self):
        combined, shares, gross, eligible = hedge_terms("0.47", "0.49", "10")
        self.assertEqual(combined, Decimal("0.96"))
        self.assertTrue(eligible)
        self.assertEqual(shares, Decimal("10") / Decimal("0.96"))
        self.assertEqual(gross, Decimal("0.04") * shares)

    def test_pair_above_ceiling_is_rejected(self):
        combined, _, gross, eligible = hedge_terms("0.50", "0.49", "10")
        self.assertEqual(combined, Decimal("0.99"))
        self.assertFalse(eligible)
        self.assertGreater(gross, Decimal("0"))

    def test_pair_above_one_has_negative_edge(self):
        _, _, gross, eligible = hedge_terms("0.52", "0.51", "10")
        self.assertFalse(eligible)
        self.assertLess(gross, Decimal("0"))


class AskDepthSignalTests(unittest.TestCase):
    def test_trade_configuration(self):
        self.assertEqual(TRADE_COOLDOWN_SECONDS, 1.0)
        self.assertEqual(MIN_ENTRY_PRICE, Decimal("0.20"))
        self.assertEqual(MAX_ENTRY_PRICE, Decimal("0.80"))

    def test_up_depth_over_7000_triggers_up(self):
        activity = {"depth": {
            "Up": {"asks": [[0.50, 4000], [0.51, 3001]]},
            "Down": {"asks": [[0.50, 4000]]},
        }}
        self.assertEqual(ask_depth_signal(activity), ("Up", Decimal("7001")))

    def test_down_depth_over_7000_triggers_down(self):
        activity = {"depth": {
            "Up": {"asks": [[0.50, 4000]]},
            "Down": {"asks": [[0.50, 7001]]},
        }}
        self.assertEqual(ask_depth_signal(activity), ("Down", Decimal("7001")))

    def test_larger_side_wins_when_both_qualify(self):
        activity = {"depth": {
            "Up": {"asks": [[0.50, 8000]]},
            "Down": {"asks": [[0.50, 9000]]},
        }}
        self.assertEqual(ask_depth_signal(activity), ("Down", Decimal("9000")))

    def test_exactly_7000_does_not_trigger(self):
        activity = {"depth": {
            "Up": {"asks": [[0.50, 7000]]}, "Down": {"asks": []},
        }}
        self.assertIsNone(ask_depth_signal(activity))


class KrakenUpEntryTests(unittest.TestCase):
    def test_configured_entry_price_boundaries_are_allowed(self):
        self.assertTrue(entry_price_allowed(MIN_ENTRY_PRICE))
        self.assertTrue(entry_price_allowed(MAX_ENTRY_PRICE))

    def test_price_outside_configured_range_is_rejected(self):
        self.assertFalse(entry_price_allowed(MIN_ENTRY_PRICE - Decimal("0.01")))
        self.assertFalse(entry_price_allowed(MAX_ENTRY_PRICE + Decimal("0.01")))


class UpBidTrendTests(unittest.TestCase):
    def test_rising_up_bid(self):
        self.assertEqual(up_bid_trend([(7, "0.45"), (10, "0.47")], 10),
                         ("RISING", Decimal("0.02")))

    def test_falling_up_bid(self):
        self.assertEqual(up_bid_trend([(7, "0.47"), (10, "0.45")], 10),
                         ("FALLING", Decimal("-0.02")))

    def test_small_up_bid_move_is_flat(self):
        self.assertEqual(up_bid_trend([(7, "0.450"), (10, "0.455")], 10),
                         ("FLAT", Decimal("0.005")))

    def test_open_position_pnl_uses_executable_bid(self):
        self.assertEqual(marked_position_pnl("0.50", "20", "0.53"),
                         Decimal("0.60"))

    def test_losing_open_position_pnl(self):
        self.assertEqual(marked_position_pnl("0.60", "10", "0.55"),
                         Decimal("-0.50"))

    def test_positive_five_dollar_move_triggers_up(self):
        self.assertEqual(kraken_entry_signal(100000, 100005), ("Up", 5))

    def test_move_below_five_dollars_does_not_trigger(self):
        self.assertIsNone(kraken_entry_signal(100000, 100004.99))

    def test_negative_five_dollar_move_triggers_down(self):
        self.assertEqual(kraken_entry_signal(100000, 99995), ("Down", -5))


if __name__ == "__main__":
    unittest.main()
