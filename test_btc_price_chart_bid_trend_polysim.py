import unittest

from btc_price_chart_bid_trend_polysim import (
    MARKET_SECONDS,
    bid_trend_action,
    is_large_buy,
    market_from_payload,
)


class BidTrendStrategyTests(unittest.TestCase):
    def test_copies_only_up_or_down_buys_strictly_over_100_shares(self):
        self.assertTrue(is_large_buy(
            {"outcome": "Up", "side": "BUY", "size": 100.01}))
        self.assertTrue(is_large_buy(
            {"outcome": "Down", "side": "buy", "size": 101}))
        self.assertFalse(is_large_buy(
            {"outcome": "Up", "side": "BUY", "size": 100}))
        self.assertFalse(is_large_buy(
            {"outcome": "Down", "side": "SELL", "size": 500}))

    def test_discovers_current_five_minute_market(self):
        now = 1_800_000_123
        start = now // MARKET_SECONDS * MARKET_SECONDS
        slug = f"btc-updown-5m-{start}"
        payload = [{
            "slug": slug,
            "active": True,
            "closed": False,
            "conditionId": "condition",
            "outcomes": '["Up", "Down"]',
            "clobTokenIds": '["up-token", "down-token"]',
        }]

        self.assertEqual(
            market_from_payload(payload, now),
            (slug, start + MARKET_SECONDS,
             {"Up": "up-token", "Down": "down-token"}),
        )

    def test_buys_only_above_point_two(self):
        self.assertEqual(bid_trend_action("0.201"), "BUY_UP")
        self.assertEqual(bid_trend_action("0.200"), "HOLD")

    def test_buys_down_below_negative_point_two(self):
        self.assertEqual(bid_trend_action("-0.201"), "BUY_DOWN")
        self.assertEqual(bid_trend_action("-0.200"), "HOLD")

    def test_holds_one_position_above_exit_level(self):
        self.assertEqual(bid_trend_action("0.051", "Up"), "HOLD")

    def test_sells_at_or_below_point_zero_five(self):
        self.assertEqual(bid_trend_action("0.050", "Up"), "SELL")
        self.assertEqual(bid_trend_action("-0.010", "Up"), "SELL")

    def test_down_exits_when_trend_recovers_to_negative_point_zero_five(self):
        self.assertEqual(bid_trend_action("-0.050", "Down"), "SELL")
        self.assertEqual(bid_trend_action("-0.051", "Down"), "HOLD")


if __name__ == "__main__":
    unittest.main()
