import unittest

from btc_price_chart import (directional_jump, jump_direction, market_from_payload,
                             position_on_up_axis, price_trend,
                             threshold_from_history)


class PriceChartTests(unittest.TestCase):
    def test_jump_direction(self):
        self.assertEqual(jump_direction([(0, 100), (1, 121)], 20), ("Up", 21))
        self.assertEqual(jump_direction([(0, 100), (1, 79)], 20), ("Down", -21))
        self.assertIsNone(jump_direction([(0, 100), (1, 119)], 20))

    def test_directional_multi_tick_jump(self):
        self.assertEqual(directional_jump([4, 6], 10), ("Up", 10))
        self.assertEqual(directional_jump([-3, -4, -5], 10), ("Down", -12))
        self.assertIsNone(directional_jump([7, -4, -4], 10))
        self.assertEqual(directional_jump([2, -11], 10), ("Down", -11))

    def test_position_is_projected_onto_up_axis(self):
        self.assertEqual(position_on_up_axis("Up", 0.37), 0.37)
        self.assertAlmostEqual(position_on_up_axis("Down", 0.62), 0.38)

    def test_price_trend(self):
        self.assertEqual(price_trend([(0, 100), (5, 103), (10, 106)])[0], "UP")
        self.assertEqual(price_trend([(0, 106), (5, 103), (10, 100)])[0], "DOWN")
        self.assertEqual(price_trend([(0, 100), (10, 100.5)])[0], "FLAT")

    def test_price_to_beat_is_first_twap_at_boundary(self):
        data = [{"timestamp": 999000, "value": 99},
                {"timestamp": 1000000, "value": 101},
                {"timestamp": 1001000, "value": 102}]
        self.assertEqual(threshold_from_history(data, 1000), 101)
        self.assertIsNone(threshold_from_history(data[2:], 1000))

    def test_current_market_token_mapping(self):
        now = 1234
        start = now // 300 * 300
        slug = f"btc-updown-5m-{start}"
        payload = {"markets": [{
            "slug": slug, "active": True, "closed": False,
            "condition_id": "c",
            "markets": [{"condition_id": "c", "outcomes": '["Down","Up"]',
                         "token_ids": '["down-token","up-token"]'}],
        }]}
        market = market_from_payload(payload, now)
        self.assertEqual(market, (slug, start + 300,
                                  {"Down": "down-token", "Up": "up-token"}))


if __name__ == "__main__":
    unittest.main()
