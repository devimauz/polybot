import unittest
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from polymarket_soon_expiry_scanner import (
    PAGE_SIZE, add_live_ask_prices, qualifying_rows,
)


class SoonExpiryScannerTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)
        self.deadline = self.now + timedelta(hours=1)

    def test_gamma_page_size_matches_server_cap(self):
        self.assertEqual(PAGE_SIZE, 100)

    def market(self, minutes, prices='["0.91", "0.09"]'):
        live_prices = [Decimal(value) for value in json.loads(prices)]
        return {
            "question": "Will BTC go up?", "slug": "btc-test",
            "endDate": (self.now + timedelta(minutes=minutes)).isoformat(),
            "outcomes": '["Up", "Down"]', "outcomePrices": prices,
            "clobTokenIds": '["up-token", "down-token"]',
            "_liveAskPrices": live_prices,
            "liquidityNum": "1234.5",
        }

    def test_above_ninety_cents_is_included(self):
        rows = qualifying_rows([self.market(30)], self.now, self.deadline,
                               Decimal("0.90"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "Up")
        self.assertEqual(rows[0]["price"], Decimal("0.91"))

    def test_exactly_ninety_cents_is_excluded(self):
        rows = qualifying_rows([self.market(30, '["0.90", "0.10"]')],
                               self.now, self.deadline, Decimal("0.90"))
        self.assertEqual(rows, [])

    def test_market_outside_horizon_is_excluded(self):
        rows = qualifying_rows([self.market(61)], self.now, self.deadline,
                               Decimal("0.90"))
        self.assertEqual(rows, [])

    def test_non_up_down_outcome_is_excluded_by_default(self):
        market = self.market(30)
        market["outcomes"] = '["Yes", "No"]'
        rows = qualifying_rows([market], self.now, self.deadline, Decimal("0.90"))
        self.assertEqual(rows, [])

    def test_all_outcomes_mode_includes_yes(self):
        market = self.market(30)
        market["outcomes"] = '["Yes", "No"]'
        rows = qualifying_rows([market], self.now, self.deadline,
                               Decimal("0.90"), up_down_only=False)
        self.assertEqual(rows[0]["outcome"], "Yes")

    def test_cached_gamma_price_is_not_used(self):
        market = self.market(30, '["0.99", "0.01"]')
        market["_liveAskPrices"] = [Decimal("0.89"), Decimal("0.11")]
        rows = qualifying_rows([market], self.now, self.deadline, Decimal("0.90"))
        self.assertEqual(rows, [])

    def test_clob_best_ask_is_minimum_ask_level(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return [{
                    "asset_id": "up-token",
                    "asks": [{"price": "0.99", "size": "10"},
                             {"price": "0.92", "size": "2"}],
                }, {
                    "asset_id": "down-token",
                    "asks": [{"price": "0.08", "size": "5"}],
                }]

        class Client:
            def post(self, *args, **kwargs):
                return Response()

        market = self.market(30)
        market.pop("_liveAskPrices")
        add_live_ask_prices(Client(), [market])
        self.assertEqual(market["_liveAskPrices"],
                         [Decimal("0.92"), Decimal("0.08")])


if __name__ == "__main__":
    unittest.main()
