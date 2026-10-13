import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from polymarket_btc5m_logger import CoinbaseFeed, JsonlWriter, parse_market, summarize_book


class PolymarketLoggerTests(unittest.TestCase):
    def test_parse_market_maps_outcomes_to_tokens(self):
        row = {"slug": "btc-updown-5m-600", "outcomes": '["Down","Up"]',
               "clobTokenIds": '["down-id","up-id"]', "conditionId": "condition",
               "active": True, "closed": False, "acceptingOrders": True}
        market = parse_market(row, row["slug"], 600)
        self.assertEqual(market["tokens"], {"Down": "down-id", "Up": "up-id"})
        self.assertEqual(market["end"], 900)

    def test_summarize_book_sorts_and_keeps_best_sizes(self):
        book = {"timestamp": "123", "bids": [{"price": "0.4", "size": "8"},
                                                {"price": "0.45", "size": "3"}],
                "asks": [{"price": "0.55", "size": "4"},
                         {"price": "0.5", "size": "6"}]}
        result = summarize_book(book)
        self.assertEqual((result["bid"], result["bid_size"]), (0.45, 3.0))
        self.assertEqual((result["ask"], result["ask_size"]), (0.5, 6.0))
        self.assertAlmostEqual(result["spread"], 0.05)

    def test_full_book_ignores_invalid_and_empty_levels(self):
        result = summarize_book({"bids": [{"price": "0.4", "size": "0"}],
                                 "asks": [{"price": "bad", "size": "2"}]}, True)
        self.assertEqual(result["bids"], [])
        self.assertEqual(result["asks"], [])
        self.assertIsNone(result["mid"])

    def test_coinbase_ticker_snapshot_has_price_and_age(self):
        feed = CoinbaseFeed()
        self.assertTrue(feed.handle({"type": "ticker", "product_id": "BTC-USD",
                                    "price": "70123.45", "time": "2026-10-12T00:00:00Z"},
                                   received=100.0))
        result = feed.snapshot(now=100.125)
        self.assertEqual(result["price"], 70123.45)
        self.assertEqual(result["exchange_ts"], "2026-10-12T00:00:00Z")
        self.assertEqual(result["age_ms"], 125.0)

    def test_writer_creates_one_file_per_window(self):
        with TemporaryDirectory() as folder:
            writer = JsonlWriter(Path(folder))
            writer.write({"event": "snapshot", "slug": "btc-updown-5m-600"})
            writer.write({"event": "snapshot", "slug": "btc-updown-5m-900"})
            writer.write({"event": "resolution", "slug": "btc-updown-5m-600",
                          "winner": "Up"})
            writer.close()
            first = (Path(folder) / "btc-updown-5m-600.jsonl").read_text().splitlines()
            second = (Path(folder) / "btc-updown-5m-900.jsonl").read_text().splitlines()
            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 1)


if __name__ == "__main__":
    unittest.main()
