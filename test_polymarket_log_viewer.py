import unittest

from polymarket_log_viewer import row_values, snapshot_rows, winner_of


class PolymarketLogViewerTests(unittest.TestCase):
    def test_filters_snapshots_and_finds_resolution(self):
        rows = [{"event": "market"}, {"event": "snapshot", "ts": "x"},
                {"event": "resolution", "winner": "Up"}]
        self.assertEqual(len(snapshot_rows(rows)), 1)
        self.assertEqual(winner_of(rows), "Up")

    def test_row_values_extracts_nested_prices(self):
        row = {"event": "snapshot", "ts": "2026-10-13T01:25:00.123Z",
               "seconds_into_window": 2.5,
               "Up": {"bid": .4, "ask": .41, "bid_size": 2, "ask_size": 3},
               "Down": {"bid": .59, "ask": .6, "bid_size": 4, "ask_size": 5},
               "coinbase": {"price": 81234.5, "age_ms": 22}}
        values = row_values(row)
        self.assertEqual(values[2:4], ("0.400", "0.410"))
        self.assertEqual(values[10], "81234.50")
        self.assertEqual(values[11], "22")


if __name__ == "__main__":
    unittest.main()
