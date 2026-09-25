import unittest

from dashboard import summarize


class PerformanceTests(unittest.TestCase):
    def position(self, asset, **values):
        return {"asset": asset, "slug": "btc-updown-5m-1789510500", "outcome": "Up",
                "size": 10, "avgPrice": .8, "initialValue": 8, "currentValue": 9,
                "realizedPnl": -.1, **values}

    def test_open_gain_is_not_reported_as_won(self):
        result = summarize([self.position("a")], [], [])
        self.assertAlmostEqual(result["metrics"]["realized"], -.1)
        self.assertEqual(result["metrics"]["unrealized"], 1)
        self.assertEqual(result["metrics"]["wins"], 0)
        self.assertAlmostEqual(result["positions"][0]["pnl"], .9)

    def test_closed_wins_losses_and_curve(self):
        result = summarize([], [self.position("a", realizedPnl=2, timestamp=100),
                                self.position("b", realizedPnl=-3, timestamp=200)], [])
        self.assertEqual(result["metrics"]["wins"], 1)
        self.assertEqual(result["metrics"]["losses"], 1)
        self.assertEqual(result["metrics"]["realized"], -1)
        self.assertEqual(result["curve"][-1]["value"], -1)
        self.assertEqual(result["metrics"]["position_value"], 0)

    def test_claimable_not_counted_as_realized_or_closed_win(self):
        result = summarize([self.position("a", redeemable=True, currentValue=10)], [], [])
        self.assertEqual(result["metrics"]["claimable"], 10)
        self.assertEqual(result["metrics"]["wins"], 0)
        self.assertEqual(result["metrics"]["realized"], -.1)

    def test_excludes_altcoins_and_duplicate_assets(self):
        p = self.position("a")
        result = summarize([p, dict(p), self.position("b", slug="eth-updown-5m-1789510500")],
                           [dict(p, realizedPnl=100)], [])
        self.assertEqual(len(result["positions"]), 1)
        self.assertEqual(result["metrics"]["position_value"], 9)


if __name__ == "__main__":
    unittest.main()
