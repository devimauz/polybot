import unittest

from btc5m_momentum_strategy import momentum_signal, sweep_asks


class MomentumStrategyTests(unittest.TestCase):
    def test_sweep_uses_multiple_levels(self):
        fill = sweep_asks({"asks": [{"price": .5, "size": 100},
                                     {"price": .6, "size": 100}]}, 100)
        self.assertAlmostEqual(fill["shares"], 100 + 50 / .6)
        self.assertEqual(fill["worst_price"], .6)

    def test_insufficient_depth_skips(self):
        self.assertIsNone(sweep_asks({"asks": [{"price": .5, "size": 10}]}, 100))

    def test_signal_follows_coinbase_move(self):
        books = {"Up": {"ask": .6, "asks": [{"price": .6, "size": 1000}]}}
        signal = momentum_signal(100, 100.03, books)
        self.assertEqual(signal["side"], "Up")
        self.assertGreater(signal["move_bps"], 2)

    def test_small_move_is_filtered(self):
        self.assertIsNone(momentum_signal(100, 100.01, {}))


if __name__ == "__main__":
    unittest.main()
