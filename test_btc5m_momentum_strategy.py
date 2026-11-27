import unittest
from argparse import Namespace

from btc5m_momentum_backtest import backtest
from btc5m_momentum_polymarket import (
    actual_fill_from_response, build_parser, fill_within_limit,
    is_confirmed_fok_kill,
)
from btc5m_momentum_strategy import fee_for, momentum_guard, momentum_signal, sweep_asks


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

    def test_default_value_cap_accepts_82_and_rejects_83(self):
        at_cap = {"Up": {"ask": .82, "asks": [{"price": .82, "size": 100}]}}
        above_cap = {"Up": {"ask": .83, "asks": [{"price": .83, "size": 100}]}}
        self.assertIsNotNone(momentum_signal(100, 100.03, at_cap))
        self.assertIsNone(momentum_signal(100, 100.03, above_cap))

    def test_crypto_taker_fee_formula(self):
        self.assertAlmostEqual(fee_for(.5, 10), .175)
        self.assertAlmostEqual(fee_for(.8, 10), .112)

    def test_live_defaults_match_frozen_strategy(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.stake, 2)
        self.assertEqual(args.entry_second, 150)
        self.assertEqual(args.minimum_move_bps, 2)
        self.assertEqual(args.ask_min, .45)
        self.assertEqual(args.ask_max, .82)
        self.assertEqual(args.slippage, .05)
        self.assertFalse(args.trend_guard)
        self.assertEqual(args.fok_retry_seconds, 0)
        self.assertEqual(args.fok_max_attempts, 1)

    def test_backtest_rejects_top_ask_above_fok_limit(self):
        first = {"seconds_into_window": 0, "coinbase": {"price": 100}}
        decision = {"seconds_into_window": 150, "coinbase": {"price": 100.03},
                    "Up": {"ask": .50, "ask_size": 100}, "Down": {}}
        execution = {"seconds_into_window": 151, "coinbase": {"price": 100.03},
                     "Up": {"ask": .60, "ask_size": 100}, "Down": {}}
        window = {"slug": "test", "first": first, "decision": decision,
                  "execution_rows": [execution], "winner": "Up"}
        args = Namespace(seed=42, delay_min_ms=300, delay_max_ms=800,
                         stake=2, minimum_move_bps=2, ask_min=.45,
                         ask_max=.90, slippage=.03)
        self.assertEqual(backtest([window], args), [])

    def test_guard_accepts_continuation(self):
        result = momentum_guard("Up", 100.01, 100, .70, .72)
        self.assertTrue(result["passed"])
        self.assertGreater(result["signed_recent_bps"], 0)

    def test_guard_rejects_coinbase_or_bid_reversal(self):
        self.assertFalse(momentum_guard("Up", 99.97, 100, .70, .72)["passed"])
        self.assertFalse(momentum_guard("Down", 99.97, 100, .60, .66)["passed"])

    def test_guard_needs_complete_history(self):
        self.assertIsNone(momentum_guard("Up", 100, None, .70, .72))

    def test_only_exact_fok_kill_is_retryable(self):
        import httpx
        from py_clob_client_v2.exceptions import PolyApiException
        exact = {"error": "order couldn't be fully filled. FOK orders are fully filled or killed."}
        self.assertTrue(is_confirmed_fok_kill(
            PolyApiException(httpx.Response(400, json=exact))))
        self.assertTrue(is_confirmed_fok_kill(exact))
        self.assertFalse(is_confirmed_fok_kill(
            PolyApiException(httpx.Response(400, json={"error": "not enough balance"}))))
        self.assertFalse(is_confirmed_fok_kill(TimeoutError()))

    def test_retry_depth_never_uses_asks_above_original_limit(self):
        book = {"asks": [{"price": .75, "size": 1},
                          {"price": .76, "size": 100}]}
        self.assertIsNone(fill_within_limit(book, 2, .75))
        self.assertIsNotNone(fill_within_limit(book, 2, .76))

    def test_actual_fill_uses_matched_amounts_and_fee_on_top(self):
        fill = actual_fill_from_response(
            {"makingAmount": "2", "takingAmount": "2.5"}, 412.75)
        self.assertEqual(fill["paid"], 2)
        self.assertEqual(fill["shares"], 2.5)
        self.assertAlmostEqual(fill["effective_price"], .8)
        self.assertAlmostEqual(fill["fee"], .028)
        self.assertAlmostEqual(fill["all_in_cost"], 2.028)
        self.assertEqual(fill["submit_to_response_ms"], 412.75)

    def test_actual_fill_rejects_malformed_response(self):
        self.assertIsNone(actual_fill_from_response(None, 100))
        self.assertIsNone(actual_fill_from_response({"makingAmount": "2"}, 100))
        self.assertIsNone(actual_fill_from_response(
            {"makingAmount": "bad", "takingAmount": "2.5"}, 100))
        self.assertIsNone(actual_fill_from_response(
            {"makingAmount": "2", "takingAmount": "0"}, 100))
        self.assertIsNone(actual_fill_from_response(
            {"makingAmount": "2", "takingAmount": "2.5"}, float("nan")))


if __name__ == "__main__":
    unittest.main()
