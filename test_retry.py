import time
import unittest
from unittest.mock import Mock, patch

import httpx
from py_clob_client_v2.exceptions import PolyApiException
from botqwenlocal import Bot


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.bot = Bot.__new__(Bot)
        self.bot.client = Mock()
        self.signed = object()
        self.error = PolyApiException(error_msg="Request exception!")
        self.sleep = patch("bot.time.sleep").start()
        self.addCleanup(patch.stopall)

    def test_disconnect_then_success_reuses_signed_order(self):
        response = {"success": True, "status": "matched"}
        self.bot.client.post_order.side_effect = [self.error, response]
        self.assertEqual(self.bot.post_with_retry(self.signed, time.time() + 100), response)
        self.assertEqual(self.bot.client.post_order.call_count, 2)
        for call in self.bot.client.post_order.call_args_list:
            self.assertIs(call.args[0], self.signed)
        self.sleep.assert_called_once_with(1)

    def test_persistent_disconnect_is_bounded(self):
        self.bot.client.post_order.side_effect = self.error
        self.assertIsNone(self.bot.post_with_retry(self.signed, time.time() + 100))
        self.assertEqual(self.bot.client.post_order.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1, 2])

    def test_no_retry_near_close(self):
        self.bot.client.post_order.side_effect = self.error
        self.assertIsNone(self.bot.post_with_retry(self.signed, time.time() + 10.5))
        self.bot.client.post_order.assert_called_once()
        self.sleep.assert_not_called()

    def test_duplicate_after_disconnect_remains_uncertain(self):
        duplicate = PolyApiException(httpx.Response(400, json={"error": "order is invalid. Duplicated."}))
        self.bot.client.post_order.side_effect = [self.error, duplicate]
        self.assertIsNone(self.bot.post_with_retry(self.signed, time.time() + 100))
        self.assertEqual(self.bot.client.post_order.call_count, 2)

    def test_validation_error_does_not_retry(self):
        self.bot.client.post_order.side_effect = PolyApiException(httpx.Response(400, json={"error": "invalid"}))
        with self.assertRaises(PolyApiException):
            self.bot.post_with_retry(self.signed, time.time() + 100)
        self.bot.client.post_order.assert_called_once()

    def test_rejection_response_after_disconnect_is_not_assumed_unfilled(self):
        self.bot.client.post_order.side_effect = [self.error, {"success": False, "errorMsg": "FOK not filled"}]
        self.assertIsNone(self.bot.post_with_retry(self.signed, time.time() + 100))


if __name__ == "__main__":
    unittest.main()
