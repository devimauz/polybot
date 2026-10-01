import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from botqwenlocal import Bot, quote_book
from multi_bot import MultiBot
from test_bot import book


class PriceLogTests(unittest.TestCase):
    def test_per_asset_append_and_fill_prices(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            directory = Path(folder) / "prices"
            for asset in ("btc", *MultiBot.assets):
                bot = (Bot if asset == "btc" else MultiBot).__new__(Bot if asset == "btc" else MultiBot)
                bot.asset = asset
                bot.live = True
                bot.price_log_dir = directory
                q = quote_book(book(), time.time())
                bot.log_prices("market", "Up", q)
                bot.log_prices("market", "Up", q, "buy_matched", {
                    "makingAmount": "2.1", "takingAmount": "2.5", "orderID": "order"})
                bot.live = False
                bot.log_prices("market", "Down", q, "buy_simulated")
                rows = [json.loads(line) for line in (directory / f"{asset.upper()}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["buy_price"], "0.83")
                self.assertEqual(rows[0]["bid_price"], "0.81")
                self.assertEqual(rows[0]["sell_price"], "0.81")
                self.assertEqual(rows[1]["fill_price"], "0.84")
                self.assertEqual(rows[1]["asset"], asset.upper())
                self.assertEqual(rows[2]["mode"], "dry-run")
                self.assertIsNone(rows[2]["fill_price"])
            self.assertEqual(len(list(directory.glob("*.jsonl"))), 5)

    def test_unwritable_log_does_not_interrupt_bot(self):
        bot = Bot.__new__(Bot)
        bot.live = True
        bot.price_log_dir = Mock()
        bot.price_log_dir.mkdir.side_effect = PermissionError()
        with self.assertLogs("btc-bot", level="WARNING"):
            bot.log_prices("market", "Up", quote_book(book(), time.time()))


if __name__ == "__main__":
    unittest.main()
