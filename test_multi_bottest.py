import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from bottest import Bot, Journal, Settings, argument_parser, settings_from_args
from multi_bottest import MultiBot
from test_bottest import book


class MultiBotTests(unittest.TestCase):
    def setUp(self):
        self.bot = MultiBot.__new__(MultiBot)
        self.bot.live = False
        self.bot.settings = Settings()
        self.bot.asset_states = {}
        self.bot.market = None
        self.bot.signals = {}
        self.bot.journal = Journal(":memory:")
        self.addCleanup(self.bot.journal.db.close)
        self.bot.client = Mock()
        self.bot.log_prices = Mock()
        self.bot.get = Mock(side_effect=self.get)
        self.clock = patch("bottest.time.time", return_value=1789653400).start()
        self.addCleanup(patch.stopall)

    def get(self, url, **params):
        if url.endswith("/book"):
            return book()
        slug = params["slug"]
        return [{"slug": slug, "active": True, "closed": False, "acceptingOrders": True,
                 "endDate": datetime.fromtimestamp(1789653600, timezone.utc).isoformat(),
                 "outcomes": ["Up", "Down"], "clobTokenIds": [slug + "up", slug + "down"],
                 "conditionId": slug + "condition"}]

    def test_all_assets_share_protection_but_have_independent_markets(self):
        self.assertTrue(self.bot.cycle())
        self.assertTrue(self.bot.cycle())
        self.assertEqual(self.bot.journal.db.execute("SELECT count(*) FROM orders").fetchone()[0], 5)
        self.assertEqual(set(self.bot.asset_states), set(MultiBot.assets))
        self.bot.client.create_market_order.assert_not_called()
        self.bot.client.post_order.assert_not_called()
        discoveries = [c for c in self.bot.get.call_args_list if c.args[0].endswith("/markets")]
        self.assertEqual(len(discoveries), 5)
        quote_logs = [c for c in self.bot.log_prices.call_args_list if len(c.args) == 3]
        self.assertEqual(len(quote_logs), 20)

    def test_confirmation_timers_survive_switching_assets(self):
        self.bot.settings = Settings(confirmation_seconds=2)
        with patch("bottest.time.monotonic", return_value=10):
            self.bot.cycle()
        self.assertEqual(self.bot.journal.db.execute("SELECT count(*) FROM orders").fetchone()[0], 0)
        with patch("bottest.time.monotonic", return_value=12):
            self.bot.cycle()
        self.assertEqual(self.bot.journal.db.execute("SELECT count(*) FROM orders").fetchone()[0], 5)

    def test_data_failure_continues_other_assets(self):
        self.bot.step = Mock(side_effect=[httpx.ConnectError("offline")] + [None] * 4)
        self.assertFalse(self.bot.cycle())
        self.assertEqual(self.bot.step.call_count, 5)

    def test_cli_settings_are_shared_with_bottest(self):
        args = argument_parser().parse_args(["--max-spread", "0.02", "--confirmation-seconds", "2", "--amount", "3"])
        settings = settings_from_args(args)
        self.assertEqual(str(settings.max_spread), "0.02")
        self.assertEqual(settings.confirmation_seconds, 2)
        self.assertEqual(args.amount, 3)
        self.assertFalse(args.live)
        from decimal import Decimal
        self.assertEqual(settings.min_midpoint, Decimal("0.88"))
        self.assertEqual(settings.max_midpoint, Decimal("0.89"))
        overridden = settings_from_args(argument_parser().parse_args([
            "--min-midpoint", "0.90", "--max-midpoint", "0.94", "--max-price", "0.93"]))
        self.assertEqual(overridden.max_midpoint, Decimal("0.94"))

    def test_background_settlement_uses_correct_asset_file(self):
        with tempfile.TemporaryDirectory() as directory, patch("bottest.ROOT", Path(directory)):
            bot = MultiBot(False)
            bot.price_log_dir = Path(directory) / "logs"
            self.assertEqual(bot.journal_path.name, "multi-bottest-dry-run.sqlite3")
            slug = "xrp-updown-5m-1789653000"
            bot.journal.db.execute("INSERT INTO bottest_markets(market,end,condition_id,tokens) VALUES (?,0,'c',?)",
                                   (slug, json.dumps({"Up": "up", "Down": "down"})))
            bot.journal.db.commit()
            done = threading.Event()

            def settle(worker):
                Bot.settle_one(worker)
                done.set()

            market = {"condition_id": "c", "closed": True,
                      "tokens": [{"token_id": "up", "outcome": "Up", "winner": True}]}
            try:
                with patch.object(MultiBot, "get", return_value=market), patch.object(MultiBot, "settle_one", settle):
                    bot.start_maintenance()
                    self.assertTrue(done.wait(3), "Background settlement did not finish")
                    bot.close()
                rows = [json.loads(line) for line in (bot.price_log_dir / "XRP.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["asset"] == "XRP" for row in rows))
                self.assertFalse((bot.price_log_dir / "SOL.jsonl").exists())
                self.assertFalse(bot.worker.is_alive())
            finally:
                bot.close()


if __name__ == "__main__":
    unittest.main()
