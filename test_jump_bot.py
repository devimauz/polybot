import time
import unittest
from decimal import Decimal
from unittest.mock import patch

from jump_bot import JumpBot, JumpJournal, MarketFeed


class JumpBotTests(unittest.TestCase):
    def setUp(self):
        self.bot = JumpBot.__new__(JumpBot)
        self.bot.history = {"Up": __import__("collections").deque(),
                            "Down": __import__("collections").deque()}

    def test_detects_ten_cent_rise_inside_300ms(self):
        self.assertIsNone(self.bot._record_and_detect("Up", Decimal("0.40"), 1.0))
        self.assertIsNone(self.bot._record_and_detect("Up", Decimal("0.49"), 1.2))
        self.assertEqual(self.bot._record_and_detect("Up", Decimal("0.50"), 1.299),
                         Decimal("0.10"))

    def test_does_not_use_sample_older_than_300ms(self):
        self.bot._record_and_detect("Down", Decimal("0.30"), 1.0)
        self.assertIsNone(self.bot._record_and_detect("Down", Decimal("0.40"), 1.301))

    def test_journal_allows_only_one_open_position(self):
        journal = JumpJournal(":memory:")
        self.addCleanup(journal.db.close)
        self.assertTrue(journal.reserve_position("btc-updown-5m-1000", "Down"))
        self.assertFalse(journal.reserve_position("btc-updown-5m-1300", "Up"))
        journal.update_position("btc-updown-5m-1000", "open",
                                entry_price=Decimal("0.40"), shares=Decimal("25"))
        position = journal.active()
        self.assertEqual((position.outcome, position.entry_price, position.shares),
                         ("Down", Decimal("0.40"), Decimal("25")))
        journal.update_position(position.market, "closed_tp")
        self.assertTrue(journal.reserve_position("btc-updown-5m-1300", "Up"))

    def test_live_status_shows_quotes_and_flat_state(self):
        from bot import Quote
        self.bot._status_width = 0
        quote = lambda mid, bid, ask: Quote(Decimal(mid), Decimal(ask),
                                            Decimal(ask), "0.01", False, Decimal(bid))
        quotes = {"Up": quote("0.51", "0.50", "0.52"),
                  "Down": quote("0.49", "0.48", "0.50")}
        with patch("jump_bot.time.time", return_value=1050), \
                patch("jump_bot.sys.stdout.write") as write:
            self.bot._live_status("btc-updown-5m-1000", quotes,
                                  {"Up": Decimal("0.10"), "Down": None})
        line = write.call_args.args[0]
        self.assertIn("UP bid/ask/mid 0.50/0.52/0.510", line)
        self.assertIn("Δ300ms +0.100", line)
        self.assertIn("| FLAT", line)

    def test_websocket_book_and_price_change_update_quotes(self):
        feed = MarketFeed()
        tokens = {"up-token": "Up", "down-token": "Down"}
        feed.tokens = tokens
        feed.generation = 1
        feed._handle([
            {"asset_id": "up-token",  # initial snapshots omit event_type
             "bids": [{"price": "0.50", "size": "10"}],
             "asks": [{"price": "0.52", "size": "10"}]},
            {"event_type": "book", "asset_id": "down-token",
             "bids": [{"price": "0.48", "size": "10"}],
             "asks": [{"price": "0.50", "size": "10"}]},
        ], tokens, 1)
        self.assertEqual(feed.quotes()["Up"].probability, Decimal("0.51"))
        feed._handle({"event_type": "price_change", "price_changes": [
            {"asset_id": "up-token", "best_bid": "0.60", "best_ask": "0.62"}
        ]}, tokens, 1)
        self.assertEqual(feed.quotes()["Up"].probability, Decimal("0.61"))


if __name__ == "__main__":
    unittest.main()
