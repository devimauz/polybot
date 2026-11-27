import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from polymarket_btc5m_logger import (
    CHAINLINK_TWAP_TOPIC,
    POLYBOLT_TWAP_CHANNEL,
    ChainlinkTwapFeed,
    CoinbaseFeed,
    JsonlWriter,
    Logger,
    PolymarketBookFeed,
    parse_market,
    retry_delay_seconds,
    summarize_book,
)


class PolymarketLoggerTests(unittest.TestCase):
    def test_retry_delay_exponentially_backs_off_and_caps(self):
        delays = [retry_delay_seconds(failures, 0.1) for failures in range(1, 9)]
        self.assertEqual(delays, [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0])

    def test_retry_delay_never_shortens_normal_poll(self):
        self.assertEqual(retry_delay_seconds(0, 2.5), 2.5)
        self.assertEqual(retry_delay_seconds(1, 2.5), 2.5)
        self.assertEqual(retry_delay_seconds(20, 45.0), 45.0)

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

    def test_chainlink_twap_subscription_uses_compact_string_filter(self):
        subscription = ChainlinkTwapFeed.subscription()["subscriptions"][0]
        self.assertEqual(subscription["topic"], CHAINLINK_TWAP_TOPIC)
        self.assertEqual(subscription["filters"], '{"symbol":"btc/usd"}')

    def test_chainlink_twap_snapshot_keeps_source_and_receive_times(self):
        feed = ChainlinkTwapFeed()
        message = {
            "topic": CHAINLINK_TWAP_TOPIC,
            "timestamp": 1792066000123,
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 1792066000000,
                "value": "70123.456789012345678",
                "full_accuracy_value": "70123456789012345678000",
                "window_s": 60,
            },
        }
        self.assertTrue(feed.handle(message, received=1792066000.25))
        result = feed.snapshot(now=1792066000.375)
        self.assertAlmostEqual(result["price"], 70123.45678901235)
        self.assertEqual(result["source_ts"], 1792066000000)
        self.assertEqual(result["source_ts_epoch"], 1792066000.0)
        self.assertEqual(result["age_ms"], 125.0)
        self.assertEqual(result["window_seconds"], 60)

    def test_chainlink_twap_decodes_e18_fallback(self):
        feed = ChainlinkTwapFeed()
        message = {
            "topic": CHAINLINK_TWAP_TOPIC,
            "payload": {
                "symbol": "btc/usd",
                "full_accuracy_value": "70123456789012345678000",
                "timestamp": 1792066000000,
            },
        }
        self.assertTrue(feed.handle(message, received=1792066000.25))
        self.assertAlmostEqual(feed.snapshot()["price"], 70123.45678901235)

    def test_polybolt_twap_auth_subscription_and_decimal_fallback(self):
        feed = ChainlinkTwapFeed("fake-key", "fake-secret", "fake-passphrase")
        self.assertEqual(feed.polybolt_auth(), {
            "op": "auth",
            "rid": "chainlink-auth",
            "auth": {
                "apiKey": "fake-key",
                "secret": "fake-secret",
                "passphrase": "fake-passphrase",
            },
        })
        self.assertEqual(feed.polybolt_subscription(), {
            "op": "subscribe",
            "rid": "chainlink-twap",
            "subscriptions": [{
                "channel": POLYBOLT_TWAP_CHANNEL,
                "filter": {"symbol": "btcusd", "window_seconds": 60},
            }],
        })
        message = {
            "channel": POLYBOLT_TWAP_CHANNEL,
            "ts": 1792066000123,
            "payload": {
                "symbol": "btcusd",
                "full_accuracy_value": "70123.45678901",
                "timestamp": 1792066000000,
                "window_seconds": 60,
                "source": "chainlink",
            },
        }
        self.assertTrue(feed.handle(message, received=1792066000.25))
        result = feed.snapshot(now=1792066000.375)
        self.assertAlmostEqual(result["price"], 70123.45678901)
        self.assertEqual(result["source"], "polymarket_polybolt")
        self.assertEqual(result["topic"], POLYBOLT_TWAP_CHANNEL)
        self.assertEqual(result["symbol"], "btcusd")
        self.assertEqual(result["source_ts_epoch"], 1792066000.0)
        self.assertEqual(result["age_ms"], 125.0)

    def test_polybolt_requires_all_three_credentials(self):
        with self.assertRaisesRegex(ValueError, "requires POLY_API_KEY"):
            ChainlinkTwapFeed("key-only", None, None)

    def test_polybolt_history_snapshot_seeds_newest_twap(self):
        feed = ChainlinkTwapFeed("fake-key", "fake-secret", "fake-passphrase")
        self.assertTrue(feed.handle({
            "channel": POLYBOLT_TWAP_CHANNEL,
            "seq": 3,
            "snapshot": True,
            "payload": {
                "symbol": "btcusd",
                "window_seconds": 60,
                "data": [
                    {"timestamp": 1792065999000, "value": "70122.5"},
                    {"timestamp": 1792066000000, "value": "70123.5"},
                ],
            },
        }, received=1792066000.25))
        result = feed.snapshot(now=1792066000.30)
        self.assertEqual(result["price"], 70123.5)
        self.assertEqual(result["source_ts_epoch"], 1792066000.0)
        self.assertEqual(result["seq"], 3)
        self.assertTrue(result["snapshot"])

    def test_twap_rejects_wrong_window(self):
        feed = ChainlinkTwapFeed("fake-key", "fake-secret", "fake-passphrase")
        self.assertFalse(feed.handle({
            "channel": POLYBOLT_TWAP_CHANNEL,
            "payload": {
                "symbol": "btcusd", "value": 70_000,
                "timestamp": 1792066000000, "window_seconds": 30,
            },
        }, received=1792066000.25))

    def test_market_book_subscription_enables_price_changes(self):
        message = PolymarketBookFeed.subscription(["up-token", "down-token"])
        self.assertEqual(message, {
            "assets_ids": ["up-token", "down-token"],
            "type": "market",
            "custom_feature_enabled": True,
        })

    def test_market_book_applies_initial_books_and_price_changes(self):
        feed = PolymarketBookFeed()
        feed.subscribe({"Up": "up-token", "Down": "down-token"})
        initial = [
            {
                "event_type": "book", "asset_id": "up-token",
                "timestamp": "100000", "hash": "up-1",
                "bids": [{"price": "0.42", "size": "5"},
                         {"price": "0.40", "size": "8"}],
                "asks": [{"price": "0.45", "size": "7"}],
            },
            {
                "event_type": "book", "asset_id": "down-token",
                "timestamp": "100000", "hash": "down-1",
                "bids": [{"price": "0.54", "size": "7"}],
                "asks": [{"price": "0.57", "size": "5"}],
            },
        ]
        self.assertTrue(feed.handle(initial, received=100.0))

        # A batch is applied in wire order. Size zero removes an aggregate
        # level, while a positive size replaces the level's current size.
        change = {
            "event_type": "price_change", "timestamp": "100100",
            "price_changes": [
                {"asset_id": "up-token", "side": "BUY", "price": "0.42",
                 "size": "0", "hash": "up-2"},
                {"asset_id": "up-token", "side": "BUY", "price": "0.43",
                 "size": "6", "hash": "up-3"},
                {"asset_id": "down-token", "side": "SELL", "price": "0.56",
                 "size": "4", "hash": "down-2"},
            ],
        }
        self.assertTrue(feed.handle(change, received=100.1))
        books = feed.snapshot(full_book=True, now=100.25)

        self.assertEqual(books["Up"]["bid"], 0.43)
        self.assertEqual(books["Up"]["bid_size"], 6.0)
        self.assertEqual(books["Up"]["bids"], [
            {"price": 0.43, "size": 6.0},
            {"price": 0.4, "size": 8.0},
        ])
        self.assertEqual(books["Down"]["ask"], 0.56)
        self.assertEqual(books["Down"]["ask_size"], 4.0)
        self.assertEqual(books["Down"]["book_age_ms"], 150.0)
        self.assertTrue(books["Up"]["book_ready"])

    def test_market_book_ignores_preimage_delta_and_stale_window(self):
        feed = PolymarketBookFeed()
        feed.subscribe({"Up": "old-up", "Down": "old-down"})
        delta = {
            "event_type": "price_change",
            "price_changes": [{"asset_id": "old-up", "side": "BUY",
                               "price": "0.4", "size": "10"}],
        }
        self.assertFalse(feed.handle(delta, received=100.0))
        self.assertFalse(feed.snapshot()["Up"]["book_ready"])

        feed.subscribe({"Up": "new-up", "Down": "new-down"})
        stale = {"event_type": "book", "asset_id": "old-up",
                 "bids": [{"price": "0.4", "size": "10"}], "asks": []}
        self.assertFalse(feed.handle(stale, received=100.1))
        books = feed.snapshot()
        self.assertEqual(set(books), {"Up", "Down"})
        self.assertFalse(books["Up"]["book_ready"])

    def test_same_generation_reconnect_requires_new_full_book_image(self):
        feed = PolymarketBookFeed()
        feed.subscribe({"Up": "up-token", "Down": "down-token"})
        old_book = {
            "event_type": "book", "asset_id": "up-token",
            "bids": [{"price": "0.48", "size": "2"}],
            "asks": [{"price": "0.49", "size": "3"}],
        }
        self.assertTrue(feed.handle(old_book, received=100.0))
        self.assertTrue(feed.snapshot(now=100.1)["Up"]["book_ready"])

        tokens, generation, old_session = feed._subscription_state()
        new_session = feed._begin_session(generation)
        self.assertIsNotNone(new_session)
        self.assertFalse(feed.snapshot(now=100.2)["Up"]["book_ready"])

        # A packet decoded from the previous socket cannot refill the cache.
        self.assertFalse(feed._handle(
            old_book, tokens, generation, old_session, received=100.3
        ))
        # A delta on the replacement socket is also ignored until its initial
        # full-book image has arrived.
        delta = {
            "event_type": "price_change",
            "price_changes": [{"asset_id": "up-token", "side": "BUY",
                               "price": "0.50", "size": "4"}],
        }
        self.assertFalse(feed._handle(
            delta, tokens, generation, new_session, received=100.4
        ))
        self.assertTrue(feed._handle(
            old_book, tokens, generation, new_session, received=100.5
        ))
        self.assertTrue(feed.snapshot(now=100.6)["Up"]["book_ready"])

    def test_market_book_liveness_timeout_boundary(self):
        self.assertFalse(PolymarketBookFeed.traffic_timed_out(10.0, 39.999))
        self.assertTrue(PolymarketBookFeed.traffic_timed_out(10.0, 40.0))
        self.assertTrue(PolymarketBookFeed.traffic_timed_out(
            10.0, 12.0, timeout=2.0
        ))

    def test_market_book_readiness_requires_both_exact_market_images(self):
        feed = PolymarketBookFeed()
        tokens = {"Up": "up-token", "Down": "down-token"}
        feed.subscribe(tokens)
        feed.handle({
            "event_type": "book", "asset_id": "up-token",
            "bids": [], "asks": [],
        }, received=100.0)

        self.assertFalse(feed.books_ready(tokens))
        self.assertFalse(feed.wait_for_books(tokens, timeout=0.0))

        feed.handle({
            "event_type": "book", "asset_id": "down-token",
            "bids": [], "asks": [],
        }, received=100.1)
        self.assertTrue(feed.books_ready(tokens))
        self.assertTrue(feed.wait_for_books(tokens, timeout=0.0))

        # Readiness for an old token pair cannot satisfy a new-window wait.
        feed.subscribe({"Up": "new-up", "Down": "new-down"})
        self.assertFalse(feed.books_ready(tokens))

    def test_logger_snapshots_cached_books_without_http_book_calls(self):
        writer = Mock()
        client = Mock()
        client.get.side_effect = AssertionError("snapshot must not use REST")
        feed = PolymarketBookFeed()
        feed.subscribe({"Up": "up-token", "Down": "down-token"})
        feed.handle([
            {"event_type": "book", "asset_id": "up-token",
             "bids": [{"price": "0.48", "size": "2"}],
             "asks": [{"price": "0.49", "size": "3"}]},
            {"event_type": "book", "asset_id": "down-token",
             "bids": [{"price": "0.51", "size": "3"}],
             "asks": [{"price": "0.52", "size": "2"}]},
        ], received=700.0)
        logger = Logger(client, writer, full_book=True, market_books=feed)
        market = {"slug": "btc-updown-5m-600", "start": 600, "end": 900,
                  "condition_id": "condition", "tokens": {
                      "Up": "up-token", "Down": "down-token"}}
        with patch("polymarket_btc5m_logger.time.time",
                   side_effect=[700.1, 700.1, 700.2, 700.2]):
            logger.snapshot(market)
            logger.snapshot(market)

        self.assertEqual(writer.write.call_count, 2)
        self.assertEqual(client.get.call_count, 0)
        first, second = [call.args[0] for call in writer.write.call_args_list]
        self.assertEqual(first["Up"]["ask"], 0.49)
        self.assertEqual(first["Up"]["asks"], [{"price": 0.49, "size": 3.0}])
        self.assertEqual(second["Up"]["hash"], first["Up"]["hash"])
        self.assertGreater(second["ts_epoch"], first["ts_epoch"])

    def test_snapshot_timestamp_covers_future_feed_receive_times(self):
        writer = Mock()
        books = PolymarketBookFeed()
        books.subscribe({"Up": "up-token", "Down": "down-token"})
        books.handle([
            {"event_type": "book", "asset_id": "up-token",
             "bids": [{"price": "0.48", "size": "2"}],
             "asks": [{"price": "0.49", "size": "3"}]},
            {"event_type": "book", "asset_id": "down-token",
             "bids": [{"price": "0.51", "size": "3"}],
             "asks": [{"price": "0.52", "size": "2"}]},
        ], received=700.4)
        coinbase = CoinbaseFeed()
        coinbase.handle({"type": "ticker", "product_id": "BTC-USD",
                         "price": "70123", "time": "exchange-time"},
                        received=700.5)
        chainlink = ChainlinkTwapFeed()
        chainlink.handle({
            "topic": CHAINLINK_TWAP_TOPIC,
            "payload": {"symbol": "btc/usd", "value": "70122",
                        "timestamp": 700500},
        }, received=700.6)
        logger = Logger(Mock(), writer, full_book=False, coinbase=coinbase,
                        chainlink_twap=chainlink, market_books=books)
        market = {"slug": "btc-updown-5m-600", "start": 600, "end": 900,
                  "condition_id": "condition", "tokens": {
                      "Up": "up-token", "Down": "down-token"}}

        with patch("polymarket_btc5m_logger.time.time",
                   side_effect=[700.1, 700.2]):
            logger.snapshot(market)

        row = writer.write.call_args.args[0]
        self.assertEqual(row["ts_epoch"], 700.6)
        self.assertGreaterEqual(row["ts_epoch"], row["Up"]["book_received_ts"])
        self.assertGreaterEqual(row["ts_epoch"], row["coinbase"]["received_ts"])
        self.assertGreaterEqual(
            row["ts_epoch"], row["chainlink_twap_60s"]["received_ts"]
        )
        self.assertEqual(row["Up"]["book_age_ms"], 200.0)
        self.assertEqual(row["coinbase"]["age_ms"], 100.0)
        self.assertEqual(row["chainlink_twap_60s"]["age_ms"], 0.0)

    def test_one_shot_never_writes_a_partial_book_snapshot(self):
        writer = Mock()
        feed = PolymarketBookFeed()
        tokens = {"Up": "up-token", "Down": "down-token"}
        market = {"slug": "btc-updown-5m-600", "start": 600, "end": 900,
                  "condition_id": "condition", "tokens": tokens}
        feed.subscribe(tokens)
        feed.handle({
            "event_type": "book", "asset_id": "up-token",
            "bids": [{"price": "0.48", "size": "2"}],
            "asks": [{"price": "0.49", "size": "3"}],
        }, received=700.0)
        logger = Logger(Mock(), writer, full_book=False, market_books=feed)

        self.assertFalse(logger.snapshot_once_when_ready(market, timeout=0.0))
        writer.write.assert_not_called()

        feed.handle({
            "event_type": "book", "asset_id": "down-token",
            "bids": [{"price": "0.51", "size": "3"}],
            "asks": [{"price": "0.52", "size": "2"}],
        }, received=700.1)
        with patch("polymarket_btc5m_logger.time.time",
                   side_effect=[700.2, 700.2, 700.2]):
            self.assertTrue(logger.snapshot_once_when_ready(market, timeout=0.0))

        row = writer.write.call_args.args[0]
        self.assertTrue(row["Up"]["book_ready"])
        self.assertTrue(row["Down"]["book_ready"])

    def test_resolution_failure_is_async_and_keeps_discovery_retry_state(self):
        request_started = threading.Event()
        release_request = threading.Event()

        def blocked_failure(*_args, **_kwargs):
            request_started.set()
            release_request.wait(2.0)
            raise RuntimeError("settlement endpoint unavailable")

        client = Mock()
        client.get.side_effect = blocked_failure
        logger = Logger(client, Mock(), full_book=False)
        logger.pending_resolutions["btc-updown-5m-600"] = "condition"
        logger.last_discovery_slug = "btc-updown-5m-900"
        logger.last_discovery_attempt = 101.25
        result = []
        caller = threading.Thread(
            target=lambda: result.append(logger.check_resolutions(900.0))
        )
        try:
            caller.start()
            self.assertTrue(request_started.wait(1.0))
            caller.join(0.25)
            self.assertFalse(
                caller.is_alive(),
                "resolution HTTP must not block the snapshot-loop caller",
            )
            self.assertEqual(result, [True])
            self.assertTrue(logger._resolution_thread.is_alive())
        finally:
            release_request.set()
            caller.join(1.0)
            if logger._resolution_thread is not None:
                logger._resolution_thread.join(1.0)

        with self.assertLogs("polymarket-btc5m-logger", level="WARNING") as logs:
            self.assertFalse(logger.check_resolutions(900.1))
        self.assertIn("next batch eligible at 930.000", "\n".join(logs.output))
        self.assertEqual(logger.last_discovery_slug, "btc-updown-5m-900")
        self.assertEqual(logger.last_discovery_attempt, 101.25)
        self.assertEqual(logger.last_resolution_check, 900.0)
        self.assertEqual(client.get.call_count, 1)
        logger.close()

    def test_missing_market_discovery_is_rate_limited(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = []
        client = Mock()
        client.get.return_value = response
        logger = Logger(client, Mock(), full_book=False,
                        market_books=PolymarketBookFeed())

        self.assertIsNone(logger.discover(600.0))
        self.assertIsNone(logger.discover(600.1))
        self.assertEqual(client.get.call_count, 1)
        self.assertIsNone(logger.discover(605.0))
        self.assertEqual(client.get.call_count, 2)

    def test_discovery_resubscribes_tokens_at_window_change(self):
        def response(rows):
            result = Mock()
            result.raise_for_status.return_value = None
            result.json.return_value = rows
            return result

        first = {"slug": "btc-updown-5m-600", "outcomes": '["Up","Down"]',
                 "clobTokenIds": '["up-1","down-1"]',
                 "conditionId": "condition-1"}
        second = {"slug": "btc-updown-5m-900", "outcomes": '["Up","Down"]',
                  "clobTokenIds": '["up-2","down-2"]',
                  "conditionId": "condition-2"}
        client = Mock()
        client.get.side_effect = [response([first]), response([second])]
        feed = PolymarketBookFeed()
        logger = Logger(client, Mock(), full_book=False, market_books=feed)

        self.assertEqual(logger.discover(600.0)["tokens"]["Up"], "up-1")
        self.assertEqual(feed.tokens, {"up-1": "Up", "down-1": "Down"})
        self.assertEqual(logger.discover(900.0)["tokens"]["Up"], "up-2")
        self.assertEqual(feed.tokens, {"up-2": "Up", "down-2": "Down"})
        self.assertEqual(logger.pending_resolutions,
                         {"btc-updown-5m-600": "condition-1",
                          "btc-updown-5m-900": "condition-2"})
        self.assertGreaterEqual(feed.generation, 2)

    def test_discovered_condition_survives_logger_restart(self):
        row = {"slug": "btc-updown-5m-600", "outcomes": '["Up","Down"]',
               "clobTokenIds": '["up-token","down-token"]',
               "conditionId": "condition-600"}
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [row]
        client = Mock()
        client.get.return_value = response

        with TemporaryDirectory() as folder:
            directory = Path(folder)
            first_writer = JsonlWriter(directory)
            first = Logger(client, first_writer, full_book=False)
            self.assertIsNotNone(first.discover(600.0))
            state_path = first.resolution_state_path
            self.assertTrue(state_path.is_file())
            self.assertEqual(json.loads(state_path.read_text()), {
                "version": 1,
                "pending": {"btc-updown-5m-600": "condition-600"},
            })
            first_writer.close()

            second_writer = JsonlWriter(directory)
            second = Logger(Mock(), second_writer, full_book=False)
            self.assertEqual(second.pending_resolutions, {
                "btc-updown-5m-600": "condition-600",
            })
            self.assertFalse(Path(str(state_path) + ".tmp").exists())
            second_writer.close()

    def test_resolution_sidecar_validates_each_loaded_entry(self):
        with TemporaryDirectory() as folder:
            writer = JsonlWriter(Path(folder))
            state_path = Path(folder) / ".btc5m-pending-resolutions.json"
            state_path.write_text(json.dumps({
                "version": 1,
                "pending": {
                    "btc-updown-5m-600": "valid-condition",
                    "btc-updown-5m-601": "wrong-boundary",
                    "not-a-market": "condition",
                    "btc-updown-5m-900": "  whitespace  ",
                    "btc-updown-5m-1200": 123,
                },
            }), encoding="utf-8")

            with self.assertLogs(
                    "polymarket-btc5m-logger", level="WARNING") as logs:
                logger = Logger(Mock(), writer, full_book=False)
            self.assertEqual(logger.pending_resolutions, {
                "btc-updown-5m-600": "valid-condition",
            })
            self.assertIn("ignored 4 invalid", "\n".join(logs.output))
            writer.close()

    def test_resolution_poll_waits_for_window_end_then_persists_deletion(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "tokens": [{"outcome": "Up", "winner": True}],
        }
        client = Mock()
        client.get.return_value = response

        with TemporaryDirectory() as folder:
            directory = Path(folder)
            writer = JsonlWriter(directory)
            logger = Logger(client, writer, full_book=False)
            self.assertTrue(logger._remember_resolution(
                "btc-updown-5m-600", "condition-600"
            ))

            self.assertFalse(logger.check_resolutions(899.999))
            self.assertEqual(client.get.call_count, 0)
            self.assertTrue(logger.check_resolutions(900.0))
            logger._resolution_thread.join(1.0)
            self.assertFalse(logger._resolution_thread.is_alive())
            self.assertFalse(logger.check_resolutions(900.1))
            self.assertEqual(client.get.call_count, 1)
            self.assertEqual(logger.pending_resolutions, {})
            self.assertEqual(json.loads(
                logger.resolution_state_path.read_text(encoding="utf-8")
            )["pending"], {})
            logger.close()
            writer.close()

            restarted_writer = JsonlWriter(directory)
            restarted = Logger(Mock(), restarted_writer, full_book=False)
            self.assertEqual(restarted.pending_resolutions, {})
            restarted_writer.close()

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
