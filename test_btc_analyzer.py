from btc_analyzer import Feed


def feed():
    result = Feed()
    result.tokens = {"u": "Up", "d": "Down"}
    return result


def test_book_unsorted_and_empty_sides():
    f = feed()
    f.process({"event_type": "book", "asset_id": "u", "timestamp": "100",
               "bids": [{"price": "0.4", "size": "2"}, {"price": "0.5", "size": "3"}],
               "asks": [{"price": "0.9", "size": "0"}, {"price": "0.6", "size": "3"}]})
    q = f.snapshot()["quotes"]["Up"]
    assert (q.bid, q.ask) == (0.5, 0.6)
    f.process({"event_type": "book", "asset_id": "u", "timestamp": "101", "bids": [], "asks": []})
    q = f.snapshot()["quotes"]["Up"]
    assert q.bid is None and q.ask is None


def test_batched_changes_both_outcomes_and_old_event():
    f = feed()
    f.process({"event_type": "price_change", "timestamp": "200", "price_changes": [
        {"asset_id": "u", "best_bid": "0.6", "best_ask": "0.62"},
        {"asset_id": "d", "best_bid": "0.38", "best_ask": "0.4"}]})
    f.process({"event_type": "best_bid_ask", "timestamp": "100", "asset_id": "u", "best_bid": "0.1", "best_ask": "0.2"})
    quotes = f.snapshot()["quotes"]
    assert quotes["Up"].bid == 0.6
    assert quotes["Down"].ask == 0.4
    assert quotes["Up"].updates == 1


def test_sentinels_unknown_token_and_snapshot_isolation():
    f = feed()
    f.process([{"event_type": "best_bid_ask", "asset_id": "u", "best_bid": "0", "best_ask": "1"},
               {"event_type": "best_bid_ask", "asset_id": "old", "best_bid": "0.3", "best_ask": "0.4"}])
    q = f.snapshot()["quotes"]["Up"]
    assert q.bid is None and q.ask is None
    q.bid = 0.9
    assert f.snapshot()["quotes"]["Up"].bid is None


def test_discovery_maps_outcomes_by_name_and_resets(monkeypatch):
    f = feed()
    monkeypatch.setattr("btc_analyzer.time.time", lambda: 600)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return [{"slug": "btc-updown-5m-600", "acceptingOrders": True,
                     "outcomes": '["Down", "Up"]', "clobTokenIds": '["new-d", "new-u"]'}]
    class Client:
        def get(self, *args, **kwargs):
            return Response()
    f.discover(Client())
    assert f.tokens == {"new-d": "Down", "new-u": "Up"}
    assert f.end == 900
    assert all(q.updates == 0 for q in f.snapshot()["quotes"].values())
