#!/usr/bin/env python3
"""pm_lag_probe.py - millisecond-resolution lag measurement on the live BTC 5m market.

Answers one question: when the Coinbase tape moves, how many milliseconds does
the Polymarket CLOB book take to reprice, and is that lag longer than OUR
achievable action time (streaming detect + CLOB REST order RTT)?

Streams (all with local time_ns stamps):
  - Polymarket CLOB market WS (ws-subscriptions-clob): book snapshots +
    price_change deltas -> touch (best bid/ask) of both tokens, emitted on every
    touch change (plus 250ms heartbeat).
  - Coinbase Exchange WS (ws-direct): BTC-USD matches (every trade: px/sz/side)
    and ticker (best bid/ask + exchange ISO ts) -> local-vs-exchange offset.
  - RTT probe every 5s: GET clob.polymarket.com/book?token_id=... timed in ms
    (proxy for order-POST latency).
  - gamma resolution of the current window's clobTokenIds at each boundary.

Output: ~/poly/laglog/lag-<window_start>.jsonl, one file per 5m window, gzipped
after 4h, oldest purged if free disk < 2 GB. systemd: pm-lag-probe.service.
"""
import json, os, ssl, sys, threading, time, urllib.request, urllib.error, gzip, glob
import websocket  # websocket-client 1.8.x

WORK = os.path.expanduser("~/poly/laglog")
os.makedirs(WORK, exist_ok=True)
GAMMA = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_REST = "https://clob.polymarket.com"
CB_WS = "wss://ws-feed.exchange.coinbase.com"  # ws-direct refuses this AWS IP
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) lag-probe"}

lock = threading.Lock()
toks = {"U": None, "D": None, "ws": 0, "slug": None, "cond": None}
stop = {"pm": False, "cb": False}


def emit(obj):
    obj["t_ns"] = time.time_ns()
    w = (obj["t_ns"] // 10**9 // 300) * 300
    with lock:
        fh = emit._f.get(w)
        if fh is None:
            emit._f[w] = fh = open(os.path.join(WORK, f"lag-{w}.jsonl"), "a", buffering=1)
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
emit._f = {}


def resolve(w):
    slug = f"btc-updown-5m-{w}"
    try:
        req = urllib.request.Request(f"{GAMMA}/events?slug={slug}", headers=HDR)
        e = json.load(urllib.request.urlopen(req, timeout=8))
        m = e[0]["markets"][0]
        t = json.loads(m["clobTokenIds"])
        emit({"src": "meta", "ws": w, "slug": slug, "cond": m.get("conditionId"),
              "tokU": t[0], "tokD": t[1]})
        return {"U": t[0], "D": t[1], "ws": w, "slug": slug, "cond": m.get("conditionId")}
    except Exception as ex:
        emit({"src": "meta_err", "ws": w, "err": str(ex)[:120]})
        return None


pending = {}
pm_ws_ref = {"ws": None}


def resolve_loop():
    cur = None
    while True:
        now = int(time.time())
        w = (now // 300) * 300
        # pre-resolve the NEXT window in its last 30s so the switch is instant
        if w + 300 not in pending and now >= w + 300 - 30:
            r = resolve(w + 300)
            if r:
                pending[w + 300] = r
            elif now >= w + 300:
                pending[w + 300] = None  # market not live yet; live-resolve below
        if w != cur:
            cur = w
            r = pending.pop(w, None) or resolve(w)
            while r is None and (int(time.time()) // 300) * 300 == w:
                time.sleep(2.0)
                r = resolve(w)
            if r:
                toks.update(r)
                old = pm_ws_ref["ws"]
                if old:
                    try:
                        old.close()   # forces run_forever to return -> resubscribes
                    except Exception:
                        pass
        time.sleep(1.0)


def pm_touch_of(book):
    try:
        bb = max((float(x["price"]) for x in book["bids"] if float(x["size"]) > 0), default=0.0)
        ba = min((float(x["price"]) for x in book["asks"] if float(x["size"]) > 0), default=1.0)
    except Exception:
        bb, ba = 0.0, 1.0
    return bb, ba


def on_message(ws, msg):
    if not msg:
        return  # keepalive, NOT a closure (documented pitfall)
    try:
        data = json.loads(msg)
    except Exception:
        return
    events = data if isinstance(data, list) else [data]
    for ev in events:
        et = ev.get("event_type")
        if et == "book":
            a = ev.get("asset_id")
            with books.lock:
                books.d[a] = {"bids": ev.get("bids", []), "asks": ev.get("asks", [])}
            touch_emit("book", ev.get("timestamp"))
        elif et == "price_change":
            a = ev.get("asset_id")
            ch = ev.get("changes") or [ev]
            with books.lock:
                b = books.d.setdefault(a, {"bids": [], "asks": []})
                for c in ch:
                    side = "bids" if str(c.get("side", "BUY")).upper() == "BUY" else "asks"
                    px = str(c.get("price"))
                    lst = b[side]
                    for x in lst:
                        if str(x["price"]) == px:
                            x["size"] = c.get("size", "0")
                            break
                    else:
                        if float(c.get("size", 0) or 0) > 0:
                            lst.append({"price": px, "size": c.get("size")})
            touch_emit("pc", ev.get("timestamp"))
        elif et == "last_trade_price":
            emit({"src": "pm_trade", "asset": ev.get("asset_id"),
                  "p": ev.get("price"), "sz": ev.get("size"), "side": ev.get("side"),
                  "ets": ev.get("timestamp")})


class Books:
    lock = threading.RLock()
    d = {}
books = Books()


def touch_emit(ev, ets):
    U, D = toks["U"], toks["D"]
    if not U or not D:
        return
    with books.lock:
        bu = books.d.get(U); bd = books.d.get(D)
    if not bu or not bd:
        return
    bbU, baU = pm_touch_of(bu)
    bbD, baD = pm_touch_of(bd)
    t = (bbU, baU, bbD, baD)
    if t == getattr(touch_emit, "last", None) and time.time_ns() - getattr(touch_emit, "last_ns", 0) < 250 * 10**6:
        return
    touch_emit.last = t
    touch_emit.last_ns = time.time_ns()
    emit({"src": "pm", "ev": ev, "ets": ets, "bbU": bbU, "baU": baU,
          "bbD": bbD, "baD": baD})


def on_error(ws, e):
    emit({"src": "ws_err", "feed": "pm", "err": str(e)[:100]})


def pm_loop():
    sub_at = None
    while True:
        try:
            if not toks["U"]:
                time.sleep(1.5); continue
            if sub_at != toks["ws"]:
                ws = websocket.WebSocketApp(CLOB_WS, on_message=on_message,
                                            on_error=on_error)
                ws.toks = (toks["U"], toks["D"], toks["ws"])
                def _open(wsv=ws):
                    wsv.send(json.dumps({"assets_ids": [wsv.toks[0], wsv.toks[1]],
                                         "type": "market"}))
                ws.on_open = _open
                sub_at = toks["ws"]
                pm_ws_ref["ws"] = ws
                threading.Thread(target=pm_ping, args=(ws,), daemon=True).start()
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                               ping_interval=25, ping_timeout=10)
                sub_at = None
        except Exception as e:
            emit({"src": "pm_loop_err", "err": str(e)[:120]})
        time.sleep(2.0)


def pm_ping(ws):
    while True:
        time.sleep(10.0)
        try:
            if ws.sock and ws.sock.connected:
                ws.send("PING")
            else:
                break
        except Exception:
            break


# ============== 15-MINUTE MARKET MIRROR (btc-updown-15m) ==============
# Mirrors the 5m plumbing onto the 15-min slug: its own WS connection, own book
# store, own touch emitter. Rows land in the same per-5m files with distinct src
# names (meta15 / pm15 / pm15_trade) so old analyzers are untouched.
toks15 = {"U": None, "D": None, "ws": 0, "slug": None, "cond": None}
pending15 = {}
pm15_ws_ref = {"ws": None}

class BookStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.d = {}
books15 = BookStore()

def resolve15(w):
    slug = f"btc-updown-15m-{w}"
    try:
        req = urllib.request.Request(f"{GAMMA}/events?slug={slug}", headers=HDR)
        e = json.load(urllib.request.urlopen(req, timeout=8))
        m = e[0]["markets"][0]
        t = json.loads(m["clobTokenIds"])
        emit({"src": "meta15", "ws": w, "slug": slug, "cond": m.get("conditionId"),
              "tokU": t[0], "tokD": t[1]})
        return {"U": t[0], "D": t[1], "ws": w, "slug": slug, "cond": m.get("conditionId")}
    except Exception as ex:
        emit({"src": "meta15_err", "ws": w, "err": str(ex)[:120]})
        return None

def resolve_loop15():
    cur = None
    while True:
        try:
            now = int(time.time())
            w = (now // 900) * 900
            if w + 900 not in pending15 and now >= w + 900 - 60:
                pending15[w + 900] = resolve15(w + 900)
            if w != cur:
                cur = w
                r = pending15.pop(w, None) or resolve15(w)
                while r is None and (int(time.time()) // 900) * 900 == w:
                    time.sleep(2.0)
                    r = resolve15(w)
                if r:
                    toks15.update(r)
                    old = pm15_ws_ref["ws"]
                    if old:
                        try:
                            old.close()
                        except Exception:
                            pass
        except Exception as e:
            emit({"src": "meta15_loop_err", "err": str(e)[:120]})
        time.sleep(1.0)

def touch_emit15(ev, ets):
    U, D = toks15["U"], toks15["D"]
    if not U or not D:
        return
    with books15.lock:
        bu = books15.d.get(U); bd = books15.d.get(D)
    if not bu or not bd:
        return
    bbU, baU = pm_touch_of(bu)
    bbD, baD = pm_touch_of(bd)
    t = (bbU, baU, bbD, baD)
    if t == getattr(touch_emit15, "last", None) and time.time_ns() - getattr(touch_emit15, "last_ns", 0) < 250 * 10**6:
        return
    touch_emit15.last = t
    touch_emit15.last_ns = time.time_ns()
    emit({"src": "pm15", "ev": ev, "ets": ets, "bbU": bbU, "baU": baU,
          "bbD": bbD, "baD": baD})

def on_error15(ws, e):
    emit({"src": "ws_err", "feed": "pm15", "err": str(e)[:100]})

def on_message15(ws, msg):
    if not msg:
        return
    try:
        data = json.loads(msg)
    except Exception:
        return
    for ev in (data if isinstance(data, list) else [data]):
        et = ev.get("event_type")
        if et == "book":
            a = ev.get("asset_id")
            with books15.lock:
                books15.d[a] = {"bids": ev.get("bids", []), "asks": ev.get("asks", [])}
            touch_emit15("book", ev.get("timestamp"))
        elif et == "price_change":
            a = ev.get("asset_id")
            ch = ev.get("changes") or [ev]
            with books15.lock:
                b = books15.d.setdefault(a, {"bids": [], "asks": []})
                for c in ch:
                    side = "bids" if str(c.get("side", "BUY")).upper() == "BUY" else "asks"
                    px = str(c.get("price"))
                    lst = b[side]
                    for x in lst:
                        if str(x["price"]) == px:
                            x["size"] = c.get("size", "0")
                            break
                    else:
                        if float(c.get("size", 0) or 0) > 0:
                            lst.append({"price": px, "size": c.get("size")})
            touch_emit15("pc", ev.get("timestamp"))
        elif et == "last_trade_price":
            emit({"src": "pm15_trade", "asset": ev.get("asset_id"),
                  "p": ev.get("price"), "sz": ev.get("size"), "side": ev.get("side"),
                  "ets": ev.get("timestamp")})

def pm15_loop():
    sub_at = None
    while True:
        try:
            if not toks15["U"]:
                time.sleep(1.5); continue
            if sub_at != toks15["ws"]:
                ws = websocket.WebSocketApp(CLOB_WS, on_message=on_message15,
                                            on_error=on_error15)
                ws.toks = (toks15["U"], toks15["D"], toks15["ws"])
                def _open15(wsv=ws):
                    wsv.send(json.dumps({"assets_ids": [wsv.toks[0], wsv.toks[1]],
                                         "type": "market"}))
                ws.on_open = _open15
                sub_at = toks15["ws"]
                pm15_ws_ref["ws"] = ws
                threading.Thread(target=pm_ping, args=(ws,), daemon=True).start()
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                               ping_interval=25, ping_timeout=10)
                sub_at = None
        except Exception as e:
            emit({"src": "pm15_loop_err", "err": str(e)[:120]})
        time.sleep(2.0)
# ==================== end 15-min mirror ====================


def cb_loop():
    while True:
        try:
            def on_cb(ws, msg):
                if not msg:
                    return
                try:
                    m = json.loads(msg)
                except Exception:
                    return
                ty = m.get("type")
                if ty in ("match", "last_match"):
                    emit({"src": "cb", "tp": "m", "p": float(m["price"]),
                          "sz": float(m["size"]), "side": m["side"],
                          "xt": m.get("time")})
                elif ty == "ticker":
                    emit({"src": "cb", "tp": "t", "p": float(m.get("price") or 0),
                          "bb": float(m.get("best_bid") or m.get("bid") or 0),
                          "ba": float(m.get("best_ask") or m.get("ask") or 0),
                          "xt": m.get("timestamp")})

            def on_cb_err(ws, e):
                emit({"src": "ws_err", "feed": "cb", "err": str(e)[:100]})

            ws = websocket.WebSocketApp(CB_WS, on_message=on_cb, on_error=on_cb_err)
            def _open2(wsv=ws):
                wsv.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"],
                                     "channels": ["matches", "ticker"]}))
            ws.on_open = _open2
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED})
        except Exception as e:
            emit({"src": "cb_loop_err", "err": str(e)[:120]})
        time.sleep(2.0)


def rtt_loop():
    while True:
        try:
            U = toks["U"]
            if U:
                t0 = time.time()
                req = urllib.request.Request(
                    f"{CLOB_REST}/book?token_id={U}", headers=HDR)
                r = urllib.request.urlopen(req, timeout=5).read()
                ms = (time.time() - t0) * 1000
                emit({"src": "rtt", "ms": round(ms, 1), "kind": "book"})
                # write-path latency: POST an invalid order to the real endpoint.
                # Expect 401/400 quickly; the ROUND TRIP is what we measure.
                try:
                    t1 = time.time()
                    req2 = urllib.request.Request(
                        f"{CLOB_REST}/order", data=b"{}",
                        headers=dict(HDR, **{"Content-Type": "application/json"}),
                        method="POST")
                    urllib.request.urlopen(req2, timeout=5)
                except urllib.error.HTTPError:
                    emit({"src": "rtt", "ms": round((time.time() - t1) * 1000, 1),
                          "kind": "order"})
                except Exception:
                    pass
        except Exception as e:
            emit({"src": "rtt", "ms": None, "err": str(e)[:80]})
        time.sleep(5.0)


def housekeeping():
    while True:
        try:
            now = time.time()
            for fp in glob.glob(os.path.join(WORK, "lag-*.jsonl")):
                if now - os.path.getmtime(fp) > 4 * 3600:
                    with open(fp, "rb") as fin, gzip.open(fp + ".gz", "wb") as fo:
                        fo.writelines(fin)
                    os.remove(fp)
            for w, fh in list(emit._f.items()):
                if now - w > 3600:
                    try:
                        fh.close()
                    except Exception:
                        pass
                    del emit._f[w]
            free = os.statvfs(WORK).f_bavail * 4096 / 10**9
            if free < 2.0:
                olds = sorted(glob.glob(os.path.join(WORK, "lag-*.jsonl.gz")),
                              key=os.path.getmtime)
                for fp in olds[: max(1, len(olds) // 5)]:
                    os.remove(fp)
        except Exception as e:
            emit({"src": "hk_err", "err": str(e)[:100]})
        time.sleep(300.0)


def main():
    threading.Thread(target=resolve_loop, daemon=True).start()
    threading.Thread(target=pm_loop, daemon=True).start()
    threading.Thread(target=resolve_loop15, daemon=True).start()
    threading.Thread(target=pm15_loop, daemon=True).start()
    threading.Thread(target=cb_loop, daemon=True).start()
    threading.Thread(target=rtt_loop, daemon=True).start()
    threading.Thread(target=housekeeping, daemon=True).start()
    # heartbeat so the analyzer knows the process is alive between events
    while True:
        emit({"src": "hb", "toks_ok": bool(toks["U"]), "toks15_ok": bool(toks15["U"])})
        time.sleep(30.0)


if __name__ == "__main__":
    main()
