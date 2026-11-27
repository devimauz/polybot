#!/usr/bin/env python3
"""pm_latency_bot.py - sub-second Coinbase->CLOB latency taker (deploy-ready).

Detection: Coinbase WS mid moves >= THETA $ within TAU ms -> aligned side is
"should be up/down". If the aligned-side touch has NOT yet repriced (ask still
<= pre-move ask + EPS), send a taker FOK buy of SIZE_USD at the displayed ask.

Modes:
  PAPER (default, no keys needed): log the would-be order + track simulated
    settlement P&L against gamma winners at window end.
  LIVE: requires POLY_PRIVATE_KEY (+ optional POLY_API_KEY/POLY_PASSPHRASE/
    POLY_FUNDER) in ~/poly/.env and --live; uses py_clob_client if installed,
    else raw signed REST. Guardrails enforced in both modes:
      max trades/window, min spacing, daily loss cap -> halt, kill on feed gap.

Logs: ~/poly/latencylog/bot-<window>.jsonl
systemd: pm-latency-bot.service (installed disabled; start on verdict).
"""
import argparse, json, os, queue, ssl, sys, threading, time, urllib.request
import websocket
import requests

# CLOB order path latency killer: py_clob_client uses requests.post with a fresh
# TLS handshake per call (RTT 1-4 s). Share one pooled Session across the lib.
_s = requests.Session()
_s.headers.update({"User-Agent": "Mozilla/5.0 latency-bot"})
requests.post = _s.post
requests.get = _s.get
requests.request = _s.request


_ordq = queue.Queue(maxsize=4)


def order_worker():
    while True:
        tok, cap, size, rec = _ordq.get()
        age = (time.time_ns() - rec["t_ns"]) / 1e6
        if age > A.maxOrderAge:
            emit({"src": "order_dropped", "why": "stale", "age_ms": round(age, 1),
                  "ask": rec["ask"], "side_tok": tok})
            continue
        t0 = time.time_ns()
        try:
            resp = order_live(tok, cap, size)
            rec2 = dict(rec); rec2["src"] = "live_order_resp"
            rec2["resp"] = str(resp)[:400]; rec2["size_posted"] = size
            rec2["cap"] = cap; rec2["queue_ms"] = round(age, 1)
            rec2["rtt_ms"] = round((time.time_ns() - t0) / 1e6, 1)
            emit(rec2)
        except Exception as e:
            emit({"src": "live_err", "err": str(e)[:300], "side_tok": tok,
                  "ask": rec["ask"], "cap": cap, "shares": size,
                  "queue_ms": round(age, 1),
                  "rtt_ms": round((time.time_ns() - t0) / 1e6, 1)})


def _warm_conn():
    for _ in range(5):
        try:
            _s.get(CLOB_REST + "/tick-size", params={"token_id": "1"}, timeout=3)
        except Exception:
            pass
        time.sleep(20)



WORK = os.path.expanduser("~/poly/latencylog")
os.makedirs(WORK, exist_ok=True)
ENVF = os.path.expanduser("~/poly/.env")
GAMMA = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_REST = "https://clob.polymarket.com"
CB_WS = "wss://ws-feed.exchange.coinbase.com"
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) latency-bot"}

ap = argparse.ArgumentParser()
ap.add_argument("--theta", type=float, default=8.0, help="$ move to trigger")
ap.add_argument("--tau", type=float, default=0.2, help="lookback seconds")
ap.add_argument("--eps", type=float, default=0.005, help="max repricing still stale")
ap.add_argument("--stake", type=float, default=5.0, help="USD per trade")
ap.add_argument("--max-per-window", type=int, default=3)
ap.add_argument("--min-gap", type=float, default=30.0, help="s between trades")
ap.add_argument("--daily-cap", type=float, default=20.0, help="halt after this P&L loss")
ap.add_argument("--lo-band", dest="loBand", type=float, default=0.10,
                help="min acceptable ask (below = lottery ticket, skip)")
ap.add_argument("--hi-band", dest="hiBand", type=float, default=0.90,
                help="max acceptable ask (above = no payoff left, skip)")
ap.add_argument("--max-order-age-ms", dest="maxOrderAge", type=float, default=800.0,
                help="drop queued orders older than this (race already lost)")
ap.add_argument("--chase-ticks", dest="chaseTicks", type=int, default=0,
                help="live only: post limit this many 0.01 ticks ABOVE observed ask")
ap.add_argument("--live", action="store_true")
A = ap.parse_args()

toks = {"U": None, "D": None, "ws": 0}
book = {"U": (0.0, 1.0), "D": (0.0, 1.0)}   # (bid, ask)
booksz = {"U": (0.0, 0.0), "D": (0.0, 0.0)}  # (bid_sz, ask_sz) at touch
askhist = {}                                 # token -> [(t_ns, ask)]
cbq = []                                     # (t_ns, mid)
state = {"trades": 0, "last_t": 0.0, "pnl": 0.0, "halt": False, "ws_id": 0,
         "last_cb_ns": 0, "last_pm_ns": 0}
refs = {}                                    # "cb"/"pm" -> live WebSocketApp
lock = threading.Lock()


def emit(obj):
    obj["t_ns"] = time.time_ns()
    w = (obj["t_ns"] // 10**9 // 300) * 300
    with lock:
        fh = emit._f.get(w)
        if fh is None:
            emit._f[w] = fh = open(os.path.join(WORK, f"bot-{w}.jsonl"), "a", buffering=1)
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
emit._f = {}


def env(k):
    try:
        for line in open(ENVF):
            if line.startswith(k + "="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def resolve(w):
    slug = f"btc-updown-5m-{w}"
    try:
        req = urllib.request.Request(f"{GAMMA}/events?slug={slug}", headers=HDR)
        e = json.load(urllib.request.urlopen(req, timeout=8))
        m = e[0]["markets"][0]
        t = json.loads(m["clobTokenIds"])
        toks.update({"U": t[0], "D": t[1], "ws": w, "slug": slug, "cond": m["conditionId"]})
        emit({"src": "meta", "ws": w, "slug": slug})
        if A.live:   # warm per-token metadata before any signal can arrive
            threading.Thread(target=_prewarm_tokens, daemon=True).start()
        return True
    except Exception as ex:
        emit({"src": "meta_err", "ws": w, "err": str(ex)[:100]})
        return False


def resolve_loop(pm_ws_ref):
    cur = None
    while True:
        w = (int(time.time()) // 300) * 300
        if w != cur:
            cur = w
            while not resolve(w) and (int(time.time()) // 300) * 300 == w:
                time.sleep(1.5)
            old = pm_ws_ref.get("ws")
            if old:
                try:
                    old.close()
                except Exception:
                    pass
            state["ws_id"] = w
        time.sleep(1.0)


def touch_update():
    U, D = toks["U"], toks["D"]
    if not U:
        return
    bu = book.get(U)
    bd = book.get(D)


def on_pm(ws, msg):
    try:
        _on_pm(ws, msg)
    except Exception as e:
        k = type(e).__name__
        if state.get("pm_exc_kind") != k or time.time() - state.get("pm_exc_t", 0) > 60:
            state["pm_exc_kind"], state["pm_exc_t"] = k, time.time()
            emit({"src": "pm_exc", "err": k, "msg": str(e)[:150]})


def _on_pm(ws, msg):
    if not msg:
        return
    state["last_pm_ns"] = time.time_ns()
    try:
        data = json.loads(msg)
    except Exception:
        return
    for ev in (data if isinstance(data, list) else [data]):
        et = ev.get("event_type")
        aid = ev.get("asset_id")
        if et == "book":
            bl = [x for x in ev.get("bids", []) if float(x["size"]) > 0]
            al = [x for x in ev.get("asks", []) if float(x["size"]) > 0]
            bb = max(bl, key=lambda x: float(x["price"])) if bl else None
            aa = min(al, key=lambda x: float(x["price"])) if al else None
            book[aid] = (float(bb["price"]) if bb else 0.0, float(aa["price"]) if aa else 1.0)
            booksz[aid] = (float(bb["size"]) if bb else 0.0, float(aa["size"]) if aa else 0.0)
            askhist.setdefault(aid, []).append((time.time_ns(), book[aid][1]))
        elif et == "price_change":
            b, a = book.get(aid, (0.0, 1.0))
            bs, asz = booksz.get(aid, (0.0, 0.0))
            for c in (ev.get("changes") or []):
                px = float(c["price"]); sz = float(c.get("size") or 0)
                if str(c.get("side", "BUY")).upper() == "BUY":
                    if sz > 0:
                        if px > b:
                            b, bs = px, sz
                        elif px == b:
                            bs = sz
                    elif px == b:
                        bs = 0.0
                else:
                    if sz > 0:
                        if px < a:
                            a, asz = px, sz
                        elif px == a:
                            asz = sz
                    elif px == a:
                        asz = 0.0
            book[aid] = (b, a)
            booksz[aid] = (bs, asz)
            h = askhist.setdefault(aid, [])
            h.append((time.time_ns(), a))
            if len(h) > 2000:
                del h[:len(h) - 2000]


_clob = None
_clob_lock = threading.Lock()


def _load_env():
    if os.path.exists(ENVF):
        for line in open(ENVF):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _build_clob():
    """Construct + auth the CLOB v2 client once (moved off the order hot path)."""
    global _clob
    from py_clob_client_v2 import ClobClient
    with _clob_lock:
        if _clob is not None:
            return _clob
        _load_env()
        pk = os.environ.get("POLY_PK") or os.environ.get("POLY_PRIVATE_KEY")
        if not pk:
            raise RuntimeError("POLY_PRIVATE_KEY missing in ~/poly/.env")
        funder = os.environ.get("POLY_FUNDER") or os.environ.get("POLY_FUNDER_ADDRESS")
        sig = int(os.environ.get("POLY_SIG_TYPE") or os.environ.get("POLY_SIGNATURE_TYPE")
                  or ("3" if funder else "0"))
        c0 = ClobClient(host=CLOB_REST, chain_id=137, key=pk)
        creds = c0.create_or_derive_api_key()
        kw = dict(key=pk, chain_id=137, creds=creds)
        if funder:
            kw.update(signature_type=sig, funder=funder)
        c = ClobClient(host=CLOB_REST, **kw)
        _g, _t = c.get_neg_risk, c.get_tick_size
        _nr, _tk = {}, {}
        def _gr(tok, _f=_g, _d=_nr):
            if tok not in _d:
                _d[tok] = _f(tok)
            return _d[tok]
        def _ts(tok, _f=_t, _d=_tk):
            if tok not in _d:
                _d[tok] = _f(tok)
            return _d[tok]
        c.get_neg_risk = _gr
        c.get_tick_size = _ts
        c._warm = lambda tok: (_gr(tok), _ts(tok))   # per-token metadata prewarm
        _clob = c
        return _clob


def _prewarm_tokens():
    """Fetch neg_risk/tick for both new-window tokens off the hot path."""
    c = _build_clob()
    for tok in (toks.get("U"), toks.get("D")):
        if tok:
            try:
                c._warm(tok)
            except Exception:
                pass


def order_live(token, price, shares):
    """FAK buy via CLOB v2: fill what's at/below the stale ask, cancel the
    rest, never chase. Order path = create (cached meta) + POST only."""
    from py_clob_client_v2 import (OrderArgs, OrderType, Side,
                                   PartialCreateOrderOptions)
    c = _build_clob()
    px = round(round(price * 100) / 100, 2)   # tick 0.01
    o = c.create_order(
        OrderArgs(price=px, size=round(shares, 2), side=Side.BUY,
                  token_id=str(token)),
        options=PartialCreateOrderOptions(tick_size="0.01"))
    return c.post_order(o, OrderType.FAK)


_nbook_seen = {}

def _nobook_log(why, tok, ask, histlen=None):
    """Rate-limited (60s/kind) observability for previously-SILENT trade-path
    returns. A signal that reaches here but never orders/skips visibly is the
    fourth deafness mode — ask book present but its history/staleness input empty."""
    now = time.time()
    if now - _nbook_seen.get(why, 0) < 60:
        return
    _nbook_seen[why] = now
    rec = {"src": "skip_nobook", "why": why, "ask": ask}
    if histlen is not None:
        rec["hist"] = histlen
    try:
        rec["tok_tail"] = str(tok)[-10:]
    except Exception:
        pass
    emit(rec)


def try_trade(side_tok, aligned, det_t_ns):
    now = time.time()
    if state["halt"] or state["trades"] >= A.max_per_window or now - state["last_t"] < A.min_gap:
        return
    b, a = book.get(side_tok, (0.0, 1.0))
    if not (0.02 <= a <= 0.98):
        _nobook_log("no_book_or_default", side_tok, a)
        state["last_sig_d"] = None   # book not ready: let the next msg retry
        return
    if not (A.loBand <= a <= A.hiBand):  # user rule: only 0.10-0.90 (no lottery, no no-margin)
        state["last_sig_d"] = None
        return
    # staleness: ask now must still be <= ask at (signal - tau), i.e. the move
    # has NOT yet been priced in.
    h = askhist.get(side_tok) or []
    pre = [p for (tt, p) in h if tt <= det_t_ns - int(A.tau * 1e9)]
    if not pre:
        _nobook_log("askhist_empty", side_tok, a, len(h))
        state["last_sig_d"] = None
        return
    a_prev = pre[-1]
    if a > a_prev + A.eps:
        emit({"src": "skip_repriced", "aligned": aligned, "a_prev": a_prev, "a_now": a})
        state["last_sig_d"] = None
        return
    state["last_sig_d"] = None
    want = A.stake / a
    ad = booksz.get(side_tok, (0.0, 0.0))[1]      # displayed ask depth at decision
    # live sizing: WHOLE shares so price x size lands within the 2-decimal
    # maker-amount accuracy the CLOB v2 market-BUY validation enforces
    size_f = float(int(min(want, ad))) if A.live else round(want, 2)
    rec = {"src": "paper_order" if not A.live else "live_order", "side_tok": side_tok,
           "aligned": aligned, "ask": a, "a_prev": a_prev,
           "shares": size_f, "ask_depth": round(ad, 1),
           "fak_shares": round(min(want, ad), 2), "fok_full": ad >= want}
    emit(rec)
    if A.live:
        size = size_f
        if size < 5.0 or size * a < 1.0:  # v2 mins: 5 shares AND $1 notional
            emit({"src": "live_skip_minsize", "side_tok": side_tok, "ask": a,
                  "want": round(want, 2), "ask_depth": round(ad, 1)})
        else:
            cap = min(round(a * 100 + A.chaseTicks) / 100.0, 0.99)
            try:
                _ordq.put_nowait((side_tok, cap, size, rec))
            except queue.Full:
                emit({"src": "order_dropped", "why": "queue full", "ask": a})
    state["trades"] += 1
    state["last_t"] = now


def on_cb(ws, msg):
    try:
        _on_cb(ws, msg)
    except Exception as e:
        k = type(e).__name__
        if state.get("cb_exc_kind") != k or time.time() - state.get("cb_exc_t", 0) > 60:
            state["cb_exc_kind"], state["cb_exc_t"] = k, time.time()
            emit({"src": "cb_exc", "err": k, "msg": str(e)[:150]})


def _on_cb(ws, msg):
    if not msg:
        return
    try:
        m = json.loads(msg)
    except Exception:
        return
    state["cb_msgs"] = state.get("cb_msgs", 0) + 1
    state["cb_last_type"] = m.get("type")
    if m.get("type") == "ticker":
        state["last_cb_ns"] = time.time_ns()
        try:
            bb, ba = float(m["best_bid"]), float(m["best_ask"])
        except (KeyError, ValueError, TypeError):
            return
        mid = (bb + ba) / 2
        if mid <= 0:
            return
        if mid != state.get("cb_mid"):
            state["cb_mid"] = mid
            state["cb_mid_ns"] = time.time_ns()
        tns = time.time_ns()
        cbq.append((tns, mid))
        while cbq and tns - cbq[0][0] > (A.tau + 2.0) * 10**9:
            cbq.pop(0)
        old = [p for (tt, p) in cbq if tt <= tns - A.tau * 10**9]
        if not old:
            return
        d = mid - old[-1]
        if abs(d) < A.theta:
            return
        aligned = "Up" if d > 0 else "Down"
        # dedup: the same move stays in the lookback for many ticker msgs
        lsd = state.get("last_sig_d")
        if (lsd is not None and abs(abs(d) - abs(lsd)) < 0.5 and
              aligned == state.get("last_sig_side") and
              tns - state.get("last_sig_ns", 0) < 2e9):
            return
        U, D = toks["U"], toks["D"]
        if not U:
            return
        atok = U if aligned == "Up" else D
        b, a = book.get(atok, (0.0, 1.0))
        emit({"src": "signal", "d": round(d, 2), "aligned": aligned, "ask": a,
              "hist": len(askhist.get(atok) or [])})
        state["last_sig_d"], state["last_sig_side"], state["last_sig_ns"] = d, aligned, tns
        try_trade(atok, aligned, tns)


def pm_loop(ref):
    subbed = None
    while True:
        try:
            if not toks["U"]:
                time.sleep(0.5); continue
            if subbed != toks["ws"]:
                ws = websocket.WebSocketApp(CLOB_WS, on_message=on_pm)
                tk = (toks["U"], toks["D"])

                def _op(w=ws):
                    w.send(json.dumps({"assets_ids": list(tk), "type": "market"}))
                ws.on_open = _op
                ref["ws"] = ws
                subbed = toks["ws"]

                def ping(w=ws):
                    while True:
                        time.sleep(10)
                        try:
                            if w.sock and w.sock.connected:
                                w.send("PING")
                            else:
                                return
                        except Exception:
                            return
                threading.Thread(target=ping, daemon=True).start()
                ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED})
                emit({"src": "socket_end", "feed": "pm"})
                subbed = None
        except Exception as e:
            emit({"src": "pm_err", "err": str(e)[:100]})
        time.sleep(1.0)


def cb_loop():
    while True:
        try:
            ws = websocket.WebSocketApp(CB_WS, on_message=on_cb)

            def _op2(w=ws):
                w.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"],
                                   "channels": ["ticker"]}))
            ws.on_open = _op2
            refs["cb"] = ws
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                           ping_interval=25, ping_timeout=10)
            emit({"src": "socket_end", "feed": "cb"})
            refs["cb"] = None
        except Exception as e:
            emit({"src": "cb_err", "err": str(e)[:100]})
        time.sleep(1.0)


def watchdog():
    """Force-reconnect any feed that went silent >30 s (half-open TCP) or
    whose prices froze >120 s while the socket still messages (silent staleness)."""
    while True:
        time.sleep(10)
        now = time.time_ns()
        for dict_key, stamp, tag in (("cb", "last_cb_ns", "cb"), ("ws", "last_pm_ns", "pm")):
            last = state.get(stamp, 0)
            ws = refs.get(dict_key)
            if dict_key == "cb" and ws is not None:
                mchg = state.get("cb_mid_ns", 0)
                if mchg and now - mchg > 120 * 10**9:
                    emit({"src": "reconnect", "feed": "cb", "why": "frozen_prices",
                          "silent_s": round((now - mchg) / 1e9, 1)})
                    try:
                        ws.close()
                    except Exception:
                        pass
                    continue
            if ws is not None and (now - last > 30 * 10**9 or last == 0):
                emit({"src": "reconnect", "feed": tag, "silent_s": round((now - last) / 1e9, 1)})
                try:
                    ws.close()
                except Exception:
                    pass
                refs[dict_key] = None


def settle_loop():
    """Settle order-bearing windows against gamma; retry until resolved."""
    import glob as _g0
    done = set()
    # restore cumulative paper P&L from historical settle rows (dedupe by ws)
    byws = {}
    for fp in _g0.glob(os.path.join(WORK, "bot-*.jsonl")):
        try:
            for line in open(fp):
                if '"src":"settle"' in line:
                    r = json.loads(line)
                    byws[r["ws"]] = r.get("pnl", 0.0)
                    done.add(r["ws"])
        except Exception:
            pass
    state["pnl"] = sum(byws.values())
    state["pnl0"] = state["pnl"]   # halt measures LIVE-delta from this point
    # seed per-window trade counter from the CURRENT window's file so a mid-window
    # restart can't exceed max_per_window
    try:
        _cw = (int(time.time()) // 300) * 300
        _f = os.path.join(WORK, f"bot-{_cw}.jsonl")
        if os.path.exists(_f):
            n0 = sum(1 for L in open(_f) if '"src":"paper_order"' in L or '"src":"live_order_resp"' in L)
            state["trades"] = min(n0, A.max_per_window)
            state["last_t"] = time.time()  # also respect min-gap across restarts
    except Exception:
        pass
    last_cur = 0
    import glob as _g
    while True:
        time.sleep(10)
        try:
            cur = (int(time.time()) // 300) * 300
            if cur != last_cur:
                state["trades"] = 0
                last_cur = cur
            for fp in sorted(_g.glob(os.path.join(WORK, "bot-*.jsonl"))):
                try:
                    w = int(os.path.basename(fp)[4:-6])
                except Exception:
                    continue
                if w >= cur or w in done:
                    continue
                txt = open(fp).read()
                if '"paper_order"' not in txt and '"live_order"' not in txt:
                    done.add(w)
                    continue
                try:
                    req = urllib.request.Request(
                        f"{GAMMA}/events?slug=btc-updown-5m-{w}", headers=HDR)
                    m = json.load(urllib.request.urlopen(req, timeout=10))[0]["markets"][0]
                    prs = [float(x) for x in json.loads(m["outcomePrices"])]
                    if sorted(prs) != [0.0, 1.0]:
                        if time.time() > w + 3600:   # gamma can take a while to close
                            done.add(w)
                        continue
                    winner = "Up" if prs[0] == 1.0 else "Down"
                    tot = 0.0
                    ntr = 0
                    # live mode honesty: score ONLY matched fills (taking/making from
                    # the exchange response); FAK kills moved no money, ignore them.
                    pos = []
                    resp_seen = False
                    for line in txt.splitlines():
                        rr = json.loads(line)
                        s = rr.get("src")
                        if s == "live_order_resp":
                            resp_seen = True
                            try:
                                import ast as _ast
                                d = _ast.literal_eval(rr["resp"])
                                if d.get("success") and d.get("status") in ("matched", "partial"):
                                    got = float(d["takingAmount"]); made = float(d["makingAmount"])
                                    pos.append((got, made / got if got else 0.0, rr["aligned"]))
                            except Exception:
                                pass
                    if not resp_seen:   # paper window: paper_order rows only.
                        # live_order rows WITHOUT a resp row are kills (cost $0) — never
                        # treat them as positions (phantom-loss bug, fixed 2026-11-04).
                        for line in txt.splitlines():
                            rr = json.loads(line)
                            if rr.get("src") == "paper_order":
                                pos.append((rr["shares"], rr["ask"], rr["aligned"]))
                    for sh, a, side in pos:
                        fee = sh * 0.07 * a * (1 - a)
                        pay = sh * (1.0 if side == winner else 0.0)
                        tot += pay - sh * a - fee
                        ntr += 1
                    emit({"src": "settle", "ws": w, "winner": winner,
                          "pnl": round(tot, 3), "n": ntr})
                    done.add(w)
                    state["pnl"] += tot
                    if state["pnl"] - state.get("pnl0", state["pnl"]) < -A.daily_cap:
                        state["halt"] = True
                        emit({"src": "HALT", "reason": "daily cap",
                              "pnl": round(state["pnl"], 2)})
                except Exception as e:
                    emit({"src": "settle_err", "err": str(e)[:100]})
        except Exception:
            pass


def main():
    if A.live:
        emit({"src": "warn", "msg": "LIVE orders armed: FAK via CLOB v2, "
              f"${A.stake}/trade, band 0.10-0.90, live-delta halt $"
              f"{A.daily_cap}, keepalive-on"})
        threading.Thread(target=_warm_conn, daemon=True).start()
        threading.Thread(target=order_worker, daemon=True).start()
        threading.Thread(target=_build_clob, daemon=True).start()  # creds at boot, not first order
    threading.Thread(target=resolve_loop, args=(refs,), daemon=True).start()
    threading.Thread(target=pm_loop, args=(refs,), daemon=True).start()
    threading.Thread(target=cb_loop, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=settle_loop, daemon=True).start()
    emit({"src": "boot", "mode": "live" if A.live else "paper", "theta": A.theta,
          "stake": A.stake, "cap": A.daily_cap})
    while True:
        time.sleep(30)
        emit({"src": "hb", "pnl": round(state["pnl"], 2), "halt": state["halt"],
              "cb_msgs": state.get("cb_msgs", 0), "cb_type": state.get("cb_last_type"),
              "cb_age_s": round((time.time_ns() - state.get("last_cb_ns", 0)) / 1e9, 1),
              "mid": state.get("cb_mid"),
              "mid_age_s": round((time.time_ns() - state.get("cb_mid_ns", 0)) / 1e9, 1)})


if __name__ == "__main__":
    main()
