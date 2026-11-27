"""btc5m_endgame_bot.py - LIVE bot for the Polymarket BTC 5-minute "endgame
favourite" strategy, validated by btc5m_endgame_backtest.py.

THE STRATEGY
------------
Polymarket 5-minute crypto binaries settle on the Chainlink 60-second TWAP:
the price to beat is the 60s TWAP just before the window opens, and the
settlement is the 60s TWAP over the window's final 60 seconds.  Part of the
settlement window is therefore already observed and locked in before the
window ends, and near expiry the market's price for the favourite does not
fully reflect that lock-in.

Backtest result on the logged order books (price_logs/polymarket, 269
resolved windows, ~22.4 h of L2 data, SOURCE-ALIGNED clock):

    rule:  T >= 262s into the window, buy the FAVOURITE side
           when its best ask is in [0.80, 0.96), both legs still live
    trades 38 over 269 windows      mean ask 0.881   realised win rate 0.974
    net +$19.07 on $5 stakes         ROI +10.0%       cluster-t +3.20
    bootstrap CI95 on avg P&L        [+0.15, +0.72]   family-wise p < 0.001
    worst single trade -$5.04 (the TWAP settlement overrode a confident
    spot market that had the losing side at 0.90 -> 0.97)

$5 IS THE EXCHANGE'S PRACTICAL FLOOR
------------------------------------
The per-market `min_order_size` is 5 SHARES.  At the 0.95 end of the band
with the 0.03 slippage cushion a $5 stake buys 5.10 shares - it clears, but
with no room to spare.  A $2 stake buys 2.11 shares and is REJECTED
outright, so all 38 orders would bounce and the strategy would place zero
trades.  The bot reads `min_order_size` from the live book and skips with a
logged reason rather than sending an order the exchange will bounce.

CONDITIONAL SIZING (enabled by default; set STAKE_CHEAP = STAKE to disable)
-----------------------------------------------------------------------
At a roughly CONSTANT win rate p across the band, expected profit per
dollar is (p - ask)/ask, so a CHEAPER favourite carries much more ROI per
dollar: at p=0.974 an entry at 0.81 earns +18.9%/dollar against +3.2% at
0.94.  Kelly says to put more capital in the cheap bucket, so this bot
stakes $15 when the favourite's ask is below 0.85 and $5 otherwise.

    flat $5  : net +$19.07  ROI +10.0%  t +3.20  CI [+0.15, +0.72]
    sized    : net +$44.72  ROI +14.0%  t +4.57  CI [+0.66, +1.66]

This needs NO win-rate gap in the cheap bucket - the outcome placebo found
none (p=0.66) and none is required.  It is the same edge allocated better,
not a new edge.  Worst trade and max drawdown are unchanged (-$5.04) because
the single loss happened to land in the dear bucket.  Break-even cheap-bucket
win rate for the 3x stake is 0.8912 (observed 13/13).

THE TIMESTAMP CAVEAT (this bit us once already)
------------------------------------------------
The first version of this study used the logger's own `seconds_into_window`
stamp, which is written BEFORE the logger's REST calls complete and is
therefore ~0.96 s optimistic.  That look-ahead inflated the result to
t=+3.32 and had to be corrected to t=+1.83 before the rule above was chosen.
Everything here and in the backtest uses the source-aligned clock
(max of the frame's book timestamps and the coinbase exchange timestamp).

THE HONEST CAVEAT
-----------------
N = 38 trades is a small sample.  The edge survives leave-one-out, P&L
concentration (top 5 = 28% of P&L), an 8-block stability split, a
chronological split (18/18 out of sample), a permutation family-wise test
and latency/fee/depth stress - but it is NOT yet proven.  THIS BOT DEFAULTS
TO PAPER MODE.

Run:
    python btc5m_endgame_bot.py --paper     paper orders on polysimulator
    python btc5m_endgame_bot.py              dry run, signals only
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR

import httpx
import numpy as np
from dotenv import load_dotenv

from bot import (Bot, D, Journal, LOG, ROOT, configure_simulator_pacing,
                 is_fok_rejection)
from polymarket_feed import MarketFeed

# ---------------------------------------------------------------- parameters
STAKE = D("5")            # dollars per trade (the exchange's practical minimum)
# --- conditional sizing overlay -------------------------------------------
# With a roughly CONSTANT win rate p across the band, EV per dollar is
# (p - ask)/ask, so a CHEAPER favourite carries more ROI per dollar:
#   p=0.974:  ask 0.82 -> +18.9%/dollar,  ask 0.93 -> +4.3%/dollar
# So the same dollar buys more expected profit at a low price.  Kelly says
# to put more capital there.  This does NOT require the cheap bucket to win
# MORE often - the placebo found no win-rate gap (p=0.66), and none is
# needed.  Break-even cheap-bucket win rate for the 3x stake is 0.8912
# (observed 13/13 = 1.000).  Disable with STAKE_CHEAP = STAKE.
#
# $5 is the smallest stake the exchange accepts on EVERY trade in the
# sample: at the 0.95 cap with the 0.03 slippage cushion it buys 5.10
# shares against the 5-share minimum.  A $2 stake would buy only 2.11
# shares and be REJECTED outright.
STAKE_CHEAP = D("15")    # stake when the favourite's ask is below CHEAP_ASK
CHEAP_ASK = D("0.85")     # cheap-bucket threshold
FAV_ASK_MIN = D("0.80")   # favourite ask must be at least this
FAV_ASK_MAX = D("0.96")   # and strictly below this
ENTRY_SECONDS = 262      # enter only once the window is this far in
MAX_SPREAD = D("0.04")    # require a tight two-sided book
MIN_DEPTH_USD = D("10")   # require this much displayed size at the ask
# Polymarket rejects orders below the market's `min_order_size`, which is a
# SHARE count (observed: 5).  A stake that cannot buy 5 shares at the ask
# is rejected outright, so the stake must satisfy
#     stake >= min_order_shares * (ask + slippage)
# With a 0.95 cap and a 0.03 slippage cushion the smallest safe stake is
# 5 * 0.98 = $4.90.  $10 (the default) is comfortably clear.
MIN_ORDER_SHARES = 5.0
FAV_SLIPPAGE = D("0.03")  # FOK worst-price cushion (the book is stable here)
MAX_LEG_ATTEMPTS = 3      # re-quote retries per leg
HOLD_SECONDS = 0.0        # optional: ask must stay in band this long first

DASH_EVENT_KEYWORDS = ("SIGNAL", "filled", "rejected", "aborted", "not matched",
                       "DRY RUN", "ERROR", "CRITICAL", "Waiting for active")


class EndgameJournal(Journal):
    """Journal that records the fill price so realised P&L is auditable."""

    def __init__(self, path):
        super().__init__(path)
        # The base schema has no condition_id column; the endgame bot needs it
        # to map a slug back to the market it reserved.
        self._migrate("orders", {"condition_id": "TEXT NOT NULL DEFAULT ''"})

    def condition_for(self, market):
        try:
            cols = [c[1] for c in self.db.execute("PRAGMA table_info(orders)")]
            if "condition_id" not in cols:
                return ""
            row = self.db.execute(
                "SELECT condition_id FROM orders WHERE market=? AND condition_id!='' LIMIT 1",
                (market,)).fetchone()
            return row[0] if row else ""
        except Exception:
            return ""

    def finish(self, market, outcome, status, detail):
        def do():
            self.db.execute(
                "UPDATE orders SET status=?, detail=? WHERE market=? AND outcome=?",
                (status, detail, market, outcome))
            if status == "matched":
                cost = shares = None
                try:
                    resp = json.loads(detail) if detail.startswith("{") else {}
                    if resp.get("makingAmount") and resp.get("takingAmount"):
                        cost = D(str(resp["makingAmount"]))
                        shares = D(str(resp["takingAmount"]))
                except (ValueError, KeyError, ArithmeticError):
                    cost = shares = None
                if cost is not None and shares is not None and cost > 0 and shares > 0:
                    self.db.execute(
                        "UPDATE orders SET filled=1, fill_price=?, shares=? "
                        "WHERE market=? AND outcome=?",
                        (float(cost / shares), float(shares), market, outcome))
            self.db.commit()
        return self._retry(do)

    def reserve(self, market, outcome, amount=STAKE, *, once_per_market=False,
                condition_id=""):
        # Commit before sending: a timeout or crash must never cause a duplicate.
        def do():
            self.db.execute("BEGIN IMMEDIATE")
            try:
                day = datetime.now(timezone.utc).date().isoformat()
                if (self.seen_market(market) if once_per_market
                        else self.seen(market, outcome)):
                    self.db.rollback()
                    return False
                self.db.execute(
                    "INSERT INTO orders(market, outcome, day, amount, status, detail, condition_id) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (market, outcome, day, float(amount), "reserved", "",
                     str(condition_id or "")))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(do)


class EndgameBot(Bot):
    """Buy the favourite late in a BTC 5-minute window when it is underpriced."""

    amount = STAKE
    journal_type = EndgameJournal
    journal_prefix = "endgame-"
    price_log_dir = ROOT / "price_logs" / "endgame"
    last_quotes = ("", {})

    def __init__(self, live, asset="btc"):
        super().__init__(live, asset)
        self.feed = MarketFeed()
        self.timer_market = None
        self.above_since = {}

    def polymarket_quotes(self, tokens):
        self.feed.subscribe(tokens)
        return self.feed.quotes()

    def market_status(self, condition_id):
        cached = _STATUS_CACHE.get(condition_id)
        now = time.time()
        if cached and now - cached[0] < 60:
            return cached[1]
        result = (None, None)
        try:
            m = self.simulator_get(f"markets/{condition_id}")
            prices = None
            if isinstance(m, dict):
                nested = m.get("markets")
                if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                    prices = nested[0].get("outcome_prices")
                if prices is None:
                    prices = m.get("outcome_prices")
                winner = m.get("resolved_outcome")
                if not winner and prices and len(prices) >= 2:
                    if max(prices) >= 0.995 or min(prices) <= 0.005:
                        winner = "Up" if prices[0] >= prices[1] else "Down"
                if m.get("closed") or m.get("resolved") or winner:
                    result = ("closed", winner)
        except Exception:
            pass
        _STATUS_CACHE[condition_id] = (now, result)
        return result

    # ------------------------------------------------------------------ step
    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        self.feed.subscribe(tokens)
        if self.timer_market != slug:
            self.timer_market = slug
        if self.journal.seen_market(slug):
            now = time.monotonic()
            if now >= getattr(self, "_next_wait_log", 0):
                LOG.info("%s already traded this window; next window in %.0fs",
                         self.asset.upper(), max(0, end - time.time()))
                self._next_wait_log = now + 15
            return

        seconds_in = 300.0 - (end - time.time())
        if seconds_in < ENTRY_SECONDS or seconds_in > 300:
            return

        quotes = self.feed.quotes()
        if len(quotes) < 2:
            return
        new_state = {}
        for outcome, q in quotes.items():
            self.log_prices(slug, outcome, q)
            new_state[outcome] = (q.ask, q.bid)
        self.last_quotes = (slug, new_state)
        self._next_quote_log = time.time()

        # Build the raw leg view and hand it to the shared decision function
        # - the same function the backtest and the replay harness execute.
        legs = {}
        for outcome, q in quotes.items():
            legs[outcome] = {
                "ask": float(q.ask) if q.ask is not None else None,
                "bid": float(q.bid) if q.bid is not None else None,
                "asksz": self.depth_at_ask(tokens, outcome),
            }
        p_up = 0.5 * (quotes["Up"].probability +
                      (1.0 - quotes["Down"].probability)) if set(quotes) == {"Up", "Down"} \
            else max(q.probability for q in quotes.values())
        # the feed's probability is the midpoint of that leg; recompute the
        # pair-consistent "Up" probability the same way the backtest does.
        if set(quotes) == {"Up", "Down"}:
            p_up = 0.5 * ((float(quotes["Up"].ask) + float(quotes["Up"].bid)) / 2.0 +
                          (1.0 - (float(quotes["Down"].ask) + float(quotes["Down"].bid)) / 2.0))
        decision = decide_entry(legs, p_up)
        if decision is None:
            self.above_since.pop("Up", None)
            self.above_since.pop("Down", None)
            return
        fav_out, fav_ask, fav_bid = decision
        fav_q = quotes[fav_out]

        if HOLD_SECONDS > 0:
            # Optional: require the quote to have held inside the band for
            # HOLD_SECONDS before entering.  0 = enter on the first qualifying
            # quote, which is the rule the backtest validated.
            now = time.monotonic()
            first = self.above_since.get(fav_out)
            if first is None:
                self.above_since[fav_out] = now
                return
            if now - first < HOLD_SECONDS:
                return

        # --- exchange minimum-order-size guard ---------------------------------
        # Polymarket rejects any order below the market's `min_order_size`
        # (a SHARE count, observed 5).  Check it at the WORST price the
        # order could fill at (ask + slippage cushion) so we never send an
        # order the exchange will bounce.
        stake = stake_for_ask(fav_ask)
        min_sh = self.min_order_shares(tokens, fav_out)
        worst_price = D(str(fav_ask)) + FAV_SLIPPAGE
        need_usd = D(str(min_sh)) * worst_price
        if stake < float(need_usd):
            LOG.warning("SKIP %s %s: stake $%.2f buys %.2f shares at %.3f, "
                        "below the %g-share minimum ($%.2f needed)",
                        slug, fav_out, stake, stake / float(worst_price),
                        float(worst_price), min_sh, float(need_usd))
            return

        LOG.info("SIGNAL %s: %s ask=%.3f bid=%.3f, %.0fs into the window; "
                 "buying $%.2f %s",
                 slug, fav_out, fav_ask, fav_bid, seconds_in, stake, fav_out)
        if not self._buy_leg(slug, end, tokens, fav_out, stake=stake):
            LOG.warning("%s %s order did not fill", slug, fav_out)

    def depth_at_ask(self, tokens, outcome):
        """Best-ask size from the raw book, or None when unknown.

        tokens maps outcome -> token id; feed.books is keyed by token id.
        """
        try:
            token = tokens.get(outcome)
            if token is None:
                return None
            book = self.feed.books.get(str(token))
            if not book:
                return None
            asks = book.get("asks") or {}
            if not asks:
                return None
            best = min(asks)
            return float(asks[best])
        except Exception:
            return None
        return None

    def min_order_shares(self, tokens, outcome):
        """Polymarket's per-market minimum order size, in SHARES.

        The exchange rejects any order below it.  It is published on the
        market/book payload as `min_order_size` (observed value: "5") and
        it is a SHARE count, not a dollar amount - so a $2 stake at a 0.90
        ask buys only 2.2 shares and is rejected outright.
        """
        try:
            token = tokens.get(outcome)
            if token is None:
                return MIN_ORDER_SHARES
            book = self.feed.books.get(str(token))
            if not book:
                return MIN_ORDER_SHARES
            raw = book.get("min_order_size")
            if raw is None:
                return MIN_ORDER_SHARES
            v = float(raw)
            return v if v > 0 else MIN_ORDER_SHARES
        except Exception:
            return MIN_ORDER_SHARES

    def _buy_leg(self, slug, end, tokens, outcome, stake=None):
        if self.journal.seen(slug, outcome):
            return True
        cid = self.condition_id or ""
        amount = self.amount if stake is None else D(str(stake))
        if not self.journal.reserve(slug, outcome, amount, condition_id=cid):
            return False
        last_detail = "no fill in attempts"
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= end:
                self.journal.finish(slug, outcome, "unknown", "market closed before fill")
                return False
            q = self.feed.quotes().get(outcome)
            if q is None:
                LOG.info("%s attempt %d: waiting for Polymarket book", outcome, attempt)
                time.sleep(0.1)
                continue
            if q.ask > FAV_ASK_MAX:
                LOG.info("%s attempt %d: ask %.3f above cap; waiting", outcome, attempt,
                         q.ask)
                time.sleep(0.1)
                continue
            tick = D(q.tick)
            limit = min(D("0.99"), q.ask + FAV_SLIPPAGE)
            limit = (limit / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
            signed = {
                "market_id": cid,
                "side": "BUY",
                "outcome": outcome,
                "amount": str(amount),
                "price": str(limit),
                "order_type": "market",
                "time_in_force": "FOK",
            }
            if not self.live:
                self.journal.finish(slug, outcome, "simulated",
                                    f"paper: ask {q.ask}, cap {FAV_ASK_MAX}")
                LOG.info("PAPER: buy $%s %s %s at %s (ask %.3f)",
                         amount, self.asset.upper(), outcome, limit, q.ask)
                self.log_prices(slug, outcome, q, "buy_simulated")
                return True
            try:
                resp = self.post_with_retry(signed, end)
            except Exception as exc:
                last_detail = f"submit error: {exc}"
                LOG.warning("%s submit failed: %s", outcome, exc)
                time.sleep(0.2)
                continue
            if resp is None:
                last_detail = "no response"
                time.sleep(0.2)
                continue
            if is_fok_rejection(resp):
                last_detail = json.dumps(resp)[:300]
                LOG.info("%s FOK rejected: %s", outcome, last_detail)
                time.sleep(0.1)
                continue
            self.journal.finish(slug, outcome, "matched", json.dumps(resp))
            LOG.info("%s %s filled: %s", self.asset.upper(), outcome,
                     json.dumps(resp)[:200])
            return True
        self.journal.finish(slug, outcome, "rejected",
                            f"{last_detail} (cap {FAV_ASK_MAX})")
        return False


_STATUS_CACHE = {}


def stake_for_ask(ask):
    """The stake to use for a favourite quote at `ask`.

    Kept as a module-level pure function so the backtest, the replay
    harness and the live bot all size a trade identically.

    Conditional sizing: a cheaper favourite carries more expected profit
    per dollar (EV per dollar = (p - ask)/ask at a roughly constant win
    rate), so the cheap bucket gets a larger stake.  The threshold CHEAP_ASK
    and the multiplier STAKE_CHEAP/STAKE were validated out of sample; see
    ENDGAME_EDGE.md.
    """
    ask = float(ask)
    if ask < float(CHEAP_ASK):
        return float(STAKE_CHEAP)
    return float(STAKE)


def decide_entry(legs, p_up, *, fav_ask_min=None, fav_ask_max=None,
                 max_spread=None, min_depth_usd=None):
    """The single source of truth for the entry decision.

    legs  : {outcome: {"ask": float|None, "bid": float|None, "asksz": float|None}}
            raw best-ask / best-bid / size-at-ask for each leg
    p_up  : the market's implied probability that "Up" wins, in (0, 1)

    Returns (fav_outcome, ask, bid) or None when there is no trade.

    Kept as a module-level pure function so that the backtest, the replay
    harness and the live bot all execute exactly the same logic.  The band
    comparison is done in plain float space on purpose: comparing a float
    0.99 against Decimal("0.99") is True (0.99 is stored just below the
    decimal), which would let a 0.99 quote through a "< 0.99" cap.
    """
    fav_ask_min = float(FAV_ASK_MIN) if fav_ask_min is None else float(fav_ask_min)
    fav_ask_max = float(FAV_ASK_MAX) if fav_ask_max is None else float(fav_ask_max)
    max_spread = float(MAX_SPREAD) if max_spread is None else float(max_spread)
    min_depth_usd = float(MIN_DEPTH_USD) if min_depth_usd is None else float(min_depth_usd)

    if not (0.002 < p_up < 0.998):
        return None

    # The FAVOURITE is decided by the market's own implied probability, not by
    # which leg happens to have the nicest book.  If the favourite's own book
    # is not tradable there is no trade - we never substitute the underdog.
    fav_out = "Up" if p_up >= 0.5 else "Down"
    other_out = "Down" if fav_out == "Up" else "Up"
    if fav_out not in legs or other_out not in legs:
        return None

    fl, ol = legs[fav_out], legs[other_out]

    # The traded leg needs a sane, tight, deep two-sided book.
    ask, bid, asz = fl.get("ask"), fl.get("bid"), fl.get("asksz")
    if ask is None or bid is None or asz is None:
        return None
    if not all(np.isfinite(v) for v in (ask, bid, asz)):
        return None
    if not (0 < bid <= ask < 1):
        return None
    if (ask - bid) > max_spread:
        return None
    if asz * ask < min_depth_usd:
        return None

    # The stake must be FILLABLE: a FOK order for more dollars than the book
    # shows at the ask is rejected for insufficient size.  The conditional
    # sizing overlay can stake 3x the base, so the displayed depth has to
    # cover THAT stake, not just the base one.
    if asz * ask < stake_for_ask(ask):
        return None

    # The opposite leg must still be trading: if it has been resolved away
    # (ask pinned at 0.001 or 0.999) the market has effectively decided and
    # the two-sided book we are trading into is gone.
    o_ask = ol.get("ask")
    if o_ask is None or not np.isfinite(o_ask) or not (0.001 < o_ask < 0.999):
        return None

    if not (fav_ask_min <= ask < fav_ask_max):
        return None
    return fav_out, float(ask), float(bid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", "--live", dest="live", action="store_true",
                    help="submit PAPER orders to polysimulator (default: dry run, "
                         "no orders at all)")
    ap.add_argument("--asset", default="btc", help="asset slug (default btc)")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="require the favourite's ask to stay in band this long "
                         "before entering (0 = enter on the first qualifying "
                         "quote, which is what the backtest validated)")
    args = ap.parse_args()
    global HOLD_SECONDS
    HOLD_SECONDS = args.hold_seconds

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = EndgameBot(live=args.live, asset=args.asset)
    configure_simulator_pacing(bot)
    LOG.info("%s %s endgame-favourite bot; $%s per trade; entry at T>=%ss, "
             "favourite ask in [%s,%s)",
             "POLYSIMULATOR PAPER" if args.live else "DRY RUN",
             args.asset.upper(), STAKE, ENTRY_SECONDS, FAV_ASK_MIN, FAV_ASK_MAX)
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.05 <= poll <= 60:
            raise ValueError("Polling must be 0.05-60 seconds")
        while True:
            try:
                bot.step()
            except httpx.HTTPError as exc:
                LOG.warning("%s market data request failed (%s)", args.asset.upper(),
                            type(exc).__name__)
            time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s. Check configuration, journal and account "
                  "before retrying.", type(exc).__name__, exc)
        return 1
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
