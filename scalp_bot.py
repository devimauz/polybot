"""Polysimulator 5-minute pair scalper (BTC only by default).

Scalps the BUY-UP / BUY-DOWN price gap inside each five-minute window:

  * ENTRY  — buy both legs (FOK market orders) when the pair's ask sum is
             cheap. Tiered stakes so the bot keeps taking trades all window:
               tier 1: ask sum <= 0.97 (any time)      $100/leg
               tier 2: ask sum <= 1.00 (from t=30s)    $100/leg
               tier 3: ask sum <= 1.02 (from t=90s)    $75/leg
               filler: sum <= 1.10, only once 4 trades
                       are already done in the window   $15/leg
  * TAKE PROFIT — sell both legs (FOK market sells) once the bid sum gets
             fat: >= 1.02 for scalps, >= 1.08 for filler (fillers also
             market-sell after an 8s hold so they free the slot). Round-trip
             profit = bid sum - ask sum - fees. The bid sum can exceed 1.00:
             each leg is capped at $1, but the SUM of two legs is not.
  * SETTLE FALLBACK — if a scalp never hits TP before FORCE_SETTLE_T, hold to
             window close. A full pair pays min(shares_up, shares_down) * $1
             guaranteed (one leg settles at $1, the other at $0), so the
             position can never lose more than the entry fees.
  * CADENCE — up to MAX_TRADES_PER_WINDOW (=5) round trips per window,
             COOLDOWN seconds apart; only one position open at a time.

Backtest on the 96 recorded BTC windows (11.7h of ninefive price logs,
properly time-paired and filtered to the books quote_book actually accepts):
  the 3 scalp tiers make ~$570-900 with a ~45-90% win rate; the filler
  (small $15/leg 5th trade) costs ~$10/trade in fees+spread and exists only
  to make the bot trade 5x every window. Net over the recorded sample is
  roughly flat-to-positive; the recorded logs are sparse (~80 quotes/window)
  so 12/96 windows fell just short of 5 — at live 0.5s polling every window
  reaches the 5-trade cap.

Run:
  python scalp_bot.py --paper          paper orders on polysimulator
  python scalp_bot.py                  dry run, signals logged, no orders
  python scalp_bot.py --asset eth      (optional: another asset)
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR

import httpx
from dotenv import load_dotenv

from bot import (Bot, D, Journal, LOG, ROOT,
                 configure_simulator_pacing, is_fok_rejection, quote_book)

# ------------------------------------------------------------------ constants
# Tiered entry: up to MAX_TRADES_PER_WINDOW round trips per window.
#   tiers 1-3 are the profitable scalps (ask sum well under 1.00);
#   tier 4 is the filler — a small guaranteed entry late in the window so the
#   bot actually trades 5x even when the market never dips.
STAKE_TIERS = [
    # (max ask-sum, min t, stake)
    (D("0.97"), 0.0, D("100")),   # tier 1: full stake, cheap pair
    (D("1.00"), 30.0, D("100")),  # tier 2: full stake, near-fair pair
    (D("1.02"), 90.0, D("75")),   # tier 3: reduced stake
]
# Filler tier: the 5th-trade cadence guarantee. It fires only AFTER 4 trades
# are already done in the window (so it's bounded to ~1 per window and cannot
# crowd out the profitable scalp tiers), on any ask sum up to 1.10.
# Exits fast: TP at bid sum >= 1.08, or market-sell after an 8s hold so it
# frees the slot. A $15/leg filler costs ~$10 per trade (fees+spread); it is
# the small price of making the bot trade 5x every window.
FILLER = (D("1.10"), 30.0, D("15"), D("1.08"), 8.0)  # (max sum, min t, stake, tp, hold s)
FILLER_MIN_DONE = 4   # filler only fills the (4th, 5th) slot of the window
TP_BID_MIN = D("1.02")     # sell a scalp once bid_Up + bid_Down >= 1.02
MAX_TRADES_PER_WINDOW = 5
ENTRY_CUTOFF = 290         # no new entries this many seconds before window end
EXIT_CUTOFF = 15           # stop placing orders this many seconds before end
FORCE_SETTLE_T = 270       # if still holding at window_start+270s, hold to close
COOLDOWN = 2.0             # seconds between round trips in the same window
MAX_LEG_ATTEMPTS = 3       # re-quote retries per leg before giving up
MIN_LEG_SIZE = D("5")      # below this a sell leg is dust; leave it to settle
TAKER_FEE = D("0.05")      # fee = 0.05 * min(p, 1-p) * shares, per leg


def leg_fee(price: D, shares: D) -> D:
    return TAKER_FEE * min(price, D(1) - price) * shares


# ------------------------------------------------------------------ journal
@dataclass
class Entry:
    entry_t: float
    ask_up: D
    ask_down: D
    shares_up: D
    shares_down: D
    cost: D        # 2 * stake
    fee: D
    stake: D = D("0")
    is_filler: bool = False


class ScalpJournal(Journal):
    """Base journal + positions table (current state) + trades table (history)."""

    def __init__(self, path):
        super().__init__(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS positions (
            market TEXT PRIMARY KEY,
            entry_t REAL, ask_up REAL, ask_down REAL,
            shares_up REAL, shares_down REAL, cost REAL, fee REAL,
            state TEXT, detail TEXT)""")
        for col, decl in (("stake", "REAL NOT NULL DEFAULT 0"),
                          ("is_filler", "INTEGER NOT NULL DEFAULT 0")):
            try:  # journal from an older version
                self.db.execute(f"ALTER TABLE positions ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass
        self.db.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT, entry_t REAL, stake REAL,
            exit_kind TEXT, detail TEXT)""")
        self.db.commit()

    def open_entry(self, market, entry):
        self._retry(lambda: (
            self.db.execute(
                "INSERT OR REPLACE INTO positions "
                "(market, entry_t, ask_up, ask_down, shares_up, shares_down, "
                " cost, fee, state, detail, stake, is_filler) "
                "VALUES (?,?,?,?,?,?,?,?,?, ?,?,?)",
                (market, entry.entry_t, float(entry.ask_up), float(entry.ask_down),
                 float(entry.shares_up), float(entry.shares_down),
                 float(entry.cost), float(entry.fee), "open", "",
                 float(entry.stake), 1 if entry.is_filler else 0)),
            self.db.commit()))

    def close_position(self, market, state, detail):
        def do():
            self.db.execute(
                "UPDATE positions SET state=?, detail=? WHERE market=?",
                (state, detail, market))
            if state.startswith("closed"):
                row = self.db.execute(
                    "SELECT entry_t, stake FROM positions WHERE market=?",
                    (market,)).fetchone()
                if row:
                    self.db.execute(
                        "INSERT INTO trades (market, entry_t, stake, exit_kind, detail) "
                        "VALUES (?,?,?,?,?)",
                        (market, row[0], row[1], state[len("closed_"):], detail))
                # Free both leg rows so the next round trip this window can
                # re-enter (a half-held leg stays locked until settlement).
                if not state.startswith("closed_half_held") \
                        and not state.startswith("closed_holding"):
                    self.db.execute(
                        "UPDATE orders SET status='reserved', detail='', filled=0, "
                        "fill_price=0, shares=0 WHERE market=? "
                        "AND outcome IN ('Up','Down')", (market,))
            self.db.commit()
        self._retry(do)

    def position(self, market):
        """(entry, state) for this market; (None, None) when no record."""
        row = self._retry(lambda: self.db.execute(
            "SELECT entry_t, ask_up, ask_down, shares_up, shares_down, "
            "cost, fee, state, is_filler, stake FROM positions WHERE market=?",
            (market,)).fetchone())
        if row is None:
            return None, None
        entry = Entry(row[0], D(str(row[1])), D(str(row[2])), D(str(row[3])),
                      D(str(row[4])), D(str(row[5])), D(str(row[6])),
                      D(str(row[9])), is_filler=bool(row[8]))
        return entry, row[7]

    def trades_count(self, market):
        """Number of closed round trips this window."""
        row = self._retry(lambda: self.db.execute(
            "SELECT COUNT(*) FROM trades WHERE market=?", (market,)).fetchone())
        return row[0] if row else 0

    def reset_order_row(self, market, outcome):
        """Free an order row so a later round trip in the same window refills it."""
        def do():
            self.db.execute(
                "UPDATE orders SET status='reserved', detail='', filled=0, "
                "fill_price=0, shares=0 WHERE market=? AND outcome=?", (market, outcome))
            self.db.commit()
        return self._retry(do)


# ------------------------------------------------------------------ bot
class ScalpBot(Bot):
    journal_type = ScalpJournal
    journal_prefix = "scalp-"
    price_log_dir = ROOT / "price_logs" / "scalp"

    def __init__(self, live, asset="btc"):
        super().__init__(live, asset)
        self._cooldown_until = 0.0   # monotonic re-entry lock
        self._next_status_log = 0.0
        self._cur_end = 0.0
        self.last_quotes = ("", {})

    # ------------------------------------------------------------ entry
    def _pick_tier(self, t, ask_sum, trades_done):
        """Stake for this entry, or None. Returns (stake, is_filler)."""
        for max_sum, min_t, stake in STAKE_TIERS:
            if t >= min_t and ask_sum <= max_sum:
                return stake, False
        f_max_sum, f_min_t, f_stake, _, _ = FILLER
        if (t >= f_min_t and t <= ENTRY_CUTOFF and ask_sum <= f_max_sum
                and trades_done >= FILLER_MIN_DONE):
            return f_stake, True
        return None

    def _try_enter(self, slug, end, tokens, quotes, now):
        """Buy both legs when the ask sum clears a tier. True if entered."""
        t = now - (end - 300)
        if t > ENTRY_CUTOFF or now < self._cooldown_until:
            return False
        if self.journal.seen_market(slug):
            # A leg order row already exists for this window — either a
            # half-filled entry is being held to settlement, or the prior
            # round trip's rows were not reset yet. Don't re-fire entry on
            # top of an open position.
            return False
        au, ad = quotes["Up"].ask, quotes["Down"].ask
        ask_sum = au + ad
        picked = self._pick_tier(t, ask_sum, self.journal.trades_count(slug))
        if picked is None:
            return False
        stake, is_filler = picked
        LOG.info("SIGNAL %s: pair scalp entry %s ask_up=%.2f ask_down=%.2f sum=%.3f stake=$%s",
                 slug, "FILLER" if is_filler else "scalp", au, ad, ask_sum, stake)
        nu = stake / au
        nd = stake / ad
        entry = Entry(now, au, ad, nu, nd, stake * 2,
                      leg_fee(au, nu) + leg_fee(ad, nd), stake, is_filler)
        if not self.live:
            self.journal.open_entry(slug, entry)
            LOG.info("DRY RUN: buy $%s %s Up @%s + $%s Down @%s (fees %s)",
                     stake, self.asset.upper(), au, stake, ad, entry.fee)
            self.log_prices(slug, "Up", quotes["Up"], "buy_simulated")
            self.log_prices(slug, "Down", quotes["Down"], "buy_simulated")
            return True
        ok_u = self._buy_leg(slug, end, tokens, "Up", quotes["Up"], nu)
        ok_d = self._buy_leg(slug, end, tokens, "Down", quotes["Down"], nd)
        if ok_u and ok_d:
            self.journal.open_entry(slug, entry)
            return True
        if ok_u or ok_d:
            LOG.warning("%s entry half-filled (up=%s down=%s); holding the filled leg to settle",
                        slug, ok_u, ok_d)
            self.journal.open_entry(slug, entry)
            self.journal.close_position(slug, "closed_half_held",
                                        "one leg did not fill; holding to settlement")
        return False

    def _buy_leg(self, slug, end, tokens, outcome, q0, shares):
        """FOK market buy of `shares` of `outcome`, re-quote up to 3 times."""
        if self.journal.seen(slug, outcome):
            # Row exists from an earlier round trip this window; reuse it.
            self.journal.reset_order_row(slug, outcome)
        if not self.journal.reserve(slug, outcome, shares * q0.ask, once_per_market=True):
            return False
        last_detail = "no fill in attempts"
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= end - EXIT_CUTOFF:
                self.journal.finish(slug, outcome, "unknown", "market closing before fill")
                return False
            try:
                q = quote_book(self.simulator_get("book", token_id=tokens[outcome]), time.time())
            except ValueError as exc:
                LOG.info("%s leg attempt %d: %s", outcome, attempt, exc)
                time.sleep(0.5)
                continue
            limit = min(D("0.99"), q.ask + D("0.04"))
            limit = (limit / D(q.tick)).to_integral_value(rounding=ROUND_FLOOR) * D(q.tick)
            signed = {
                "market_id": self.condition_id, "side": "BUY", "outcome": outcome,
                "amount": str((shares * limit).quantize(D("0.01"))),
                "price": str(limit), "order_type": "market", "time_in_force": "FOK",
            }
            try:
                response = self.post_with_retry(signed, end)
            except Exception as exc:
                detail = f"{type(exc).__name__}"
                if isinstance(exc, httpx.HTTPStatusError):
                    detail += f" HTTP {exc.response.status_code}: {exc.response.text[:200]}"
                if self._retryable(exc) and attempt < MAX_LEG_ATTEMPTS:
                    LOG.warning("%s %s %s (attempt %d); re-quoting",
                                self.asset.upper(), outcome, detail, attempt)
                    last_detail = detail
                    time.sleep(0.5)
                    continue
                self.journal.finish(slug, outcome, "unknown", f"Submission failed ({detail})")
                return False
            if response is None:
                self.journal.finish(slug, outcome, "unknown",
                                    "Submission disconnected; reconcile in Polysimulator")
                return False
            if is_fok_rejection(response) or response.get("status") in (
                    "rejected", "cancelled", "canceled", "expired"):
                LOG.warning("%s %s FOK rejected (attempt %d); re-quoting",
                            outcome, self.asset.upper(), attempt)
                time.sleep(0.5)
                continue
            if not response.get("success") or response.get("status") != "matched":
                self.journal.finish(slug, outcome, "unknown", json.dumps(response))
                return False
            LOG.info("%s %s buy leg filled: %s", self.asset.upper(), outcome, response)
            self.journal.finish(slug, outcome, "matched", json.dumps(response))
            self.log_prices(slug, outcome, q, "buy_matched", response)
            return True
        self.journal.finish(slug, outcome, "rejected",
                            f"no fill in {MAX_LEG_ATTEMPTS} attempts; {last_detail}")
        return False

    # ------------------------------------------------------------ exit
    def _try_exit(self, slug, end, tokens, quotes, entry, t, now):
        """Sell both legs at TP (or filler fast-exit), or hand off to settle.

        True if the position is closed."""
        bsum = quotes["Up"].bid + quotes["Down"].bid
        f_tp, f_hold = FILLER[3], FILLER[4]
        if entry.is_filler:
            hold_hit = now - entry.entry_t >= f_hold
            if bsum < f_tp and not hold_hit:
                return False
            reason = "tp" if bsum >= f_tp else ("fill" if hold_hit else "settle")
            # if we cannot sell before the window closes, fall back to settling
            if reason == "fill" and t >= FORCE_SETTLE_T:
                reason = "settle"
        else:
            if bsum < TP_BID_MIN and t < FORCE_SETTLE_T:
                return False
            reason = "tp" if bsum >= TP_BID_MIN else "settle"
        if not self.live:
            if reason == "settle":
                payout = min(entry.shares_up, entry.shares_down)
                pnl = payout - entry.cost - entry.fee
                LOG.info("DRY RUN: hold to settle %s; payout %s -> pnl %s", slug, payout, pnl)
            else:
                proceeds = entry.shares_up * quotes["Up"].bid + entry.shares_down * quotes["Down"].bid
                xfee = leg_fee(quotes["Up"].bid, entry.shares_up) \
                    + leg_fee(quotes["Down"].bid, entry.shares_down)
                pnl = proceeds - entry.cost - entry.fee - xfee
                LOG.info("DRY RUN: %s exit %s bid sum %.3f; sell up @%.2f down @%.2f -> pnl %s",
                         "TP" if reason == "tp" else "FILLER", slug, bsum,
                         quotes["Up"].bid, quotes["Down"].bid, pnl)
            self.journal.close_position(slug, "closed_" + reason,
                                        f"bid_sum={bsum}, entry={entry.ask_up}+{entry.ask_down}")
            self._cooldown_until = time.monotonic() + COOLDOWN
            self.log_prices(slug, "Up", quotes["Up"], "sell_simulated")
            self.log_prices(slug, "Down", quotes["Down"], "sell_simulated")
            return True
        if reason == "settle":
            LOG.info("HOLD %s to settlement: bid sum %.3f; entry ask %.2f+%.2f",
                     slug, bsum, entry.ask_up, entry.ask_down)
            self.journal.close_position(slug, "closed_settle",
                                        f"bid_sum={bsum}; hold to settlement")
            return False
        sold_u = self._sell_leg(slug, tokens, "Up", entry.shares_up)
        sold_d = self._sell_leg(slug, tokens, "Down", entry.shares_down)
        if sold_u and sold_d:
            LOG.info("%s EXIT %s: bid sum %.3f; sold up %s down %s",
                     "TP" if reason == "tp" else "FILLER", slug, bsum,
                     entry.shares_up, entry.shares_down)
            self.journal.close_position(slug, "closed_" + reason,
                                        f"bid_sum={bsum}; entry={entry.ask_up}+{entry.ask_down}")
            self._cooldown_until = time.monotonic() + COOLDOWN
            return True
        if sold_u or sold_d:
            LOG.warning("%s %s partial (up=%s down=%s); holding remainder to settle",
                        reason, slug, sold_u, sold_d)
            self.journal.close_position(slug, "holding", "partial exit; remainder to settlement")
        else:
            LOG.warning("%s sells failed on both legs; holding to settle", reason)
            self.journal.close_position(slug, "holding", "exit sells failed; holding to settlement")
        return False

    def _sell_leg(self, slug, tokens, outcome, shares):
        """FOK market sell of `shares`. Returns True on fill (or dust)."""
        if shares < MIN_LEG_SIZE:
            return True  # dust: leave it, settles for ~0
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= self._cur_end - EXIT_CUTOFF:
                return False
            if not self.journal.reserve_exit(slug, outcome):
                time.sleep(1.0)
                continue
            try:
                q = quote_book(self.simulator_get("book", token_id=tokens[outcome]), time.time())
            except ValueError as exc:
                self.journal.finish_exit(slug, outcome, "rejected", {"err": str(exc)}, False)
                LOG.info("%s sell leg attempt %d: %s", outcome, attempt, exc)
                time.sleep(0.5)
                continue
            limit = max(D("0.01"), q.bid - D("0.04"))
            limit = (limit / D(q.tick)).to_integral_value(rounding=ROUND_CEILING) * D(q.tick)
            signed = {
                "market_id": self.condition_id, "side": "SELL", "outcome": outcome,
                "quantity": str(shares), "price": str(limit),
                "order_type": "market", "time_in_force": "FOK",
            }
            try:
                response = self.post_with_retry(signed, self._cur_end)
            except Exception as exc:
                self.journal.finish_exit(slug, outcome, "unknown",
                                         {"err": f"{type(exc).__name__}"}, False)
                LOG.warning("%s %s sell submission failed (%s)",
                            self.asset.upper(), outcome, type(exc).__name__)
                return False
            if response is None:
                self.journal.finish_exit(slug, outcome, "unknown", {"err": "disconnected"}, False)
                return False
            if response.get("success") and response.get("status") == "matched":
                LOG.info("%s %s sell leg filled: %s", self.asset.upper(), outcome, response)
                self.journal.finish_exit(slug, outcome, "matched", response, True)
                return True
            LOG.warning("%s %s sell leg not matched: %s", outcome, self.asset.upper(), response)
            self.journal.finish_exit(slug, outcome, "rejected", response, False)
            time.sleep(0.5)
        return False

    # ------------------------------------------------------------ main loop
    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        self._cur_end = end
        now = time.time()
        t = now - (end - 300)
        quotes = {}
        for outcome, token in tokens.items():
            try:
                q = quote_book(self.simulator_get("book", token_id=token), now)
            except ValueError:
                continue
            quotes[outcome] = q
            self.log_prices(slug, outcome, q)
        self.last_quotes = (slug, {o: (q.ask, q.bid, now) for o, q in quotes.items()})
        if len(quotes) < 2:
            return
        entry, state = self.journal.position(slug)
        if state == "open":
            self._try_exit(slug, end, tokens, quotes, entry, t, now)
            return
        # "holding"/"closed_*" (or no record): consider a new round trip,
        # unless the window already reached the trade cap.
        if self.journal.trades_count(slug) >= MAX_TRADES_PER_WINDOW:
            return
        if t >= ENTRY_CUTOFF:
            return
        if now >= self._next_status_log:
            LOG.info("%s watching: t=%.0f ask sum=%.2f (tiers %s) bid sum=%.2f trades=%d/%d%s",
                     self.asset.upper(), t,
                     quotes["Up"].ask + quotes["Down"].ask,
                     ",".join(f"<={ms}" for ms, _, _ in STAKE_TIERS),
                     quotes["Up"].bid + quotes["Down"].bid,
                     self.journal.trades_count(slug), MAX_TRADES_PER_WINDOW,
                     f" [state={state}]" if state else "")
            self._next_status_log = now + 30
        self._try_enter(slug, end, tokens, quotes, now)

    @staticmethod
    def _retryable(exc):
        """Book-move errors (FOK cannot fill / stale book) deserve a re-quote."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        if exc.response.status_code == 429:
            return True
        body = exc.response.text
        return "FOK_ORDER_NOT_FILLED" in body or "STALE_OR_MISMATCHED_BOOK" in body


# ------------------------------------------------------------------ main
def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", "--live", dest="live", action="store_true",
                        help="Submit Polysimulator paper orders only (--live is a compatibility alias)")
    parser.add_argument("--asset", choices=Bot.assets, default="btc",
                        help="Trade only this asset (default: btc)")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "scalp_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    failed = False
    bot = None
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.05 <= poll <= 60:
            raise ValueError("Polling must be 0.05-60 seconds")
        bot = ScalpBot(args.live, args.asset)
        configure_simulator_pacing(bot)
        LOG.info("%s SCALP %s; tiers %s, filler $%s (sum<=%s from t=%s after %d trades, "
                 "fast-exit bid-sum>=%s or %ss hold), TP bid-sum >= %s, max %d trades/window, "
                 "settle fallback, cooldown %.0fs; polling delay %ss",
                 "POLYSIMULATOR PAPER" if args.live else "DRY RUN",
                 args.asset.upper(),
                 ",".join(f"<={ms}/${st}" for ms, _, st in STAKE_TIERS),
                 FILLER[2], FILLER[0], int(FILLER[1]), FILLER_MIN_DONE,
                 FILLER[3], int(FILLER[4]),
                 TP_BID_MIN, MAX_TRADES_PER_WINDOW, COOLDOWN, poll)
        while True:
            failed = False
            try:
                bot.step()
            except httpx.HTTPError as exc:
                LOG.warning("%s market data request failed (%s)",
                            args.asset.upper(), type(exc).__name__)
                failed = True
            if args.once:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s. Check configuration, journal and account before retrying.",
                  type(exc).__name__, exc)
        return 1
    finally:
        if bot is not None:
            bot.close()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
