"""Final-minute direction bot using Polymarket prices and Polysimulator orders.

For each of the five assets (BTC, ETH, SOL, BNB, DOGE) watch the
five-minute UP/DOWN market. During the final 60 seconds, buy UP when its best
ask stays above 90c (and no higher than 98c) for five continuous seconds. Apply
the same rule to DOWN. Buy only the triggered side, with at most one trade per
market per asset.

Run:
  python ninefive_bot.py --paper    paper orders on polysimulator
  python ninefive_bot.py            dry run, signals logged, no orders
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR

import httpx
from dotenv import load_dotenv

from bot import (Bot, D, Journal, LOG, ROOT,
                 configure_simulator_pacing, is_fok_rejection)
from polymarket_feed import MarketFeed

STAKE = D("100")          # dollars on the triggered side
FAV_ASK_MAX = D("0.98")   # highest qualifying outcome ask
FAV_ASK_MIN = D("0.90")   # qualifying ask must be strictly above this
TRADE_WINDOW_SECONDS = 60 # only enter during the final minute
ABOVE_HOLD_SECONDS = 5.0  # required continuous time above 90c
FAV_SLIPPAGE = D("0.06")  # FOK worst-price cushion on the favorite (book moves fast)
MAX_LEG_ATTEMPTS = 3      # re-quote retries per leg before giving up

DASH_EVENT_KEYWORDS = ("SIGNAL", "filled", "rejected", "aborted", "not matched",
                       "DRY RUN", "pair", "ERROR", "CRITICAL", "Waiting for active")


class NineFiveJournal(Journal):
    def market_filled(self, market):
        """True only when the market has at least one actually-filled order."""
        return self._retry(lambda: self.db.execute(
            "SELECT 1 FROM orders WHERE market=? AND filled=1 LIMIT 1", (market,)).fetchone() is not None)

    def condition_for(self, market):
        """condition_id stored on any order row for this market (may be empty)."""
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
        # Base behavior, plus: when the response is a real fill, persist the
        # fill price / share count so the journal (and dashboard) is accurate.
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

    def reserve(self, market, outcome, amount=STAKE, *, once_per_market=False, condition_id=""):
        # Commit before sending: a timeout or crash must never cause a duplicate.
        def do():
            cols = [c[1] for c in self.db.execute("PRAGMA table_info(orders)")]
            if "condition_id" not in cols:
                self.db.execute("ALTER TABLE orders ADD COLUMN condition_id TEXT")
                self.db.commit()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                day = datetime.now(timezone.utc).date().isoformat()
                if (self.seen_market(market) if once_per_market else self.seen(market, outcome)):
                    self.db.rollback()
                    return False
                self.db.execute(
                    "INSERT INTO orders(market, outcome, day, amount, status, detail, condition_id) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (market, outcome, day, float(amount), "reserved", "", condition_id))
                self.db.commit()
                return True
            except BaseException:
                self.db.rollback()
                raise
        return self._retry(do)


_MARKET_STATUS_CACHE = {}


class NineFiveBot(Bot):
    """Direction bot: $100 after a side stays above 90c for five seconds."""

    amount = STAKE
    journal_type = NineFiveJournal
    journal_prefix = "ninefive-"
    price_log_dir = ROOT / "price_logs" / "ninefive"
    last_quotes = ("", {})

    def __init__(self, live, asset="btc"):
        super().__init__(live, asset)
        self.feed = MarketFeed()
        self.above_since = {"Up": None, "Down": None}
        self.timer_market = None

    def polymarket_quotes(self, tokens):
        """Latest pushed Polymarket BBOs; never falls back to Polysimulator."""
        self.feed.subscribe(tokens)
        return self.feed.quotes()

    def market_status(self, condition_id):
        """Resolve a (possibly closed) market: 'closed'|'resolved'|None + winner."""
        cached = _MARKET_STATUS_CACHE.get(condition_id)
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
        _MARKET_STATUS_CACHE[condition_id] = (now, result)
        return result

    def step(self):
        market = self.discover(time.time())
        if not market:
            LOG.info("Waiting for active %s five-minute market", self.asset.upper())
            return
        slug, end, tokens = market
        self.feed.subscribe(tokens)
        if self.timer_market != slug:
            self.above_since = {"Up": None, "Down": None}
            self.timer_market = slug
        if self.journal.seen_market(slug):
            now = time.monotonic()
            if now >= getattr(self, "_next_wait_log", 0):
                LOG.info("%s pair already placed this session; next session in %.0fs",
                         self.asset.upper(), max(0, end - time.time()))
                self._next_wait_log = now + 15
            return
        seconds_remaining = end - time.time()
        if seconds_remaining <= 0 or seconds_remaining > TRADE_WINDOW_SECONDS:
            return
        now = time.time()
        quotes = self.feed.quotes()
        new_state = {}
        log_line_due = now >= getattr(self, "_next_quote_log", 0)
        for outcome, q in quotes.items():
            if log_line_due:
                LOG.info("%s %s Polymarket midpoint=%.1f%% ask=%.3f",
                         self.asset.upper(), outcome, q.probability * 100, q.ask)
            self.log_prices(slug, outcome, q)
            quotes[outcome] = q
            new_state[outcome] = (q.ask, time.time())
        self.last_quotes = (slug, new_state)
        self._next_quote_log = now + 10
        if len(quotes) < 2:
            return
        for outcome, q in quotes.items():
            if FAV_ASK_MIN < q.ask <= FAV_ASK_MAX:
                if self.above_since[outcome] is None:
                    self.above_since[outcome] = now
            else:
                self.above_since[outcome] = None
        eligible = [(q.ask, outcome, q) for outcome, q in quotes.items()
                    if self.above_since[outcome] is not None
                    and now - self.above_since[outcome] >= ABOVE_HOLD_SECONDS]
        if not eligible:
            return
        _, fav_out, fav_q = max(eligible)
        held = now - self.above_since[fav_out]
        LOG.info("SIGNAL %s: %s ask=%.2f stayed above %.2f for %.2fs with %.1fs left; "
                 "buying only %s for $%s", slug, fav_out, fav_q.ask,
                 FAV_ASK_MIN, held, seconds_remaining, fav_out, STAKE)
        if not self._buy_leg(slug, end, tokens, fav_out, FAV_ASK_MAX, FAV_SLIPPAGE):
            LOG.warning("%s %s order did not fill", slug, fav_out)

    def _buy_leg(self, slug, end, tokens, outcome, max_price, slippage=D("0.05")):
        """One leg of the pair: FOK market buy capped at max_price, re-quote retries."""
        if self.journal.seen(slug, outcome):
            LOG.info("%s leg already placed", outcome)
            return True
        if not self.journal.reserve(slug, outcome, self.amount, condition_id=self.condition_id or ""):
            return False
        last_detail = "no fill in attempts"
        for attempt in range(1, MAX_LEG_ATTEMPTS + 1):
            if time.time() >= end:
                self.journal.finish(slug, outcome, "unknown", "market closed before fill")
                return False
            q = self.feed.quotes().get(outcome)
            if q is None:
                LOG.info("%s leg attempt %d: waiting for Polymarket book", outcome, attempt)
                time.sleep(0.1)
                continue
            if q.ask > max_price:
                LOG.info("%s leg attempt %d: ask %.2f above cap %s; waiting",
                         outcome, attempt, q.ask, max_price)
                time.sleep(0.1)
                continue
            tick = D(q.tick)
            limit = min(D("0.99"), q.ask + slippage)
            limit = (limit / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
            signed = {
                "market_id": self.condition_id, "side": "BUY",
                "outcome": outcome, "amount": str(self.amount),
                "price": str(limit), "order_type": "market",
                "time_in_force": "FOK",
            }
            if not self.live:
                self.journal.finish(slug, outcome, "simulated", f"dry run, ask {q.ask}, cap {max_price}")
                LOG.info("DRY RUN: buy $%s %s %s at %s", self.amount, self.asset.upper(), outcome, limit)
                self.log_prices(slug, outcome, q, "buy_simulated")
                return True
            try:
                response = self.post_with_retry(signed, end)
            except Exception as exc:
                detail = f"{type(exc).__name__}"
                if isinstance(exc, httpx.HTTPStatusError):
                    detail += f" HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                if self._retryable(exc) and attempt < MAX_LEG_ATTEMPTS:
                    LOG.warning("%s %s %s (attempt %d); re-quoting", self.asset.upper(), outcome, detail, attempt)
                    last_detail = detail
                    time.sleep(0.1)
                    continue
                self.journal.finish(slug, outcome, "unknown", f"Submission failed ({detail}); reconcile in Polysimulator")
                LOG.warning("%s %s submission failed (%s); leg abandoned",
                            self.asset.upper(), outcome, detail)
                return False
            if response is None:
                self.journal.finish(slug, outcome, "unknown", "Submission disconnected or market closing; reconcile in Polysimulator")
                return False
            if is_fok_rejection(response) or response.get("status") in ("rejected", "cancelled", "canceled", "expired"):
                LOG.warning("%s %s FOK rejected (attempt %d); re-quoting", self.asset.upper(), outcome, attempt)
                time.sleep(0.1)
                continue
            if not response.get("success") or response.get("status") != "matched":
                # Deterministic server answer that is not a fill: record it and
                # move on instead of stopping every asset worker.
                self.journal.finish(slug, outcome, "unknown", json.dumps(response))
                LOG.warning("%s %s order not matched: %s; leg abandoned",
                            self.asset.upper(), outcome, response.get("status"))
                return False
            LOG.info("%s %s direction order filled: %s", self.asset.upper(), outcome, response)
            self.journal.finish(slug, outcome, "matched", json.dumps(response))
            self.log_prices(slug, outcome, q, "buy_matched", response)
            return True
        self.journal.finish(slug, outcome, "rejected", f"no fill in {MAX_LEG_ATTEMPTS} attempts (cap {max_price}); {last_detail}")
        return False

    @staticmethod
    def _retryable(exc):
        """Book-move errors (FOK cannot fill / stale book) deserve a re-quote."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        if exc.response.status_code == 429:
            return True
        body = exc.response.text
        return "FOK_ORDER_NOT_FILLED" in body or "STALE_OR_MISMATCHED_BOOK" in body


def journal_positions(journals, resolver=None, limit=10):
    """Newest positions from the journals, with the window's final outcome.

    resolver(condition_id) -> (status, winner) so closed windows (which drop
    out of the active market list) still resolve their outcome.  The SQL limit
    is important for the live dashboard: resolving the entire trade history
    before its first render can otherwise leave the terminal blank for minutes.
    """
    rows = []
    seen = set()
    now = time.time()
    for j in journals:
        try:
            cols = [c[1] for c in j.db.execute("PRAGMA table_info(orders)")]
            has_cid = "condition_id" in cols
            sql = ("SELECT market, outcome, fill_price, shares"
                   + (", condition_id" if has_cid else ", ''") +
                   " FROM orders WHERE status='matched' AND shares>0 "
                   "ORDER BY market DESC LIMIT " + str(max(1, int(limit) * 2)))
            for market, outcome, fill_price, shares, *rest in j.db.execute(sql):
                if (market, outcome) in seen:
                    continue
                seen.add((market, outcome))
                condition_id = rest[0] if rest else ""
                asset, start_s = market.rsplit("-updown-5m-", 1)
                # Expired windows with no stored condition_id cannot be resolved
                # remotely; do not pin them at "open" forever.
                if now - int(start_s) > 15 * 60 and not condition_id:
                    continue
                winner = None
                if condition_id and resolver is not None:
                    try:
                        _status, winner = resolver(condition_id)
                    except Exception:
                        winner = None
                if not winner and condition_id:
                    # Fresh window: try the active market list as a fallback.
                    try:
                        payload = j.simulator_get("markets/updown", asset=asset.upper(), interval="5M")
                        mkts = payload if isinstance(payload, list) else payload.get("markets", [])
                        for m in mkts:
                            if m.get("slug") != market:
                                continue
                            prices = m.get("outcome_prices")
                            if prices and len(prices) >= 2 and (max(prices) >= 0.995 or min(prices) <= 0.005):
                                winner = "Up" if prices[0] >= prices[1] else "Down"
                            break
                    except Exception:
                        pass
                pnl = None
                if winner:
                    pnl = (shares if outcome == winner else 0.0) - fill_price * shares
                rows.append({"market": market, "outcome": outcome, "start": int(start_s),
                             "winner": winner, "fill": fill_price, "shares": shares, "pnl": pnl})
        except Exception:
            continue
    rows.sort(key=lambda r: r["start"], reverse=True)
    return rows


def run_dashboard(args, assets):
    """Live rich terminal dashboard; the bot itself keeps writing ninefive_bot.log."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    load_dotenv(ROOT / ".env")
    import logging
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(ROOT / "ninefive_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots = [NineFiveBot(args.live, asset) for asset in assets]
    try:
        configure_simulator_pacing(bots[0])
    except Exception:
        pass
    LOG.info("%s FINAL-MINUTE dashboard; $%s on triggered side, ask >%.2f up to %.2f for %.1fs",
             "POLYSIMULATOR PAPER" if args.live else "DRY RUN",
             STAKE, FAV_ASK_MIN, FAV_ASK_MAX, ABOVE_HOLD_SECONDS)
    log_file = ROOT / "ninefive_bot.log"
    cursor = 0
    try:
        cursor = log_file.stat().st_size
    except OSError:
        pass
    events = []
    live_positions = []
    next_positions_refresh = 0.0
    try:
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            console.print(Text("Starting 95c dashboard...", style="bold yellow"))
            console.show_cursor(False)
            try:
                while True:
                    futures = [(b, workers.submit(b.step)) for b in bots]
                    for bot, fut in futures:
                        try:
                            fut.result()
                        except httpx.HTTPError as exc:
                            LOG.warning("%s market data request failed (%s)", bot.asset.upper(), type(exc).__name__)
                    now = time.time()
                    if now >= next_positions_refresh:
                        try:
                            payload = bots[0].simulator_get("account/positions", status="OPEN")
                            live_positions = (payload if isinstance(payload, list)
                                              else payload.get("positions", []))
                        except Exception as exc:
                            LOG.warning("Live position refresh failed (%s)", type(exc).__name__)
                        next_positions_refresh = now + 2
                    # new log lines -> event feed
                    try:
                        if log_file.stat().st_size < cursor:
                            cursor = 0  # rotated
                        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                            fh.seek(cursor)
                            for line in fh:
                                cursor = fh.tell()
                                if any(k in line for k in DASH_EVENT_KEYWORDS):
                                    text = line.strip()
                                    if text:
                                        events.append(text)
                        events[:] = events[-14:]
                    except OSError:
                        pass
                    table = Table(expand=True, pad_edge=False, header_style="bold yellow")
                    for col, justify in (("asset", "left"), ("close", "right"),
                                         ("up ask", "right"), ("down ask", "right"),
                                         ("favorite", "right"), ("state", "left")):
                        table.add_column(col, justify=justify)
                    now = time.time()
                    for bot in bots:
                        mk = getattr(bot, "market", None)
                        if not mk:
                            table.add_row(bot.asset.upper(), "-", "-", "-", "-", Text("waiting", style="dim"))
                            continue
                        slug, end, tokens = mk
                        t = max(0, end - now)
                        asks = {}
                        lq_slug, lq = getattr(bot, "last_quotes", ("", {}))
                        if lq_slug == slug:
                            for outcome, (ask, ts) in lq.items():
                                if now - ts < 5:
                                    asks[outcome] = ask
                        up_s = f"{asks['Up']:.2f}" if "Up" in asks else "-"
                        down_s = f"{asks['Down']:.2f}" if "Down" in asks else "-"
                        if "Up" in asks and "Down" in asks:
                            fav_out, fav_q = (("Up", asks["Up"]) if asks["Up"] > asks["Down"]
                                              else ("Down", asks["Down"]))
                            fav_s = f"{fav_out} {fav_q:.2f}"
                            since = bot.above_since.get(fav_out)
                            held = max(0.0, now - since) if since is not None else 0.0
                            if (0 < t <= TRADE_WINDOW_SECONDS
                                    and held >= ABOVE_HOLD_SECONDS):
                                state = Text("TRIGGER", style="bold green")
                            elif t > TRADE_WINDOW_SECONDS:
                                state = Text("waiting final minute", style="yellow")
                            elif since is not None:
                                state = Text(f"arming {held:.1f}/{ABOVE_HOLD_SECONDS:.0f}s",
                                             style="yellow")
                            else:
                                state = Text("price outside range", style="white")
                        else:
                            fav_s, state = "-", Text("thin book", style="dim")
                        if bot.journal.market_filled(slug):
                            state = Text("pair filled", style="bold magenta")
                        elif bot.journal.seen_market(slug):
                            state = Text("attempt failed", style="dim red")
                        table.add_row(bot.asset.upper(), f"{t:4.0f}s", up_s, down_s, fav_s, state)
                    pos_lines = []
                    for p in live_positions[-10:]:
                        try:
                            qty = D(str(p.get("quantity") or 0))
                            entry = D(str(p.get("avg_entry_price") or 0))
                        except (ValueError, ArithmeticError):
                            continue
                        if qty <= 0 or str(p.get("status", "")).upper() != "OPEN":
                            continue
                        outcome = p.get("outcome", "?")
                        market_id = p.get("market_id", "")
                        mark, source = None, "Polysim"
                        for bot in bots:
                            if bot.condition_id == market_id:
                                quote = bot.feed.quotes().get(outcome)
                                if quote is not None:
                                    mark, source = quote.bid, "PM bid"
                                break
                        if mark is None and p.get("current_price") is not None:
                            try:
                                mark = D(str(p["current_price"]))
                            except (ValueError, ArithmeticError):
                                pass
                        question = str(p.get("market_question") or market_id)[:38]
                        base = f"{question} | {outcome} {qty:.2f} sh @ {entry:.3f}"
                        if mark is None:
                            pos_lines.append(Text(base + " | mark waiting", style="yellow"))
                        else:
                            value, pnl = qty * mark, qty * (mark - entry)
                            pos_lines.append(Text(
                                base + f" | {source} {mark:.3f} | value ${value:.2f} | PnL {pnl:+.2f}",
                                style="green" if pnl >= 0 else "red"))
                    pos_panel = Panel(
                        Group(*pos_lines) if pos_lines else Text("no open Polysimulator positions", style="dim"),
                        title="Live Polysimulator Positions (marked to Polymarket)",
                        title_align="left")
                    ev_lines = [Text(e, style="red" if "ERROR" in e or "rejected" in e else "cyan")
                                for e in events[-8:]]
                    ev_panel = Panel(Group(*ev_lines) if ev_lines else Text("no events yet", style="dim"),
                                     title="Events (SIGNAL / fills / errors)", title_align="left")
                    header = Text(f"FINAL-MINUTE bot - {'PAPER' if args.live else 'dry run'} - "
                                  f"${STAKE} one side - ask >{FAV_ASK_MIN} to {FAV_ASK_MAX} "
                                  f"for {ABOVE_HOLD_SECONDS:.0f}s - "
                                  f"{time.strftime('%H:%M:%S')}", style="bold")
                    # Render off-screen, then replace the terminal frame in one
                    # write. This avoids both scrolling and the flash caused by
                    # clearing the console before drawing the next frame.
                    with console.capture() as capture:
                        console.print(Group(header, table, pos_panel, ev_panel))
                    sys.stdout.write("\x1b[H" + capture.get() + "\x1b[J")
                    sys.stdout.flush()
                    time.sleep(0.5)
            finally:
                console.show_cursor(True)
    finally:
        for b in bots:
            b.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", "--live", dest="live", action="store_true",
                        help="Submit Polysimulator paper orders only (--live is a compatibility alias)")
    parser.add_argument("--asset", choices=Bot.assets,
                        help="Trade only this asset (default: all of " + "/".join(Bot.assets) + ")")
    parser.add_argument("--once", action="store_true", help="Check once, then exit")
    parser.add_argument("--gui", action="store_true",
                        help="Show the live rich terminal dashboard (log file still records everything)")
    args = parser.parse_args()
    assets = (args.asset,) if args.asset else Bot.assets
    if args.gui:
        try:
            run_dashboard(args, assets)
            return 0
        except KeyboardInterrupt:
            LOG.info("Stopped")
            return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "ninefive_bot.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bots, failed = [], False
    try:
        poll = float(os.environ.get("POLL_SECONDS", "0.5"))
        if not 0.05 <= poll <= 60:
            raise ValueError("Polling must be 0.05-60 seconds")
        for asset in assets:
            bots.append(NineFiveBot(args.live, asset))
        configure_simulator_pacing(bots[0])
        LOG.info("%s FINAL-MINUTE %s; Polymarket live prices -> Polysimulator orders; "
                 "$%s on triggered side, ask >%.2f up to %.2f for %.1fs; polling delay %ss",
                 "POLYSIMULATOR PAPER" if args.live else "DRY RUN",
                 "/".join(asset.upper() for asset in assets), STAKE, FAV_ASK_MIN,
                 FAV_ASK_MAX, ABOVE_HOLD_SECONDS, poll)
        with ThreadPoolExecutor(max_workers=len(bots)) as workers:
            while True:
                failed = False
                futures = [(bot, workers.submit(bot.step)) for bot in bots]
                for bot, future in futures:
                    try:
                        future.result()
                    except httpx.HTTPError as exc:
                        LOG.warning("%s market data request failed (%s)", bot.asset.upper(), type(exc).__name__)
                        failed = True
                if args.once:
                    break
                time.sleep(poll)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s. Check configuration, journal and account before retrying.", type(exc).__name__, exc)
        return 1
    finally:
        for bot in bots:
            bot.close()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
