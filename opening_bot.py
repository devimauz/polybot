"""Buy both sides of a new five-minute Polysimulator market, then take 10% gross profit.

Default: BTC, $5 per side, first 30 seconds only. Use --paper to submit orders.
The two buys are separate FOK orders; one can fill while the other rejects.
"""
import argparse
import json
import logging
import os
import sqlite3
import time
import uuid
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import httpx
from dotenv import load_dotenv

from bot import Bot, ROOT, pace_simulator
from edge_trader import book_quote, meets_minimum_size

LOG = logging.getLogger("opening-bot")
API = "https://api.polysimulator.com/v1/"


def profit_target(price, tick):
    return (price * Decimal("1.10") / tick).to_integral_value(rounding=ROUND_CEILING) * tick


class OpeningBot:
    # Reuse only simulator discovery and authenticated GETs, not the threshold strategy.
    discover = Bot.discover
    simulator_get = Bot.simulator_get

    def __init__(self, asset="btc", amount=Decimal("5"), paper=False,
                 entry_window=30, db_path=None, http_client=None):
        self.asset, self.amount, self.paper = asset, amount, paper
        self.entry_window = entry_window
        self.api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Set POLYSIM_API_KEY in .env")
        self._owns_http = http_client is None
        self.http = http_client or httpx.Client(timeout=10)
        self.market = None
        self.condition_id = None
        name = "opening-paper.sqlite3" if paper else "opening-dry-run.sqlite3"
        self.db = sqlite3.connect(db_path or ROOT / name, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS trades (
            market TEXT, outcome TEXT, cid TEXT, end REAL, tick TEXT,
            entry_state TEXT, shares TEXT, price TEXT, tp_state TEXT,
            retry_at REAL DEFAULT 0, detail TEXT, PRIMARY KEY(market,outcome))""")
        self.db.commit()

    def post(self, body):
        pace_simulator()
        response = self.http.post(API + "orders", json=body,
                                  headers={"X-API-Key": self.api_key, "Idempotency-Key": str(uuid.uuid4())})
        response.raise_for_status()
        return response.json()

    def update(self, slug, outcome, **fields):
        assignments = ",".join(f"{key}=?" for key in fields)
        self.db.execute(f"UPDATE trades SET {assignments} WHERE market=? AND outcome=?",
                        (*fields.values(), slug, outcome))
        self.db.commit()

    def error_detail(self, exc):
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}: " + exc.response.text.replace(self.api_key, "[REDACTED]")[:2000]
        return type(exc).__name__

    def enter(self, slug, end, outcome, token, direct_tick=None):
        start = end - 300
        if not start <= time.time() < start + self.entry_window:
            return
        previous = self.db.execute("SELECT entry_state,retry_at FROM trades WHERE market=? AND outcome=?",
                                   (slug, outcome)).fetchone()
        if previous and (previous["entry_state"] != "retry" or previous["retry_at"] > time.time()):
            return
        if direct_tick is not None:
            # Fast path: the caller already has a live Polymarket WebSocket
            # quote. 0.99 is the binary-contract ceiling and intentionally
            # disables the usual ask-relative slippage cap.
            tick = Decimal(str(direct_tick))
            if not tick.is_finite() or not 0 < tick < 1:
                raise ValueError("invalid direct quote tick")
            limit = (Decimal("0.99") / tick).to_integral_value(
                rounding=ROUND_FLOOR) * tick
        else:
            data = self.simulator_get("book", token_id=token)
            ask, tick = book_quote(data, time.time())
            if not ask:
                return
            # Leave room for a 10% target below the $1 payout ceiling.
            ceiling = min(Decimal("0.99") / Decimal("1.10"), Decimal(str(ask[0])) + Decimal("0.05"))
            limit = (ceiling / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
            if limit < Decimal(str(ask[0])):
                LOG.info("SKIP %s %s: price too high for 10%% target", slug, outcome)
                return
            enough, minimum, shares = meets_minimum_size(data, self.amount, limit)
            if not enough:
                LOG.info("SKIP %s %s: $%s covers %s shares; minimum %s", slug, outcome, self.amount, shares, minimum)
                return
        # Check again after data fetching; never chase a market outside the opening window.
        pace_simulator()
        if not start <= time.time() < start + self.entry_window:
            return
        cursor = self.db.execute("""INSERT INTO trades
            (market,outcome,cid,end,tick,entry_state,detail) VALUES (?,?,?,?,?,'reserved','')
            ON CONFLICT(market,outcome) DO UPDATE SET entry_state='reserved',detail=''
            WHERE trades.entry_state='retry' AND trades.retry_at<=?""",
            (slug, outcome, self.condition_id, end, str(tick), time.time()))
        self.db.commit()
        if not cursor.rowcount:
            return
        if not self.paper:
            self.update(slug, outcome, entry_state="dry-run", detail=f"BUY amount={self.amount} limit={limit}")
            LOG.info("DRY RUN BUY %s %s $%s limit=%s; TP would use actual fill +10%%",
                     slug, outcome, self.amount, limit)
            return
        try:
            body = {"market_id": self.condition_id, "side": "BUY", "outcome": outcome,
                    "amount": str(self.amount), "price": str(limit), "order_type": "market", "time_in_force": "FOK"}
            # Already paced immediately before reservation: don't delay this entry again.
            response = self.http.post(API + "orders", json=body,
                                      headers={"X-API-Key": self.api_key, "Idempotency-Key": str(uuid.uuid4())})
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("status", "")).lower() not in ("filled", "matched"):
                raise ValueError("entry not confirmed filled")
            price, quantity = Decimal(str(payload["price"])), Decimal(str(payload["quantity"]))
            if not price.is_finite() or not quantity.is_finite() or not 0 < price <= limit or quantity <= 0:
                raise ValueError("invalid fill response")
            self.update(slug, outcome, entry_state="filled", shares=str(quantity), price=str(price),
                        tp_state="needed", detail=json.dumps(payload))
            LOG.info("FILLED %s %s shares=%s price=%s", slug, outcome, quantity, price)
        except Exception as exc:
            retry = False
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    code = exc.response.json().get("error") or exc.response.json().get("code")
                    retry = exc.response.status_code in (400, 404, 429) and code in (
                        "FOK_ORDER_NOT_FILLED_ERROR", "MARKET_NOT_FOUND", "RATE_LIMIT_EXCEEDED")
                except (ValueError, AttributeError):
                    pass
            self.update(slug, outcome, entry_state="retry" if retry else "unknown",
                        retry_at=time.time() + 2, detail=self.error_detail(exc))
            LOG.warning("BUY %s %s: %s (%s)", slug, outcome, self.error_detail(exc),
                        "retry within opening window" if retry else "reserved; inspect journal")

    def place_profits(self):
        # Pending work survives restarts; resting GTC sells execute on the server.
        rows = self.db.execute("""SELECT * FROM trades
            WHERE entry_state='filled' AND tp_state='needed' AND retry_at<=?""", (time.time(),)).fetchall()
        for row in rows:
            slug, outcome = row["market"], row["outcome"]
            if time.time() >= row["end"]:
                self.update(slug, outcome, tp_state="expired")
                continue
            target = profit_target(Decimal(row["price"]), Decimal(row["tick"]))
            if target >= 1:
                self.update(slug, outcome, tp_state="unreachable")
                LOG.warning("TP unreachable below $1: %s %s", slug, outcome)
                continue
            cursor = self.db.execute("""UPDATE trades SET tp_state='reserved'
                WHERE market=? AND outcome=? AND tp_state='needed'""", (slug, outcome))
            self.db.commit()
            if not cursor.rowcount:
                continue
            try:
                payload = self.post({"market_id": row["cid"], "side": "SELL", "outcome": outcome,
                                     "quantity": row["shares"], "price": str(target),
                                     "order_type": "limit", "time_in_force": "GTC"})
                status = str(payload.get("status", "")).lower()
                order_id = payload.get("order_id", payload.get("id"))
                if status not in ("open", "pending", "live", "placed", "accepted", "filled", "matched", "partially_filled") or order_id is None:
                    raise ValueError("TP order not confirmed accepted")
                self.update(slug, outcome, tp_state="submitted", detail=json.dumps(payload))
                LOG.info("TP SELL %s %s target=%s order=%s status=%s", slug, outcome, target, order_id, status)
            except Exception as exc:
                # Only explicit rate-limit rejections may be automatically retried.
                retry = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429
                self.update(slug, outcome, tp_state="needed" if retry else "unknown",
                            retry_at=time.time() + 10, detail=self.error_detail(exc))
                LOG.warning("TP %s %s: %s; %s", slug, outcome, self.error_detail(exc),
                            "retrying later" if retry else "inspect journal and simulator")

    def step(self):
        now = time.time()
        if now % 300 < self.entry_window:
            expected = f"{self.asset}-updown-5m-{int(now)//300*300}"
            if self.market is None or self.market[0] != expected:
                self.market = self.discover(now)
            if self.market:
                slug, end, tokens = self.market
                try:
                    # Attempt both entries before spending API requests on the exits.
                    for outcome, token in tokens.items():
                        try:
                            self.enter(slug, end, outcome, token)
                        except (httpx.HTTPError, ValueError, KeyError) as exc:
                            LOG.warning("Entry scan %s %s: %s", slug, outcome, self.error_detail(exc))
                finally:
                    self.place_profits()
                return
        self.place_profits()

    def close(self):
        if self._owns_http:
            self.http.close()
        self.db.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="Submit Polysimulator orders (default: dry run)")
    parser.add_argument("--asset", choices=Bot.assets, default="btc")
    parser.add_argument("--amount", type=Decimal, default=Decimal("5"), help="Dollars per side")
    parser.add_argument("--entry-window", type=float, default=30, help="Seconds after market start to allow entry")
    parser.add_argument("--once", action="store_true", help="One check, without waiting for the next window")
    args = parser.parse_args()
    if not args.amount.is_finite() or args.amount <= 0 or not 0 < args.entry_window <= 60:
        parser.error("amount must be positive; entry-window must be 0-60 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    trader = None
    try:
        trader = OpeningBot(args.asset, args.amount, args.paper, args.entry_window)
        LOG.info("%s %s: $%s per side; 10%% gross TP; first %ss only",
                 "PAPER" if args.paper else "DRY RUN", args.asset.upper(), args.amount, args.entry_window)
        if time.time() % 300 >= args.entry_window:
            LOG.info("Waiting for next five-minute opening")
        while True:
            try:
                trader.step()
            except httpx.HTTPError as exc:
                LOG.warning("Market request failed: %s", type(exc).__name__)
            if args.once:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        LOG.info("Stopped; submitted take-profit orders remain active on Polysimulator")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        if trader is not None:
            trader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
