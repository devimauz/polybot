"""Polysimulator edge trader: paper buys held until resolution.

Run normally for simulated orders, or --dry-run --once to inspect signals.
"""
import argparse
import json
import logging
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import httpx
from dotenv import load_dotenv
from bot import Journal, pace_simulator

ROOT = Path(__file__).resolve().parent
API = "https://api.polysimulator.com"
LOG = logging.getLogger("edge-trader")
STAKE = 1.65
MIN_REM, MAX_REM = 90, 420
MIN_EDGE = 0.18
MAX_PRICE = 0.72
BL = 0.35
MAX_PER_WINDOW = 4.00
WALLET_FLOOR = 1
ASSETS = ("btc", "eth", "sol", "xrp", "doge", "bnb")


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def decode_list(value):
    return json.loads(value) if isinstance(value, str) else value


def meets_minimum_size(data, amount, limit):
    minimum = Decimal(str(data["min_order_size"]))
    if not minimum.is_finite() or minimum <= 0:
        raise ValueError("invalid minimum order size")
    # Check at the worst permitted price so the stake covers the minimum
    # even if the quote moves before execution. The API rounds shares down.
    shares = (Decimal(str(amount)) / limit).quantize(Decimal("0.0001"), rounding=ROUND_FLOOR)
    return shares >= minimum, minimum, shares


def book_quote(data, now):
    if data.get("stale") is True:
        raise ValueError("stale order book")
    stamp = data["timestamp"]
    try:
        stamp = float(stamp) / 1000
    except (ValueError, TypeError):
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("book timestamp has no timezone")
        stamp = parsed.timestamp()
    if not -5 <= now - stamp <= 30:
        raise ValueError("stale order book")
    asks = [(float(x["price"]), float(x["size"])) for x in data.get("asks", [])]
    asks = [(p, size) for p, size in asks if 0 < p < 1 and math.isfinite(size) and size > 0]
    tick = Decimal(str(data["tick_size"]))
    if not tick.is_finite() or not 0 < tick < 1:
        raise ValueError("invalid tick size")
    return min(asks, default=None), tick


class EdgeTrader:
    def __init__(self, dry_run=False):
        self.api_key = os.environ.get("POLYSIM_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Set POLYSIM_API_KEY in the local .env")
        self.dry_run = dry_run
        self.http = httpx.Client(timeout=12)
        name = "edge-polysimulator-dry-run.sqlite3" if dry_run else "edge-polysimulator.sqlite3"
        self.journal = Journal(ROOT / name)
        self.spent = 0.0

    def api_get(self, path, **params):
        pace_simulator()
        response = self.http.get(API + path, params=params, headers={"X-API-Key": self.api_key})
        response.raise_for_status()
        return response.json()

    def balance(self):
        # Simulator balances are dollars, not token base units.
        value = float(self.api_get("/v1/account/balance")["balance"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid simulator balance")
        return value

    def spot_and_sigma(self, asset):
        # Preserve the original Binance spot/one-minute volatility model.
        params = {"symbol": asset.upper() + "USDT"}
        response = self.http.get("https://api.binance.com/api/v3/ticker/price", params=params)
        response.raise_for_status()
        spot = float(response.json()["price"])
        response = self.http.get("https://api.binance.com/api/v3/klines",
                                 params={**params, "interval": "1m", "limit": 20})
        response.raise_for_status()
        closes = [float(c[4]) for c in response.json()]
        if not math.isfinite(spot) or spot <= 0 or any(not math.isfinite(c) or c <= 0 for c in closes):
            raise ValueError("invalid Binance prices")
        returns = [(b - a) / a for a, b in zip(closes, closes[1:])]
        return spot, statistics.pstdev(returns) if len(returns) > 1 else 0

    def discover(self, asset, duration, start, seconds):
        slug = f"{asset}-updown-{duration}-{start}"
        payload = self.api_get("/v1/markets/updown", asset=asset.upper(), interval=duration.upper())
        rows = payload if isinstance(payload, list) else payload.get("markets", payload.get("items", []))
        for row in rows:
            if row.get("slug") != slug or not row.get("active") or row.get("closed") or row.get("resolved"):
                continue
            end = datetime.fromisoformat(row["end_date"].replace("Z", "+00:00")).timestamp()
            if abs(end - (start + seconds)) > 1:
                raise ValueError("market window mismatch")
            cid = row["condition_id"]
            nested = next(m for m in row["markets"] if m.get("condition_id") == cid)
            outcomes, tokens = decode_list(nested["outcomes"]), decode_list(nested["token_ids"])
            if len(outcomes) != 2 or set(outcomes) != {"Up", "Down"} or len(tokens) != 2 or len(set(tokens)) != 2:
                raise ValueError("unexpected outcomes")
            return slug, cid, end, dict(zip(outcomes, tokens))
        return None

    def buy(self, slug, cid, outcome, limit, end):
        if not MIN_REM <= end - time.time() <= MAX_REM:
            return False
        # One entry per window, including restarts and ambiguous timeouts.
        if not self.journal.reserve(slug, "entry", Decimal(str(STAKE))):
            return False
        if self.dry_run:
            self.journal.finish(slug, "entry", "simulated", f"{outcome} limit={limit}")
            LOG.info("DRY RUN %s %s $%.2f limit=%s", slug, outcome, STAKE, limit)
            return True
        body = {"market_id": cid, "side": "BUY", "outcome": outcome,
                "amount": str(STAKE), "price": str(limit),
                "order_type": "market", "time_in_force": "FOK"}
        try:
            pace_simulator()
            if not MIN_REM <= end - time.time() <= MAX_REM:
                self.journal.finish(slug, "entry", "skipped", "entry window elapsed")
                return False
            response = self.http.post(API + "/v1/orders", json=body,
                                      headers={"X-API-Key": self.api_key,
                                               "Idempotency-Key": str(uuid.uuid4())})
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status", "")).lower()
            self.journal.finish(slug, "entry", status or "unknown", json.dumps(payload))
            if status in ("filled", "matched"):
                self.spent += STAKE
                LOG.info("PAPER FILLED %s %s price=%s shares=%s", slug, outcome,
                         payload.get("price"), payload.get("quantity"))
            elif status in ("rejected", "cancelled", "canceled", "expired"):
                LOG.warning("Paper order status=%s; attempt retained", status)
            else:
                raise RuntimeError("Unconfirmed paper order")
        except Exception as exc:
            rejected = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404, 422, 429)
            detail = {"exception": type(exc).__name__, "order": body}
            if isinstance(exc, httpx.HTTPStatusError):
                detail["http_status"] = exc.response.status_code
                # Keep the server's explanation, but never persist echoed credentials.
                message = exc.response.text.replace(self.api_key, "[REDACTED]")
                message = re.sub(r"ps_live_[A-Za-z0-9_-]+", "[REDACTED]", message)
                detail["response"] = message[:4000]
                detail["retry_after"] = exc.response.headers.get("Retry-After")
            self.journal.finish(slug, "entry", "rejected" if rejected else "unknown", json.dumps(detail))
            LOG.warning("Paper order %s %s %s: %s; no automatic resubmission",
                        "rejected" if rejected else "uncertain", slug, outcome, json.dumps(detail))
            if not rejected:
                raise RuntimeError("Check the simulator and journal before restarting") from None
        return True

    def cycle(self):
        balance = self.balance()  # Failure prevents orders in this cycle.
        LOG.info("PAPER balance=$%.2f session_spent=$%.2f", balance, self.spent)
        if balance < WALLET_FLOOR:
            LOG.warning("STOP balance below $%.2f floor", WALLET_FLOOR)
            return False
        for duration, seconds in (("15m", 900), ("5m", 300)):
            for asset in ASSETS:
                now = time.time()
                start = int(now) // seconds * seconds
                if not MIN_REM <= start + seconds - now <= MAX_REM:
                    continue
                slug = f"{asset}-updown-{duration}-{start}"
                if self.journal.seen(slug, "entry"):
                    continue
                try:
                    market = self.discover(asset, duration, start, seconds)
                    if market is None:
                        continue
                    slug, cid, end, tokens = market
                    spot, sigma = self.spot_and_sigma(asset)
                    strike = float(self.api_get(f"/prices/ptb/{cid}")["price"])
                    if not math.isfinite(strike) or strike <= 0:
                        raise ValueError("invalid strike")
                    for outcome, token in tokens.items():
                        data = self.api_get("/v1/book", token_id=token)
                        ask, tick = book_quote(data, time.time())
                        remaining = end - time.time()
                        if not MIN_REM <= remaining <= MAX_REM or sigma <= 0:
                            break
                        p_up = norm_cdf((spot - strike) / (spot * sigma * math.sqrt(remaining / 60)))
                        model = p_up if outcome == "Up" else 1 - p_up
                        if not ask or ask[1] <= 1 or ask[0] > MAX_PRICE or model - ask[0] < MIN_EDGE:
                            continue
                        # Enforce the declared maximum even with slippage.
                        cap = min(Decimal(str(MAX_PRICE)), Decimal(str(ask[0])) + Decimal("0.05"))
                        limit = (cap / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
                        if limit < Decimal(str(ask[0])) or STAKE > MAX_PER_WINDOW:
                            continue
                        enough, minimum, shares = meets_minimum_size(data, STAKE, limit)
                        if not enough:
                            LOG.info("SKIP %s %s: $%.2f covers %s shares at limit %s; minimum=%s shares",
                                     slug, outcome, STAKE, shares, limit, minimum)
                            continue
                        balance = self.balance()
                        if balance - STAKE - 0.05 < WALLET_FLOOR:
                            LOG.warning("Insufficient paper balance above the floor")
                            return False
                        LOG.info("EDGE %s %s model=%.3f blend=%.3f ask=%.3f", slug, outcome,
                                 model, ask[0] + (model - ask[0]) * BL, ask[0])
                        if self.buy(slug, cid, outcome, limit, end):
                            break
                except (httpx.HTTPError, ValueError, KeyError, StopIteration) as exc:
                    LOG.warning("Scan skipped %s (%s)", slug, type(exc).__name__)
        return True

    def close(self):
        self.http.close()
        self.journal.db.close()


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log signals without paper orders")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "edge_trader.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    trader = None
    try:
        trader = EdgeTrader(args.dry_run)
        LOG.info("EDGE TRADER %s stake=$%.2f; hold until resolution",
                 "DRY RUN" if args.dry_run else "POLYSIMULATOR PAPER", STAKE)
        while trader.cycle():
            if args.once:
                break
            time.sleep(4)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__, exc)
        return 1
    finally:
        if trader is not None:
            trader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
