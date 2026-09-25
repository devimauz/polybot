"""Read-only account data and profit calculations for the desktop GUI."""

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def number(value):
    result = float(value or 0)
    if not math.isfinite(result):
        raise ValueError("Invalid numeric API value")
    return result


def journal_orders():
    path = ROOT / "live.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT market,outcome,day,amount,status,detail FROM orders ORDER BY rowid DESC").fetchall()
    orders = []
    for row in rows:
        if not row["market"].startswith("btc-updown-5m-"):
            continue
        status = row["status"]
        try:
            detail = json.loads(row["detail"])
        except (ValueError, TypeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        if status == "submitted":
            status = "matched" if detail.get("success") and detail.get("status") == "matched" else "review"
        orders.append({"slug": row["market"], "outcome": row["outcome"],
                       "timestamp": int(row["market"].rsplit("-", 1)[1]),
                       "amount": row["amount"], "status": status,
                       "order_id": detail.get("orderID", "")})
    return orders


def summarize(open_rows, closed_rows, orders):
    tracked = {(o["slug"], o["outcome"]) for o in orders}
    current_assets = {p.get("asset") for p in open_rows}
    positions = []
    for closed, rows in ((False, open_rows), (True, closed_rows)):
        seen = set()
        for p in rows:
            slug = p.get("slug", "")
            asset = p.get("asset")
            if not re.fullmatch(r"btc-updown-5m-\d+", slug):
                continue
            if asset in seen or (closed and asset in current_assets):
                continue
            seen.add(asset)
            realized = number(p.get("realizedPnl"))
            initial = number(p.get("initialValue"))
            current = number(p.get("currentValue"))
            unrealized = 0 if closed else current - initial
            redeemable = bool(p.get("redeemable")) and not closed
            state = "closed" if closed else "claimable" if redeemable else "open"
            stamp = int(p.get("timestamp") or (int(slug.rsplit("-", 1)[1]) + 300))
            positions.append({"slug": slug, "title": p.get("title", slug),
                              "outcome": p.get("outcome", ""), "state": state,
                              "shares": number(p.get("size") if not closed else p.get("totalBought")),
                              "avg_price": number(p.get("avgPrice")), "value": 0 if closed else current,
                              "realized": realized, "unrealized": unrealized,
                              "pnl": realized + unrealized, "timestamp": stamp,
                              "tracked": (slug, p.get("outcome")) in tracked})
    closed = [p for p in positions if p["state"] == "closed"]
    wins = sum(p["realized"] > 0 for p in closed)
    losses = sum(p["realized"] < 0 for p in closed)
    curve, running = [], 0
    for p in sorted(closed, key=lambda p: p["timestamp"]):
        running += p["realized"]
        curve.append({"time": p["timestamp"], "value": round(running, 6)})
    return {"positions": sorted(positions, key=lambda p: p["timestamp"], reverse=True),
            "orders": orders, "curve": curve,
            "metrics": {"realized": sum(p["realized"] for p in positions),
                        "closed_pnl": sum(p["realized"] for p in closed),
                        "unrealized": sum(p["unrealized"] for p in positions),
                        "position_value": sum(p["value"] for p in positions),
                        "claimable": sum(p["value"] for p in positions if p["state"] == "claimable"),
                        "wins": wins, "losses": losses, "closed_count": len(closed),
                        "matched": sum(o["status"] == "matched" for o in orders),
                        "review": sum(o["status"] in ("unknown", "reserved", "review") for o in orders)}}


class Dashboard:
    def __init__(self):
        load_dotenv(ROOT / ".env")
        self.wallet = os.environ.get("POLY_FUNDER", "")
        self.lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.stop = threading.Event()
        self.http = httpx.Client(timeout=12)
        self.clob = None
        self.data = {"loading": True, "errors": [], "updated_at": None}

    def pages(self, endpoint, limit):
        rows = []
        for page in range(20):
            params = {"user": self.wallet, "limit": limit, "offset": page * limit, "_": uuid.uuid4().hex}
            if endpoint == "positions":
                params["sizeThreshold"] = 0
            else:
                params.update(sortBy="TIMESTAMP", sortDirection="DESC")
            r = self.http.get("https://data-api.polymarket.com/" + endpoint, params=params)
            r.raise_for_status()
            batch = r.json()
            if not isinstance(batch, list):
                raise ValueError("Invalid position response")
            rows.extend(batch)
            if len(batch) < limit:
                return rows, False
        return rows, True

    def cash(self):
        if self.clob is None:
            from py_clob_client_v2 import ClobClient
            key = os.environ.get("POLY_PRIVATE_KEY")
            if not key:
                return None
            client = ClobClient("https://clob.polymarket.com", chain_id=137, key=key,
                                signature_type=int(os.environ.get("POLY_SIGNATURE_TYPE", "0")), funder=self.wallet)
            client.set_api_creds(client.derive_api_key())
            self.clob = client
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        result = self.clob.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return number(result["balance"]) / 1_000_000

    def refresh(self):
        if not self.refresh_lock.acquire(blocking=False):
            return
        try:
            errors = []
            if not re.fullmatch(r"0x[0-9a-fA-F]{40}", self.wallet):
                raise ValueError("Set POLY_FUNDER in the local .env file")
            opened, partial_open = self.pages("positions", 500)
            closed, partial_closed = self.pages("closed-positions", 50)
            data = summarize(opened, closed, journal_orders())
            tracked = {(o["slug"], o["outcome"]) for o in data["orders"]}
            data["bot_view"] = summarize(
                [p for p in opened if (p.get("slug"), p.get("outcome")) in tracked],
                [p for p in closed if (p.get("slug"), p.get("outcome")) in tracked], data["orders"])
            cash = None
            try:
                cash = self.cash()
                if cash is None:
                    errors.append("Cash unavailable: wallet signing credentials are not configured.")
            except Exception:
                errors.append("Cash balance unavailable. Position results are still shown.")
            if partial_open or partial_closed:
                errors.append("History limit reached. Totals cover the retrieved positions only.")
            data.update(loading=False, cash=cash, wallet=self.wallet,
                        updated_at=time.time(), errors=errors, partial=partial_open or partial_closed)
            with self.lock:
                self.data = data
        except Exception as exc:
            with self.lock:
                self.data = {**self.data, "loading": False,
                             "errors": ["Refresh failed (%s). Showing last available data." % type(exc).__name__]}
        finally:
            self.refresh_lock.release()

    def run(self):
        while not self.stop.is_set():
            self.refresh()
            self.stop.wait(15)
