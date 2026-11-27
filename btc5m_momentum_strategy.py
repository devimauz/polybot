"""Shared signal and execution model for the BTC 5-minute momentum strategy."""

from __future__ import annotations

import math

FEE_RATE = 0.07
DEFAULT_STAKE = 2.0
DEFAULT_ENTRY_SECOND = 150.0
DEFAULT_MINIMUM_MOVE_BPS = 2.0
DEFAULT_ASK_MIN = 0.45
DEFAULT_ASK_MAX = 0.82
DEFAULT_FOK_ALLOWANCE = 0.05


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fee_for(price, shares):
    """Return the current Polymarket crypto taker fee in USDC."""
    return round(shares * FEE_RATE * price * (1 - price), 5)


def sweep_asks(book, stake):
    """Spend exactly stake across recorded asks; return None if depth is insufficient."""
    stake = finite(stake)
    if stake is None or stake <= 0:
        raise ValueError("stake must be positive and finite")
    levels = book.get("asks") or []
    if not levels and book.get("ask") is not None and book.get("ask_size") is not None:
        levels = [{"price": book["ask"], "size": book["ask_size"]}]
    clean = []
    for level in levels:
        price, size = finite(level.get("price")), finite(level.get("size"))
        if price is not None and size is not None and 0 < price < 1 and size > 0:
            clean.append((price, size))
    remaining, shares, fees, worst = stake, 0.0, 0.0, None
    for price, size in sorted(clean):
        dollars = min(remaining, price * size)
        quantity = dollars / price
        shares += quantity
        fees += fee_for(price, quantity)
        remaining -= dollars
        worst = price
        if remaining <= 1e-8:
            return {"stake": stake, "shares": shares, "fee": fees,
                    "cost": stake + fees, "avg_price": stake / shares,
                    "worst_price": worst}
    return None


def momentum_signal(open_price, current_price, books, stake=DEFAULT_STAKE,
                    minimum_move_bps=DEFAULT_MINIMUM_MOVE_BPS,
                    ask_min=DEFAULT_ASK_MIN, ask_max=DEFAULT_ASK_MAX):
    open_price, current_price = finite(open_price), finite(current_price)
    if open_price is None or current_price is None or open_price <= 0:
        return None
    move_bps = (current_price / open_price - 1) * 10_000
    if abs(move_bps) < minimum_move_bps:
        return None
    side = "Up" if move_bps > 0 else "Down"
    book = books.get(side) or {}
    best_ask = finite(book.get("ask"))
    if best_ask is None or not ask_min <= best_ask <= ask_max:
        return None
    fill = sweep_asks(book, stake)
    if fill is None:
        return None
    return {"side": side, "move_bps": move_bps, "best_ask": best_ask, **fill}


def momentum_guard(side, current_coinbase, prior_coinbase, current_bid, prior_bid,
                   minimum_recent_bps=-2.0, minimum_bid_change=-0.05):
    """Confirm that neither Coinbase nor the selected token is reversing.

    All inputs must come from at or before the decision tick.  ``prior_coinbase``
    is the 10-second reference and ``prior_bid`` is the 20-second reference for
    the strategy's default settings.  None is returned when confirmation data
    is incomplete so a guarded live bot can fail closed.
    """
    if side not in ("Up", "Down"):
        raise ValueError("side must be Up or Down")
    values = [finite(value) for value in
              (current_coinbase, prior_coinbase, current_bid, prior_bid)]
    if any(value is None for value in values) or values[1] <= 0:
        return None
    current_coinbase, prior_coinbase, current_bid, prior_bid = values
    direction = 1 if side == "Up" else -1
    signed_recent_bps = direction * (current_coinbase / prior_coinbase - 1) * 10_000
    bid_change = current_bid - prior_bid
    return {
        "passed": (signed_recent_bps >= minimum_recent_bps
                   and bid_change >= minimum_bid_change),
        "signed_recent_bps": signed_recent_bps,
        "bid_change": bid_change,
        "minimum_recent_bps": minimum_recent_bps,
        "minimum_bid_change": minimum_bid_change,
    }

