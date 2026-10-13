"""Shared signal and execution model for the BTC 5-minute momentum strategy."""

from __future__ import annotations

import math

FEE_RATE = 0.05


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fee_for(price, shares):
    return FEE_RATE * min(price, 1 - price) * shares


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


def momentum_signal(open_price, current_price, books, stake=100.0,
                    minimum_move_bps=2.0, ask_min=0.45, ask_max=0.90):
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

