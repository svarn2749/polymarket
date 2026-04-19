from __future__ import annotations

from ..sources.base import OrderBook


def fill_price(*, side: str, book: OrderBook, model: str) -> float | None:
    """Return a single fill price for a paper trade.

    Models:
      - "mid":          fill at mid (optimistic, matches backtest)
      - "realistic":    fill at ask (buy) / bid (sell) — cross the spread
      - "half_spread":  fill halfway between mid and the touch
    """
    bid, ask, mid = book.best_bid, book.best_ask, book.mid
    if bid is None or ask is None or mid is None:
        return None
    if side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    if model == "mid":
        return mid
    if model == "realistic":
        return ask if side == "buy" else bid
    if model == "half_spread":
        half = (ask - bid) / 4.0
        return mid + half if side == "buy" else mid - half
    raise ValueError(f"unknown fill model: {model!r}")
