from __future__ import annotations

from .base import (
    FeeModel,
    FlatFee,
    KalshiFee,
    Market,
    MarketSource,
    OrderBook,
    Position,
)
from .kalshi import KalshiSource
from .polymarket import PolymarketSource

__all__ = [
    "FeeModel",
    "FlatFee",
    "KalshiFee",
    "KalshiSource",
    "Market",
    "MarketSource",
    "OrderBook",
    "PolymarketSource",
    "Position",
    "get_source",
]


def get_source(name: str) -> MarketSource:
    name = name.lower()
    if name == "polymarket":
        return PolymarketSource()
    if name == "kalshi":
        return KalshiSource()
    raise ValueError(f"unknown source: {name!r} (expected 'polymarket' or 'kalshi')")
