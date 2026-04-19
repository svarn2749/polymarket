from __future__ import annotations

from dataclasses import dataclass

from .sources.base import FeeModel, FlatFee, OrderBook


@dataclass
class FillEstimate:
    side: str  # "buy" or "sell"
    target_notional: float
    filled_notional: float
    filled_size: float
    avg_price: float
    reference_price: float  # mid when available, else touch
    slippage_per_share: float  # avg_price - reference (buy) or reference - avg_price (sell)
    fee_per_share: float
    cost_per_share: float  # slippage + fee, always non-negative for a round-trip side
    fully_filled: bool


_ZERO_FEE = FlatFee(rate=0.0)


def estimate_fill(
    book: OrderBook,
    *,
    side: str,
    target_notional: float,
    fee_model: FeeModel | None = None,
) -> FillEstimate | None:
    """Walk the book to fill `target_notional` dollars on `side`.

    Returns None if the book has no levels on the needed side.
    Slippage is measured against mid when both sides exist, else against the touch.
    """
    if side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    fees = fee_model or _ZERO_FEE
    levels = book.asks if side == "buy" else book.bids
    if not levels:
        return None

    reference = book.mid if book.mid is not None else levels[0][0]

    remaining = target_notional
    filled_notional = 0.0
    filled_size = 0.0
    for price, size in levels:
        level_notional = price * size
        if remaining <= level_notional:
            take_notional = remaining
            take_size = take_notional / price
            filled_notional += take_notional
            filled_size += take_size
            remaining = 0.0
            break
        filled_notional += level_notional
        filled_size += size
        remaining -= level_notional

    if filled_size == 0:
        return None

    avg_price = filled_notional / filled_size
    if side == "buy":
        slippage = avg_price - reference
    else:
        slippage = reference - avg_price

    fee_per_share = fees.per_share(side=side, price=avg_price)
    return FillEstimate(
        side=side,
        target_notional=target_notional,
        filled_notional=filled_notional,
        filled_size=filled_size,
        avg_price=avg_price,
        reference_price=reference,
        slippage_per_share=slippage,
        fee_per_share=fee_per_share,
        cost_per_share=max(slippage, 0.0) + fee_per_share,
        fully_filled=remaining == 0.0,
    )


def round_trip_cost_per_share(
    book: OrderBook,
    *,
    side: str,
    target_notional: float,
    fee_model: FeeModel | None = None,
) -> float | None:
    """Total entry+exit cost per share for opening `side` at target notional
    and unwinding the same size at the opposite side of the book.
    """
    opposite = "sell" if side == "buy" else "buy"
    entry = estimate_fill(
        book, side=side, target_notional=target_notional, fee_model=fee_model
    )
    if entry is None:
        return None
    exit_notional = entry.filled_size * (book.mid or entry.reference_price)
    exit_fill = estimate_fill(
        book, side=opposite, target_notional=exit_notional, fee_model=fee_model
    )
    if exit_fill is None:
        return None
    return entry.cost_per_share + exit_fill.cost_per_share
