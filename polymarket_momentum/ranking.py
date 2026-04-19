from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from .cost_model import round_trip_cost_per_share
from .sources.base import Market, MarketSource, OrderBook
from .strategy import momentum_signal


@dataclass
class RankedBet:
    source: str
    market_id: str
    title: str
    slug: str
    side: str  # "buy" (Yes) or "sell" (Yes, equivalent to buying No)
    token_id: str
    ref_price: float
    signal: float
    edge_per_share: float
    cost_per_share: float
    net_edge_per_share: float
    best_bid: float | None
    best_ask: float | None
    spread: float | None


def _latest_signal(path: Path, *, lookback_hours: int, resample_rule: str) -> float | None:
    df = pd.read_csv(path, parse_dates=["ts"])
    if len(df) < lookback_hours + 1:
        return None
    prices = df.set_index("ts")["price"].sort_index()
    bars = prices.resample(resample_rule).last().ffill().dropna()
    if len(bars) < lookback_hours + 1:
        return None
    sig = momentum_signal(bars, lookback_hours).iloc[-1]
    if pd.isna(sig):
        return None
    return float(sig)


def rank_bets(
    source: MarketSource,
    *,
    prices_dir: Path,
    markets_csv: Path,
    target_notional: float = 100.0,
    lookback_hours: int = 24,
    resample_rule: str = "1h",
    min_abs_signal: float = 0.02,
    max_markets: int = 50,
    max_workers: int = 8,
) -> list[RankedBet]:
    """Rank markets by expected edge net of transaction costs.

    - Signal: latest momentum over `lookback_hours` on hourly bars.
    - Edge per share: signal * reference_price (cents-per-share proxy).
    - Cost per share: round-trip slippage walking the book for `target_notional`,
      plus `source.fee_model` per-share fee.
    """
    markets = pd.read_csv(markets_csv)
    markets["id"] = markets["id"].astype(str)
    markets = markets.drop_duplicates(subset=["id"], keep="last")

    candidates: list[tuple[Market, float]] = []
    for _, row in markets.iterrows():
        path = prices_dir / f"{row['id']}.csv"
        if not path.exists():
            continue
        sig = _latest_signal(path, lookback_hours=lookback_hours, resample_rule=resample_rule)
        if sig is None or abs(sig) < min_abs_signal:
            continue
        candidates.append(
            (
                Market(
                    source=row.get("source", source.name),
                    id=str(row["id"]),
                    question=row.get("question", ""),
                    slug=row.get("slug", ""),
                    yes_id=str(row["yes_id"]),
                    no_id=str(row["no_id"]),
                    volume=float(row.get("volume") or 0),
                    end_date=row.get("end_date"),
                ),
                sig,
            )
        )

    candidates.sort(key=lambda x: -abs(x[1]))
    candidates = candidates[:max_markets]
    if not candidates:
        return []

    with httpx.Client(timeout=15) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            books = list(
                pool.map(
                    lambda m: _safe_book(source, m, client),
                    [m for m, _ in candidates],
                )
            )

    results: list[RankedBet] = []
    for (market, sig), book in zip(candidates, books):
        if book is None or book.mid is None:
            continue
        side = "buy" if sig > 0 else "sell"
        ref = book.mid
        edge = sig * ref
        cost = round_trip_cost_per_share(
            book,
            side=side,
            target_notional=target_notional,
            fee_model=source.fee_model,
        )
        if cost is None:
            continue
        net = abs(edge) - cost
        results.append(
            RankedBet(
                source=market.source,
                market_id=market.id,
                title=market.question,
                slug=market.slug,
                side=side,
                token_id=market.yes_id,
                ref_price=ref,
                signal=sig,
                edge_per_share=abs(edge),
                cost_per_share=cost,
                net_edge_per_share=net,
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                spread=(book.best_ask - book.best_bid)
                if (book.best_bid is not None and book.best_ask is not None)
                else None,
            )
        )

    results.sort(key=lambda r: -r.net_edge_per_share)
    return results


def _safe_book(
    source: MarketSource, market: Market, client: httpx.Client
) -> OrderBook | None:
    try:
        return source.get_order_book(market, client=client)
    except httpx.HTTPError:
        return None
