from __future__ import annotations

import asyncio
import math
import time
import traceback
from dataclasses import dataclass

import httpx
import pandas as pd

from ..sources import get_source
from ..sources.base import Market, MarketSource
from .config import PaperConfig
from .db import connect, init
from .fills import fill_price
from .ledger import (
    current_positions,
    record_equity,
    record_poll,
    record_trade,
)
from .universe import load_universe


@dataclass
class PollSummary:
    ts: int
    duration_ms: int
    n_markets: int
    n_signals: int
    n_trades: int
    n_errors: int


def _compute_signal(prices: pd.Series, lookback_hours: int) -> float | None:
    # Resample to 1h, require at least (lookback+1) bars.
    hourly = prices.resample("1h").last().ffill().dropna()
    if len(hourly) < lookback_hours + 1:
        return None
    now_price = float(hourly.iloc[-1])
    then_price = float(hourly.iloc[-1 - lookback_hours])
    if then_price <= 0:
        return None
    return now_price / then_price - 1.0


def _desired_size(
    *,
    signal: float,
    threshold: float,
    strategy: str,
    mid: float,
    position_size_usd: float,
) -> float:
    if abs(signal) <= threshold or mid <= 0:
        return 0.0
    direction = math.copysign(1.0, signal)
    if strategy == "reversion":
        direction = -direction
    return direction * position_size_usd / mid


def _evaluate_market(
    source: MarketSource,
    market: Market,
    *,
    config: PaperConfig,
    current_size: float,
    client: httpx.Client,
) -> tuple[float | None, float | None, float | None, tuple[str, float, float] | None]:
    """Evaluate one market. Returns (signal, mid, decided_direction, trade_tuple).

    trade_tuple is (side, size_delta, fill_price) if a trade should happen, else None.
    """
    df = source.get_price_history(
        market,
        lookback_days=max(2, math.ceil((config.lookback_hours + 4) / 24)),
        fidelity=60,
        client=client,
    )
    if df.empty or "price" not in df.columns:
        return None, None, None, None
    prices = df.set_index("ts")["price"]
    signal = _compute_signal(prices, config.lookback_hours)
    if signal is None:
        return None, None, None, None

    book = source.get_order_book(market, client=client)
    if book.mid is None or book.best_bid is None or book.best_ask is None:
        return signal, None, None, None

    desired_size = _desired_size(
        signal=signal,
        threshold=config.entry_threshold,
        strategy=config.strategy,
        mid=book.mid,
        position_size_usd=config.position_size_usd,
    )
    direction = 0.0 if desired_size == 0 else math.copysign(1.0, desired_size)
    delta = desired_size - current_size
    if abs(delta * book.mid) < config.min_trade_usd:
        return signal, book.mid, direction, None

    side = "buy" if delta > 0 else "sell"
    fp = fill_price(side=side, book=book, model=config.fill_model)
    if fp is None:
        return signal, book.mid, direction, None
    return signal, book.mid, direction, (side, delta, fp)


def poll_once(
    source: MarketSource,
    universe: list[Market],
    config: PaperConfig,
) -> PollSummary:
    started = time.time()
    ts = int(started)
    init(config.db_path)

    n_signals = 0
    n_trades = 0
    n_errors = 0

    limits = httpx.Limits(
        max_keepalive_connections=config.http_concurrency,
        max_connections=config.http_concurrency * 2,
    )

    with httpx.Client(timeout=20, limits=limits) as client, connect(config.db_path) as conn:
        positions = current_positions(conn)
        for market in universe:
            current_size = positions[market.id].size if market.id in positions else 0.0
            try:
                signal, mid, direction, trade = _evaluate_market(
                    source, market, config=config, current_size=current_size, client=client
                )
            except Exception:
                n_errors += 1
                traceback.print_exc()
                continue

            if signal is None:
                continue
            n_signals += 1

            if mid is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO snapshots (ts, market_id, bid, ask, mid) VALUES (?, ?, ?, ?, ?)",
                    (ts, market.id, None, None, mid),
                )
            conn.execute(
                "INSERT OR REPLACE INTO signals (ts, market_id, value, direction) VALUES (?, ?, ?, ?)",
                (ts, market.id, signal, direction if direction is not None else 0.0),
            )

            if trade is None:
                continue
            side, size_delta, fp = trade
            record_trade(
                conn,
                ts=ts,
                market_id=market.id,
                side=side,
                size=size_delta,
                price=fp,
                signal=signal,
                strategy=config.strategy,
                source=source.name,
            )
            n_trades += 1

        # Recompute positions after trades, then mark-to-market for equity.
        positions = current_positions(conn)
        realized = sum(p.realized_pnl for p in positions.values())
        unrealized = 0.0
        gross = 0.0
        for p in positions.values():
            snap_row = conn.execute(
                "SELECT mid FROM snapshots WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
                (p.market_id,),
            ).fetchone()
            if snap_row and snap_row["mid"] is not None:
                p.last_price = float(snap_row["mid"])
            unrealized += p.unrealized_pnl
            gross += p.notional

        record_equity(
            conn,
            ts=ts,
            realized=realized,
            unrealized=unrealized,
            gross_exposure=gross,
            n_positions=len(positions),
        )

        duration_ms = int((time.time() - started) * 1000)
        record_poll(
            conn,
            ts=ts,
            duration_ms=duration_ms,
            n_markets=len(universe),
            n_errors=n_errors,
            n_trades=n_trades,
            note=f"signals={n_signals}",
        )

    return PollSummary(
        ts=ts,
        duration_ms=int((time.time() - started) * 1000),
        n_markets=len(universe),
        n_signals=n_signals,
        n_trades=n_trades,
        n_errors=n_errors,
    )


async def poll_loop(config: PaperConfig, stop: asyncio.Event) -> None:
    if config.startup_delay_sec > 0:
        try:
            await asyncio.wait_for(stop.wait(), timeout=config.startup_delay_sec)
            return
        except asyncio.TimeoutError:
            pass

    source = get_source(config.source)
    while not stop.is_set():
        universe = load_universe(
            source=config.source,
            markets_csv=config.markets_csv,
            spreads_csv=config.spreads_csv,
            top_n=config.universe_top_n,
            max_spread_bps=config.max_spread_bps,
        )
        if not universe:
            print(
                f"[paper] empty universe — check {config.markets_csv} "
                f"(source={config.source}, max_spread_bps={config.max_spread_bps})"
            )
        else:
            try:
                summary = await asyncio.to_thread(poll_once, source, universe, config)
                print(
                    f"[paper] ts={summary.ts} markets={summary.n_markets} "
                    f"signals={summary.n_signals} trades={summary.n_trades} "
                    f"errors={summary.n_errors} took={summary.duration_ms}ms"
                )
            except Exception:
                print("[paper] poll_once crashed:")
                traceback.print_exc()

        try:
            await asyncio.wait_for(stop.wait(), timeout=config.poll_interval_sec)
            return
        except asyncio.TimeoutError:
            continue
