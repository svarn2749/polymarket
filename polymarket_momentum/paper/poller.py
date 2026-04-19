from __future__ import annotations

import asyncio
import math
import time
import traceback
from dataclasses import dataclass

import httpx
import pandas as pd

from ..sources import get_source
from ..sources.base import Market, MarketSource, OrderBook
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
    n_reversion_trades: int
    n_cross_sectional_trades: int
    n_errors: int

    @property
    def n_trades(self) -> int:
        return self.n_reversion_trades + self.n_cross_sectional_trades


@dataclass
class MarketEval:
    market: Market
    signal: float
    mid: float
    book: OrderBook


def _compute_signal(prices: pd.Series, lookback_hours: int) -> float | None:
    hourly = prices.resample("1h").last().ffill().dropna()
    if len(hourly) < lookback_hours + 1:
        return None
    now_price = float(hourly.iloc[-1])
    then_price = float(hourly.iloc[-1 - lookback_hours])
    if then_price <= 0:
        return None
    return now_price / then_price - 1.0


def _fetch_evals(
    source: MarketSource,
    universe: list[Market],
    *,
    config: PaperConfig,
    client: httpx.Client,
    conn,
    ts: int,
) -> tuple[list[MarketEval], int]:
    evals: list[MarketEval] = []
    errors = 0
    price_lookback_days = max(2, math.ceil((config.lookback_hours + 4) / 24))

    for market in universe:
        try:
            df = source.get_price_history(
                market,
                lookback_days=price_lookback_days,
                fidelity=60,
                client=client,
            )
        except Exception:
            errors += 1
            traceback.print_exc()
            continue
        if df.empty or "price" not in df.columns:
            continue
        prices = df.set_index("ts")["price"]
        signal = _compute_signal(prices, config.lookback_hours)
        if signal is None:
            continue

        try:
            book = source.get_order_book(market, client=client)
        except Exception:
            errors += 1
            continue
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            continue

        conn.execute(
            "INSERT OR REPLACE INTO snapshots (ts, market_id, bid, ask, mid) VALUES (?, ?, ?, ?, ?)",
            (ts, market.id, book.best_bid, book.best_ask, book.mid),
        )
        conn.execute(
            "INSERT OR REPLACE INTO signals (ts, market_id, value, direction) VALUES (?, ?, ?, ?)",
            (ts, market.id, signal, 0.0),
        )
        evals.append(MarketEval(market=market, signal=signal, mid=book.mid, book=book))
    return evals, errors


def _execute_trade(
    conn,
    *,
    ts: int,
    market_id: str,
    delta: float,
    signal: float,
    strategy: str,
    source_name: str,
    book: OrderBook,
    config: PaperConfig,
) -> bool:
    if abs(delta * (book.mid or 0)) < config.min_trade_usd:
        return False
    side = "buy" if delta > 0 else "sell"
    fp = fill_price(side=side, book=book, model=config.fill_model)
    if fp is None:
        return False
    record_trade(
        conn,
        ts=ts,
        market_id=market_id,
        side=side,
        size=delta,
        price=fp,
        signal=signal,
        strategy=strategy,
        source=source_name,
    )
    return True


def _reversion_desired_direction(
    *,
    current_size: float,
    signal: float,
    entry_threshold: float,
    exit_threshold: float,
) -> float:
    """Return the desired raw direction (-1, 0, +1) after applying reversion logic.

    Stateless when exit_threshold <= 0: position = -sign(signal) when |signal| > entry, else 0.
    With hysteresis: once in a position, hold until |signal| < exit_threshold OR signal
    crosses past the opposite entry band (flip).
    """
    # Treat reversion positions in "raw" (pre-flip) signal space so the hysteresis
    # logic mirrors sized_position_hysteresis exactly.
    raw_current = 0.0 if current_size == 0 else -math.copysign(1.0, current_size)

    use_hysteresis = exit_threshold > 0.0
    if not use_hysteresis:
        if abs(signal) > entry_threshold:
            raw_new = math.copysign(1.0, signal)
        else:
            raw_new = 0.0
    else:
        if raw_current == 0.0:
            if signal > entry_threshold:
                raw_new = 1.0
            elif signal < -entry_threshold:
                raw_new = -1.0
            else:
                raw_new = 0.0
        else:
            if raw_current > 0 and signal < -entry_threshold:
                raw_new = -1.0
            elif raw_current < 0 and signal > entry_threshold:
                raw_new = 1.0
            elif abs(signal) < exit_threshold:
                raw_new = 0.0
            else:
                raw_new = raw_current  # hold

    return -raw_new  # flip for reversion


def _apply_reversion(
    evals: list[MarketEval],
    *,
    config: PaperConfig,
    conn,
    ts: int,
    source_name: str,
) -> int:
    held = current_positions(conn, strategy="reversion")
    n_trades = 0

    # 1) Evaluate markets currently visible in the universe.
    for e in evals:
        current_size = held[e.market.id].size if e.market.id in held else 0.0
        direction = _reversion_desired_direction(
            current_size=current_size,
            signal=e.signal,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
        )
        desired = direction * config.position_size_usd / e.mid if direction != 0 else 0.0
        delta = desired - current_size
        if _execute_trade(
            conn, ts=ts, market_id=e.market.id, delta=delta, signal=e.signal,
            strategy="reversion", source_name=source_name, book=e.book, config=config,
        ):
            n_trades += 1

    # 2) Positions held but no longer in universe — can't close this cycle; leave for later.
    #    (Orderbook not available; would need a targeted fetch. TODO.)
    return n_trades


def _apply_cross_sectional(
    evals: list[MarketEval],
    *,
    config: PaperConfig,
    conn,
    ts: int,
    source_name: str,
) -> int:
    k = config.cross_sectional_top_k
    if len(evals) < 2 * k:
        return 0

    sorted_evals = sorted(evals, key=lambda e: e.signal)
    long_targets = sorted_evals[:k]         # biggest drops → bet on bounce
    short_targets = sorted_evals[-k:]       # biggest rallies → bet on fade
    eval_by_id = {e.market.id: e for e in evals}

    target_sizes: dict[str, float] = {}
    for e in long_targets:
        target_sizes[e.market.id] = +config.position_size_usd / e.mid
    for e in short_targets:
        target_sizes[e.market.id] = -config.position_size_usd / e.mid

    held = current_positions(conn, strategy="cross_sectional")
    for market_id in held:
        target_sizes.setdefault(market_id, 0.0)  # close positions that rolled off the leaderboard

    n_trades = 0
    for market_id, target in target_sizes.items():
        current_size = held[market_id].size if market_id in held else 0.0
        delta = target - current_size
        e = eval_by_id.get(market_id)
        if e is None:
            # Market no longer visible this cycle — skip close for now.
            continue
        if _execute_trade(
            conn, ts=ts, market_id=market_id, delta=delta, signal=e.signal,
            strategy="cross_sectional", source_name=source_name, book=e.book, config=config,
        ):
            n_trades += 1
    return n_trades


def _mark_to_market(conn, *, ts: int, strategy: str) -> None:
    positions = current_positions(conn, strategy=strategy)
    realized = sum(p.realized_pnl for p in positions.values())
    unrealized = 0.0
    gross = 0.0
    for p in positions.values():
        snap = conn.execute(
            "SELECT mid FROM snapshots WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
            (p.market_id,),
        ).fetchone()
        if snap and snap["mid"] is not None:
            p.last_price = float(snap["mid"])
        unrealized += p.unrealized_pnl
        gross += p.notional
    record_equity(
        conn,
        ts=ts,
        strategy=strategy,
        realized=realized,
        unrealized=unrealized,
        gross_exposure=gross,
        n_positions=len(positions),
    )


def poll_once(
    source: MarketSource,
    universe: list[Market],
    config: PaperConfig,
) -> PollSummary:
    started = time.time()
    ts = int(started)
    init(config.db_path)

    limits = httpx.Limits(
        max_keepalive_connections=config.http_concurrency,
        max_connections=config.http_concurrency * 2,
    )

    with httpx.Client(timeout=20, limits=limits) as client, connect(config.db_path) as conn:
        evals, errors = _fetch_evals(
            source, universe, config=config, client=client, conn=conn, ts=ts
        )
        reversion_trades = _apply_reversion(
            evals, config=config, conn=conn, ts=ts, source_name=source.name
        )
        cs_trades = 0
        if config.cross_sectional_enabled:
            cs_trades = _apply_cross_sectional(
                evals, config=config, conn=conn, ts=ts, source_name=source.name
            )

        _mark_to_market(conn, ts=ts, strategy="reversion")
        _mark_to_market(conn, ts=ts, strategy="cross_sectional")

        duration_ms = int((time.time() - started) * 1000)
        record_poll(
            conn,
            ts=ts,
            duration_ms=duration_ms,
            n_markets=len(universe),
            n_errors=errors,
            n_trades=reversion_trades + cs_trades,
            note=f"signals={len(evals)} rev={reversion_trades} cs={cs_trades}",
        )

    return PollSummary(
        ts=ts,
        duration_ms=int((time.time() - started) * 1000),
        n_markets=len(universe),
        n_signals=len(evals),
        n_reversion_trades=reversion_trades,
        n_cross_sectional_trades=cs_trades,
        n_errors=errors,
    )


def _age_seconds(path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _refresh_markets(source: MarketSource, config: PaperConfig) -> bool:
    print(f"[paper] refreshing markets.csv from {source.name}")
    try:
        markets = source.list_markets(
            min_volume=config.min_volume,
            limit=max(100, config.universe_top_n * 3),
        )
    except Exception as exc:
        print(f"[paper] markets refresh failed: {exc}")
        return False
    if not markets:
        print("[paper] list_markets returned 0 markets — API issue?")
        return False
    source.write_market_metadata(markets, config.markets_csv)
    print(f"[paper] wrote {len(markets)} markets to {config.markets_csv}")
    return True


def _refresh_spreads(config: PaperConfig) -> bool:
    from ..snapshot_spreads import snapshot_spreads

    if not config.markets_csv.exists():
        return False
    print(f"[paper] refreshing spreads.csv")
    try:
        snapshot_spreads(
            config.markets_csv,
            config.spreads_csv,
            source_name=config.source,
            concurrency=config.http_concurrency,
        )
        return True
    except Exception as exc:
        print(f"[paper] spreads refresh failed: {exc}")
        return False


def _maybe_refresh_metadata(source: MarketSource, config: PaperConfig) -> None:
    max_age = config.refresh_interval_days * 86_400

    markets_age = _age_seconds(config.markets_csv)
    if markets_age is None or markets_age > max_age:
        if _refresh_markets(source, config):
            _refresh_spreads(config)
            return

    spreads_age = _age_seconds(config.spreads_csv)
    if spreads_age is None or spreads_age > max_age:
        _refresh_spreads(config)


async def poll_loop(config: PaperConfig, stop: asyncio.Event) -> None:
    if config.startup_delay_sec > 0:
        try:
            await asyncio.wait_for(stop.wait(), timeout=config.startup_delay_sec)
            return
        except asyncio.TimeoutError:
            pass

    source = get_source(config.source)
    while not stop.is_set():
        await asyncio.to_thread(_maybe_refresh_metadata, source, config)
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
                    f"signals={summary.n_signals} "
                    f"rev={summary.n_reversion_trades} cs={summary.n_cross_sectional_trades} "
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
