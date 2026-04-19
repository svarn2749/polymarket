from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import momentum_signal, sized_position


@dataclass
class BacktestResult:
    bars: pd.DataFrame
    stats: dict


def backtest_market(
    prices: pd.Series,
    *,
    lookback_hours: int = 24,
    entry_threshold: float = 0.05,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_market_spread_bps: float | None = None,
    resample_rule: str = "1h",
    strategy: str = "momentum",
    rebalance_every_hours: int = 1,
) -> BacktestResult:
    if strategy not in {"momentum", "reversion"}:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if rebalance_every_hours < 1:
        raise ValueError("rebalance_every_hours must be >= 1")

    prices = prices.sort_index()
    bars = prices.resample(resample_rule).last().ffill().dropna()

    signal = momentum_signal(bars, lookback_hours)
    position = sized_position(signal, entry_threshold=entry_threshold)
    if strategy == "reversion":
        position = -position

    if rebalance_every_hours > 1:
        mask = np.arange(len(position)) % rebalance_every_hours == 0
        position = position.where(mask).ffill().fillna(0.0)

    # PnL: position held over the bar, realized at next bar's price change.
    # Shift position to avoid look-ahead (signal at t trades at t, earns t->t+1 move).
    price_change = bars.diff().fillna(0.0)
    gross_pnl = position.shift(1).fillna(0.0) * price_change

    turnover = position.diff().abs().fillna(0.0)
    # Precedence: per_market_spread_bps (when provided) replaces slippage_bps.
    effective_slip = slippage_bps if per_market_spread_bps is None else per_market_spread_bps
    cost_bps = fee_bps + effective_slip
    fee_cost = turnover * bars * (cost_bps / 10_000.0)
    net_pnl = gross_pnl - fee_cost
    equity = net_pnl.cumsum()

    frame = pd.DataFrame(
        {
            "price": bars,
            "signal": signal,
            "position": position,
            "gross_pnl": gross_pnl,
            "fee": fee_cost,
            "pnl": net_pnl,
            "equity": equity,
        }
    )

    stats = _summarize(frame, resample_rule=resample_rule)
    return BacktestResult(bars=frame, stats=stats)


def _summarize(frame: pd.DataFrame, *, resample_rule: str) -> dict:
    pnl = frame["pnl"]
    bars_per_year = _bars_per_year(resample_rule)
    sharpe = (
        pnl.mean() / pnl.std() * np.sqrt(bars_per_year)
        if pnl.std() > 0
        else 0.0
    )
    equity = frame["equity"]
    drawdown = (equity - equity.cummax()).min()
    entries = (frame["position"].diff().fillna(frame["position"]) != 0) & (
        frame["position"] != 0
    )
    turnover = float(frame["position"].diff().abs().sum())
    return {
        "n_bars": int(len(frame)),
        "n_entries": int(entries.sum()),
        "turnover": turnover,
        "total_pnl": float(pnl.sum()),
        "total_fees": float(frame["fee"].sum()),
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "sharpe_annualized": float(sharpe),
        "max_drawdown": float(drawdown),
        "exposure": float((frame["position"] != 0).mean()),
    }


def _bars_per_year(resample_rule: str) -> float:
    unit = pd.tseries.frequencies.to_offset(resample_rule)
    seconds = pd.Timedelta(unit).total_seconds()
    return 365 * 86_400 / seconds


def backtest_directory(
    data_dir: Path,
    *,
    train_frac: float | None = None,
    split: str = "full",
    spreads: dict[str, float] | None = None,
    **kwargs,
) -> pd.DataFrame:
    if split not in {"full", "train", "test"}:
        raise ValueError(f"split must be full|train|test, got {split!r}")

    rows: list[dict] = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["ts"])
        if len(df) < 96:
            continue
        if train_frac is not None and split != "full":
            cut = int(len(df) * train_frac)
            df = df.iloc[:cut] if split == "train" else df.iloc[cut:]
        if len(df) < 48:
            continue
        prices = df.set_index("ts")["price"]
        market_kwargs = dict(kwargs)
        if spreads is not None and path.stem in spreads:
            market_kwargs["per_market_spread_bps"] = float(spreads[path.stem])
        result = backtest_market(prices, **market_kwargs)
        rows.append({"market_id": path.stem, **result.stats})
    return pd.DataFrame(rows)
