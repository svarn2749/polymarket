from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, _bars_per_year
from .strategy import momentum_signal


def load_panel(data_dir: Path, resample: str = "1h") -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["ts"])
        if df.empty:
            continue
        s = (
            df.set_index("ts")["price"]
            .sort_index()
            .resample(resample)
            .last()
        )
        # ffill only within lifespan: between first and last non-NaN observation.
        first = s.first_valid_index()
        last = s.last_valid_index()
        if first is None or last is None:
            continue
        alive = s.loc[first:last].ffill()
        s.loc[first:last] = alive
        series[path.stem] = s

    if not series:
        return pd.DataFrame()

    panel = pd.concat(series, axis=1).sort_index()
    return panel


def backtest_cross_sectional(
    panel: pd.DataFrame,
    *,
    lookback_hours: int,
    top_k: int,
    strategy: str,
    fee_bps: float = 0.0,
    slippage_bps: float = 200.0,
) -> BacktestResult:
    if strategy not in {"momentum", "reversion"}:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if panel.empty:
        raise ValueError("empty panel")

    panel = panel.sort_index()
    # Infer resample rule from index spacing for annualization.
    freq = pd.infer_freq(panel.index) or "1h"

    # Per-market trailing return.
    signal = panel.apply(lambda col: momentum_signal(col, lookback_hours))

    # Row-wise rank; markets with NaN signal are excluded from the row's ranking.
    valid_count = signal.notna().sum(axis=1)
    ranks = signal.rank(axis=1, method="first", na_option="keep")

    # Number of markets per row (for the long leg boundary).
    n_per_row = valid_count.astype(float)
    # Positions: +1 if rank in top-k, -1 if rank in bottom-k, else 0.
    long_mask = ranks > (n_per_row.values[:, None] - top_k)
    short_mask = ranks <= top_k
    long_mask = long_mask & signal.notna()
    short_mask = short_mask & signal.notna()

    # Rows where we don't have enough markets: NaN everything (skip).
    enough = valid_count >= (2 * top_k)

    weight = 0.5 / top_k
    positions = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    positions = positions.where(~long_mask, weight)
    positions = positions.where(~short_mask, -weight)
    if strategy == "reversion":
        positions = -positions
    # Invalidate rows with insufficient breadth.
    positions.loc[~enough, :] = np.nan

    # PnL: position(t-1) * (price(t) - price(t-1)), summed across markets.
    price_change = panel.diff()
    shifted_pos = positions.shift(1)
    gross_per_market = shifted_pos * price_change
    gross_pnl = gross_per_market.sum(axis=1, min_count=1)

    # Costs on per-market turnover, charged in bps of that market's price.
    # Treat NaN (no-data) rows as flat: fill positions with 0 for turnover calc.
    pos_for_turn = positions.fillna(0.0)
    turnover = pos_for_turn.diff().abs()
    cost_bps = fee_bps + slippage_bps
    fee_per_market = turnover * panel * (cost_bps / 10_000.0)
    fee = fee_per_market.sum(axis=1, min_count=1).fillna(0.0)

    net_pnl = gross_pnl.fillna(0.0) - fee
    equity = net_pnl.cumsum()

    # Row-level summaries for the BacktestResult.bars frame.
    n_long = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)
    gross_exposure = pos_for_turn.abs().sum(axis=1)
    net_exposure = pos_for_turn.sum(axis=1)

    frame = pd.DataFrame(
        {
            "n_valid": valid_count.astype(int),
            "n_long": n_long.astype(int),
            "n_short": n_short.astype(int),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "gross_pnl": gross_pnl.fillna(0.0),
            "fee": fee,
            "pnl": net_pnl,
            "equity": equity,
        }
    )

    stats = _summarize_cs(frame, positions=pos_for_turn, resample_rule=freq)
    return BacktestResult(bars=frame, stats=stats)


def _summarize_cs(
    frame: pd.DataFrame,
    *,
    positions: pd.DataFrame,
    resample_rule: str,
) -> dict:
    pnl = frame["pnl"]
    bars_per_year = _bars_per_year(resample_rule)
    sharpe = (
        pnl.mean() / pnl.std() * np.sqrt(bars_per_year)
        if pnl.std() > 0
        else 0.0
    )
    equity = frame["equity"]
    drawdown = float((equity - equity.cummax()).min())
    turnover_total = float(positions.diff().abs().sum().sum())
    active_mask = frame["n_long"] > 0
    return {
        "n_bars": int(len(frame)),
        "n_active_bars": int(active_mask.sum()),
        "turnover": turnover_total,
        "total_pnl": float(pnl.sum()),
        "total_fees": float(frame["fee"].sum()),
        "hit_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "frac_positive_bars": float((pnl[active_mask] > 0).mean())
        if active_mask.any()
        else 0.0,
        "sharpe_annualized": float(sharpe),
        "max_drawdown": drawdown,
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "mean_net_exposure": float(frame["net_exposure"].mean()),
    }


def sweep_cross_sectional(
    panel: pd.DataFrame,
    *,
    lookbacks: list[int],
    top_ks: list[int],
    strategies: list[str],
    fee_bps: float = 0.0,
    slippage_bps: float = 200.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in strategies:
        for lookback in lookbacks:
            for top_k in top_ks:
                res = backtest_cross_sectional(
                    panel,
                    lookback_hours=lookback,
                    top_k=top_k,
                    strategy=strategy,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "lookback": lookback,
                        "top_k": top_k,
                        "total_pnl": res.stats["total_pnl"],
                        "sharpe": res.stats["sharpe_annualized"],
                        "max_drawdown": res.stats["max_drawdown"],
                        "frac_positive_bars": res.stats["frac_positive_bars"],
                        "turnover": res.stats["turnover"],
                        "n_bars": res.stats["n_bars"],
                    }
                )
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return out.reset_index(drop=True)
