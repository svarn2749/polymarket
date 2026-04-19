from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .backtest import backtest_directory


GRID = {
    "strategy": ["momentum", "reversion"],
    "lookback_hours": [24, 72],
    "entry_threshold": [0.10, 0.20],
    "exit_threshold": [None, 0.05, 0.10],  # None = legacy stateless
    "rebalance_every_hours": [1, 6],
}


def _summarize_config(
    data_dir: Path,
    params: dict,
    *,
    fee_bps: float,
    slippage_bps: float,
    spreads: dict[str, float] | None = None,
    train_frac: float | None = None,
    split: str = "full",
) -> dict | None:
    per_market = backtest_directory(
        data_dir,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spreads=spreads,
        train_frac=train_frac,
        split=split,
        **params,
    )
    if per_market.empty:
        return None
    return {
        **params,
        "n_markets": int(len(per_market)),
        "mean_sharpe": float(per_market["sharpe_annualized"].mean()),
        "median_sharpe": float(per_market["sharpe_annualized"].median()),
        "mean_pnl": float(per_market["total_pnl"].mean()),
        "median_pnl": float(per_market["total_pnl"].median()),
        "frac_profitable": float((per_market["total_pnl"] > 0).mean()),
        "mean_turnover": float(per_market["turnover"].mean()),
        "mean_fees_paid": float(per_market["total_fees"].mean()),
    }


def _iter_combos(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        # Respect hysteresis semantics: exit must not exceed entry.
        if params.get("exit_threshold") is not None:
            if params["exit_threshold"] > params["entry_threshold"]:
                continue
        combos.append(params)
    return combos


def run_sweep(
    data_dir: Path,
    *,
    slippage_bps: float,
    fee_bps: float,
    grid: dict[str, list] = GRID,
    workers: int | None = None,
) -> pd.DataFrame:
    combos = _iter_combos(grid)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []

    if workers <= 1:
        for i, params in enumerate(combos, start=1):
            print(f"  [{i}/{len(combos)}] {params}")
            row = _summarize_config(data_dir, params, fee_bps=fee_bps, slippage_bps=slippage_bps)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows)

    print(f"sweeping {len(combos)} combos with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_summarize_config, data_dir, p, fee_bps=fee_bps, slippage_bps=slippage_bps): p
            for p in combos
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  [{done}/{len(combos)}] FAILED {futures[fut]}: {exc}")
                continue
            if row is not None:
                rows.append(row)
            if done % 10 == 0 or done == len(combos):
                print(f"  [{done}/{len(combos)}] done")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep strategy params to find turnover/edge sweet spot.")
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default cpu_count-1)")
    parser.add_argument("--out", type=Path, default=Path("data/sweep.csv"))
    args = parser.parse_args()

    result = run_sweep(
        args.data,
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
        workers=args.workers,
    )
    if result.empty:
        print("no data")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(f"\nswept {len(result)} combos, wrote {args.out}")

    cols = [
        "strategy", "lookback_hours", "entry_threshold", "exit_threshold",
        "rebalance_every_hours", "mean_sharpe", "mean_pnl",
        "frac_profitable", "mean_turnover",
    ]
    print("\ntop 10 by mean Sharpe:")
    print(result.sort_values("mean_sharpe", ascending=False).head(10)[cols].to_string(index=False))

    print("\ntop 10 by median Sharpe (robust to outlier markets):")
    cols2 = [
        "strategy", "lookback_hours", "entry_threshold", "exit_threshold",
        "rebalance_every_hours", "median_sharpe", "median_pnl",
        "frac_profitable", "mean_turnover",
    ]
    print(result.sort_values("median_sharpe", ascending=False).head(10)[cols2].to_string(index=False))


if __name__ == "__main__":
    main()
