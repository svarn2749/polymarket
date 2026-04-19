from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from .backtest import backtest_directory


GRID = {
    "strategy": ["momentum", "reversion"],
    "lookback_hours": [12, 24, 72],
    "entry_threshold": [0.05, 0.10, 0.20],
    "rebalance_every_hours": [1, 6, 24],
}


def run_sweep(
    data_dir: Path,
    *,
    slippage_bps: float,
    fee_bps: float,
    grid: dict[str, list] = GRID,
) -> pd.DataFrame:
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows: list[dict] = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        print(f"  [{i}/{len(combos)}] {params}")
        per_market = backtest_directory(
            data_dir,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            **params,
        )
        if per_market.empty:
            continue
        rows.append(
            {
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
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep strategy params to find turnover/edge sweet spot.")
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument("--out", type=Path, default=Path("data/sweep.csv"))
    args = parser.parse_args()

    result = run_sweep(
        args.data,
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
    )
    if result.empty:
        print("no data")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\nswept {len(result)} combos, wrote {args.out}")

    print("\ntop 10 by mean Sharpe:")
    cols = ["strategy", "lookback_hours", "entry_threshold", "rebalance_every_hours",
            "mean_sharpe", "mean_pnl", "frac_profitable", "mean_turnover"]
    print(result.sort_values("mean_sharpe", ascending=False).head(10)[cols].to_string(index=False))

    print("\ntop 10 by median Sharpe (robust to outlier markets):")
    cols2 = ["strategy", "lookback_hours", "entry_threshold", "rebalance_every_hours",
             "median_sharpe", "median_pnl", "frac_profitable", "mean_turnover"]
    print(result.sort_values("median_sharpe", ascending=False).head(10)[cols2].to_string(index=False))


if __name__ == "__main__":
    main()
