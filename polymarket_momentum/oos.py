from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .backtest import backtest_directory
from .sweep import GRID, _iter_combos


def load_spreads(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, dtype={"market_id": str})
    return dict(zip(df["market_id"].astype(str), df["spread_bps"].astype(float)))


def _summarize(per_market: pd.DataFrame) -> dict:
    return {
        "n_markets": int(len(per_market)),
        "mean_sharpe": float(per_market["sharpe_annualized"].mean()),
        "median_sharpe": float(per_market["sharpe_annualized"].median()),
        "mean_pnl": float(per_market["total_pnl"].mean()),
        "frac_profitable": float((per_market["total_pnl"] > 0).mean()),
        "mean_turnover": float(per_market["turnover"].mean()),
    }


def _oos_one(
    data_dir: Path,
    params: dict,
    *,
    train_frac: float,
    fee_bps: float,
    slippage_bps: float,
    spreads: dict[str, float] | None,
) -> dict | None:
    train = backtest_directory(
        data_dir, train_frac=train_frac, split="train",
        fee_bps=fee_bps, slippage_bps=slippage_bps, spreads=spreads, **params,
    )
    test = backtest_directory(
        data_dir, train_frac=train_frac, split="test",
        fee_bps=fee_bps, slippage_bps=slippage_bps, spreads=spreads, **params,
    )
    if train.empty or test.empty:
        return None
    return {
        **params,
        **{f"train_{k}": v for k, v in _summarize(train).items()},
        **{f"test_{k}": v for k, v in _summarize(test).items()},
    }


def run_oos(
    data_dir: Path,
    *,
    train_frac: float,
    fee_bps: float,
    slippage_bps: float,
    grid: dict[str, list] = GRID,
    spreads: dict[str, float] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    combos = _iter_combos(grid)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []

    if workers <= 1:
        for i, params in enumerate(combos, start=1):
            print(f"  [{i}/{len(combos)}] {params}")
            row = _oos_one(data_dir, params, train_frac=train_frac,
                           fee_bps=fee_bps, slippage_bps=slippage_bps, spreads=spreads)
            if row is not None:
                rows.append(row)
        return pd.DataFrame(rows)

    print(f"oos sweep of {len(combos)} combos with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _oos_one, data_dir, p,
                train_frac=train_frac, fee_bps=fee_bps,
                slippage_bps=slippage_bps, spreads=spreads,
            ): p for p in combos
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
    parser = argparse.ArgumentParser(description="Out-of-sample validation via time split.")
    parser.add_argument("--data", type=Path, default=Path("data/prices"))
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=200.0)
    parser.add_argument(
        "--spreads-csv",
        type=Path,
        default=None,
        help="CSV with market_id,spread_bps; per-market overrides for --slippage-bps",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/oos.csv"))
    args = parser.parse_args()

    spreads = load_spreads(args.spreads_csv) if args.spreads_csv else None

    result = run_oos(
        args.data,
        train_frac=args.train_frac,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spreads=spreads,
        workers=args.workers,
    )
    if result.empty:
        print("no data")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(f"\nswept {len(result)} combos ({args.train_frac:.0%} train / {1 - args.train_frac:.0%} test), wrote {args.out}")

    corr = result[["train_mean_sharpe", "test_mean_sharpe"]].corr().iloc[0, 1]
    rank_corr = result[["train_mean_sharpe", "test_mean_sharpe"]].corr(method="spearman").iloc[0, 1]
    print(f"\ntrain/test mean-Sharpe correlation: pearson={corr:.3f} spearman={rank_corr:.3f}")
    print("  >0.5 = signal probably real, 0-0.5 = weak, <0 = overfit")

    cols = [
        "strategy", "lookback_hours", "entry_threshold", "exit_threshold",
        "rebalance_every_hours",
        "train_mean_sharpe", "test_mean_sharpe",
        "train_mean_pnl", "test_mean_pnl",
        "train_frac_profitable", "test_frac_profitable",
    ]
    print("\ntop 10 train configs — how they hold up on test:")
    top_train = result.sort_values("train_mean_sharpe", ascending=False).head(10)
    print(top_train[cols].to_string(index=False))

    best_train = result.sort_values("train_mean_sharpe", ascending=False).iloc[0]
    best_test = result.sort_values("test_mean_sharpe", ascending=False).iloc[0]
    print("\nbest-on-train config:")
    print(f"  params: {best_train[list(GRID.keys())].to_dict()}")
    print(f"  train Sharpe {best_train['train_mean_sharpe']:.2f} -> test Sharpe {best_train['test_mean_sharpe']:.2f}")
    print(f"\nbest-on-test config: {best_test[list(GRID.keys())].to_dict()}")
    print(f"  test Sharpe {best_test['test_mean_sharpe']:.2f} (train was {best_test['train_mean_sharpe']:.2f})")


if __name__ == "__main__":
    main()
