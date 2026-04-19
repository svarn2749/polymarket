from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

from .paper.config import _default_markets_csv, _default_spreads_csv
from .sources import get_source
from .sources.base import Market


def _row_to_market(row: dict, default_source: str) -> Market:
    return Market(
        source=str(row.get("source") or default_source),
        id=str(row["id"]),
        question=str(row.get("question") or ""),
        slug=str(row.get("slug") or ""),
        # Tolerate both the new sources schema (`yes_id`) and the legacy one (`yes_token_id`).
        yes_id=str(row.get("yes_id") or row.get("yes_token_id") or ""),
        no_id=str(row.get("no_id") or row.get("no_token_id") or ""),
        volume=float(row.get("volume") or 0),
        end_date=row.get("end_date"),
        closed=bool(row.get("closed") or False),
    )


def snapshot_spreads(
    markets_csv: Path,
    out_csv: Path,
    *,
    source_name: str = "polymarket",
    concurrency: int = 8,
) -> pd.DataFrame:
    markets_df = pd.read_csv(markets_csv, dtype=str)
    source = get_source(source_name)

    limits = httpx.Limits(
        max_keepalive_connections=concurrency, max_connections=concurrency * 2
    )
    rows: list[dict] = []
    skipped = 0
    errors = 0

    with httpx.Client(timeout=15, limits=limits) as client:
        def fetch_one(row_dict: dict) -> tuple[str, dict | None, bool]:
            market = _row_to_market(row_dict, source_name)
            if not market.yes_id:
                return market.id, None, False
            try:
                book = source.get_order_book(market, client=client)
            except httpx.HTTPError:
                return market.id, None, True
            if book.best_bid is None or book.best_ask is None or book.mid is None:
                return market.id, None, False
            spread_bps = (book.best_ask - book.best_bid) / book.mid * 10_000
            return market.id, {
                "bid": book.best_bid,
                "ask": book.best_ask,
                "mid": book.mid,
                "spread_bps": spread_bps,
            }, False

        records = markets_df.to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fetch_one, r) for r in records]
            done = 0
            for fut in as_completed(futures):
                market_id, book, err = fut.result()
                done += 1
                if err:
                    errors += 1
                    continue
                if book is None:
                    skipped += 1
                    continue
                rows.append({"market_id": market_id, **book})
                if done % 25 == 0:
                    print(f"  [{done}/{len(records)}] ok={len(rows)} skipped={skipped} errors={errors}")

    df = pd.DataFrame(rows, columns=["market_id", "bid", "ask", "mid", "spread_bps"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"wrote {out_csv}: {len(df)} tradeable / {skipped} un-tradeable / {errors} errors"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot live orderbook spreads per market."
    )
    parser.add_argument("--source", default="polymarket")
    parser.add_argument("--markets", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    markets_path = args.markets or _default_markets_csv()
    out_path = args.out or _default_spreads_csv()

    df = snapshot_spreads(
        markets_path,
        out_path,
        source_name=args.source,
        concurrency=args.concurrency,
    )
    if df.empty:
        print("no spreads captured")
        return
    q = df["spread_bps"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    print("spread_bps quantiles:")
    for k, v in q.items():
        print(f"  p{int(k * 100):>2}: {v:8.1f} bps")


if __name__ == "__main__":
    main()
