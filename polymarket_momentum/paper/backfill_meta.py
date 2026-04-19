"""Fill market_meta for markets that appear in trades but aren't in the current
universe. One-shot — run via `python -m polymarket_momentum.paper.backfill_meta`.
"""
from __future__ import annotations

import argparse
import time

import httpx

from ..sources.polymarket import GAMMA_URL
from .config import PaperConfig
from .db import connect, init


def _fetch_one(client: httpx.Client, market_id: str) -> dict | None:
    try:
        resp = client.get(f"{GAMMA_URL}/{market_id}")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill market_meta from Polymarket for historical positions.")
    parser.add_argument("--source", default="polymarket")
    args = parser.parse_args()

    config = PaperConfig.from_env()
    init(config.db_path)

    # Phase 1: read missing IDs (short read transaction).
    with connect(config.db_path) as conn:
        missing = [
            r["market_id"]
            for r in conn.execute(
                "SELECT DISTINCT t.market_id FROM trades t "
                "LEFT JOIN market_meta m ON t.market_id = m.market_id "
                "WHERE m.market_id IS NULL"
            )
        ]
    if not missing:
        print("no missing metadata")
        return
    print(f"backfilling {len(missing)} markets from Polymarket")

    # Phase 2: HTTP fetches — no DB lock held, so poller can run freely.
    fetched: list[tuple[str, str, str]] = []
    with httpx.Client(timeout=15) as client:
        for i, mid in enumerate(missing, 1):
            payload = _fetch_one(client, mid)
            if payload is None:
                print(f"  [{i}/{len(missing)}] {mid} — not found")
                continue
            question = payload.get("question") or ""
            slug = payload.get("slug") or ""
            fetched.append((mid, question, slug))
            print(f"  [{i}/{len(missing)}] {mid} — {question[:60]}")

    # Phase 3: single short write transaction.
    ts = int(time.time())
    with connect(config.db_path) as conn:
        for mid, question, slug in fetched:
            conn.execute(
                "INSERT OR REPLACE INTO market_meta (market_id, source, question, slug, updated_ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (mid, args.source, question, slug, ts),
            )
    print(f"backfilled {len(fetched)}/{len(missing)}")


if __name__ == "__main__":
    main()
