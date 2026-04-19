from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..sources.base import Market
from ..topic import derive_topic


def load_universe(
    *,
    source: str,
    markets_csv: Path,
    spreads_csv: Path | None,
    top_n: int,
    max_spread_bps: float,
    min_days_to_expiry: float = 0.0,
    exclude_topics: str = "",
) -> list[Market]:
    if not markets_csv.exists():
        return []

    df = pd.read_csv(markets_csv, dtype={"id": str, "yes_id": str, "no_id": str})
    if "source" in df.columns:
        df = df[df["source"].astype(str) == source]
    df = df.sort_values("volume", ascending=False)

    if spreads_csv is not None and spreads_csv.exists() and max_spread_bps > 0:
        spreads = pd.read_csv(spreads_csv, dtype={"market_id": str})
        allowed = set(spreads.loc[spreads["spread_bps"] <= max_spread_bps, "market_id"])
        df = df[df["id"].astype(str).isin(allowed)]

    if min_days_to_expiry > 0 and "end_date" in df.columns:
        ends = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=min_days_to_expiry)
        # Keep markets without an end_date (NaT) and those expiring after the cutoff.
        df = df[ends.isna() | (ends > cutoff)]

    excluded = {t.strip().lower() for t in exclude_topics.split(",") if t.strip()}
    if excluded:
        topics = [derive_topic(s, q) for s, q in zip(df.get("slug", ""), df.get("question", ""))]
        df = df.assign(_topic=topics)
        df = df[~df["_topic"].str.lower().isin(excluded)]
        df = df.drop(columns=["_topic"])

    df = df.head(top_n)

    markets: list[Market] = []
    for row in df.to_dict(orient="records"):
        markets.append(
            Market(
                source=str(row.get("source") or source),
                id=str(row["id"]),
                question=str(row.get("question") or ""),
                slug=str(row.get("slug") or ""),
                yes_id=str(row["yes_id"]),
                no_id=str(row["no_id"]),
                volume=float(row.get("volume") or 0),
                end_date=(row.get("end_date") or None),
                closed=bool(row.get("closed") or False),
            )
        )
    return markets
