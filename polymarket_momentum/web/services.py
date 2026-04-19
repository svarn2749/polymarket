from __future__ import annotations

from dataclasses import dataclass

from ..ranking import RankedBet, rank_bets
from ..sources import get_source
from ..sources.base import Position
from .config import Settings


@dataclass
class PositionsView:
    source: str
    positions: list[Position]
    error: str | None


@dataclass
class RankingView:
    source: str
    bets: list[RankedBet]
    target_notional: float
    fee_model_name: str
    lookback_hours: int
    error: str | None


def load_positions(settings: Settings) -> PositionsView:
    try:
        source = get_source(settings.source)
        positions = source.fetch_positions()
    except Exception as exc:  # surface upstream / auth errors to the UI
        return PositionsView(source=settings.source, positions=[], error=str(exc))
    return PositionsView(source=settings.source, positions=positions, error=None)


def load_ranking(settings: Settings) -> RankingView:
    try:
        source = get_source(settings.source)
    except Exception as exc:
        return RankingView(
            source=settings.source,
            bets=[],
            target_notional=settings.target_notional,
            fee_model_name="?",
            lookback_hours=settings.lookback_hours,
            error=str(exc),
        )

    if not settings.markets_csv.exists():
        return RankingView(
            source=settings.source,
            bets=[],
            target_notional=settings.target_notional,
            fee_model_name=source.fee_model.name,
            lookback_hours=settings.lookback_hours,
            error=f"markets metadata not found at {settings.markets_csv} — run `fetch` first",
        )
    try:
        bets = rank_bets(
            source,
            prices_dir=settings.prices_dir,
            markets_csv=settings.markets_csv,
            target_notional=settings.target_notional,
            lookback_hours=settings.lookback_hours,
            resample_rule=settings.resample_rule,
            min_abs_signal=settings.min_abs_signal,
            max_markets=settings.max_markets,
        )
    except Exception as exc:
        return RankingView(
            source=settings.source,
            bets=[],
            target_notional=settings.target_notional,
            fee_model_name=source.fee_model.name,
            lookback_hours=settings.lookback_hours,
            error=str(exc),
        )
    return RankingView(
        source=settings.source,
        bets=bets,
        target_notional=settings.target_notional,
        fee_model_name=source.fee_model.name,
        lookback_hours=settings.lookback_hours,
        error=None,
    )
