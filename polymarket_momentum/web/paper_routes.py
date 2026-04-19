from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..paper.config import PaperConfig
from ..paper.db import connect
from ..paper.ledger import current_positions

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/paper")

STRATEGIES = ("reversion", "cross_sectional")
STRATEGY_LABELS = {
    "reversion": "Per-market reversion",
    "cross_sectional": "Cross-sectional reversion",
}


@dataclass
class MarketMeta:
    question: str = ""
    slug: str = ""
    source: str = ""


@dataclass
class PaperPositionRow:
    market_id: str
    question: str
    slug: str
    source: str
    direction: str
    size: float
    avg_entry_price: float
    last_price: float | None
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    notional: float


@dataclass
class EquityPoint:
    ts: int
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    gross_exposure: float
    n_positions: int


@dataclass
class PaperTradeRow:
    ts: int
    ts_iso: str
    market_id: str
    question: str
    slug: str
    source: str
    side: str
    size: float
    price: float
    signal: float | None
    strategy: str | None


@dataclass
class StrategyView:
    name: str
    label: str
    positions: list[PaperPositionRow]
    recent_trades: list[PaperTradeRow]
    latest_equity: EquityPoint | None
    realized_from_closed: float  # PnL from fully-closed positions (not in positions table)


@dataclass
class PaperView:
    config: dict
    strategies: list[StrategyView]
    last_poll: dict | None


def _config_from_request(request: Request) -> PaperConfig:
    config = getattr(request.app.state, "paper_config", None)
    return config or PaperConfig.from_env()


def _load_market_meta(config: PaperConfig) -> dict[str, MarketMeta]:
    """Load market metadata from SQLite (authoritative, persists across universe
    refreshes), overlaying current markets.csv if present. SQLite holds metadata
    for every market the poller has ever observed — so positions opened before
    a universe refresh still render cleanly.
    """
    out: dict[str, MarketMeta] = {}
    if config.db_path.exists():
        with connect(config.db_path) as conn:
            for r in conn.execute(
                "SELECT market_id, question, slug, source FROM market_meta"
            ):
                out[str(r["market_id"])] = MarketMeta(
                    question=str(r["question"] or ""),
                    slug=str(r["slug"] or ""),
                    source=str(r["source"] or config.source),
                )
    if config.markets_csv.exists():
        df = pd.read_csv(config.markets_csv, dtype=str)
        for row in df.to_dict(orient="records"):
            # markets.csv "wins" over SQLite only when SQLite has no record yet
            # (fresh universe entry whose first poll hasn't completed).
            mid = str(row["id"])
            if mid not in out:
                out[mid] = MarketMeta(
                    question=str(row.get("question") or ""),
                    slug=str(row.get("slug") or ""),
                    source=str(row.get("source") or config.source),
                )
    return out


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _direction_label(size: float) -> str:
    if size > 0:
        return "long YES"
    if size < 0:
        return "short YES"
    return "flat"


def _strategy_view(
    conn,
    *,
    strategy: str,
    meta: dict[str, MarketMeta],
    config: PaperConfig,
) -> StrategyView:
    all_positions = current_positions(conn, strategy=strategy)
    open_rows: list[PaperPositionRow] = []
    realized_from_closed = 0.0
    for pos in all_positions.values():
        if abs(pos.size) <= 1e-9:
            realized_from_closed += pos.realized_pnl
            continue
        snap = conn.execute(
            "SELECT mid FROM snapshots WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
            (pos.market_id,),
        ).fetchone()
        if snap and snap["mid"] is not None:
            pos.last_price = float(snap["mid"])
        m = meta.get(pos.market_id) or MarketMeta(source=config.source)
        open_rows.append(
            PaperPositionRow(
                market_id=pos.market_id,
                question=m.question,
                slug=m.slug,
                source=m.source,
                direction=_direction_label(pos.size),
                size=pos.size,
                avg_entry_price=pos.avg_entry_price,
                last_price=pos.last_price,
                realized_pnl=pos.realized_pnl,
                unrealized_pnl=pos.unrealized_pnl,
                total_pnl=pos.realized_pnl + pos.unrealized_pnl,
                notional=pos.notional,
            )
        )
    open_rows.sort(key=lambda r: -abs(r.total_pnl))

    eq_row = conn.execute(
        "SELECT ts, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions "
        "FROM equity WHERE strategy = ? ORDER BY ts DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    latest_equity = (
        EquityPoint(
            ts=int(eq_row["ts"]),
            realized_pnl=float(eq_row["realized_pnl"]),
            unrealized_pnl=float(eq_row["unrealized_pnl"]),
            total_pnl=float(eq_row["total_pnl"]),
            gross_exposure=float(eq_row["gross_exposure"]),
            n_positions=int(eq_row["n_positions"]),
        )
        if eq_row
        else None
    )

    trade_rows = conn.execute(
        "SELECT ts, market_id, side, size, price, signal, strategy "
        "FROM trades WHERE strategy = ? ORDER BY id DESC LIMIT 25",
        (strategy,),
    ).fetchall()
    trades: list[PaperTradeRow] = []
    for r in trade_rows:
        market_id = str(r["market_id"])
        m = meta.get(market_id) or MarketMeta(source=config.source)
        trades.append(
            PaperTradeRow(
                ts=int(r["ts"]),
                ts_iso=_fmt_ts(int(r["ts"])),
                market_id=market_id,
                question=m.question,
                slug=m.slug,
                source=m.source,
                side=str(r["side"]),
                size=float(r["size"]),
                price=float(r["price"]),
                signal=(float(r["signal"]) if r["signal"] is not None else None),
                strategy=(str(r["strategy"]) if r["strategy"] is not None else None),
            )
        )

    return StrategyView(
        name=strategy,
        label=STRATEGY_LABELS.get(strategy, strategy),
        positions=open_rows,
        recent_trades=trades,
        latest_equity=latest_equity,
        realized_from_closed=realized_from_closed,
    )


def _build_view(config: PaperConfig) -> PaperView:
    if not config.db_path.exists():
        return PaperView(
            config=_config_summary(config),
            strategies=[
                StrategyView(
                    name=s, label=STRATEGY_LABELS.get(s, s),
                    positions=[], recent_trades=[],
                    latest_equity=None, realized_from_closed=0.0,
                )
                for s in STRATEGIES
            ],
            last_poll=None,
        )

    meta = _load_market_meta(config)

    with connect(config.db_path) as conn:
        strategies = [_strategy_view(conn, strategy=s, meta=meta, config=config) for s in STRATEGIES]

        poll_row = conn.execute(
            "SELECT ts, duration_ms, n_markets, n_errors, n_trades, note "
            "FROM poll_log ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if poll_row:
            last_poll = dict(poll_row)
            last_poll["ts_iso"] = _fmt_ts(int(poll_row["ts"]))
        else:
            last_poll = None

    return PaperView(
        config=_config_summary(config),
        strategies=strategies,
        last_poll=last_poll,
    )


def _config_summary(config: PaperConfig) -> dict:
    return {
        "source": config.source,
        "strategy": config.strategy,
        "lookback_hours": config.lookback_hours,
        "entry_threshold": config.entry_threshold,
        "fill_model": config.fill_model,
        "position_size_usd": config.position_size_usd,
        "poll_interval_sec": config.poll_interval_sec,
        "universe_top_n": config.universe_top_n,
        "max_spread_bps": config.max_spread_bps,
        "db_path": str(config.db_path),
        "enabled": config.enabled,
        "cross_sectional_enabled": config.cross_sectional_enabled,
        "cross_sectional_top_k": config.cross_sectional_top_k,
    }


@router.get("", response_class=HTMLResponse)
def paper_dashboard(request: Request):
    config = _config_from_request(request)
    view = _build_view(config)
    return templates.TemplateResponse(
        request,
        "paper.html",
        {"view": view, "settings": {"source": config.source}},
    )


@router.get("/api/positions")
def api_positions(request: Request, strategy: str | None = None):
    config = _config_from_request(request)
    view = _build_view(config)
    out = {
        s.name: [asdict(p) for p in s.positions] for s in view.strategies
    }
    if strategy is not None:
        return JSONResponse({"strategy": strategy, "positions": out.get(strategy, [])})
    return JSONResponse({"by_strategy": out})


@router.get("/api/equity")
def api_equity(request: Request, strategy: str | None = None, limit: int = 500):
    config = _config_from_request(request)
    if not config.db_path.exists():
        return JSONResponse({"points": []})
    with connect(config.db_path) as conn:
        if strategy is None:
            rows = conn.execute(
                "SELECT ts, strategy, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions "
                "FROM equity ORDER BY ts ASC LIMIT ?",
                (limit * len(STRATEGIES),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, strategy, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions "
                "FROM equity WHERE strategy = ? ORDER BY ts ASC LIMIT ?",
                (strategy, limit),
            ).fetchall()
    return JSONResponse({"points": [dict(r) for r in rows]})


@router.get("/api/health")
def api_health(request: Request):
    config = _config_from_request(request)
    view = _build_view(config)
    return JSONResponse(
        {
            "config": view.config,
            "last_poll": view.last_poll,
            "strategies": {
                s.name: {
                    "latest_equity": asdict(s.latest_equity) if s.latest_equity else None,
                    "n_open_positions": len(s.positions),
                    "realized_from_closed": s.realized_from_closed,
                }
                for s in view.strategies
            },
        }
    )
