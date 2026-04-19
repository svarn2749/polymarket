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
    direction: str  # "long YES" / "short YES"
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
class PaperView:
    config: dict
    positions: list[PaperPositionRow]
    latest_equity: EquityPoint | None
    recent_trades: list[PaperTradeRow]
    last_poll: dict | None


def _config_from_request(request: Request) -> PaperConfig:
    config = getattr(request.app.state, "paper_config", None)
    return config or PaperConfig.from_env()


def _load_market_meta(config: PaperConfig) -> dict[str, MarketMeta]:
    if not config.markets_csv.exists():
        return {}
    df = pd.read_csv(config.markets_csv, dtype=str)
    out: dict[str, MarketMeta] = {}
    for row in df.to_dict(orient="records"):
        out[str(row["id"])] = MarketMeta(
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


def _build_view(config: PaperConfig) -> PaperView:
    if not config.db_path.exists():
        return PaperView(
            config=_config_summary(config),
            positions=[],
            latest_equity=None,
            recent_trades=[],
            last_poll=None,
        )

    meta = _load_market_meta(config)

    def _meta(market_id: str) -> MarketMeta:
        return meta.get(str(market_id)) or MarketMeta(source=config.source)

    with connect(config.db_path) as conn:
        positions_dict = current_positions(conn)

        # Enrich positions with latest mid from snapshots.
        positions_rows: list[PaperPositionRow] = []
        for pos in positions_dict.values():
            snap = conn.execute(
                "SELECT mid FROM snapshots WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
                (pos.market_id,),
            ).fetchone()
            if snap and snap["mid"] is not None:
                pos.last_price = float(snap["mid"])
            m = _meta(pos.market_id)
            positions_rows.append(
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
        positions_rows.sort(key=lambda r: -abs(r.total_pnl))

        eq_row = conn.execute(
            "SELECT ts, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions "
            "FROM equity ORDER BY ts DESC LIMIT 1"
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
            "FROM trades ORDER BY id DESC LIMIT 25"
        ).fetchall()
        trades: list[PaperTradeRow] = []
        for r in trade_rows:
            market_id = str(r["market_id"])
            m = _meta(market_id)
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

        poll_row = conn.execute(
            "SELECT ts, duration_ms, n_markets, n_errors, n_trades, note "
            "FROM poll_log ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        last_poll = dict(poll_row) if poll_row else None

    return PaperView(
        config=_config_summary(config),
        positions=positions_rows,
        latest_equity=latest_equity,
        recent_trades=trades,
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
def api_positions(request: Request):
    config = _config_from_request(request)
    view = _build_view(config)
    return JSONResponse({"positions": [asdict(p) for p in view.positions]})


@router.get("/api/equity")
def api_equity(request: Request, limit: int = 500):
    config = _config_from_request(request)
    if not config.db_path.exists():
        return JSONResponse({"points": []})
    with connect(config.db_path) as conn:
        rows = conn.execute(
            "SELECT ts, realized_pnl, unrealized_pnl, total_pnl, gross_exposure, n_positions "
            "FROM equity ORDER BY ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return JSONResponse(
        {"points": [dict(r) for r in rows]}
    )


@router.get("/api/health")
def api_health(request: Request):
    config = _config_from_request(request)
    view = _build_view(config)
    return JSONResponse(
        {
            "config": view.config,
            "last_poll": view.last_poll,
            "latest_equity": asdict(view.latest_equity) if view.latest_equity else None,
        }
    )
