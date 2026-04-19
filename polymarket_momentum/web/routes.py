from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .config import load_settings
from .services import load_positions, load_ranking

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


def _resolve_settings(source: str | None):
    settings = load_settings()
    if source:
        settings = replace(settings, source=source)
    return settings


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, source: str | None = Query(default=None)):
    settings = _resolve_settings(source)
    ranking = load_ranking(settings)
    positions = load_positions(settings)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "ranking": ranking,
            "positions": positions,
            "settings": settings,
        },
    )


@router.get("/api/ranking")
def api_ranking(source: str | None = Query(default=None)):
    settings = _resolve_settings(source)
    view = load_ranking(settings)
    return JSONResponse(
        {
            "source": view.source,
            "target_notional": view.target_notional,
            "fee_model": view.fee_model_name,
            "lookback_hours": view.lookback_hours,
            "error": view.error,
            "bets": [asdict(b) for b in view.bets],
        }
    )


@router.get("/api/positions")
def api_positions(source: str | None = Query(default=None)):
    settings = _resolve_settings(source)
    view = load_positions(settings)
    return JSONResponse(
        {
            "source": view.source,
            "error": view.error,
            "positions": [asdict(p) for p in view.positions],
        }
    )
