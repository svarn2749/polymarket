from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..paper.config import PaperConfig
from ..paper.db import init as init_paper_db
from ..paper.poller import poll_loop
from .paper_routes import router as paper_router
from .routes import router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PaperConfig.from_env()
    app.state.paper_config = config
    init_paper_db(config.db_path)

    stop = asyncio.Event()
    task: asyncio.Task | None = None
    if config.enabled:
        task = asyncio.create_task(poll_loop(config, stop))
        print(
            f"[paper] poller started: source={config.source} "
            f"strategy={config.strategy} poll={config.poll_interval_sec}s "
            f"db={config.db_path}"
        )
    else:
        print("[paper] PAPER_ENABLED=false — poller not started")

    try:
        yield
    finally:
        stop.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="Polymarket Momentum", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(router)
    app.include_router(paper_router)
    return app


app = create_app()
