import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from ami_rag.api.routes import admin as admin_routes
from ami_rag.api.routes import metrics as metrics_routes
from ami_rag.api.routes import rag as rag_routes
from ami_rag.observability import init_telemetry
from ami_rag.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    worker_task = None
    if settings.WORKER_ENABLED:
        from ami_rag.core.factory import get_pipeline
        from ami_rag.workers.ingest_worker import IngestWorker, _build_default_deps

        docs_repo, state_repo, queue, asset_store = _build_default_deps(settings)
        worker = IngestWorker(
            runner=await get_pipeline(),
            docs_repo=docs_repo,
            state_repo=state_repo,
            queue=queue,
            asset_store=asset_store,
            settings=settings,
        )
        worker_task = asyncio.create_task(worker.run_forever())
    yield
    if worker_task is not None:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task
    from ami_rag.core.factory import close_pipeline

    await close_pipeline()


def create_app() -> FastAPI:
    # Tracing is limited to the manual `rag.retrieval` span of /v2/rag (init_telemetry +
    # observability.track_retrieval). Disable FastAPI >=0.142 native telemetry: it would
    # trace every route (/metrics, /admin, ...) and push metrics/logs to /v1/metrics and
    # /v1/logs of the OTLP endpoint (Tempo: 404) plus a duplicate traces exporter.
    app = FastAPI(
        title="AMI RAG API",
        version="0.1.0",
        lifespan=lifespan,
        telemetry={
            "auto_configure": False,
            "tracing": False,
            "metrics": False,
            "logs": False,
        },
    )
    app.include_router(rag_routes.router, prefix="/v2/rag")
    app.include_router(admin_routes.router, prefix="/admin")
    app.include_router(metrics_routes.router)

    @app.get("/")
    async def root():
        return {"service": "ami-rag", "version": "0.1.0"}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ok"}

    return app


init_telemetry()
app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
