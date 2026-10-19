import asyncio
import logging
import os
import time
import uuid
from functools import partial

from ami_rag.observability import (
    INGEST_DOCUMENTS,
    INGEST_DURATION_SECONDS,
    INGEST_EVENTS_TOTAL,
    INGEST_IN_FLIGHT,
    INGEST_STREAM_LAG,
    INGEST_STREAM_PENDING,
    observe_ingest_stage,
)
from ami_rag.queue.events import EVENT_DELETED, RagEvent
from ami_rag.queue.streams import RagStreamQueue
from ami_rag.settings import Settings, get_settings
from ami_rag.sources import SOURCE_MINIO_PARSE, select_source, source_hash
from ami_rag.storage.doc_status import (
    STATUS_INDEXED,
    DocStatusStore,
    effective_status,
)

logger = logging.getLogger(__name__)

_STATS_INTERVAL_SECONDS = 15
_COUNTS_INTERVAL_SECONDS = 60


class IngestWorker:
    """Consume RAG ingest events from Redis Streams and drive the vector pipeline.

    created/updated -> ensure_pending + runner.run(doc_id, from_stage="parse")
    (parse -> describe -> chunk -> embed -> indexed, DocStatusStore ghi ở mỗi
    stage). Unchanged source_hash (đã indexed, cùng embed model/chunker) skip.
    deleted -> runner.delete_doc + registry row.
    """

    def __init__(
        self,
        runner,
        docs_repo,
        state_repo: DocStatusStore,
        queue: RagStreamQueue,
        asset_store,
        settings: Settings | None = None,
        logger_=None,
    ):
        self.runner = runner
        self.docs_repo = docs_repo
        self.state_repo = state_repo
        self.queue = queue
        self.asset_store = asset_store
        self.settings = settings or get_settings()
        self.log = logger_ or logger
        self._seen_statuses: set[str] = set()

    async def _purge(self, doc_id: str) -> None:
        with observe_ingest_stage("delete"):
            await self.runner.delete_doc(doc_id)

    async def handle_event(self, event: RagEvent) -> dict | None:
        """Process one event. Returns ingest info when the document was (re)indexed,
        None when nothing was indexed (deleted, missing, empty, unchanged)."""
        doc_id = event.document_id

        if event.event == EVENT_DELETED:
            await self._purge(doc_id)
            await asyncio.to_thread(self.state_repo.delete, doc_id)
            self.log.info("deleted document %s from RAG index", doc_id)
            return None

        doc = await asyncio.to_thread(self.docs_repo.find_by_id, doc_id)
        if doc is None:
            self.log.warning("document %s not found in Mongo, skipping", doc_id)
            return None

        source = select_source(doc)
        if source == SOURCE_MINIO_PARSE:
            if not doc.get("file_path"):
                raise ValueError(f"document {doc_id} has no file_path in MinIO to parse")
        elif not (doc.get("content") or "").strip():
            self.log.warning("document %s has empty content, skipping", doc_id)
            return None

        h = source_hash(doc, source)
        prev = await asyncio.to_thread(self.state_repo.get, doc_id)
        if (
            prev
            and effective_status(
                prev, self.settings.EMBED_MODEL, self.settings.CHUNKER_VERSION
            )
            == STATUS_INDEXED
            and prev.get("content_hash") == h
        ):
            self.log.debug("document %s unchanged, skipping", doc_id)
            return None

        if prev is None:
            await asyncio.to_thread(
                partial(
                    self.state_repo.ensure_pending,
                    doc_id,
                    doc.get("file_path") or "",
                    h,
                )
            )

        outcome = await self.runner.run(doc_id, from_stage="parse")
        if not outcome.ok:
            raise RuntimeError(
                f"pipeline failed at stage '{outcome.stage}': {outcome.error}"
            )
        self.log.info(
            "indexed document %s (type=%s, source=%s, chunks=%s)",
            doc_id,
            doc.get("document_type"),
            source,
            outcome.chunk_count,
        )
        return {
            "source": source,
            "source_hash": h,
            "chunk_count": outcome.chunk_count,
        }

    async def _process_message(self, message_id: str, event: RagEvent) -> None:
        attempts = await asyncio.to_thread(
            self.state_repo.begin_attempt, event.document_id, event.content_hash or ""
        )
        INGEST_IN_FLIGHT.inc()
        started = time.perf_counter()
        source = "none"
        ran = False
        try:
            try:
                result = await self.handle_event(event)
                ran = True
            except Exception as exc:
                self.log.error(
                    "failed to process event %s for document %s: %s",
                    event.event,
                    event.document_id,
                    exc,
                )
                if not ran:
                    # begin_attempt đã +1 attempts ở đầu delivery; mark_failed
                    # chỉ ghi trạng thái (tăng nữa là đếm đôi, cắt ngắn budget
                    # WORKER_MAX_DELIVERY / --max-attempts của CLI retry).
                    await asyncio.to_thread(
                        partial(
                            self.state_repo.mark_failed,
                            event.document_id,
                            str(exc),
                            "ingest",
                            increment_attempt=False,
                        )
                    )
                if attempts >= self.settings.WORKER_MAX_DELIVERY:
                    INGEST_EVENTS_TOTAL.labels(event=event.event, result="failed").inc()
                    await self.queue.ack(message_id)
                    self.log.error(
                        "document %s marked failed after %s attempts",
                        event.document_id,
                        attempts,
                    )
                else:
                    INGEST_EVENTS_TOTAL.labels(event=event.event, result="retry").inc()
                return
            if result:
                source = result.get("source") or "none"
                INGEST_EVENTS_TOTAL.labels(event=event.event, result="processed").inc()
            else:
                INGEST_EVENTS_TOTAL.labels(event=event.event, result="skipped").inc()
                await asyncio.to_thread(self.state_repo.release_attempt, event.document_id)
            await self.queue.ack(message_id)
        finally:
            INGEST_DURATION_SECONDS.labels(source=source).observe(time.perf_counter() - started)
            INGEST_IN_FLIGHT.dec()

    async def _update_queue_gauges(self) -> None:
        try:
            pending, lag = await self.queue.stats()
            INGEST_STREAM_PENDING.set(pending)
            INGEST_STREAM_LAG.set(lag)
        except Exception as exc:
            self.log.warning("queue gauge update failed: %s", exc)

    async def _update_document_gauges(self) -> None:
        try:
            counts = await asyncio.to_thread(self.state_repo.counts)
            for status in self._seen_statuses - set(counts):
                INGEST_DOCUMENTS.labels(status=status).set(0)
            for status, n in counts.items():
                INGEST_DOCUMENTS.labels(status=status).set(n)
            self._seen_statuses = set(counts)
        except Exception as exc:
            self.log.warning("document gauge update failed: %s", exc)

    async def run_forever(self) -> None:
        await self.queue.ensure_group()
        self.log.info(
            "ingest worker started (stream=%s group=%s consumer=%s)",
            self.settings.RAG_STREAM,
            self.settings.RAG_CONSUMER_GROUP,
            self.queue._consumer,
        )
        next_stats = next_counts = 0.0
        while True:
            now = time.monotonic()
            if now >= next_stats:
                next_stats = now + _STATS_INTERVAL_SECONDS
                await self._update_queue_gauges()
            if now >= next_counts:
                next_counts = now + _COUNTS_INTERVAL_SECONDS
                await self._update_document_gauges()
            try:
                batch = await self.queue.read_batch(
                    count=self.settings.WORKER_BATCH,
                    block_ms=self.settings.WORKER_POLL_BLOCK_MS,
                )
            except Exception as exc:
                self.log.error("queue read failed: %s, retrying", exc)
                await asyncio.sleep(5)
                continue
            for message_id, event in batch:
                await self._process_message(message_id, event)


def _build_default_deps(settings: Settings):
    """Returns (docs_repo, state_repo, queue, asset_store)."""
    from ami_rag.core.factory import get_asset_store
    from ami_rag.queue.streams import RagStreamQueue
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    consumer = settings.RAG_CONSUMER_NAME or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    docs_repo = MongoDocumentRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.ORG_DB,
        collection_name=settings.DOC_COLLECTION,
    )
    state_repo = DocStatusStore(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.RAG_DB,
        collection_name=settings.RAG_DOCUMENTS_COLLECTION,
    )
    # XREADGROUP BLOCK holds the connection for WORKER_POLL_BLOCK_MS; redis-py>=8 defaults
    # socket_timeout to 5s, which would fire first on an idle stream.
    redis_client = aioredis_from_url(
        settings.REDIS_URL,
        socket_timeout=settings.WORKER_POLL_BLOCK_MS / 1000 + 5,
        health_check_interval=30,
    )
    queue = RagStreamQueue(
        redis_client=redis_client,
        stream=settings.RAG_STREAM,
        group=settings.RAG_CONSUMER_GROUP,
        consumer=consumer,
        retry_idle_ms=settings.WORKER_RETRY_IDLE_MS,
    )
    return docs_repo, state_repo, queue, get_asset_store()


def aioredis_from_url(url: str, **kwargs):
    import redis.asyncio as aioredis

    return aioredis.from_url(url, decode_responses=True, **kwargs)


async def run_worker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    # Standalone worker only: the embedded worker shares the API's /metrics.
    from prometheus_client import start_http_server

    start_http_server(settings.WORKER_METRICS_PORT)
    from ami_rag.core.factory import build_pipeline

    docs_repo, state_repo, queue, asset_store = _build_default_deps(settings)
    worker = IngestWorker(
        runner=build_pipeline(settings, docs_repo=docs_repo),
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=queue,
        asset_store=asset_store,
        settings=settings,
    )
    try:
        await worker.run_forever()
    finally:
        from ami_rag.core.factory import close_pipeline

        await close_pipeline()


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
