"""Observability: Prometheus metrics + OpenTelemetry tracing for the retrieval API.

Two layers, initialised once:

- **Metrics (Prometheus)**: counters/histograms prefixed ``multimodal_rag_retrieval_``,
  scraped via ``/metrics`` and rendered in the "Multimodal RAG Retrieval" Grafana
  dashboard. Day/hour-of-day labels use a fixed timezone (``Asia/Ho_Chi_Minh``);
  the dashboard MUST use the same fixed timezone (not ``browser``).
- **Tracing (OpenTelemetry, OTLP/HTTP)**: one ``rag.retrieval`` span per search with
  the verbatim query and the full document list, plus child spans per pipeline
  stage. Full payloads are also written to a structured log line carrying the
  ``trace_id`` so they can be joined in Grafana.

``track_retrieval`` wraps a whole search; ``observe_stage`` times one stage
(``embed_query`` / ``vector_search`` / ``rerank``).
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedInputTooLong,
    EmbedModelMismatch,
    EmbedServerOOM,
    EmbedServerUnreachable,
)

logger = logging.getLogger(__name__)

DASHBOARD_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "multimodal-rag-retrieval")

# --- Prometheus metrics (registry: default) --------------------------------

REQUESTS_TOTAL = Counter(
    "multimodal_rag_retrieval_requests_total",
    "Total number of retrieval requests.",
    ["mode"],
)
# shortcut: nhãn `day`/`hod` sinh series mới mỗi ngày (1/ngày, 24/ngày cho
# day+hour) và không bao giờ hết hạn - chấp nhận vì corpus nội bộ và dashboard
# drill-down theo ngày dùng chúng; khi Prometheus phình to, bỏ 3 counter này
# và cho Grafana group theo time thay vì label.
REQUESTS_BY_DAY = Counter(
    "multimodal_rag_retrieval_requests_by_day",
    "Retrieval requests per calendar day (Asia/Ho_Chi_Minh). Dashboard timezone MUST match.",
    ["day"],
)
REQUESTS_BY_HOUR_OF_DAY = Counter(
    "multimodal_rag_retrieval_requests_by_hour_of_day",
    "Retrieval requests per hour of day (Asia/Ho_Chi_Minh). Dashboard timezone MUST match.",
    ["hod"],
)
REQUESTS_BY_DAY_HOUR = Counter(
    "multimodal_rag_retrieval_requests_by_day_hour",
    "Retrieval requests per day+hour combination (Asia/Ho_Chi_Minh), for day drill-down.",
    ["day", "hod"],
)
ERRORS_TOTAL = Counter(
    "multimodal_rag_retrieval_errors_total",
    "Total number of failed retrieval requests.",
    ["mode"],
)
DURATION_SECONDS = Histogram(
    "multimodal_rag_retrieval_duration_seconds",
    "Full round-trip retrieval duration in seconds (query + rerank + resolve).",
    ["mode"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)
DOCS_RETURNED = Histogram(
    "multimodal_rag_retrieval_docs_returned",
    "Number of documents returned per retrieval request.",
    buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50),
)
REQUESTS_IN_FLIGHT = Gauge(
    "multimodal_rag_retrieval_requests_in_flight",
    "Retrieval requests currently being processed.",
)
STAGE_DURATION_SECONDS = Histogram(
    "multimodal_rag_retrieval_stage_duration_seconds",
    "Duration of one retrieval pipeline stage (embed_query, vector_search, rerank).",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
CHUNKS_RETRIEVED = Histogram(
    "multimodal_rag_retrieval_chunks_retrieved",
    "Number of chunks returned by vector search before reranking.",
    buckets=(0, 1, 5, 10, 20, 30, 40, 60, 80, 100),
)
RERANK_FALLBACK_TOTAL = Counter(
    "multimodal_rag_retrieval_rerank_fallback_total",
    "Rerank fell back to unscored chunks (reason: error | empty).",
    ["reason"],
)
RETRIEVAL_DOCS_BY_MODALITY_TOTAL = Counter(
    "multimodal_rag_retrieval_docs_by_modality_total",
    "Documents returned per modality (text | image | table | equation).",
    ["modality"],
)
PRESIGN_FAILURES_TOTAL = Counter(
    "multimodal_rag_retrieval_presign_failures_total",
    "Presigned asset URL generation failures.",
)
SEARCH_MODE_TOTAL = Counter(
    "multimodal_rag_retrieval_search_mode_total",
    "Retrieval searches by effective search mode (dense | hybrid).",
    ["mode"],
)
ARTIFACT_URLS_RETURNED_TOTAL = Counter(
    "multimodal_rag_retrieval_artifact_urls_returned_total",
    "artifact_urls entries returned in RetrievalResponseV2 documents.",
)
EMBED_FAILURES_TOTAL = Counter(
    "multimodal_rag_retrieval_embed_failures_total",
    "Query embed failures in the retrieval path "
    "(reason: unreachable | auth | input_too_long | oom | model_mismatch | circuit_open | other).",
    ["reason"],
)

# Closed ``reason`` label set (see EMBED_FAILURES_TOTAL / INGEST_EMBED_FAILURES_TOTAL):
# one value per embed exception class in ami_rag.core.embedder + "other".
_EMBED_FAILURE_REASONS: tuple[tuple[type[BaseException], str], ...] = (
    (EmbedServerUnreachable, "unreachable"),
    (EmbedCircuitOpen, "circuit_open"),
    (EmbedAuthError, "auth"),
    (EmbedInputTooLong, "input_too_long"),
    (EmbedServerOOM, "oom"),
    (EmbedModelMismatch, "model_mismatch"),
)


def embed_failure_reason(exc: BaseException) -> str:
    """Map an embed exception to the closed ``reason`` label set (fallback ``other``)."""
    for cls, reason in _EMBED_FAILURE_REASONS:
        if isinstance(exc, cls):
            return reason
    return "other"

# --- Ingest worker metrics (prefix multimodal_rag_ingest_) ------------------

_INGEST_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800)

INGEST_EVENTS_TOTAL = Counter(
    "multimodal_rag_ingest_events_total",
    "Ingest events handled (result: processed | skipped | failed | retry).",
    ["event", "result"],
)
INGEST_IN_FLIGHT = Gauge(
    "multimodal_rag_ingest_in_flight",
    "Ingest events currently being processed.",
)
INGEST_DURATION_SECONDS = Histogram(
    "multimodal_rag_ingest_duration_seconds",
    "Full duration of one ingest event (source: minio_parse | mongo_text | none).",
    ["source"],
    buckets=_INGEST_BUCKETS,
)
INGEST_STAGE_DURATION_SECONDS = Histogram(
    "multimodal_rag_ingest_stage_duration_seconds",
    "Duration of one ingest stage (delete). Timing of the vector pipeline stages (parse/describe/chunk/embed) lives in the DocStatusStore registry, not Prometheus.",
    ["stage"],
    buckets=_INGEST_BUCKETS,
)
INGEST_PARSE_FAILURES_TOTAL = Counter(
    "multimodal_rag_ingest_parse_failures_total",
    "Document parse failures.",
    ["document_type"],
)
INGEST_CRAWL_IMAGE_FAILURES_TOTAL = Counter(
    "multimodal_rag_ingest_crawl_image_failures_total",
    "Crawl ('Hình ảnh' section) image download/upload failures (reason: http | redirect | connect | ssl | minio | other).",
    ["reason"],
)
INGEST_ITEMS_TOTAL = Counter(
    "multimodal_rag_ingest_items_total",
    "Content-list items ingested per modality.",
    ["modality"],
)
INGEST_PAGES_TOTAL = Counter(
    "multimodal_rag_ingest_pages_total",
    "Pages ingested.",
)
INGEST_STREAM_PENDING = Gauge(
    "multimodal_rag_ingest_stream_pending",
    "Ingest stream messages delivered but not yet acked.",
)
INGEST_STREAM_LAG = Gauge(
    "multimodal_rag_ingest_stream_lag",
    "Ingest stream messages not yet delivered to the consumer group.",
)
INGEST_DOCUMENTS = Gauge(
    "multimodal_rag_ingest_documents",
    "Documents in the ingest registry by status.",
    ["status"],
)
INGEST_EMBED_FAILURES_TOTAL = Counter(
    "multimodal_rag_ingest_embed_failures_total",
    "Embed failures in the ingest embed stage "
    "(reason: unreachable | auth | input_too_long | oom | model_mismatch | circuit_open | other).",
    ["reason"],
)
INGEST_EMBED_ITEMS_TOTAL = Counter(
    "multimodal_rag_ingest_embed_items_total",
    "Items embedded during ingest (denominator of the embed cache hit rate).",
)
INGEST_EMBED_CACHE_HITS_TOTAL = Counter(
    "multimodal_rag_ingest_embed_cache_hits_total",
    "Ingest embed items already in the embedding cache (numerator of the hit rate).",
)

# --- OpenTelemetry ---------------------------------------------------------


def init_telemetry() -> None:
    """Initialise the OTel TracerProvider (OTLP/HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT``).

    Without the env var no exporter is registered and spans are no-ops.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        # When ``endpoint`` is passed to the constructor the exporter uses it verbatim,
        # so the /v1/traces path must be appended here (Tempo returns 404 otherwise).
        url = normalize_otlp_traces_url(endpoint)
        exporter = OTLPSpanExporter(endpoint=url)
        # The exporter's requests.Session honours HTTP(S)_PROXY, and a corporate proxy
        # answers 404 for host.docker.internal/NodePort targets. Tempo is reachable directly.
        session = getattr(exporter, "_session", None)
        if session is not None:
            session.trust_env = False
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OTel tracing enabled, exporting OTLP/HTTP to %s", url)
    else:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set - tracing is a no-op")
    trace.set_tracer_provider(provider)


def normalize_otlp_traces_url(endpoint: str) -> str:
    base = endpoint.rstrip("/").removesuffix("/v1/traces")
    return f"{base}/v1/traces"


tracer = trace.get_tracer("multimodal-rag-retrieval")

# Safety net against the ~4MB OTLP request limit, not a display limit. 0 disables it.
_DEFAULT_MAX_EVENT_ATTR_CHARS = 2_000_000


def _parse_max_event_attr_chars() -> int:
    raw = os.environ.get("TRACE_OUTPUT_MAX_LEN", "").strip()
    if not raw:
        return _DEFAULT_MAX_EVENT_ATTR_CHARS
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid TRACE_OUTPUT_MAX_LEN=%r; falling back to %d",
            raw,
            _DEFAULT_MAX_EVENT_ATTR_CHARS,
        )
        return _DEFAULT_MAX_EVENT_ATTR_CHARS


def _log_documents_enabled() -> bool:
    """``TRACE_OUTPUT_DOCUMENTS=0`` tắt payload document trong span event + log
    (còn num_docs/duration/trace_id để join Tempo) - cắt volume log/trace khi
    dashboard không cần xem nội dung document."""
    raw = os.environ.get("TRACE_OUTPUT_DOCUMENTS", "").strip().lower()
    return raw not in ("0", "false", "no")


_MAX_EVENT_ATTR_CHARS = _parse_max_event_attr_chars()


def _record_request_counters(mode: str) -> None:
    REQUESTS_TOTAL.labels(mode=mode).inc()
    now = datetime.now(DASHBOARD_TIMEZONE)
    day = now.strftime("%Y-%m-%d")
    hod = now.strftime("%H")
    REQUESTS_BY_DAY.labels(day=day).inc()
    REQUESTS_BY_HOUR_OF_DAY.labels(hod=hod).inc()
    REQUESTS_BY_DAY_HOUR.labels(day=day, hod=hod).inc()


def _truncate_documents_json(payload: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """Serialise the payload; if over the limit, shrink ``page_content`` but keep valid JSON."""
    limit = _MAX_EVENT_ATTR_CHARS if max_chars is None else max_chars
    docs_json = json.dumps(payload, ensure_ascii=False, default=str)
    if limit <= 0 or len(docs_json) <= limit or not payload:
        return docs_json
    for per_doc in (2000, 1000, 500, 200, 100):
        shrunk = [
            {**doc, "text": doc["text"][:per_doc] + "...[truncated]"}
            if len(doc["text"]) > per_doc
            else doc
            for doc in payload
        ]
        docs_json = json.dumps(shrunk, ensure_ascii=False, default=str)
        if len(docs_json) <= limit:
            return docs_json
    first = payload[0]
    return json.dumps(
        [{**first, "text": first["text"][:100] + "...[truncated]"}],
        ensure_ascii=False,
        default=str,
    )


def _docs_payload(documents: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for doc in documents:
        item = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
        payload.append(
            {
                "text": item.get("text") or "",
                "score": item.get("score"),
                "metadata": item.get("metadata"),
            }
        )
    return payload


class RetrievalTracker:
    """Handle yielded by :func:`track_retrieval` to report the final documents."""

    def __init__(self, span: Span, started: float) -> None:
        self.span = span
        self._started = started

    def set_documents(self, documents: list[Any]) -> None:
        DOCS_RETURNED.observe(len(documents))
        self.span.set_attribute("output.num_docs", len(documents))
        ctx = self.span.get_span_context()
        extra: dict[str, Any] = {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
        if _log_documents_enabled():
            payload = _docs_payload(documents)
            self.span.add_event(
                "retrieval.output",
                attributes={"documents_json": _truncate_documents_json(payload)},
            )
            extra["documents"] = payload
        logger.info(
            "rag_retrieval_output num_docs=%d duration=%.4fs",
            len(documents),
            time.perf_counter() - self._started,
            extra=extra,
        )


@contextmanager
def track_retrieval(
    query: str, mode: str, version: int, top_k: int | None
) -> Iterator[RetrievalTracker]:
    """Record request counters, in-flight gauge, duration, errors and the root span."""
    _record_request_counters(mode)
    REQUESTS_IN_FLIGHT.inc()
    attributes: dict[str, str | int] = {
        "input.query": query,
        "input.mode": mode,
        "input.version": version,
    }
    if top_k is not None:
        attributes["input.top_k"] = top_k
    span = tracer.start_span("rag.retrieval", attributes=attributes)
    started = time.perf_counter()
    try:
        # Make the root span current so stage spans nest under it.
        with trace.use_span(span, end_on_exit=False):
            yield RetrievalTracker(span, started)
    except BaseException as exc:
        ERRORS_TOTAL.labels(mode=mode).inc()
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
    else:
        span.set_status(Status(StatusCode.OK))
    finally:
        DURATION_SECONDS.labels(mode=mode).observe(time.perf_counter() - started)
        REQUESTS_IN_FLIGHT.dec()
        span.end()


@contextmanager
def observe_stage(stage: str) -> Iterator[Span]:
    """Time one pipeline stage and emit a child span ``rag.<stage>``."""
    started = time.perf_counter()
    with tracer.start_as_current_span(f"rag.{stage}") as span:
        try:
            yield span
        finally:
            STAGE_DURATION_SECONDS.labels(stage=stage).observe(time.perf_counter() - started)


@contextmanager
def observe_ingest_stage(stage: str) -> Iterator[Span]:
    """Time one ingest stage and emit a child span ``ingest.<stage>``."""
    started = time.perf_counter()
    with tracer.start_as_current_span(f"ingest.{stage}") as span:
        try:
            yield span
        finally:
            INGEST_STAGE_DURATION_SECONDS.labels(stage=stage).observe(time.perf_counter() - started)
