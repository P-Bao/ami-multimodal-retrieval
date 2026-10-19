import json
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from ami_rag import observability as obs
from ami_rag.api.routes import rag as rag_routes

from .test_api import app, client  # noqa: F401  (fixtures)
from .test_vector_pipeline import pipeline  # noqa: F401  (fixture: pipeline w/ parser)

P = "multimodal_rag_retrieval_"
BODY = {"messages": [{"role": "user", "content": "học phí"}]}


def _val(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(P + name, labels or None) or 0.0


async def test_search_records_request_metrics(client):  # noqa: F811
    before = {
        "req": _val("requests_total", mode="vector"),
        "dur": _val("duration_seconds_count", mode="vector"),
        "docs": _val("docs_returned_count"),
        "chunks": _val("chunks_retrieved_count"),
        "embed": _val("stage_duration_seconds_count", stage="embed_query"),
        "search": _val("stage_duration_seconds_count", stage="vector_search"),
        "rerank": _val("stage_duration_seconds_count", stage="rerank"),
    }
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert _val("requests_total", mode="vector") == before["req"] + 1
    assert _val("duration_seconds_count", mode="vector") == before["dur"] + 1
    assert _val("docs_returned_count") == before["docs"] + 1
    assert _val("chunks_retrieved_count") == before["chunks"] + 1
    assert _val("stage_duration_seconds_count", stage="embed_query") == before["embed"] + 1
    assert _val("stage_duration_seconds_count", stage="vector_search") == before["search"] + 1
    assert _val("stage_duration_seconds_count", stage="rerank") == before["rerank"] + 1
    assert _val("requests_in_flight") == 0


async def test_stream_endpoint_counts_as_request(client):  # noqa: F811
    before = _val("requests_total", mode="vector")
    resp = await client.post("/v2/rag/stream", json=BODY)
    assert resp.status_code == 200
    assert _val("requests_total", mode="vector") == before + 1


async def test_rerank_fallback_on_error(client, app):  # noqa: F811
    from .conftest import FakeRerank

    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank(fail=True)
    before = _val("rerank_fallback_total", reason="error")
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert _val("rerank_fallback_total", reason="error") == before + 1


async def test_exception_counts_error_and_releases_in_flight(client, app):  # noqa: F811
    pipeline = app.dependency_overrides[rag_routes.get_pipeline]()  # noqa: F811
    pipeline.__dict__["embedder"] = _BoomEmbedder()

    before = _val("errors_total", mode="vector")
    before_embed = _val("embed_failures_total", reason="other")
    with pytest.raises(RuntimeError):
        await client.post("/v2/rag/", json=BODY)
    assert _val("errors_total", mode="vector") == before + 1
    # RuntimeError không phải EmbedderError -> rơi vào reason "other" (tập đóng)
    assert _val("embed_failures_total", reason="other") == before_embed + 1
    assert _val("requests_in_flight") == 0


class _BoomEmbedder:
    async def embed_query(self, query):
        raise RuntimeError("embed server down")


class _UnreachableEmbedder:
    """Dense-only embedder, embed loi theo EmbedderError để test label reason."""

    dim = 8

    async def embed_query(self, query):
        from ami_rag.core.embedder import EmbedServerUnreachable

        raise EmbedServerUnreachable("embed server down")


async def test_retrieval_embed_failure_reason_from_embedder_error(client, app):  # noqa: F811
    from ami_rag.core.embedder import EmbedServerUnreachable

    pipeline = app.dependency_overrides[rag_routes.get_pipeline]()  # noqa: F811
    pipeline.__dict__["embedder"] = _UnreachableEmbedder()

    before = _val("embed_failures_total", reason="unreachable")
    with pytest.raises(EmbedServerUnreachable):
        await client.post("/v2/rag/", json=BODY)
    assert _val("embed_failures_total", reason="unreachable") == before + 1


async def test_search_mode_counts_hybrid(client):  # noqa: F811
    before = _val("search_mode_total", mode="hybrid")
    assert (await client.post("/v2/rag/", json=BODY)).status_code == 200
    # FakeEmbedder có embed_query_sparse + sparse không rỗng -> hybrid
    assert _val("search_mode_total", mode="hybrid") == before + 1


async def test_search_mode_counts_dense_for_legacy_embedder(client, app, fake_pipeline):  # noqa: F811
    from .test_api import _LegacyEmbedder

    fake_pipeline.embedder = _LegacyEmbedder()
    before_hybrid = _val("search_mode_total", mode="hybrid")
    before_dense = _val("search_mode_total", mode="dense")
    assert (await client.post("/v2/rag/", json=BODY)).status_code == 200
    assert _val("search_mode_total", mode="dense") == before_dense + 1
    assert _val("search_mode_total", mode="hybrid") == before_hybrid


async def test_artifact_urls_returned_metric(client, app):  # noqa: F811
    from .test_api import _ImageAssetStore

    app.dependency_overrides[rag_routes.get_asset_store] = lambda: _ImageAssetStore()
    before = _val("artifact_urls_returned_total")
    resp = await client.post("/v2/rag/", json=BODY)
    total = sum(len(d["artifact_urls"]) for d in resp.json()["documents"])
    assert total > 0
    assert _val("artifact_urls_returned_total") == before + total


async def test_metrics_endpoint_exposes_series(client):  # noqa: F811
    await client.post("/v2/rag/", json=BODY)
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert P + "requests_total" in resp.text
    assert P + "stage_duration_seconds_bucket" in resp.text


I = "multimodal_rag_ingest_"
PDF_ID = "64b000000000000000000001"
TEXT_ID = "64b000000000000000000005"


def _ival(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(I + name, labels or None) or 0.0


def _ingest_worker(runner, docs_repo, state_repo, queue, asset_store, max_delivery=3):
    from ami_rag.workers.ingest_worker import IngestWorker

    class _S:
        WORKER_MAX_DELIVERY = max_delivery
        WORKER_BATCH = 10
        WORKER_POLL_BLOCK_MS = 10
        RAG_STREAM = "rag:ingest"
        RAG_CONSUMER_GROUP = "ami-rag"
        EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
        CHUNKER_VERSION = "v1"

    return IngestWorker(
        runner=runner,
        docs_repo=docs_repo,
        state_repo=state_repo,
        queue=queue,
        asset_store=asset_store,
        settings=_S(),
    )


async def test_ingest_processed_records_metrics(
    fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    worker = _ingest_worker(
        fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
    )
    before_ev = _ival("events_total", event="created", result="processed")
    before_dur = _ival("duration_seconds_count", source="mongo_text")

    await worker._process_message("1-1", RagEvent(event="created", document_id=TEXT_ID))

    assert _ival("events_total", event="created", result="processed") == before_ev + 1
    assert _ival("duration_seconds_count", source="mongo_text") == before_dur + 1
    assert _ival("in_flight") == 0


async def test_ingest_skipped_event(
    fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    worker = _ingest_worker(
        fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
    )
    before = _ival("events_total", event="created", result="skipped")
    await worker._process_message("1-1", RagEvent(event="created", document_id="missing-doc"))
    assert _ival("events_total", event="created", result="skipped") == before + 1
    assert _ival("in_flight") == 0


async def test_ingest_failure_retry_then_failed(
    fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    from ami_rag.queue.events import RagEvent

    async def boom(doc_id, *, from_stage=None, dry_run=False):
        raise RuntimeError("pipeline down")

    fake_pipeline.run = boom
    worker = _ingest_worker(
        fake_pipeline,
        fake_docs_repo,
        fake_status_store,
        fake_queue,
        fake_asset_store,
        max_delivery=2,
    )
    before_retry = _ival("events_total", event="created", result="retry")
    before_failed = _ival("events_total", event="created", result="failed")
    event = RagEvent(event="created", document_id=TEXT_ID)

    await worker._process_message("1-1", event)
    assert _ival("events_total", event="created", result="retry") == before_retry + 1
    assert _ival("events_total", event="created", result="failed") == before_failed

    await worker._process_message("1-1", event)
    assert _ival("events_total", event="created", result="failed") == before_failed + 1
    assert _ival("in_flight") == 0


async def test_ingest_gauges_updated_and_errors_swallowed(
    fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _ingest_worker(
        fake_pipeline, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
    )

    async def stats():
        return 4, 7

    fake_queue.stats = stats
    fake_status_store.counts = lambda: {"indexed": 2}
    await worker._update_queue_gauges()
    await worker._update_document_gauges()
    assert _ival("stream_pending") == 4
    assert _ival("stream_lag") == 7
    assert _ival("documents", status="indexed") == 2

    fake_status_store.counts = lambda: {"failed": 1}
    await worker._update_document_gauges()
    assert _ival("documents", status="indexed") == 0
    assert _ival("documents", status="failed") == 1

    async def bad_stats():
        raise RuntimeError("redis down")

    fake_queue.stats = bad_stats
    await worker._update_queue_gauges()  # must not raise


async def test_ingest_embed_items_and_cache_hits(
    pipeline, fake_status_store  # noqa: F811
):
    fake_status_store.ensure_pending(TEXT_ID)
    before_items = _ival("embed_items_total")
    before_hits = _ival("embed_cache_hits_total")

    outcome = await pipeline.run(TEXT_ID)

    assert outcome.ok
    chunk_count = fake_status_store.get(TEXT_ID)["chunk_count"]
    assert chunk_count > 0
    assert _ival("embed_items_total") == before_items + chunk_count
    assert _ival("embed_cache_hits_total") == before_hits  # FakeEmbedder.count_cache_hits -> 0


async def test_ingest_embed_failure_reason(pipeline, fake_status_store):  # noqa: F811
    from ami_rag.core.embedder import EmbedServerUnreachable

    class _Unreachable:
        dim = 8

        async def embed_documents_sparse(self, items):
            raise EmbedServerUnreachable("embed server down")

        def count_cache_hits(self, items):
            return 0

    fake_status_store.ensure_pending(TEXT_ID)
    pipeline.embedder = _Unreachable()
    before = _ival("embed_failures_total", reason="unreachable")

    outcome = await pipeline.run(TEXT_ID)

    assert outcome.ok is False
    assert _ival("embed_failures_total", reason="unreachable") == before + 1


def test_truncate_documents_json_keeps_valid_json():
    payload = [{"text": "x" * 5000, "score": 0.9, "metadata": {}} for _ in range(10)]
    out = obs._truncate_documents_json(payload, max_chars=6000)
    assert len(out) <= 6000
    assert len(json.loads(out)) == 10
    assert obs._truncate_documents_json(payload, max_chars=0) == json.dumps(
        payload, ensure_ascii=False
    )


def test_normalize_otlp_traces_url():
    for raw in ("http://tempo:4318", "http://tempo:4318/", "http://tempo:4318/v1/traces"):
        assert obs.normalize_otlp_traces_url(raw) == "http://tempo:4318/v1/traces"


def test_dashboard_metrics_exist_in_code():
    path = (
        Path(__file__).resolve().parents[2]
        / "monitoring/dashboards/multimodal-rag-retrieval-dashboard.json"
    )
    dash = json.loads(path.read_text(encoding="utf-8"))
    assert dash["spec"]["title"] == "Multimodal RAG Retrieval"
    text = path.read_text(encoding="utf-8")
    import re

    # Families (not samples): labeled metrics that were never observed export no sample yet.
    exported = set()
    for fam in REGISTRY.collect():
        exported.add(fam.name)
        exported.update(fam.name + sfx for sfx in ("_total", "_bucket", "_count", "_sum"))
    used = set(re.findall(r"multimodal_rag_(?:retrieval|ingest)_[a-z_]+", text))
    assert used, "dashboard uses no metrics"
    missing = {u for u in used if u not in exported}
    assert not missing, f"dashboard references unknown metrics: {missing}"


def test_alert_rule_metrics_exist_in_code():
    import re

    path = Path(__file__).resolve().parents[2] / "monitoring/helm/prometheusrule.yaml"
    text = path.read_text(encoding="utf-8")
    exported = set()
    for fam in REGISTRY.collect():
        exported.add(fam.name)
        exported.update(fam.name + sfx for sfx in ("_total", "_bucket", "_count", "_sum"))
    used = set(re.findall(r"multimodal_rag_(?:retrieval|ingest)_[a-z_]+", text))
    assert used, "alert rules use no metrics"
    missing = {u for u in used if u not in exported}
    assert not missing, f"alert rules reference unknown metrics: {missing}"


def test_embed_failure_alerts_present():
    path = Path(__file__).resolve().parents[2] / "monitoring/helm/prometheusrule.yaml"
    text = path.read_text(encoding="utf-8")
    assert "alert: RagEmbedFailures" in text
    assert "alert: RagIngestEmbedFailures" in text
