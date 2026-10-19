import pytest

from ami_rag.queue.events import RagEvent
from ami_rag.sources import source_hash
from ami_rag.workers.ingest_worker import IngestWorker

from .conftest import FakeDocsRepo, FakeEmbedder, FakeParser, FakeVectorStore

PDF_ID = "64b000000000000000000001"
TEXT_ID = "64b000000000000000000005"
TXT_ID = "64b000000000000000000006"


@pytest.fixture
def runner(fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, fake_docs_repo):
    from ami_rag.core.vector_pipeline import VectorPipeline
    from ami_rag.settings import Settings

    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        CHUNK_SIZE=64,
        CHUNK_OVERLAP=8,
        PARSER="mineru",
        PARSE_METHOD="auto",
        MINERU_BACKEND="pipeline",
    )
    return VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )


def _worker(runner, fake_docs_repo, fake_status_store, fake_queue, asset_store, max_delivery=3):
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
        docs_repo=fake_docs_repo,
        state_repo=fake_status_store,
        queue=fake_queue,
        asset_store=asset_store,
        settings=_S(),
    )


async def test_created_event_runs_pipeline_and_indexes(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store, fake_vector_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    result = await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID))

    assert result is not None
    assert result["source"] == "mongo_text"
    assert result["source_hash"] == source_hash(fake_docs_repo.docs[TEXT_ID])
    assert result["chunk_count"] > 0
    assert fake_asset_store.fetched == []  # mongo_text: không fetch file
    assert fake_vector_store.points  # vector đã upsert
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == "indexed"
    assert row["stage"] == "indexed"
    assert row["chunk_count"] == result["chunk_count"]
    assert row["embed_model"] == "Qwen/Qwen3-VL-Embedding-2B"


async def test_minio_parse_fetches_file_and_stores_content_list(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    await worker.handle_event(RagEvent(event="created", document_id=PDF_ID))

    assert fake_asset_store.fetched == ["documents/HV/a.pdf"]
    assert fake_asset_store.load_content_list(PDF_ID) is not None
    parses = [c for c in runner.parser.calls if c[0] == "parse"]
    assert len(parses) == 1 and parses[0][2] == "auto"


async def test_unchanged_indexed_source_skipped(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID))
    row = fake_status_store.get(TEXT_ID)

    # re-deliver cùng event: đã indexed + cùng content_hash -> skip
    assert await worker.handle_event(RagEvent(event="updated", document_id=TEXT_ID)) is None
    assert fake_status_store.get(TEXT_ID) == row


async def test_deleted_event_purges_vectors_assets_and_row(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store, fake_vector_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID))
    assert fake_vector_store.points

    await worker._process_message("1-1", RagEvent(event="deleted", document_id=TEXT_ID))

    assert fake_vector_store.points == {}
    assert TEXT_ID in fake_asset_store.deleted_assets
    assert fake_status_store.get(TEXT_ID) is None
    assert "1-1" in fake_queue.acked


async def test_missing_document_skipped(
    runner, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, FakeDocsRepo(), fake_status_store, fake_queue, fake_asset_store)
    result = await worker.handle_event(
        RagEvent(event="created", document_id="64b000000000000000000009")
    )
    assert result is None
    assert fake_queue.acked == []


async def test_empty_text_content_skipped(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    fake_docs_repo.docs[TEXT_ID]["content"] = "   "
    assert await worker.handle_event(RagEvent(event="created", document_id=TEXT_ID)) is None


async def test_pdf_without_file_path_fails_after_max_delivery(
    fake_status_store, fake_queue, fake_asset_store, fake_docs_repo
):
    from ami_rag.core.vector_pipeline import VectorPipeline
    from ami_rag.settings import Settings

    fake_docs_repo.docs[PDF_ID]["file_path"] = None
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        CHUNKER_VERSION="v1",
        PARSER="mineru",
    )
    runner = VectorPipeline(
        settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store, max_delivery=2)
    event = RagEvent(event="created", document_id=PDF_ID)
    await worker._process_message("1-1", event)
    assert "1-1" not in fake_queue.acked
    await worker._process_message("1-1", event)
    row = fake_status_store.get(PDF_ID)
    assert row["status"] == "failed"
    assert row["attempts"] == 2  # 1 lần +1 mỗi delivery, không đếm đôi
    assert "file_path" in row["error"]
    assert "1-1" in fake_queue.acked


async def test_failure_marks_failed_and_retries_until_max_delivery(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    async def boom(doc_id, *, from_stage=None, dry_run=False):
        raise RuntimeError("embed down")

    runner.run = boom
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store, max_delivery=2)
    event = RagEvent(event="created", document_id=TEXT_ID)

    await worker._process_message("1-1", event)
    assert "1-1" not in fake_queue.acked
    assert fake_status_store.get(TEXT_ID)["status"] == "failed"

    await worker._process_message("1-1", event)
    assert "1-1" in fake_queue.acked
    assert fake_status_store.get(TEXT_ID)["status"] == "failed"
    assert fake_status_store.get(TEXT_ID)["attempts"] == 2


async def test_skipped_event_releases_attempt(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    # unknown document: begin_attempt tạo row, skip phải drop lại
    missing = "64b000000000000000000009"
    await worker._process_message("1-1", RagEvent(event="created", document_id=missing))
    assert fake_status_store.get(missing) is None
    assert "1-1" in fake_queue.acked


async def test_worker_passes_mineru_kwargs_to_parse(
    runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store
):
    worker = _worker(runner, fake_docs_repo, fake_status_store, fake_queue, fake_asset_store)
    await worker.handle_event(RagEvent(event="created", document_id=PDF_ID))
    parses = runner.parser.calls
    assert parses and parses[0][2] == "auto"
