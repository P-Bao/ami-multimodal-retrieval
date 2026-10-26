"""Test VectorPipeline (fake embedder/vector_store/asset_store/DocsStatusStore)."""

import asyncio
import base64
from pathlib import Path

import pytest

from ami_rag.core.vector_pipeline import (
    ArtifactPaths,
    VectorPipeline,
    chunk_text,
)
from ami_rag.settings import Settings, resolve_image_describe_mode
from ami_rag.storage.doc_status import STATUS_INDEXED
from raganything.backend_data_extract import BackendDataParseError
from tests.ami_service.conftest import (
    FakeEmbedder,
    FakeParser,
    FakeVectorStore,
)

TEXT_ID = "64b000000000000000000005"
PDF_ID = "64b000000000000000000001"
CRAWL_ID = "64b000000000000000000007"

CRAWL_URLS = [
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04608-scaled.jpg",
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04739-scaled.jpg",
]
CRAWL_CONTENT = (
    "Bài crawl PTIT.\n\n"
    "Nội dung chính.\n\n"
    "**Hình ảnh:**\n\n"
    f"- {CRAWL_URLS[0]}\n"
    f"- {CRAWL_URLS[1]}\n"
)


def _add_crawl_doc(fake_docs_repo, fake_status_store):
    fake_docs_repo.docs[CRAWL_ID] = {
        "_id": CRAWL_ID,
        "title": "Crawl PTIT",
        "content": CRAWL_CONTENT,
        "document_type": "news",
        "original_file_name": None,
        "file_path": None,
        "organization_unit_id": "64b000000000000000000002",
        "metadata": {"source_url": "https://ptit.edu.vn/tin/le-tot-nghiep"},
    }
    fake_status_store.ensure_pending(CRAWL_ID)


@pytest.fixture
def pipeline(
    fake_embedder,
    fake_vector_store,
    fake_asset_store,
    fake_status_store,
    fake_docs_repo,
    fake_modal_processors,
):
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
        PARSE_TEXT_SOURCE="mineru",
    )
    return VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
        modal_processors=fake_modal_processors,
    )


@pytest.fixture(autouse=True)
def _seed_pending(fake_status_store):
    fake_status_store.ensure_pending(TEXT_ID)
    fake_status_store.ensure_pending(PDF_ID)


async def test_full_run_mongo_text(
    pipeline, fake_status_store, fake_vector_store, fake_asset_store
):
    outcome = await pipeline.run(TEXT_ID)

    assert outcome.ok and outcome.stage == "indexed"
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == STATUS_INDEXED
    assert row["chunk_count"] == outcome.chunk_count
    assert row["embed_model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert row["embed_dim"] == 8
    assert row["chunker_version"] == "v1"
    assert fake_vector_store.points
    assert all(p["doc_id"] == TEXT_ID for p in fake_vector_store.points.values())
    # artifacts ở MinIO
    assert fake_asset_store.load_content_list(TEXT_ID) is not None
    chunks_key = ArtifactPaths.chunks_key(fake_asset_store, TEXT_ID)
    assert fake_asset_store.json_objects.get(chunks_key) is not None
    # parse: text doc không fetch file
    assert fake_asset_store.fetched == []


async def test_full_run_describes_multimodal(
    pipeline, fake_status_store, fake_modal_processors, fake_asset_store
):
    await pipeline.run(PDF_ID)

    all_calls = []
    for proc in fake_modal_processors.values():
        all_calls.extend(proc.calls)
    assert all_calls, "describe stage phải gọi modal processor cho item multimodal"
    row = fake_status_store.get(PDF_ID)
    assert row["status"] == STATUS_INDEXED
    # descriptions.json đã lưu
    desc_key = ArtifactPaths.descriptions_key(fake_asset_store, PDF_ID)
    assert isinstance(fake_asset_store.json_objects.get(desc_key), list)


async def test_run_is_idempotent_no_duplicate_vectors(
    pipeline, fake_vector_store, fake_status_store
):
    await pipeline.run(TEXT_ID)
    count_after_first = len(fake_vector_store.points)
    await pipeline.run(TEXT_ID)  # stage == indexed -> early return, không upsert lại
    assert len(fake_vector_store.points) == count_after_first


async def test_resume_from_embed_reuses_chunks_json(
    pipeline, fake_status_store, fake_vector_store
):
    await pipeline.run(TEXT_ID)
    count_first = len(fake_vector_store.points)
    chunk_count = fake_status_store.get(TEXT_ID)["chunk_count"]

    # xoá vector nhưng giữ artifacts -> resume từ embed không gọi parse lại
    await fake_vector_store.delete_by_doc(pipeline.collection, TEXT_ID)
    outcome = await pipeline.run(TEXT_ID, from_stage="embed")

    assert outcome.ok
    assert outcome.chunk_count == chunk_count
    assert len(fake_vector_store.points) == count_first
    assert fake_status_store.get(TEXT_ID)["status"] == STATUS_INDEXED


async def test_resume_from_embed_builds_chunks_when_missing(
    pipeline, fake_asset_store, fake_status_store, fake_vector_store
):
    await pipeline.run(TEXT_ID)
    chunk_count = fake_status_store.get(TEXT_ID)["chunk_count"]

    # xoá chunks.json -> resume từ embed phải build lại từ content_list+descriptions
    fake_asset_store.json_objects.pop(
        ArtifactPaths.chunks_key(fake_asset_store, TEXT_ID), None
    )
    await fake_vector_store.delete_by_doc(pipeline.collection, TEXT_ID)

    outcome = await pipeline.run(TEXT_ID, from_stage="embed")
    assert outcome.ok
    assert outcome.chunk_count == chunk_count
    assert len(fake_vector_store.points) == chunk_count


async def test_stage_embed_replaces_old_vectors(
    pipeline, fake_vector_store, fake_status_store
):
    await pipeline.run(TEXT_ID)  # full: parse -> describe -> chunk -> embed
    n = len(fake_vector_store.points)
    # chạy lại chunk stage: delete_by_doc trước upsert -> không tăng số vector
    outcome = await pipeline.run(TEXT_ID, from_stage="chunk")
    assert outcome.ok
    assert len(fake_vector_store.points) == n


async def test_dry_run_no_embed_no_writes(
    pipeline, fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    await pipeline.run(TEXT_ID)  # full: tạo artifacts + vector
    points_before = dict(fake_vector_store.points)
    texts_before = list(fake_embedder.embedded_texts)
    status_before = fake_status_store.get(TEXT_ID)

    outcome = await pipeline.run(TEXT_ID, from_stage="embed", dry_run=True)

    assert outcome.ok
    assert outcome.chunk_count > 0
    assert outcome.cache_hits == 0  # FakeEmbedder không cache
    assert fake_vector_store.points == points_before
    assert fake_embedder.embedded_texts == texts_before
    assert fake_status_store.get(TEXT_ID) == status_before


async def test_run_without_pending_row_raises(pipeline):
    from ami_rag.storage.doc_status import DocStatusStore
    from tests.ami_service.fakes import FakeMongoCollection

    empty_store = DocStatusStore.__new__(DocStatusStore)
    empty_store._col = FakeMongoCollection()
    pipeline.store = empty_store
    with pytest.raises(ValueError):
        await pipeline.run("64b000000000000000000099")


async def test_parse_without_doc_raises_clear_error(
    fake_asset_store, fake_status_store, fake_docs_repo
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        CHUNKER_VERSION="v1",
        PARSER="mineru",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=None,  # không có docs_repo -> không parse được
        parser=FakeParser(),
    )
    fake_status_store.ensure_pending(TEXT_ID)
    outcome = await pipeline.run(TEXT_ID, from_stage="parse")
    assert not outcome.ok
    assert "documents" in outcome.error
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == "failed"
    assert row["error_stage"] == "parse"


def _ingest_val(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels or None) or 0.0


async def test_stage_parse_counts_items_and_pages_metrics(pipeline):
    """_stage_parse phải tăng items_total{modality} + pages_total (metric revive)."""
    before = {
        "text": _ingest_val("multimodal_rag_ingest_items_total", modality="text"),
        "image": _ingest_val("multimodal_rag_ingest_items_total", modality="image"),
        "table": _ingest_val("multimodal_rag_ingest_items_total", modality="table"),
        "pages": _ingest_val("multimodal_rag_ingest_pages_total"),
    }
    await pipeline.run(PDF_ID)
    assert (
        _ingest_val("multimodal_rag_ingest_items_total", modality="text")
        == before["text"] + 1
    )
    assert (
        _ingest_val("multimodal_rag_ingest_items_total", modality="image")
        == before["image"] + 1
    )
    assert (
        _ingest_val("multimodal_rag_ingest_items_total", modality="table")
        == before["table"] + 1
    )
    # FakeParser: text/image/table ở page_idx 0/1/2 -> 3 trang
    assert _ingest_val("multimodal_rag_ingest_pages_total") == before["pages"] + 3


async def test_parse_failure_increments_parse_failures_metric(
    fake_asset_store, fake_status_store
):
    """Parse lỗi phải tăng parse_failures_total{document_type} (revive alert)."""
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        CHUNKER_VERSION="v1",
        PARSER="mineru",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=None,  # không có docs_repo -> parse raise
        parser=FakeParser(),
    )
    fake_status_store.ensure_pending(TEXT_ID)
    before = _ingest_val(
        "multimodal_rag_ingest_parse_failures_total", document_type="unknown"
    )
    outcome = await pipeline.run(TEXT_ID, from_stage="parse")
    assert not outcome.ok
    assert (
        _ingest_val(
            "multimodal_rag_ingest_parse_failures_total", document_type="unknown"
        )
        == before + 1
    )


async def test_vector_store_error_marks_failed(
    pipeline, fake_status_store, fake_vector_store
):
    async def boom(collection, doc_id):
        raise RuntimeError("qdrant down")

    fake_vector_store.delete_by_doc = boom
    outcome = await pipeline.run(TEXT_ID, from_stage="embed")
    assert not outcome.ok
    row = fake_status_store.get(TEXT_ID)
    assert row["status"] == "failed"


async def test_delete_doc_purges_all(
    pipeline, fake_status_store, fake_vector_store, fake_asset_store
):
    await pipeline.run(TEXT_ID)
    assert fake_vector_store.points

    await pipeline.delete_doc(TEXT_ID)

    assert fake_vector_store.points == {}
    assert TEXT_ID in fake_asset_store.deleted_assets
    assert fake_status_store.get(TEXT_ID) is None


async def test_preflight_embed(pipeline, fake_vector_store):
    await pipeline.preflight_embed()
    assert pipeline.collection in fake_vector_store.collections


def test_chunk_text_budget_and_overlap():
    text = "\n\n".join(f"đoạn {i} " + "nội dung " * 10 for i in range(20))
    chunks = chunk_text(text, max_tokens=64, overlap_tokens=8)
    assert len(chunks) > 1
    assert all(len(c) <= 512 for c in chunks)  # max_chars = max(512, 64*4)


def test_chunk_text_packs_many_lines_not_one_line_per_chunk():
    """1 đoạn dài nhiều dòng -> gom nhiều dòng/chunk, không tách từng dòng."""
    line = "dòng nội dung của đoạn văn bản " * 4  # ~130 chars
    text = "\n".join(f"{i}: {line}" for i in range(40))
    chunks = chunk_text(text, max_tokens=1200, overlap_tokens=0, min_chars=0)
    assert len(chunks) < 5, "phải gom nhiều dòng, không cắt từng dòng một"
    assert sum(c.count("\n") + 1 for c in chunks) >= 40


def test_chunk_text_no_orphan_chunk_below_min_chars():
    """Chunk 'Địa chỉ: ...' trơ trọi bị gộp vào chunk kề."""
    head = "A" * 4000
    text = f"{head}\n\n- Địa chỉ: Tầng 1 nhà A1, Cơ sở đào tạo Hà Đông"
    chunks = chunk_text(text, max_tokens=1200, overlap_tokens=0, min_chars=200)
    assert all(len(c) >= 200 for c in chunks), [len(c) for c in chunks]
    assert any("Địa chỉ: Tầng 1 nhà A1" in c for c in chunks)


def test_merge_small_absorbs_trailing_orphan():
    """Chunk cuối quá ngắn (chunk trước đã đầy) -> gộp đuôi dù hơi vượt budget."""
    from ami_rag.core.vector_pipeline import _merge_small

    out = _merge_small(["B" * 300, "- Địa chỉ: nhà A1"], max_chars=512, min_chars=200)
    assert len(out) == 1
    assert out[0].endswith("- Địa chỉ: nhà A1")


def test_chunk_text_min_chars_keeps_small_document_intact():
    """Văn bản nhỏ hơn ngưỡng thì giữ nguyên 1 chunk (không gộp bừa)."""
    chunks = chunk_text("Địa chỉ: nhà A1", max_tokens=1200, overlap_tokens=0)
    assert chunks == ["Địa chỉ: nhà A1"]


def test_chunk_text_blank_line_still_separates_paragraphs():
    a = "Đoạn một " * 30
    b = "Đoạn hai " * 30
    chunks = chunk_text(f"{a}\n\n{b}", max_tokens=1200, overlap_tokens=0)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0]


def test_chunk_text_empty_returns_original():
    assert chunk_text("") == [""]
    assert chunk_text("   ") == ["   "]


async def test_mark_stage_meta_preserved_through_mark_indexed(
    pipeline, fake_status_store
):
    await pipeline.run(PDF_ID)
    row = fake_status_store.get(PDF_ID)
    assert row["source"] == "minio_parse"
    assert row["parser"] == "mineru"
    assert row["file_path"] == "documents/HV/a.pdf"
    assert row["page_count"] == 3
    assert row["counts"]["image"] == 1


async def test_mark_indexed_writes_document_oid_for_lookup(pipeline, fake_status_store):
    """docs §3: registry indexed mang document_oid (ObjectId) để $lookup."""
    from bson import ObjectId

    await pipeline.run(PDF_ID)
    row = fake_status_store.get(PDF_ID)
    assert row["document_oid"] == ObjectId(PDF_ID)


# --- Crawl: section 'Hình ảnh' -> 1 text item kèm artifact_urls (không tải ảnh) ---


async def test_stage_parse_crawl_artifact_urls_no_download(
    pipeline, fake_docs_repo, fake_status_store, fake_asset_store
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)
    content_list = intermediate.content_list

    # 1 text item duy nhất: text đã strip section ảnh, URL gốc nằm trong artifact_urls
    assert content_list == [
        {
            "type": "text",
            "text": "Bài crawl PTIT.\n\nNội dung chính.",
            "page_idx": 0,
            "artifact_urls": CRAWL_URLS,
        }
    ]
    # KHÔNG tải ảnh: không fetch file gốc, không upload asset nào lên MinIO
    assert fake_asset_store.fetched == []
    assert ("upload_assets", CRAWL_ID) not in fake_asset_store.calls


async def test_stage_parse_crawl_empty_strip_keeps_original_text(
    pipeline, fake_docs_repo, fake_status_store
):
    # Bài chỉ có section 'Hình ảnh' -> strip rỗng -> giữ text gốc (URL trong
    # text, không mất nội dung khỏi index) vẫn kèm artifact_urls.
    fake_docs_repo.docs[CRAWL_ID] = {
        "_id": CRAWL_ID,
        "title": "Crawl ảnh only",
        "content": "**Hình ảnh:**\n\n" f"- {CRAWL_URLS[0]}\n",
        "document_type": "news",
        "original_file_name": None,
        "file_path": None,
        "metadata": {},
    }

    doc = fake_docs_repo.find_by_id(CRAWL_ID)
    intermediate = await pipeline._stage_parse(CRAWL_ID, doc)

    assert intermediate.content_list == [
        {
            "type": "text",
            "text": f"**Hình ảnh:**\n\n- {CRAWL_URLS[0]}\n",
            "page_idx": 0,
            "artifact_urls": [CRAWL_URLS[0]],
        }
    ]


async def test_stage_parse_text_doc_without_section_single_text_item(
    pipeline, fake_docs_repo, fake_status_store
):
    # doc mongo_text không có section 'Hình ảnh' -> 1 text item, không artifact_urls
    doc = fake_docs_repo.find_by_id(TEXT_ID)
    intermediate = await pipeline._stage_parse(TEXT_ID, doc)

    assert intermediate.content_list == [
        {"type": "text", "text": "nội dung text E", "page_idx": 0}
    ]


async def test_full_run_crawl_doc_indexes_artifact_urls(
    pipeline,
    fake_docs_repo,
    fake_status_store,
    fake_vector_store,
    fake_asset_store,
    fake_embedder,
    fake_modal_processors,
):
    _add_crawl_doc(fake_docs_repo, fake_status_store)

    outcome = await pipeline.run(CRAWL_ID)

    assert outcome.ok and outcome.stage == "indexed"
    row = fake_status_store.get(CRAWL_ID)
    assert row["source"] == "mongo_text"
    assert row["counts"] == {"text": 1}
    payloads = [p for p in fake_vector_store.points.values() if p["doc_id"] == CRAWL_ID]
    assert len(payloads) == outcome.chunk_count
    # URL ảnh gốc đi theo payload chunk (qua Chunk.extra) — API trả thẳng
    assert all(p.get("artifact_urls") == CRAWL_URLS for p in payloads)
    # text chunk không còn URL ảnh trong content
    assert all("NL_04608" not in p["content"] for p in payloads)
    # embed caption-only (BGE-M3): không item dict image_b64 nữa
    last_batch = fake_embedder.embedded_texts[-1]
    assert all(isinstance(it, str) for it in last_batch)
    # không còn item modal -> describe lưu descriptions.json rỗng
    desc_key = ArtifactPaths.descriptions_key(fake_asset_store, CRAWL_ID)
    assert fake_asset_store.json_objects.get(desc_key) == []


async def test_qdrant_vector_store_upsert_batching():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import ChunkRecord, QdrantVectorStore

    store = QdrantVectorStore("http://fake:6333")
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id=f"c_{i}", vector=[0.1] * 8, payload={"idx": i})
        for i in range(150)
    ]
    await store.upsert("test_col", chunks, batch_size=64)
    assert store._client.upsert.call_count == 3
    call1_points = store._client.upsert.call_args_list[0].kwargs["points"]
    call2_points = store._client.upsert.call_args_list[1].kwargs["points"]
    call3_points = store._client.upsert.call_args_list[2].kwargs["points"]
    assert len(call1_points) == 64
    assert len(call2_points) == 64
    assert len(call3_points) == 22


async def test_qdrant_upsert_packs_by_byte_limit():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import (
        ChunkRecord,
        QdrantVectorStore,
        _point_estimate,
    )

    store = QdrantVectorStore("http://fake:6333", max_request_bytes=64 * 1024)
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id=f"b_{i}", vector=[0.1] * 8, payload={"text": "x" * 4096})
        for i in range(32)
    ]
    await store.upsert("test_col", chunks, batch_size=64)
    calls = store._client.upsert.call_args_list
    # payload ~4 KB/point -> batch bị chia theo byte trước khi chạm 64 điểm
    assert len(calls) > 1
    total = 0
    for call in calls:
        points = call.kwargs["points"]
        total += len(points)
        assert sum(_point_estimate(p) for p in points) <= 64 * 1024
        assert len(points) <= 64
    assert total == 32


async def test_qdrant_upsert_single_oversized_point_sent_alone():
    from unittest.mock import MagicMock

    from ami_rag.core.vector_store import ChunkRecord, QdrantVectorStore

    store = QdrantVectorStore("http://fake:6333", max_request_bytes=1024)
    store._client = MagicMock()
    chunks = [
        ChunkRecord(id="small", vector=[0.1] * 8, payload={"text": "ok"}),
        ChunkRecord(id="big", vector=[0.1] * 8, payload={"text": "x" * 8192}),
        ChunkRecord(id="small2", vector=[0.1] * 8, payload={"text": "ok"}),
    ]
    await store.upsert("test_col", chunks, batch_size=10)
    calls = store._client.upsert.call_args_list
    sizes = [len(c.kwargs["points"]) for c in calls]
    # point "big" vượt limit -> gửi riêng, không chặn hai điểm nhỏ
    assert sizes == [1, 1, 1]
    assert calls[1].kwargs["points"][0].payload["chunk_id"] == "big"


def test_build_chunks_with_duplicate_content_assigns_unique_ids(pipeline):
    # 2 text item giống nhau nhưng CÁCH NHAU bởi 1 ảnh -> vẫn 2 chunk text riêng
    # (text liên tiếp được gộp, nên trùng lặp phải nằm giữa 2 ranh giới flush).
    content_list = [
        {"type": "text", "text": "Đại học Bách Khoa", "page_idx": 1},
        {"type": "image", "img_path": "img1.png", "page_idx": 2},
        {"type": "text", "text": "Đại học Bách Khoa", "page_idx": 3},
    ]
    chunks = pipeline._build_chunks("doc_test", content_list, descriptions=[])
    assert len(chunks) == 3
    ids = [c.id for c in chunks]
    assert len(set(ids)) == 3


def test_build_chunks_merges_consecutive_text_items(pipeline):
    """Text item liên tiếp gộp 1 chunk (backend_data tách PDF theo từng dòng)."""
    content_list = [
        {"type": "text", "text": "Dòng 1", "page_idx": 0},
        {"type": "text", "text": "Dòng 2", "page_idx": 0},
        {"type": "text", "text": "Dòng 3", "page_idx": 1},
    ]
    chunks = pipeline._build_chunks("doc_test", content_list, descriptions=[])
    assert len(chunks) == 1
    assert chunks[0].content == "Dòng 1\nDòng 2\nDòng 3"
    assert chunks[0].page_idx == 0
    # trang cuối lưu ở extra để citation biết chunk trải nhiều trang
    assert chunks[0].extra["page_end"] == 1


def test_build_chunks_flushes_around_image_keeps_order(pipeline):
    """Ảnh nằm giữa 2 dải text: thứ tự đọc giữ nguyên (text - ảnh - text)."""
    content_list = [
        {"type": "text", "text": "Dòng 1", "page_idx": 0},
        {"type": "image", "img_path": "img1.png", "page_idx": 0},
        {"type": "text", "text": "Dòng 2", "page_idx": 1},
    ]
    chunks = pipeline._build_chunks("doc_test", content_list, descriptions=[])
    assert [c.modality for c in chunks] == ["text", "image", "text"]
    assert chunks[0].content == "Dòng 1"
    assert chunks[2].content == "Dòng 2"


def test_build_chunks_prefix_doc_title(pipeline):
    content_list = [{"type": "text", "text": "- Địa chỉ: Tầng 1 nhà A1", "page_idx": 0}]
    chunks = pipeline._build_chunks(
        "doc_test",
        content_list,
        descriptions=[],
        doc_title="PHÒNG CHÍNH TRỊ VÀ CÔNG TÁC SINH VIÊN",
    )
    assert chunks[0].content == (
        "PHÒNG CHÍNH TRỊ VÀ CÔNG TÁC SINH VIÊN\n- Địa chỉ: Tầng 1 nhà A1"
    )
    assert chunks[0].extra["title"] == "PHÒNG CHÍNH TRỊ VÀ CÔNG TÁC SINH VIÊN"


def test_build_chunks_no_title_prefix_when_chunk_already_starts_with_title(pipeline):
    content_list = [
        {"type": "text", "text": "# Sơ đồ cơ cấu tổ chức\nNội dung", "page_idx": 0}
    ]
    chunks = pipeline._build_chunks(
        "doc_test", content_list, descriptions=[], doc_title="Sơ đồ cơ cấu tổ chức"
    )
    assert chunks[0].content == "# Sơ đồ cơ cấu tổ chức\nNội dung"


def test_build_chunks_prefix_title_on_image_chunk(pipeline):
    content_list = [{"type": "image", "img_path": "img.png", "page_idx": 3}]
    chunks = pipeline._build_chunks(
        "doc_test", content_list, descriptions=[], doc_title="Sơ đồ khuôn viên Hà Đông"
    )
    assert chunks[0].modality == "image"
    assert chunks[0].content.startswith("Sơ đồ khuôn viên Hà Đông\n[Image Content]")


async def test_stage_parse_backend_data_rebuild(
    fake_embedder,
    fake_vector_store,
    fake_asset_store,
    fake_status_store,
    fake_docs_repo,
    monkeypatch,
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        PARSER="mineru",
        PARSE_TEXT_SOURCE="backend_data",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )

    rebuilt_list = [
        {"type": "text", "text": "text backend_data", "page_idx": 0},
        {"type": "table", "table_body": "<table/>", "page_idx": 0},
        {"type": "image", "img_path": "kept.png", "page_idx": 1},
    ]
    calls = {"rebuild": 0}

    async def fake_rebuild(content_list, file_path, extractor):
        calls["rebuild"] += 1
        return [dict(i) for i in rebuilt_list]

    import raganything.backend_data_extract as bde

    monkeypatch.setattr(bde, "rebuild_content_list", fake_rebuild)

    fake_status_store.ensure_pending(PDF_ID)
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    assert calls["rebuild"] == 1
    saved = fake_asset_store.load_content_list(PDF_ID)
    assert [
        {k: v for k, v in i.items() if k != "asset_key"} for i in saved
    ] == rebuilt_list
    # asset chỉ upload cho ảnh (text/table MinerU đã bị thay)
    assert fake_asset_store.calls.count(("upload_assets", PDF_ID)) == 1
    assert saved[2]["asset_key"] == f"rag-assets/{PDF_ID}/img2.png"


async def test_stage_parse_backend_data_failure_falls_back_to_mineru(
    fake_embedder,
    fake_vector_store,
    fake_asset_store,
    fake_status_store,
    fake_docs_repo,
    monkeypatch,
):
    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        PARSER="mineru",
        PARSE_TEXT_SOURCE="backend_data",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )

    async def boom(content_list, file_path, extractor):
        raise BackendDataParseError("ocr down")

    import raganything.backend_data_extract as bde

    monkeypatch.setattr(bde, "rebuild_content_list", boom)

    fake_status_store.ensure_pending(PDF_ID)
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    saved = fake_asset_store.load_content_list(PDF_ID)
    # fallback: giữ nguyên MinerU content_list
    assert any(i["type"] == "text" and i["text"] == "Giới thiệu" for i in saved)
    assert any(i["type"] == "table" for i in saved)


async def test_prepare_embed_items_caption_only(pipeline, fake_embedder):
    outcome = await pipeline.run(PDF_ID)
    assert outcome.ok
    # Caption-only: BGE-M3 không nhận ảnh — kể cả chunk image/table/equation
    # có asset_key, item embed vẫn luôn là str (c.content), không dict image_b64.
    last_batch = fake_embedder.embedded_texts[-1]
    assert last_batch
    assert all(isinstance(it, str) for it in last_batch)


async def test_prepare_embed_items_caption_only_keeps_modal_content(
    pipeline, fake_vector_store, fake_asset_store, fake_status_store, fake_embedder
):
    from ami_rag.core.vector_pipeline import Chunk

    chunks = [
        Chunk(
            id="c1", content="text", modality="text", page_idx=0, is_multimodal=False
        ),
        Chunk(
            id="c2",
            content="[Table Content]",
            modality="table",
            page_idx=1,
            is_multimodal=True,
            asset_key="t1.png",
        ),
        Chunk(
            id="c3",
            content="[Equation]",
            modality="equation",
            page_idx=2,
            is_multimodal=True,
            asset_key="e1.png",
        ),
        Chunk(
            id="c4",
            content="table no asset",
            modality="table",
            page_idx=3,
            is_multimodal=True,
        ),
    ]
    items = await pipeline._prepare_embed_items(chunks)
    # mọi chunk đều là str caption, đúng thứ tự
    assert items == ["text", "[Table Content]", "[Equation]", "table no asset"]


async def test_stage_embed_upserts_sparse(
    pipeline, fake_status_store, fake_vector_store
):
    seen: list = []
    original_upsert = fake_vector_store.upsert

    async def _spy(collection, records):
        seen.extend(records)
        return await original_upsert(collection, records)

    fake_vector_store.upsert = _spy
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    assert len(seen) == fake_status_store.get(PDF_ID)["chunk_count"]
    for r in seen:
        assert r.sparse is not None
        assert r.sparse == FakeEmbedder._sparse_from_text(r.payload["content"])
        assert r.sparse["values"] == [0.5] * len(r.sparse["indices"])


async def test_stage_embed_sparse_empty_dict_to_none(
    fake_vector_store, fake_asset_store, fake_status_store, fake_docs_repo
):
    """Embedder legacy (không có embed_documents_sparse) -> record.sparse luôn None."""

    class _LegacyEmbedder:
        def __init__(self, dim=8):
            self.dim = dim

        async def verify(self):
            return {"model_name": "legacy", "dim": self.dim}

        async def embed_documents(self, texts):
            return [[0.1] * self.dim for _ in texts]

        async def embed_query(self, text):
            return [0.2] * self.dim

        def count_cache_hits(self, texts):
            return 0

        async def close(self):
            pass

    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        PARSER="mineru",
    )
    pipeline = VectorPipeline(
        settings,
        embedder=_LegacyEmbedder(),
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
        parser=FakeParser(),
    )
    seen: list = []
    original_upsert = fake_vector_store.upsert

    async def _spy(collection, records):
        seen.extend(records)
        return await original_upsert(collection, records)

    fake_vector_store.upsert = _spy
    outcome = await pipeline.run(PDF_ID)

    assert outcome.ok and outcome.stage == "indexed"
    assert len(seen) == fake_status_store.get(PDF_ID)["chunk_count"]
    assert all(r.sparse is None for r in seen)


# ----------------------------------------------------------------------
# Describe ảnh: materialize MinIO -> img_path + caption func (Gemini/vision)
# ----------------------------------------------------------------------


def _image_content_list(
    asset_key="rag-assets/docX/img0.png", caption=None, footnotes=None, page=0
):
    item = {
        "type": "image",
        "img_path": "gone.png",
        "asset_key": asset_key,
        "page_idx": page,
    }
    if caption:
        item["image_caption"] = [caption]
    if footnotes:
        item["image_footnote"] = [footnotes]
    return [
        {
            "type": "text",
            "text": "Cơ cấu tổ chức của Học viện gồm các phòng ban.",
            "page_idx": page,
        },
        item,
    ]


def _describe_settings(**overrides):
    base = {
        "_env_file": None,
        "WORKSPACE": "test",
        "EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "EMBED_DIM": 8,
        "CHUNKER_VERSION": "v1",
    }
    base.update(overrides)
    return Settings(**base)


class _FakeCaptionFunc:
    """Stand-in cho caption func (Gemini/vision): ghi call + đo concurrency."""

    def __init__(
        self,
        response=(
            '{"detailed_description": "Mô tả ảnh từ caption func", '
            '"entity_info": {"entity_name": "Sơ đồ tổ chức", "entity_type": "image", '
            '"summary": "Sơ đồ tổ chức của PTIT"}}'
        ),
        fail=False,
        delay=0.0,
    ):
        self.response = response
        self.fail = fail
        self.delay = delay
        self.calls = []  # (prompt, image_data)
        self._active = 0
        self.max_active = 0

    async def __call__(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("caption func down")
            self.calls.append((prompt, image_data))
            return self.response
        finally:
            self._active -= 1


def _make_pipeline(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store, **settings_kw
):
    return VectorPipeline(
        _describe_settings(**settings_kw),
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        parser=FakeParser(),
        modal_processors={},
    )


def _real_image_processor(caption_func, chunk_size=64):
    """ImageModalProcessor THẬT chạy caption func giả (chứng minh img_path materialize)."""
    from raganything.modalprocessors import ImageModalProcessor

    return ImageModalProcessor(
        lightrag=None,
        modal_caption_func=caption_func,
        global_config={"chunk_token_size": chunk_size},
    )


def test_resolve_image_describe_mode():
    # auto: có CAPTION_LLM_API_KEY -> gemini
    assert (
        resolve_image_describe_mode(_describe_settings(CAPTION_LLM_API_KEY="k"))
        == "gemini"
    )
    # auto: không key nhưng có QWEN_VLM_MODEL -> vision
    assert (
        resolve_image_describe_mode(_describe_settings(QWEN_VLM_MODEL="Qwen/Qwen3-VL"))
        == "vision"
    )
    # auto: thiếu cả hai -> none (text thô, không LLM)
    assert resolve_image_describe_mode(_describe_settings()) == "none"
    # giá trị rõ được dùng luôn
    for mode in ("gemini", "vision", "none"):
        assert (
            resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE=mode))
            == mode
        )
    # legacy "text" (path OCR-ảnh đã xoá) -> "none"
    assert (
        resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE="text"))
        == "none"
    )
    with pytest.raises(ValueError):
        resolve_image_describe_mode(_describe_settings(IMAGE_DESCRIBE_MODE="bogus"))


async def test_describe_image_via_real_image_processor(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Materialize MinIO -> img_path thật -> ImageModalProcessor THẬT gửi đúng base64."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT")
    descriptions = await pipeline._stage_describe("docX", content_list)

    assert descriptions[0]["type"] == "image"
    # JSON response được _parse_response của ImageModalProcessor parse ra description
    assert descriptions[0]["description"] == "Mô tả ảnh từ caption func"
    assert len(caption.calls) == 1
    prompt, image_data = caption.calls[0]
    # caption func nhận đúng bytes ảnh lấy từ MinIO theo asset_key
    assert base64.b64decode(image_data) == b"\x89PNG-fake-bytes"
    # prompt chứa caption + ngữ cảnh (content source = content_list gốc)
    assert "Sơ đồ tổ chức PTIT" in prompt
    assert "Cơ cấu tổ chức của Học viện" in prompt
    # content_list gốc KHÔNG bị gán img_path mới (describe_items là bản copy)
    assert content_list[1]["img_path"] == "gone.png"
    assert "asset_key" in content_list[1]


async def test_describe_image_prompt_includes_language_and_doc_title(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Prompt caption phải có chỉ dẫn ngôn ngữ + tiêu đề doc (mô tả khớp corpus)."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
        CAPTION_LLM_LANGUAGE="vi",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    await pipeline._stage_describe(
        "docX", _image_content_list(), "Sổ tay sinh viên 2026"
    )

    prompt = caption.calls[0][0]
    assert "Write the whole response in vi." in prompt
    # yêu cầu giữ nguyên văn chữ trong ảnh, không bỏ sót phần tử
    flat = " ".join(prompt.split())
    assert "EXACTLY as it is written in the image" in flat
    assert "never translate" in flat
    assert "omitting none" in flat
    assert "Sổ tay sinh viên 2026" in prompt


async def test_describe_image_prompt_without_language_when_blank(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """CAPTION_LLM_LANGUAGE rỗng -> prompt giữ nguyên, không chèn chỉ dẫn."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
        CAPTION_LLM_LANGUAGE="",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    await pipeline._stage_describe("docX", _image_content_list())

    prompt = caption.calls[0][0]
    assert "Output language:" not in prompt
    assert "Document title" not in prompt


def test_format_image_chunk_drops_empty_summary_line(pipeline):
    """ "Tóm tắt: Không có" là nhiễu -> bỏ hẳn khi không có caption."""
    from ami_rag.core.vector_pipeline import _format_modal_chunk

    item = {"type": "image", "image_caption": [], "image_footnote": []}
    assert _format_modal_chunk("image", item, "Mô tả sơ đồ") == (
        "[Image Content]\nMô tả sơ đồ"
    )
    item2 = {
        "type": "image",
        "image_caption": ["Sơ đồ tổ chức"],
        "image_footnote": ["Ghi chú"],
    }
    assert _format_modal_chunk("image", item2, "Mô tả") == (
        "[Image Content]\nMô tả\nTóm tắt: Sơ đồ tổ chức\nGhi chú"
    )


def test_format_table_chunk_without_description_has_no_empty_line():
    """LLM chết -> mô tả rỗng: bỏ dòng "Mô tả:" thay vì để trơ trọng."""
    from ami_rag.core.vector_pipeline import _format_modal_chunk

    item = {"type": "table", "table_body": "<table><tr><td>TT</td></tr></table>"}
    out = _format_modal_chunk("table", item, "")
    assert "Mô tả:" not in out
    assert (
        out == "[Table Content]\nCaption: Không có\n<table><tr><td>TT</td></tr></table>"
    )


def test_build_chunks_drops_dict_repr_description(pipeline):
    """Processor chết trả str(modal_content) -> không được để dict-repr lọt vào chunk.

    Đây là lỗi thật đã gặp: Qwen LLM không với được, TableModalProcessor nuốt lỗi
    và trả dict-repr của item (lộ path /tmp/... của MinerU) làm "mô tả" bảng.
    """
    item = {
        "type": "table",
        "table_body": "<table><tr><td>TT</td></tr></table>",
        "img_path": "/tmp/abc123/auto/images/deadbeef.jpg",
        "page_idx": 7,
    }
    descriptions = [{"index": 0, "type": "table", "description": str(item)}]
    chunks = pipeline._build_chunks("doc_test", [item], descriptions)
    content = chunks[0].content
    assert "'type': 'table'" not in content
    assert "/tmp/abc123" not in content
    assert "Mô tả:" not in content
    assert "<table>" in content


def test_build_chunks_skips_modal_item_without_any_text(pipeline):
    """Item MinerU thiếu table_body và không caption/mô tả -> bỏ chunk rác."""
    content_list = [
        {"type": "text", "text": "Nội dung trang", "page_idx": 0},
        {
            "type": "table",
            "table_type": "simple_table",
            "page_idx": 17,
            "bbox": {"top": 1},
        },
    ]
    chunks = pipeline._build_chunks("doc_test", content_list, descriptions=[])
    assert [c.modality for c in chunks] == ["text"]


async def test_describe_item_table_dict_repr_falls_back_to_body(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """_describe_item (table) cũng phải nhận ra dict-repr -> trả text thô."""

    class _DictReprProc:
        def set_content_source(self, content_source, content_format="auto"):
            pass

        async def generate_chunk_sections(
            self, modal_content, content_type, item_info=None, entity_name=None
        ):
            return [{"description": str(modal_content), "entity_info": {}}]

    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"table": _DictReprProc()}
    item = {
        "type": "table",
        "table_body": "<table><tr><td>TT</td></tr></table>",
        "page_idx": 7,
    }

    out = await pipeline._describe_item("table", item, [item], 0, "docX")
    assert out == "<table><tr><td>TT</td></tr></table>"


async def test_describe_image_temp_dir_cleaned_after_stage(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """img_path tồn tại LÚC describe chạy; temp dir bị xoá SAU _stage_describe."""
    seen: list[tuple[str, bool]] = []

    class _ProbeProc:
        def set_content_source(self, content_source, content_format="auto"):
            pass

        async def generate_chunk_sections(
            self, modal_content, content_type, item_info=None, entity_name=None
        ):
            path = modal_content.get("img_path")
            seen.append((path, Path(path).exists()))
            return [
                {"description": "mô tả image", "entity_info": {}, "window_meta": None}
            ]

    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _ProbeProc()}

    content_list = _image_content_list()
    await pipeline._stage_describe("docX", content_list)

    assert len(seen) == 1
    path, existed_during = seen[0]
    assert existed_during, "img_path phải là file thật khi processor chạy"
    assert not Path(path).exists(), "temp dir phải bị xoá sau _stage_describe"
    # bản gốc giữ img_path cũ, không bị mutate
    assert content_list[1]["img_path"] == "gone.png"


async def test_describe_image_mode_none_raw_text(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """mode none: text thô caption + footnotes + ngữ cảnh; KHÔNG gọi caption func / fetch MinIO."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="none",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    fetched: list[str] = []
    original_get_bytes = fake_asset_store.get_bytes

    def _spy(key):
        fetched.append(key)
        return original_get_bytes(key)

    fake_asset_store.get_bytes = _spy

    content_list = _image_content_list(
        caption="Sơ đồ tổ chức PTIT", footnotes="Nguồn: PTIT"
    )
    descriptions = await pipeline._stage_describe("docX", content_list)

    text = descriptions[0]["description"]
    assert (
        text
        == "Sơ đồ tổ chức PTIT\nNguồn: PTIT\nCơ cấu tổ chức của Học viện gồm các phòng ban."
    )
    assert caption.calls == []
    assert fetched == []


async def test_describe_image_caption_func_fails_falls_back_to_raw(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Caption func raise -> processor nuốt lỗi (trả dict repr) -> phát hiện -> text thô, không crash."""
    caption = _FakeCaptionFunc(fail=True)
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list(caption="Sơ đồ tổ chức PTIT")
    descriptions = await pipeline._stage_describe("docX", content_list)

    # descriptions.json vẫn ghi, mô tả là text thô (không phải dict repr của item)
    text = descriptions[0]["description"]
    assert "Sơ đồ tổ chức PTIT" in text
    assert "Cơ cấu tổ chức của Học viện" in text
    assert not text.startswith("{")


async def test_describe_image_missing_asset_key_keeps_original(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    """Ảnh thiếu asset_key -> không materialize -> img_path cũ hỏng -> text thô, không crash."""
    caption = _FakeCaptionFunc()
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = _image_content_list()
    del content_list[1]["asset_key"]
    descriptions = await pipeline._stage_describe("docX", content_list)

    # img_path cũ ("gone.png") không tồn tại -> processor lỗi -> text thô
    text = descriptions[0]["description"]
    assert "Cơ cấu tổ chức của Học viện" in text
    assert not text.startswith("{")


async def test_describe_caption_concurrency_limited(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    caption = _FakeCaptionFunc(delay=0.01)
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        IMAGE_DESCRIBE_MODE="gemini",
        CAPTION_LLM_MAX_CONCURRENCY=2,
    )
    pipeline.modal_processors = {"image": _real_image_processor(caption)}

    content_list = [
        {
            "type": "image",
            "img_path": "gone.png",
            "asset_key": f"rag-assets/docX/img{i}.png",
            "page_idx": 0,
        }
        for i in range(5)
    ]
    descriptions = await pipeline._stage_describe("docX", content_list)

    assert len(descriptions) == 5
    assert len(caption.calls) == 5
    assert caption.max_active <= 2


async def test_describe_sets_content_source_on_processors(
    fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
):
    class _RecProc:
        def __init__(self):
            self.sources = []

        def set_content_source(self, content_source, content_format="auto"):
            self.sources.append((content_source, content_format))

        async def generate_chunk_sections(
            self, modal_content, content_type, item_info=None, entity_name=None
        ):
            return [{"description": "desc", "entity_info": {}, "window_meta": None}]

    pipeline = _make_pipeline(
        fake_embedder, fake_vector_store, fake_asset_store, fake_status_store
    )
    proc = _RecProc()
    pipeline.modal_processors = {"image": proc, "table": proc}

    content_list = _image_content_list()
    await pipeline._stage_describe("docX", content_list)

    assert proc.sources and proc.sources[0][1] == "minerU"
    assert proc.sources[0][0] is content_list


async def test_describe_vision_mode_uses_processor(
    fake_embedder,
    fake_vector_store,
    fake_asset_store,
    fake_status_store,
    fake_modal_processors,
):
    pipeline = _make_pipeline(
        fake_embedder,
        fake_vector_store,
        fake_asset_store,
        fake_status_store,
        QWEN_VLM_MODEL="Qwen/Qwen3-VL",  # auto -> vision
    )
    pipeline.modal_processors = fake_modal_processors

    descriptions = await pipeline._stage_describe("docX", _image_content_list())

    # vision mode: đi qua processor, được gán img_path file tạm (đã xoá sau stage)
    assert descriptions[0]["description"] == "mô tả image"
    calls = []
    for proc in fake_modal_processors.values():
        calls.extend(proc.calls)
    img_calls = [c for c in calls if c[0] == "image"]
    assert img_calls, "processor image phải được gọi ở mode vision"
    img_item = img_calls[0][1]
    assert img_item["img_path"] != "gone.png", "materialize phải gán img_path temp"
    assert not Path(img_item["img_path"]).exists(), "temp dir xoá sau stage"
