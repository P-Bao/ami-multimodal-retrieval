import json

import httpx
import pytest

from ami_rag.api import main as api_main
from ami_rag.api.resolver import DocResolver
from ami_rag.api.routes import admin as admin_routes
from ami_rag.api.routes import rag as rag_routes
from ami_rag.settings import get_settings

from .conftest import FakeAssetStore, FakeRerank, FakeVectorStore


@pytest.fixture
def app(
    fake_pipeline,
    fake_docs_repo,
    fake_status_store,
    fake_queue,
    fake_asset_store,
):
    app = api_main.create_app()
    fake_resolver = DocResolver(
        docs_repo=fake_docs_repo,
        minio_endpoint="localhost:9000",
        minio_access_key="x",
        minio_secret_key="x",
        minio_bucket="ami-data-documents",
    )
    fake_resolver._presign = lambda object_name: None

    # Seed chunks vào vector store của pipeline (search test)
    fake_pipeline.vector_store.points.update(
        {
            "chunk-1": {
                "doc_id": "64b000000000000000000001",
                "content": "Học phí năm 2026 là 12 triệu mỗi học kỳ.",
                "modality": "text",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-1",
                "page": 0,
            },
            "chunk-2": {
                "doc_id": "64b000000000000000000005",
                "content": "Tuyển sinh sử dụng phương thức xét tuyển kết hợp.",
                "modality": "text",
                "source_path": "64b000000000000000000005_text",
                "chunk_id": "chunk-2",
            },
            "chunk-3": {
                "doc_id": "64b000000000000000000001",
                "content": "[Image Content]\nmô tả image",
                "modality": "image",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-3",
                "asset_key": "rag-assets/64b000000000000000000001/img1.png",
                "page": 2,
                "caption": "Sơ đồ",
            },
            "chunk-4": {
                "doc_id": "64b000000000000000000001",
                "content": "[Table Content]\nCaption: Học phí",
                "modality": "table",
                "source_path": "64b000000000000000000001_a.pdf",
                "chunk_id": "chunk-4",
                "table_body": "| ngành | phí |\n|---|---|\n| CNTT | 12tr |",
                "page": 5,
                "caption": "Học phí",
            },
        }
    )

    app.dependency_overrides[rag_routes.get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank()
    app.dependency_overrides[rag_routes.get_resolver] = lambda: fake_resolver
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: fake_asset_store
    app.dependency_overrides[admin_routes.get_queue] = lambda: fake_queue
    app.dependency_overrides[admin_routes.get_state_repo] = lambda: fake_status_store
    app.dependency_overrides[admin_routes.get_docs_repo] = lambda: fake_docs_repo
    return app


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


BODY = {"messages": [{"role": "user", "content": "học phí"}]}


async def test_exact_contract_shape(client):
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "documents", "references", "meta"}
    assert body["query"] == "học phí"
    assert len(body["documents"]) >= 1
    for doc in body["documents"]:
        assert set(doc.keys()) == {
            "text",
            "score",
            "metadata",
            "reference_id",
            "doc",
            "modality",
            "artifact_url",
            "artifact_urls",
            "table_body",
            "page",
            "caption",
        }
        assert set(doc["metadata"].keys()) == {
            "source",
            "page",
            "document_id",
            "chunk_index",
            "global_id",
        }
        assert doc["score"] is not None


async def test_rerank_scores_descending(client):
    resp = await client.post("/v2/rag/", json={**BODY, "top_k": 5})
    docs = resp.json()["documents"]
    scores = [d["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


async def test_document_id_resolved_from_chunk_payload(client):
    resp = await client.post("/v2/rag/", json=BODY)
    docs = resp.json()["documents"]
    assert docs[0]["doc"] is not None
    assert docs[0]["doc"]["document_id"] in (
        "64b000000000000000000001",
        "64b000000000000000000005",
        "64b000000000000000000006",
    )


async def test_resolve_doc_id_presigns_fresh_per_call(fake_docs_repo):
    """Cache giữ field Mongo; file_url presign lại mỗi call (URL hết hạn không được trả lại)."""
    resolver = DocResolver(
        docs_repo=fake_docs_repo,
        minio_endpoint="localhost:9000",
        minio_access_key="x",
        minio_secret_key="x",
        minio_bucket="ami-data-documents",
    )
    signs = iter(["url-1", "url-2"])
    resolver._presign = lambda object_name: next(signs)
    doc_id = "64b000000000000000000001"

    first = await resolver.resolve_doc_id(doc_id)
    second = await resolver.resolve_doc_id(doc_id)

    assert first["file_url"] == "url-1"
    assert second["file_url"] == "url-2"
    assert first["document_id"] == second["document_id"]


async def test_references_dedup_by_document(client):
    resp = await client.post("/v2/rag/", json=BODY)
    references = resp.json()["references"]
    ids = [r["document_id"] for r in references]
    assert len(ids) == len(set(ids))


async def test_include_references_false(client):
    resp = await client.post("/v2/rag/", json={**BODY, "include_references": False})
    assert resp.json()["references"] == []


async def test_rerank_failure_degrades_to_unscored_chunks(client, app):
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: FakeRerank(fail=True)
    resp = await client.post("/v2/rag/", json=BODY)
    docs = resp.json()["documents"]
    assert len(docs) >= 1
    assert all(d["score"] is None for d in docs)


class _IndexedRerank:
    """Rerank giả: trả điểm theo index cố định (không phụ thuộc top_n)."""

    def __init__(self, scores: dict[int, float], fail=False):
        self.scores = scores
        self.fail = fail
        self.calls: list[int | None] = []

    async def __call__(self, query, documents, top_n=None, **kwargs):
        self.calls.append(top_n)
        if self.fail:
            raise ConnectionError("rerank down")
        limit = top_n if top_n is not None else len(documents)
        return [
            {"index": i, "relevance_score": s}
            for i, s in sorted(self.scores.items())[:limit]
        ]


_FUSION_POINTS = {
    "chunk-f1": {
        "doc_id": "64b000000000000000000001",
        "content": "text 1",
        "modality": "text",
        "chunk_id": "chunk-f1",
    },
    "chunk-f2": {
        "doc_id": "64b000000000000000000001",
        "content": "text 2",
        "modality": "text",
        "chunk_id": "chunk-f2",
    },
    "chunk-f3": {
        "doc_id": "64b000000000000000000001",
        "content": "[Image Content]\nmô tả image",
        "modality": "image",
        "chunk_id": "chunk-f3",
        "asset_key": "rag-assets/64b000000000000000000001/img1.png",
    },
    "chunk-f4": {
        "doc_id": "64b000000000000000000001",
        "content": "[Table Content]\nCaption: bảng",
        "modality": "table",
        "chunk_id": "chunk-f4",
    },
    "chunk-f5": {
        "doc_id": "64b000000000000000000001",
        "content": "text 3",
        "modality": "text",
        "chunk_id": "chunk-f5",
    },
    "chunk-f6": {
        "doc_id": "64b000000000000000000001",
        "content": "[Image Content]\nmô tả image 2",
        "modality": "image",
        "chunk_id": "chunk-f6",
        "asset_key": "rag-assets/64b000000000000000000001/img2.png",
    },
}

# index theo thứ tự search trả về (thứ tự insert): 0,1=text, 2=image,
# 3=table (cùng nhóm text), 4=text, 5=image
# text: idx0 r1, idx1 r2, idx3 r3, idx4 r4; visual: idx2 r1, idx5 r2
_FUSION_SCORES = {0: 0.9, 1: 0.8, 2: 0.14, 3: 0.7, 4: 0.6, 5: 0.13}


async def test_fusion_rrf_promotes_image_over_text(
    client, app, fake_pipeline, monkeypatch
):
    monkeypatch.setattr(get_settings(), "RERANK_FUSION", "rrf")
    fake_pipeline.vector_store.points = dict(_FUSION_POINTS)
    rerank_fake = _IndexedRerank(_FUSION_SCORES)
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: rerank_fake

    resp = await client.post("/v2/rag/", json={**BODY, "top_k": 6})
    body = resp.json()
    assert body["meta"]["fusion"] == "rrf"
    # fusion xin full scores (top_n = 6 documents)
    assert rerank_fake.calls == [6]

    docs = body["documents"]
    # image raw 0.14 (hạng 1 nhóm visual, 0.8/11) vượt text raw 0.6 (hạng 4, 1/14)
    assert docs[3]["modality"] == "image"
    assert docs[3]["score"] == pytest.approx(0.14)
    assert docs[4]["score"] == pytest.approx(0.6)
    # score vẫn là raw -> KHÔNG sort desc theo score
    raw_scores = [d["score"] for d in docs]
    assert raw_scores != sorted(raw_scores, reverse=True)


async def test_fusion_meta_default_raw(client):
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.json()["meta"]["fusion"] == "raw"


async def test_fusion_rrf_failure_falls_back_to_raw(client, app, monkeypatch):
    monkeypatch.setattr(get_settings(), "RERANK_FUSION", "rrf")
    app.dependency_overrides[rag_routes.get_rerank_func] = lambda: _IndexedRerank(
        {}, fail=True
    )
    resp = await client.post("/v2/rag/", json=BODY)
    body = resp.json()
    assert body["meta"]["fusion"] == "raw"
    assert all(d["score"] is None for d in body["documents"])


async def test_auth_required_when_key_configured(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RAG_API_KEY", "secret-key")
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 401
    resp = await client.post(
        "/v2/rag/", json=BODY, headers={"Authorization": "Bearer secret-key"}
    )
    assert resp.status_code == 200
    # token lạ (non-ASCII sau khi decode latin-1) -> 401, không 500
    # (compare_digest chạy trên bytes; httpx chặn header str non-ASCII)
    resp = await client.post(
        "/v2/rag/", json=BODY, headers={"Authorization": b"Bearer cl\xe9"}
    )
    assert resp.status_code == 401


async def test_presign_failure_gives_null_artifact_url(client, fake_asset_store):
    from ami_rag.observability import PRESIGN_FAILURES_TOTAL

    fake_asset_store.fail_presign = True
    before = PRESIGN_FAILURES_TOTAL._value.get()
    resp = await client.post("/v2/rag/", json=BODY)
    assert all(d["artifact_url"] is None for d in resp.json()["documents"])
    # presign chỉ gọi khi chunk có asset_key; không có asset -> không tăng
    assert PRESIGN_FAILURES_TOTAL._value.get() >= before


# --- artifact_urls + hybrid sparse search ------------------------------------------------


class _ImageAssetStore(FakeAssetStore):
    """Thêm list_doc_images/presign_doc_images theo contract MinioAssetStore."""

    def __init__(self):
        super().__init__()
        self.presign_doc_calls: list[str] = []

    def list_doc_images(self, doc_id):
        return [f"rag-assets/{doc_id}/img1.png"]

    def presign_doc_images(self, doc_id):
        self.presign_doc_calls.append(doc_id)
        return [f"https://minio/presigned/rag-assets/{doc_id}/img1.png"]


class _SpySparseStore(FakeVectorStore):
    """Ghi nhận kwarg sparse mà search nhận được."""

    def __init__(self):
        super().__init__()
        self.search_sparse: list = []

    async def search(self, collection, vector, top_k, *, sparse=None):
        self.search_sparse.append(sparse)
        return await super().search(collection, vector, top_k, sparse=sparse)


class _LegacyEmbedder:
    """Embedder cũ: chỉ embed_query, không có embed_query_sparse."""

    def __init__(self, dim=8):
        self.dim = dim

    async def embed_query(self, text):
        return [0.2] * self.dim


class _EmptySparseEmbedder(_LegacyEmbedder):
    async def embed_query_sparse(self, text):
        return ([0.2] * self.dim, {"indices": [], "values": []})


async def test_artifact_urls_presigned_doc_images(client, app):
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: _ImageAssetStore()
    docs = (await client.post("/v2/rag/", json=BODY)).json()["documents"]
    assert docs
    for d in docs:
        # doc đã resolve: presign_doc_images chạy cho cả crawl lẫn pdf
        assert d["artifact_urls"] == [
            f"https://minio/presigned/rag-assets/{d['metadata']['document_id']}/img1.png"
        ]
    # chunk có asset_key (ảnh pdf) vẫn giữ artifact_url riêng, trùng URL doc
    image_doc = next(d for d in docs if d["modality"] == "image")
    assert image_doc["artifact_url"] == image_doc["artifact_urls"][0]


async def test_artifact_urls_presigned_once_per_doc(client, app):
    """N chunk cùng doc -> presign_doc_images chạy 1 lần, không N lần list MinIO."""
    store = _ImageAssetStore()
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: store
    docs = (await client.post("/v2/rag/", json={**BODY, "top_k": 10})).json()[
        "documents"
    ]
    same_doc = [
        d for d in docs if d["metadata"]["document_id"] == "64b000000000000000000001"
    ]
    assert len(same_doc) >= 2
    assert store.presign_doc_calls.count("64b000000000000000000001") == 1


async def test_artifact_urls_empty_for_unresolved_doc(client, app, fake_vector_store):
    fake_vector_store.points["chunk-orphan"] = {
        "doc_id": "64b000000000000000000099",
        "content": "chunk mồ côi không resolve",
        "modality": "text",
        "chunk_id": "chunk-orphan",
    }
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: _ImageAssetStore()
    resp = await client.post("/v2/rag/", json={**BODY, "top_k": 10})
    orphan = next(
        d for d in resp.json()["documents"] if d["reference_id"] == "chunk-orphan"
    )
    assert orphan["doc"] is None
    assert orphan["artifact_urls"] == []


async def test_search_hybrid_passes_sparse_query(client, app, fake_pipeline):
    spy = _SpySparseStore()
    spy.points.update(fake_pipeline.vector_store.points)
    fake_pipeline.vector_store = spy
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert len(spy.search_sparse) == 1
    sparse = spy.search_sparse[0]
    assert isinstance(sparse, dict)
    assert sparse["indices"]


async def test_search_legacy_embedder_dense_only(client, app, fake_pipeline):
    spy = _SpySparseStore()
    spy.points.update(fake_pipeline.vector_store.points)
    fake_pipeline.vector_store = spy
    fake_pipeline.embedder = _LegacyEmbedder()
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert len(spy.search_sparse) == 1
    assert spy.search_sparse[0] is None


async def test_search_empty_sparse_falls_back_dense_only(client, app, fake_pipeline):
    spy = _SpySparseStore()
    spy.points.update(fake_pipeline.vector_store.points)
    fake_pipeline.vector_store = spy
    fake_pipeline.embedder = _EmptySparseEmbedder()
    resp = await client.post("/v2/rag/", json=BODY)
    assert resp.status_code == 200
    assert len(spy.search_sparse) == 1
    assert spy.search_sparse[0] is None


async def test_stream_includes_artifact_urls(client, app):
    app.dependency_overrides[rag_routes.get_asset_store] = lambda: _ImageAssetStore()
    resp = await client.post("/v2/rag/stream", json=BODY)
    rows = [json.loads(line) for line in resp.text.splitlines() if line]
    docs = [doc for row in rows for doc in row.get("documents", [])]
    assert docs
    assert all(d.get("artifact_urls") for d in docs)


async def test_stream_emits_ndjson(client):
    resp = await client.post("/v2/rag/stream", json=BODY)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    body = resp.text
    assert '"status": "retrieving"}' in body
    assert '"documents"' in body
    assert '"status": "done"}' in body


async def test_stream_includes_modality(client):
    resp = await client.post("/v2/rag/stream", json=BODY)
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    docs = [doc for line in lines for doc in line.get("documents", [])]
    assert docs
    assert all("modality" in d for d in docs)


async def test_admin_pipeline_status(client):
    resp = await client.get("/admin/pipeline_status")
    assert resp.status_code == 200
    body = resp.json()
    assert "doc_status_counts" in body
    assert "queue_pending" in body


async def test_admin_reprocess_failed(client, fake_status_store, fake_queue):
    fake_status_store.mark_failed("64b000000000000000000001", "boom", "embed")
    resp = await client.post("/admin/reprocess_failed")
    assert resp.status_code == 200
    assert resp.json()["requeued"] == 1
    assert len(fake_queue.published) == 1


async def test_admin_reindex_by_ids(client, fake_docs_repo, fake_queue):
    resp = await client.post(
        "/admin/reindex",
        json={"document_ids": ["64b000000000000000000001", "64b000000000000000000005"]},
    )
    assert resp.json()["published"] == 2
    assert len(fake_queue.published) == 2


async def test_admin_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "RAG_API_KEY", "secret-key")
    resp = await client.get("/admin/pipeline_status")
    assert resp.status_code == 401
    resp = await client.get(
        "/admin/pipeline_status", headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401
    resp = await client.get(
        "/admin/pipeline_status", headers={"Authorization": "Bearer secret-key"}
    )
    assert resp.status_code == 200


async def test_admin_document_status(client, fake_status_store):
    from bson import ObjectId

    fake_status_store.mark_indexed(
        "doc1",
        chunk_count=2,
        embed_model="Qwen/Qwen3-VL-Embedding-2B",
        embed_dim=8,
        chunker_version="v1",
        source="minio_parse",
        parser="mineru",
        counts={"text": 1, "table": 1},
        page_count=3,
        document_type="pdf",
        title="Sổ tay",
        organization_unit_id=ObjectId("64b000000000000000000002"),
        owner_id="sub-1",
    )
    resp = await client.get("/admin/documents/doc1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc1"
    assert body["status"] == "indexed"
    assert body["stage"] == "indexed"
    assert body["source"] == "minio_parse"
    assert body["counts"] == {"text": 1, "table": 1}
    assert body["page_count"] == 3
    assert body["parser"] == "mineru"
    assert body["error"] is None
    assert body["organization_unit_id"] == "64b000000000000000000002"
    assert body["owner_id"] == "sub-1"
    assert body["document_type"] == "pdf"
    assert body["title"] == "Sổ tay"
    assert set(body) == {
        "document_id",
        "status",
        "stage",
        "source",
        "counts",
        "page_count",
        "parser",
        "document_type",
        "title",
        "organization_unit_id",
        "owner_id",
        "error",
        "updated_at",
    }


async def test_admin_document_status_not_found(client):
    resp = await client.get("/admin/documents/missing")
    assert resp.status_code == 404


async def test_admin_document_content(client, fake_asset_store):
    fake_asset_store.content_lists["doc1"] = [
        {"type": "text", "text": "Giới thiệu", "page_idx": 0},
        {
            "type": "image",
            "asset_key": "rag-assets/doc1/i.png",
            "image_caption": ["Sơ đồ"],
            "page_idx": 1,
        },
        {
            "type": "table",
            "table_body": "| a | b |",
            "table_caption": ["Bảng 1"],
            "asset_key": "rag-assets/doc1/t.png",
            "page_idx": 2,
        },
    ]
    resp = await client.get("/admin/documents/doc1/content")
    assert resp.status_code == 200
    body = resp.json()
    assert [b["type"] for b in body["blocks"]] == ["text", "image", "table"]
    assert (
        body["blocks"][1]["asset_url"]
        == "https://minio/presigned/rag-assets/doc1/i.png"
    )
    assert len(body["tables"]) == 1
    assert body["tables"][0]["table_body"] == "| a | b |"
    assert body["markdown"] == (
        "Giới thiệu\n\n![Sơ đồ](https://minio/presigned/rag-assets/doc1/i.png)\n\n*Bảng 1*\n\n| a | b |"
    )


async def test_admin_document_content_not_found(client):
    resp = await client.get("/admin/documents/missing/content")
    assert resp.status_code == 404


def test_blocks_to_markdown_skips_image_without_url_and_renders_equation():
    blocks = admin_routes.build_content_blocks(
        [
            {"type": "image", "image_caption": ["x"]},
            {"type": "equation", "text": "E=mc^2", "page_idx": 0},
        ]
    )
    assert admin_routes.blocks_to_markdown(blocks) == "$$E=mc^2$$"


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200


# --- top_k == number of documents finally returned ---------------------------------------


def _seed_chunks(fake_vector_store, n, doc_id="64b000000000000000000001"):
    for i in range(n):
        fake_vector_store.points[f"chunk-{i}"] = {
            "doc_id": doc_id,
            "content": f"chunk {i}",
            "modality": "text",
            "source_path": "documents/HV/a.pdf",
            "chunk_id": f"chunk-{i}",
        }


async def test_top_k_is_final_document_count(client, fake_vector_store):
    _seed_chunks(fake_vector_store, 30)
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "học phí"}], "top_k": 7},
    )
    body = resp.json()
    assert len(body["documents"]) == 7
    assert body["meta"]["requested_top_k"] == 7 and body["meta"]["returned"] == 7


async def test_top_k_defaults_to_rerank_top_k(client, fake_vector_store, monkeypatch):
    monkeypatch.setattr(get_settings(), "RERANK_TOP_K", 5)
    _seed_chunks(fake_vector_store, 30)
    resp = await client.post("/v2/rag/", json=BODY)
    assert len(resp.json()["documents"]) == 5


async def test_large_top_k_caps_rerank_candidates(client, fake_vector_store):
    """top_k lớn -> số ứng viên cho rerank bị cap 2× (không 4×, không 800)."""
    _seed_chunks(fake_vector_store, 10)
    resp = await client.post("/v2/rag/", json={**BODY, "top_k": 50})
    assert resp.status_code == 200
    assert fake_vector_store.search_limits[-1] == 100


async def test_fewer_candidates_than_top_k_reported_in_meta(client, fake_vector_store):
    fake_vector_store.points.clear()
    _seed_chunks(fake_vector_store, 1)
    resp = await client.post(
        "/v2/rag/",
        json={"messages": [{"role": "user", "content": "q"}], "top_k": 10},
    )
    meta = resp.json()["meta"]
    assert (
        meta["requested_top_k"] == 10
        and meta["returned"] == 1
        and meta["candidates"] == 1
    )
