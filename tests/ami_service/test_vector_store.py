"""QdrantVectorStore named-vectors (dense + sparse hybrid) — stub client, không mạng."""

from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from ami_rag.core.vector_store import (
    ChunkRecord,
    QdrantVectorStore,
    SearchHit,
    VectorStoreError,
)


class StubClient:
    """Stub QdrantClient: chỉ các method QdrantVectorStore dùng."""

    def __init__(self):
        self.calls = []
        self.existing = {}  # name -> collection info (giả lập server state)

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self.existing]
        )

    def get_collection(self, name):
        return self.existing[name]

    def create_collection(
        self, collection_name, vectors_config, sparse_vectors_config=None
    ):
        self.calls.append(
            (
                "create_collection",
                collection_name,
                vectors_config,
                sparse_vectors_config,
            )
        )
        self.existing[collection_name] = _info(vectors_config, sparse_vectors_config)

    def upsert(self, collection_name, points, wait=None):
        self.calls.append(("upsert", collection_name, points, wait))
        return SimpleNamespace(status="completed")

    def query_points(self, **kwargs):
        self.calls.append(("query_points", kwargs))
        return SimpleNamespace(
            points=[SimpleNamespace(id="p1", score=0.9, payload={"k": "v"})]
        )


def _info(vectors, sparse_vectors=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=vectors, sparse_vectors=sparse_vectors)
        )
    )


def _legacy_info(dim=8):
    return _info(models.VectorParams(size=dim, distance=models.Distance.COSINE))


def _named_info(dim=8):
    return _info(
        {"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        {"sparse": models.SparseVectorParams()},
    )


def _store(client):
    return QdrantVectorStore("http://fake:6333", client=client)


# ------------------------------------------------------------------ protocol


class _MinimalStore:
    """Chứng minh protocol mới vẫn chấp nhận caller cũ (kwargs mặc định)."""

    async def ensure_collection(self, name, dim):
        pass

    async def upsert(self, collection, chunks):
        pass

    async def search(self, collection, vector, top_k):
        return []

    async def delete_by_doc(self, collection, doc_id):
        return 0

    async def count(self, collection, doc_id=None):
        return 0


def test_chunk_record_backward_compatible_positional():
    r = ChunkRecord("c1", [0.1, 0.2], {"doc_id": "d"})
    assert r.id == "c1" and r.vector == [0.1, 0.2]
    assert r.sparse is None and r.payload == {"doc_id": "d"}


# -------------------------------------------------------- ensure_collection


async def test_ensure_collection_creates_named_dense_and_sparse():
    stub = StubClient()
    store = _store(stub)
    await store.ensure_collection("col", 8)
    kinds = [c[0] for c in stub.calls]
    assert kinds == ["create_collection"]
    _, name, vectors_config, sparse_config = stub.calls[0]
    assert name == "col"
    dense = vectors_config["dense"]
    assert isinstance(dense, models.VectorParams)
    assert dense.size == 8 and dense.distance == models.Distance.COSINE
    assert "sparse" in sparse_config


async def test_ensure_collection_sparse_false_keeps_legacy_create():
    stub = StubClient()
    store = _store(stub)
    await store.ensure_collection("col", 8, sparse=False)
    _, _, vectors_config, sparse_config = stub.calls[0]
    assert not isinstance(vectors_config, dict)
    assert sparse_config is None


async def test_ensure_collection_existing_named_dim_match_no_recreate():
    stub = StubClient()
    stub.existing["col"] = _named_info(8)
    store = _store(stub)
    await store.ensure_collection("col", 8)
    assert stub.calls == []


async def test_ensure_collection_existing_named_dim_mismatch_raises():
    stub = StubClient()
    stub.existing["col"] = _named_info(8)
    store = _store(stub)
    with pytest.raises(VectorStoreError) as exc:
        await store.ensure_collection("col", 16)
    assert "có dim=8 nhưng cấu hình là 16" in str(exc.value)


async def test_ensure_collection_existing_legacy_with_sparse_raises():
    stub = StubClient()
    stub.existing["col"] = _legacy_info(8)
    store = _store(stub)
    with pytest.raises(VectorStoreError) as exc:
        await store.ensure_collection("col", 8)  # sparse=True default
    assert "legacy (1 vector vô danh) nhưng cần sparse" in str(exc.value)
    assert "reindex" in str(exc.value)


async def test_ensure_collection_existing_legacy_sparse_false_ok():
    stub = StubClient()
    stub.existing["col"] = _legacy_info(8)
    store = _store(stub)
    await store.ensure_collection("col", 8, sparse=False)
    assert stub.calls == []


async def test_ensure_collection_existing_named_missing_sparse_raises():
    stub = StubClient()
    stub.existing["col"] = _info(
        {"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)}
    )
    store = _store(stub)
    with pytest.raises(VectorStoreError):
        await store.ensure_collection("col", 8)


# ------------------------------------------------------------------- upsert


async def test_upsert_named_vectors_with_and_without_sparse():
    stub = StubClient()
    store = _store(stub)
    sparse = {"indices": [3, 1, 7], "values": [0.5, 0.25, 1.0]}
    chunks = [
        ChunkRecord("s1", [0.1] * 8, sparse=sparse, payload={"doc_id": "d"}),
        ChunkRecord("s2", [0.2] * 8, payload={"doc_id": "d"}),
    ]
    await store.upsert("col", chunks)
    upserts = [c for c in stub.calls if c[0] == "upsert"]
    assert len(upserts) == 1  # batch duy nhất
    points = upserts[0][2]
    assert [p.payload["chunk_id"] for p in points] == ["s1", "s2"]
    # chunk sparse: named vectors "dense" + "sparse"
    vec = points[0].vector
    assert isinstance(vec, dict) and list(vec) == ["dense", "sparse"]
    assert vec["dense"] == [0.1] * 8
    sp = vec["sparse"]
    assert sorted(sp.indices) == sorted(sparse["indices"])
    assert sp.values == sparse["values"]
    # chunk sparse=None: chỉ "dense"
    vec2 = points[1].vector
    assert isinstance(vec2, dict) and "sparse" not in vec2
    assert vec2["dense"] == [0.2] * 8


# ------------------------------------------------------------------- search


async def test_search_dense_only_legacy_collection_raw_query():
    stub = StubClient()
    stub.existing["col"] = _legacy_info(8)
    store = _store(stub)
    hits = await store.search("col", [0.1] * 8, 5)
    assert len(hits) == 1 and isinstance(hits[0], SearchHit)
    _, kwargs = stub.calls[0]
    assert kwargs["query"] == [0.1] * 8
    assert "using" not in kwargs and "prefetch" not in kwargs
    assert kwargs["limit"] == 5 and kwargs["with_payload"] is True


async def test_search_dense_only_named_collection_uses_dense():
    stub = StubClient()
    stub.existing["col"] = _named_info(8)
    store = _store(stub)
    await store.search("col", [0.1] * 8, 5)
    _, kwargs = stub.calls[-1]
    assert kwargs["using"] == "dense"
    assert "prefetch" not in kwargs


async def test_search_hybrid_prefetch_and_rrf():
    stub = StubClient()
    stub.existing["col"] = _named_info(8)
    store = _store(stub)
    sparse = {"indices": [9, 2], "values": [0.7, 0.1]}
    hits = await store.search("col", [0.1] * 8, 5, sparse=sparse)
    assert len(hits) == 1
    _, kwargs = stub.calls[-1]
    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2
    dense_p, sparse_p = prefetch
    assert (
        dense_p.using == "dense" and dense_p.query == [0.1] * 8 and dense_p.limit == 5
    )
    assert sparse_p.using == "sparse" and sparse_p.limit == 5
    assert isinstance(sparse_p.query, models.SparseVector)
    assert list(sparse_p.query.indices) == [9, 2] and list(sparse_p.query.values) == [
        0.7,
        0.1,
    ]
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert kwargs["query"].fusion == models.Fusion.RRF
    assert kwargs["limit"] == 5 and kwargs["with_payload"] is True
