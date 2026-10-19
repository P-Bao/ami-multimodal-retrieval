"""Test CLI (status / retry / reindex / cleanup) với fake runner + fake repos - 0 mạng thật.

Phủ: status mặc định 0 lời gọi embed, --doc chi tiết, --check-embed-server;
retry chọn doc failed / --max-attempts / preflight dừng sớm / một lỗi không dừng lô;
reindex --stale mặc định / --dry-run không ghi / preflight / lockfile;
cleanup dò + xoá collection legacy LightRAG (giữ registry + collection vector pipeline).
"""
from datetime import datetime, timedelta, timezone

import pytest

from ami_rag.cli import cmd_cleanup, cmd_reindex, cmd_retry, cmd_status
from ami_rag.core.embedder import EmbedModelMismatch, EmbedServerUnreachable
from ami_rag.core.pipeline import StageOutcome
from ami_rag.settings import Settings
from ami_rag.storage.doc_status import (
    LEGACY_PROCESSED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    DocStatusStore,
)
from tests.ami_service.fakes import FakeMongoCollection

MODEL = "Qwen/Qwen3-VL-Embedding-2B"


class FakeRunner:
    def __init__(
        self,
        fail_ids=None,
        chunks=10,
        cache_hits=3,
        preflight_error=None,
        asset_store=None,
    ):
        self.fail_ids = set(fail_ids or [])
        self.chunks = chunks
        self.cache_hits = cache_hits
        self.preflight_error = preflight_error
        self.preflight_calls = 0
        self.runs: list[tuple] = []
        self.asset_store = asset_store

    async def preflight_embed(self):
        self.preflight_calls += 1
        if self.preflight_error:
            raise self.preflight_error

    async def run(self, doc_id, *, from_stage=None, dry_run=False):
        self.runs.append((doc_id, from_stage, dry_run))
        return StageOutcome(
            doc_id=doc_id,
            stage="indexed",
            ok=doc_id not in self.fail_ids,
            error="" if doc_id not in self.fail_ids else f"{doc_id} boom",
            chunk_count=self.chunks,
            cache_hits=self.cache_hits,
        )


class FakeCLIAssetStore:
    """load_content_list trả content_list hoặc None (mô phỏng MinIO)."""

    def __init__(self, with_content_list: bool):
        self.with_content_list = with_content_list

    def load_content_list(self, doc_id):
        return [{"type": "text", "text": "x"}] if self.with_content_list else None


class FakeScanDocsRepo:
    """iter_all cho scan (mô phỏng MongoDocumentRepo)."""

    def __init__(self, docs: dict):
        self.docs = docs

    def find_by_id(self, doc_id):
        return self.docs.get(doc_id)

    def find_by_file_path(self, file_path):
        for doc in self.docs.values():
            if doc.get("file_path") == file_path:
                return doc
        return None

    def iter_all(self, batch_size=100, document_types=None):
        return iter(list(self.docs.values()))

    def count(self):
        return len(self.docs)


class FakeEmbedder:
    def __init__(self, model=MODEL, dim=2048, error=None):
        self.model = model
        self.dim = dim
        self.error = error
        self.verify_calls = 0

    async def verify(self):
        self.verify_calls += 1
        if self.error:
            raise self.error
        return {
            "model_name": self.model,
            "dim": self.dim,
            "server_version": "test",
            "query_instruction": "Q.",
            "document_instruction": "D.",
        }


def _make_settings(**overrides) -> Settings:
    defaults = {
        "WORKSPACE": "multimodal",
        "EMBED_MODEL": MODEL,
        "EMBED_DIM": 2048,
        "CHUNKER_VERSION": "v1",
        "CLI_STUCK_PROCESSING_MINUTES": 60,
        "EMBED_SERVER_URL": "http://localhost:8007",
        "EMBED_TIMEOUT": 5,
        "MONGO_URI": "mongodb://localhost:27017",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_store() -> DocStatusStore:
    s = DocStatusStore.__new__(DocStatusStore)
    s._col = FakeMongoCollection()
    return s


def _seed(store: DocStatusStore):
    """2 indexed (model hiện tại), 1 legacy processed, 1 failed."""
    store.ensure_pending("doc-1")
    store.ensure_pending("doc-2")
    store.ensure_pending("doc-3")
    store.ensure_pending("doc-4")
    store.mark_indexed("doc-1", chunk_count=5, embed_model=MODEL, embed_dim=2048, chunker_version="v1")
    store.mark_indexed("doc-2", chunk_count=4, embed_model=MODEL, embed_dim=2048, chunker_version="v1")
    # legacy: pipeline cũ (không embed_model) -> stale
    store._col.insert("doc-3", {"status": LEGACY_PROCESSED, "source_path": "old.pdf"})
    store.mark_failed("doc-4", "boom", "embed")


class _Args:
    def __init__(self, **kwargs):
        self.failed = False
        self.stale = False
        self.doc = None
        self.check_embed_server = False
        self.all_failed = False
        self.max_attempts = 3
        self.yes = True
        self.all = False
        self.scan = False
        self.dry_run = False
        self.from_stage = None
        self.embed_server_url = None
        self.embed_batch_size = None
        self.mongo_only = False
        self.qdrant_only = False
        self.stale_models = False
        self.purge_cache_model = []
        for key, value in kwargs.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
async def test_status_summary_counts_and_missing_pending(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 6  # 5 bản ghi + 1 thiếu -> pending

    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    out = capsys.readouterr().out
    assert "indexed" in out
    assert "failed" in out
    assert "stale" in out  # legacy doc-3
    assert "pending" in out
    assert f"embed_model={MODEL}" in out
    assert "multimodal__qwen3-vl-embedding-2b__v1" in out


async def test_status_default_makes_no_embed_calls(capsys, monkeypatch):
    """status (không --check-embed-server): 0 embed server call, 0 model call."""
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 4

    def forbidden(*_a, **_k):
        raise AssertionError("status không được gọi embed server")

    monkeypatch.setattr("ami_rag.cli._build_embedder", forbidden)
    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    assert "unreachable" not in capsys.readouterr().out


async def test_status_stuck_processing_reported(capsys):
    settings = _make_settings()
    store = _make_store()
    store.ensure_pending("doc-stuck")
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    store._col.insert("doc-stuck", {
        "status": STATUS_PROCESSING,
        "stage": "embed",
        "updated_at": old,
        "attempts": 1,
    })

    class DocsRepo:
        def count(self):
            return 1

    await cmd_status(_Args(), settings=settings, store=store, docs_repo=DocsRepo())
    assert "TREO" in capsys.readouterr().out


async def test_status_doc_detail(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    await cmd_status(
        _Args(doc="doc-4"), settings=settings, store=store,
        docs_repo=type("R", (), {"count": lambda self: 4})(),
    )
    out = capsys.readouterr().out
    assert "doc-4" in out
    assert "error (embed): boom" in out
    assert "stage:" in out


async def test_status_doc_path_resolved_via_docs_repo(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)

    class DocsRepo:
        def count(self):
            return 4

        def find_by_id(self, doc_id):
            return None

        def find_by_file_path(self, path):
            return {"_id": "doc-1", "file_path": path} if path == "a.pdf" else None

    await cmd_status(
        _Args(doc="a.pdf"), settings=settings, store=store, docs_repo=DocsRepo()
    )
    out = capsys.readouterr().out
    assert "doc_id:         doc-1" in out


async def test_status_check_embed_server_ok(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    embedder = FakeEmbedder()

    class DocsRepo:
        def count(self):
            return 4

    await cmd_status(
        _Args(check_embed_server=True), settings=settings, store=store,
        docs_repo=DocsRepo(), embedder=embedder,
    )
    out = capsys.readouterr().out
    assert embedder.verify_calls == 1
    assert "khớp config" in out


async def test_status_check_embed_server_mismatch(capsys):
    settings = _make_settings()
    store = _make_store()
    embedder = FakeEmbedder(error=EmbedModelMismatch("embed server chạy model 'other-model'"))
    await cmd_status(
        _Args(check_embed_server=True), settings=settings, store=store,
        docs_repo=type("R", (), {"count": lambda self: 0})(), embedder=embedder,
    )
    assert "LỆCH hoặc lỗi" in capsys.readouterr().out


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------
async def test_retry_all_failed_runs_only_failed():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 0
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-4"]


async def test_retry_doc_specific_non_failed_skipped():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_retry(
        _Args(doc=["doc-1"], yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 0
    assert runner.runs == []


async def test_retry_max_attempts_skips():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    store.mark_failed("doc-4", "boom 2", "embed")  # attempts = 2
    runner = FakeRunner()
    await cmd_retry(
        _Args(all_failed=True, max_attempts=2, yes=True), settings=settings,
        store=store, docs_repo=None, runner=runner,
    )
    assert runner.runs == []


async def test_retry_preflight_failure_stops_early_no_fail_marked():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("embed server unreachable"))
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 2
    assert runner.runs == []
    # không đánh fail doc nào: status giữ nguyên failed, attempts giữ nguyên
    assert store.get("doc-4")["attempts"] == 1


async def test_retry_one_failure_doesnt_stop_batch():
    settings = _make_settings()
    store = _make_store()
    store.ensure_pending("doc-a")
    store.ensure_pending("doc-b")
    store.mark_failed("doc-a", "e1", "chunk")
    store.mark_failed("doc-b", "e2", "chunk")
    runner = FakeRunner(fail_ids={"doc-a"})
    rc = await cmd_retry(
        _Args(all_failed=True, yes=True), settings=settings, store=store,
        docs_repo=None, runner=runner,
    )
    assert rc == 1
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-a", "doc-b"]


async def test_retry_from_stage_passed_to_runner():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    await cmd_retry(
        _Args(all_failed=True, from_stage="parse", yes=True), settings=settings,
        store=store, docs_repo=None, runner=runner,
    )
    assert runner.runs == [("doc-4", "parse", False)]


# --------------------------------------------------------------------------
# reindex
# --------------------------------------------------------------------------
async def test_reindex_default_stale_includes_legacy_and_model_mismatch():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    # doc-1 indexed bằng model CŨ -> stale so với config hiện tại
    store._col._docs["doc-1"]["embed_model"] = "old-embed-model"
    runner = FakeRunner()
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 0
    # doc-3 (legacy) + doc-1 (lệch embed_model) -> stale
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-1", "doc-3"]


async def test_reindex_dry_run_no_preflight_no_writes():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("must not be called"))
    rc = await cmd_reindex(
        _Args(dry_run=True), settings=settings, store=store, runner=runner
    )
    assert rc == 0
    assert runner.preflight_calls == 0
    assert all(dry_run for _, _, dry_run in runner.runs)
    # store không đổi
    assert store.get("doc-4")["status"] == STATUS_FAILED


async def test_reindex_preflight_failure_stops_early():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(preflight_error=EmbedServerUnreachable("embed server unreachable"))
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 2
    assert all(not dry_run for _, _, dry_run in runner.runs) and not runner.runs


async def test_reindex_all_runs_everything_with_preflight():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()
    rc = await cmd_reindex(
        _Args(all=True, from_stage="chunk"), settings=settings, store=store, runner=runner
    )
    assert rc == 0
    assert runner.preflight_calls == 1
    assert sorted(doc_id for doc_id, from_stage, _ in runner.runs) == [
        "doc-1", "doc-2", "doc-3", "doc-4",
    ]
    assert all(from_stage == "chunk" for _, from_stage, _ in runner.runs)


async def test_reindex_dry_run_prints_chunk_and_cache_counts(capsys):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner(chunks=7, cache_hits=2)
    await cmd_reindex(
        _Args(dry_run=True), settings=settings, store=store, runner=runner
    )
    out = capsys.readouterr().out
    # Mặc định --stale: chỉ doc-3 (legacy) là stale
    assert "1 doc" in out
    assert "7 chunk cần embed" in out
    assert "2 trúng cache" in out


async def test_reindex_lockfile_conflict_stops(monkeypatch):
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    runner = FakeRunner()

    class HeldLockfile:
        def __init__(self, path):
            pass

        def acquire(self):
            from ami_rag.core.lockfile import LockHeld

            raise LockHeld("đang giữ")

        def release(self):
            pass

    monkeypatch.setattr("ami_rag.core.lockfile.Lockfile", HeldLockfile)
    rc = await cmd_reindex(_Args(), settings=settings, store=store, runner=runner)
    assert rc == 2
    assert runner.runs == []


async def test_reindex_from_stage_invalid_raises():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    with pytest.raises(ValueError):
        await cmd_reindex(
            _Args(from_stage="indexed"), settings=settings, store=store,
            runner=FakeRunner(),
        )


# --------------------------------------------------------------------------
# reindex --scan (Phase 5)
# --------------------------------------------------------------------------
async def test_reindex_scan_creates_pending_and_selects_candidates():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    # doc-3: legacy `processed` (stale); doc-9: chưa có bản ghi -> scan tạo pending
    docs = {
        "doc-3": {"_id": "doc-3", "file_path": "old.pdf", "content": "x"},
        "doc-9": {"_id": "doc-9", "file_path": "new.pdf", "content": "y"},
    }
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=False))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    # candidates = pending (doc-9) + stale (doc-3); indexed/failed loại trừ
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-3", "doc-9"]
    assert store.get("doc-9")["status"] == STATUS_PENDING
    # content_list thiếu trong MinIO -> from_stage None (full parse)
    assert all(fs is None for _, fs, _ in runner.runs)


async def test_reindex_scan_content_list_present_uses_chunk_stage():
    settings = _make_settings()
    store = _make_store()
    _seed(store)
    docs = {
        "doc-3": {"_id": "doc-3", "file_path": "old.pdf", "content": "x"},
        "doc-9": {"_id": "doc-9", "file_path": "new.pdf", "content": "y"},
    }
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=True))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    assert sorted(doc_id for doc_id, _, _ in runner.runs) == ["doc-3", "doc-9"]
    # content_list đã có -> from_stage "chunk" (không gọi lại parse/LLM)
    assert all(fs == "chunk" for _, fs, _ in runner.runs)


async def test_reindex_scan_does_not_select_indexed_or_failed():
    settings = _make_settings()
    store = _make_store()
    _seed(store)  # doc-1/doc-2 indexed, doc-3 legacy, doc-4 failed
    docs = {"doc-1": {"_id": "doc-1", "file_path": "a.pdf"}, "doc-4": {"_id": "doc-4", "content": "x"}}
    runner = FakeRunner(asset_store=FakeCLIAssetStore(with_content_list=True))
    rc = await cmd_reindex(
        _Args(scan=True), settings=settings, store=store,
        runner=runner, docs_repo=FakeScanDocsRepo(docs),
    )
    assert rc == 0
    # chỉ doc-3 (legacy stale); indexed + failed không reindex
    assert [doc_id for doc_id, _, _ in runner.runs] == ["doc-3"]


# --------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------
class FakeMongoLegacyCollection:
    def __init__(self, count: int):
        self._count = count

    def estimated_document_count(self) -> int:
        return self._count


class FakeMongoDB:
    """Mô phỏng `Database`: list_collection_names / drop_collection."""

    def __init__(self, collections: dict[str, int], fail_drop=frozenset()):
        self.counts = dict(collections)
        self.fail_drop = set(fail_drop)
        self.dropped: list[str] = []

    def list_collection_names(self):
        return list(self.counts)

    def __getitem__(self, name):
        if name not in self.counts:
            raise KeyError(name)
        return FakeMongoLegacyCollection(self.counts[name])

    def drop_collection(self, name):
        if name in self.fail_drop:
            raise RuntimeError(f"cannot drop {name}")
        del self.counts[name]
        self.dropped.append(name)


class _FakeCollInfo:
    def __init__(self, name: str):
        self.name = name


class _FakeCollections:
    def __init__(self, names: list[str]):
        self.collections = [_FakeCollInfo(n) for n in names]


class _FakeCountResult:
    def __init__(self, count: int):
        self.count = count


class FakeQdrantClient:
    """Mô phỏng QdrantClient: get_collections / count / delete_collection."""

    def __init__(self, collections: dict[str, int], fail_delete=frozenset()):
        self.counts = dict(collections)
        self.fail_delete = set(fail_delete)
        self.deleted: list[str] = []

    def get_collections(self):
        return _FakeCollections(list(self.counts))

    def count(self, collection_name, exact=True):
        if collection_name not in self.counts:
            raise ValueError(collection_name)
        return _FakeCountResult(self.counts[collection_name])

    def delete_collection(self, collection_name):
        if collection_name in self.fail_delete:
            raise RuntimeError(f"cannot delete {collection_name}")
        del self.counts[collection_name]
        self.deleted.append(collection_name)


# registry + vector pipeline hiện tại phải sống sót trong mọi test
_KEEP_MONGO = "multimodal_rag_documents"
_KEEP_QDRANT = "multimodal__qwen3-vl-embedding-2b__v1"


def _make_legacy_dbs():
    mongo = FakeMongoDB(
        {
            _KEEP_MONGO: 120,  # registry (DocStatusStore) -> giữ
            "multimodal_full_docs": 80,  # LightRAG KV -> xoá
            "multimodal_doc_status": 120,  # LightRAG doc status -> xoá
            "multimodal_chunk_entity_relation": 500,  # LightRAG graph -> xoá
            "some_other_db_collection": 10,  # ngoài workspace -> không đụng
        }
    )
    qdrant = FakeQdrantClient(
        {
            _KEEP_QDRANT: 900,  # collection vector pipeline hiện tại -> giữ
            "multimodal__old-model__v1": 400,  # model cũ, quy ước 2 gạch -> giữ
            "multimodal_chunks": 900,  # LightRAG -> xoá
            "multimodal_entities": 700,  # LightRAG -> xoá
            "other_service_collection": 50,  # ngoài workspace -> không đụng
        }
    )
    return mongo, qdrant


async def test_cleanup_dry_run_lists_legacy_deletes_nothing(capsys):
    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(dry_run=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "multimodal_full_docs" in out
    assert "multimodal_chunks" in out
    assert "dry-run: 5 collection legacy" in out
    # không xoá gì, không đụng collection ngoài workspace
    assert not mongo.dropped and not qdrant.deleted
    assert _KEEP_MONGO in mongo.counts and _KEEP_QDRANT in qdrant.counts
    assert "some_other_db_collection" in mongo.counts
    assert "other_service_collection" in qdrant.counts


async def test_cleanup_yes_deletes_all_legacy(capsys):
    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(yes=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    assert sorted(mongo.dropped) == [
        "multimodal_chunk_entity_relation",
        "multimodal_doc_status",
        "multimodal_full_docs",
    ]
    assert sorted(qdrant.deleted) == ["multimodal_chunks", "multimodal_entities"]
    # registry + vector pipeline + ngoài workspace sống sót
    assert set(mongo.counts) == {_KEEP_MONGO, "some_other_db_collection"}
    assert set(qdrant.counts) == {_KEEP_QDRANT, "multimodal__old-model__v1", "other_service_collection"}


async def test_cleanup_confirm_declined_deletes_nothing(monkeypatch, capsys):
    mongo, qdrant = _make_legacy_dbs()
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    rc = await cmd_cleanup(
        _Args(yes=False, dry_run=False), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    assert "đã huỷ" in capsys.readouterr().out
    assert not mongo.dropped and not qdrant.deleted


async def test_cleanup_mongo_only_and_qdrant_only():
    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(yes=True, mongo_only=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    assert len(mongo.dropped) == 3 and not qdrant.deleted

    mongo2, qdrant2 = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(yes=True, qdrant_only=True), settings=_make_settings(),
        mongo_db=mongo2, qdrant_client=qdrant2,
    )
    assert rc == 0
    assert len(qdrant2.deleted) == 2 and not mongo2.dropped


async def test_cleanup_no_legacy_returns_zero(capsys):
    mongo = FakeMongoDB({_KEEP_MONGO: 5})
    qdrant = FakeQdrantClient({_KEEP_QDRANT: 5})
    rc = await cmd_cleanup(
        _Args(yes=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    assert "không có collection legacy nào cần xoá" in capsys.readouterr().out
    assert not mongo.dropped and not qdrant.deleted


async def test_cleanup_connection_error_returns_2(capsys):
    class BrokenMongo:
        def list_collection_names(self):
            raise RuntimeError("mongo down")

    rc = await cmd_cleanup(
        _Args(yes=True), settings=_make_settings(),
        mongo_db=BrokenMongo(), qdrant_client=FakeQdrantClient({}),
    )
    assert rc == 2
    assert "STOPPED" in capsys.readouterr().out


async def test_cleanup_partial_delete_failure_returns_1(capsys):
    mongo = FakeMongoDB({_KEEP_MONGO: 1, "multimodal_full_docs": 2})
    qdrant = FakeQdrantClient(
        {_KEEP_QDRANT: 1, "multimodal_chunks": 2, "multimodal_entities": 2},
        fail_delete={"multimodal_entities"},
    )
    rc = await cmd_cleanup(
        _Args(yes=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "LỖI xoá: multimodal_entities" in out
    assert "đã xoá 2/3" in out


async def test_cleanup_stale_models_drops_old_model_collections(capsys):
    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(yes=True, stale_models=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "stale-model [qdrant] multimodal__old-model__v1" in out
    # legacy LightRAG + model cũ bị xoá
    assert sorted(mongo.dropped) == [
        "multimodal_chunk_entity_relation",
        "multimodal_doc_status",
        "multimodal_full_docs",
    ]
    assert sorted(qdrant.deleted) == [
        "multimodal__old-model__v1",
        "multimodal_chunks",
        "multimodal_entities",
    ]
    # registry + collection hiện tại + ngoài workspace sống sót
    assert set(qdrant.counts) == {_KEEP_QDRANT, "other_service_collection"}


async def test_cleanup_stale_models_dry_run_keeps_current(capsys):
    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(dry_run=True, stale_models=True), settings=_make_settings(),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "stale-model [qdrant] multimodal__old-model__v1" in out
    assert "dry-run: 6 collection legacy" in out
    assert not qdrant.deleted
    assert _KEEP_QDRANT in qdrant.counts


async def test_cleanup_stale_models_qdrant_only(capsys):
    qdrant = FakeQdrantClient({_KEEP_QDRANT: 5, "multimodal__old-model__v1": 3})
    rc = await cmd_cleanup(
        _Args(yes=True, stale_models=True, qdrant_only=True),
        settings=_make_settings(), qdrant_client=qdrant,
    )
    assert rc == 0
    assert qdrant.deleted == ["multimodal__old-model__v1"]
    assert set(qdrant.counts) == {_KEEP_QDRANT}


async def test_cleanup_purge_cache_model(tmp_path, capsys):
    from ami_rag.core.embedding_cache import EmbeddingCache

    cache_path = tmp_path / "cache.db"
    cache = EmbeddingCache(cache_path)
    cache.put_many({
        "Qwen/Qwen3-VL-Embedding-2B|2048|ns|a": [0.1] * 4,
        "Qwen/Qwen3-VL-Embedding-2B|2048|ns|b": [0.2] * 4,
        "nvidia/llama-nemotron-embed-vl-1b-v2|2048|ns|c": [0.3] * 4,
    })
    cache.close()

    mongo, qdrant = _make_legacy_dbs()
    rc = await cmd_cleanup(
        _Args(yes=True, purge_cache_model=["Qwen/Qwen3-VL-Embedding-2B"]),
        settings=_make_settings(EMBED_CACHE_PATH=str(cache_path)),
        mongo_db=mongo, qdrant_client=qdrant,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "đã xoá 2 entry của model `Qwen/Qwen3-VL-Embedding-2B`" in out
    # chỉ còn entry nemotron
    cache = EmbeddingCache(cache_path)
    assert cache.count() == 1
    cache.close()
