"""Test DocStatusStore (fake Mongo collection) + lockfile."""

import time

import pytest

from ami_rag.core.lockfile import Lockfile, LockHeld
from ami_rag.storage.doc_status import (
    LEGACY_PROCESSED,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_STALE,
    DocStatusStore,
    effective_status,
)
from tests.ami_service.fakes import FakeMongoCollection


@pytest.fixture()
def store():
    return DocStatusStore.__new__(DocStatusStore)


def _make_store() -> DocStatusStore:
    s = DocStatusStore.__new__(DocStatusStore)
    s._col = FakeMongoCollection()
    return s


def test_ensure_pending_creates_once():
    s = _make_store()
    s.ensure_pending("doc-1", source_path="a.pdf", content_hash="h1")
    s.ensure_pending("doc-1", source_path="a.pdf", content_hash="h2")
    row = s.get("doc-1")
    assert row["status"] == STATUS_PENDING
    assert row["content_hash"] == "h1"  # ensure_pending không ghi đè bản ghi có sẵn


def test_mark_stage_and_indexed():
    s = _make_store()
    s.ensure_pending("doc-1")
    s.mark_stage("doc-1", "chunk", chunk_count=12, embed_model="Qwen/Qwen3-VL-Embedding-2B")
    row = s.get("doc-1")
    assert row["status"] == STATUS_PROCESSING
    assert row["stage"] == "chunk"
    assert row["chunk_count"] == 12

    s.mark_indexed("doc-1", chunk_count=12, embed_model="Qwen/Qwen3-VL-Embedding-2B", embed_dim=2048, chunker_version="v1")
    row = s.get("doc-1")
    assert row["status"] == STATUS_INDEXED
    assert row["stage"] == "indexed"
    assert row["chunk_count"] == 12
    assert row["embed_dim"] == 2048


def test_mark_failed_increments_attempts():
    s = _make_store()
    s.ensure_pending("doc-1")
    s.mark_failed("doc-1", "boom", "embed")
    s.mark_failed("doc-1", "boom again", "embed")
    row = s.get("doc-1")
    assert row["status"] == STATUS_FAILED
    assert row["attempts"] == 2
    assert row["error_stage"] == "embed"
    assert row["error"] == "boom again"


def test_mark_stale_only_indexed():
    s = _make_store()
    s.ensure_pending("doc-a")
    s.mark_indexed("doc-b", chunk_count=1, embed_model="m", embed_dim=8, chunker_version="v1")
    s.ensure_pending("doc-c")
    s.mark_failed("doc-c", "err", "chunk")
    n = s.mark_stale()
    assert n == 1
    assert s.get("doc-b")["status"] == STATUS_STALE
    assert s.get("doc-a")["status"] == STATUS_PENDING
    assert s.get("doc-c")["status"] == STATUS_FAILED


def test_legacy_processed_reads_as_stale():
    s = _make_store()
    s._col.insert("old-1", {"status": LEGACY_PROCESSED, "source_path": "x.pdf"})
    stale = s.stale_rows()
    assert [r["_id"] for r in stale] == ["old-1"]


def test_effective_status_rules():
    current = {"status": STATUS_INDEXED, "embed_model": "Qwen/Qwen3-VL-Embedding-2B", "chunker_version": "v1"}
    assert effective_status(current, "Qwen/Qwen3-VL-Embedding-2B", "v1") == STATUS_INDEXED
    # lệch embed_model -> stale
    assert effective_status(current, "other-model", "v1") == STATUS_STALE
    # lệch chunker_version -> stale
    assert effective_status(current, "Qwen/Qwen3-VL-Embedding-2B", "v2") == STATUS_STALE
    # legacy processed -> stale
    assert effective_status({"status": LEGACY_PROCESSED}, "Qwen/Qwen3-VL-Embedding-2B") == STATUS_STALE
    # pending/failed giữ nguyên
    assert effective_status({"status": STATUS_PENDING}) == STATUS_PENDING
    assert effective_status({"status": STATUS_FAILED}) == STATUS_FAILED


def test_counts_and_all_rows():
    s = _make_store()
    s.ensure_pending("doc-a")
    s.mark_indexed("doc-b", chunk_count=1, embed_model="m", embed_dim=8, chunker_version="v1")
    s.mark_failed("doc-c", "err", "chunk")
    counts = s.counts()
    assert counts == {STATUS_PENDING: 1, STATUS_INDEXED: 1, STATUS_FAILED: 1}
    assert len(s.all_rows()) == 3


# --------------------------------------------------------------------------
# Lockfile
# --------------------------------------------------------------------------
def test_lockfile_acquire_release(tmp_path):
    lock = Lockfile(tmp_path / "cli.lock")
    lock.acquire()
    assert (tmp_path / "cli.lock").exists()
    lock.release()
    assert not (tmp_path / "cli.lock").exists()


def test_lockfile_blocks_second_acquire(tmp_path):
    with Lockfile(tmp_path / "cli.lock"), pytest.raises(LockHeld):
        Lockfile(tmp_path / "cli.lock").acquire()


def test_lockfile_steals_dead_pid(tmp_path):
    lock = Lockfile(tmp_path / "cli.lock")
    # pid không tồn tại (rất khó trùng với pid sống trong test env)
    lock.path.write_text(f"pid:999999999\nheld_at:{time.time():.0f}\n", encoding="utf-8")
    lock.acquire()  # không raise - pid chết -> lấy lại khoá
    assert "pid:999999999" not in lock.path.read_text(encoding="utf-8")
    lock.release()
