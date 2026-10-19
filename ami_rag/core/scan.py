"""Scan docs cũ (legacy) -> tạo pending records trong DocStatusStore cho Phase 5 reindex.

Scan backend `documents` collection (active) -> với docs chưa có bản ghi trạng
thái trong registry mới: ensure_pending (stage "", source_path + content_hash).
Legacy rows `processed` (pipeline cũ) đọc là `stale` ở DocStatusStore - không
cần chuyển; `ami-rag reindex` chọn cả pending + stale khi --scan.
"""

import logging

from ami_rag.sources import source_hash
from ami_rag.storage.doc_status import STATUS_INDEXED, effective_status

logger = logging.getLogger(__name__)


async def scan_pending(store, docs_repo, *, embed_model: str | None = None, chunker_version: str | None = None) -> dict:
    """Scan backend docs -> ensure_pending cho docs thiếu registry row.

    Returns summary: {"total", "created", "existing", "reindexed_ok", "stale_kept"}.
    Chỉ đọc Mongo (0 LLM, 0 embed); không đánh dấu gì ngoài ensure_pending.
    """
    created = 0
    existing = 0
    for doc in docs_repo.iter_all():
        doc_id = str(doc["_id"])
        row = store.get(doc_id)
        if row is None:
            store.ensure_pending(
                doc_id,
                source_path=doc.get("file_path") or "",
                content_hash=source_hash(doc),
            )
            created += 1
        else:
            existing += 1
    logger.info("scan: created=%d existing=%d", created, existing)
    return {"created": created, "existing": existing}


def reindex_candidates(rows: list[dict], *, embed_model: str, chunker_version: str) -> list[dict]:
    """Rows đủ điều kiện reindex: pending + stale (bao gồm legacy `processed`).

    Indexed (cùng embed model/chunker) và failed loại trừ (failed -> retry).
    """
    return [
        row
        for row in rows
        if effective_status(row, embed_model, chunker_version)
        not in (STATUS_INDEXED, "failed")
    ]
