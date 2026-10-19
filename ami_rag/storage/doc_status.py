"""DocStatusStore - bảng trạng thái tài liệu (Mongo), tách khỏi vector store.

Tái dùng collection `multimodal_rag_documents` (trước đây là registry của worker),
mở rộng schema theo stage:
- Stage: parse -> describe -> chunk -> embed -> indexed. Mỗi stage ghi trạng thái
  + con trỏ để retry chạy tiếp từ stage lỗi (kết quả trung gian nằm ở MinIO).
- Trạng thái: pending | processing | indexed | failed | stale.
- Legacy rows của pipeline cũ (status "processed", không có embed_model) được đọc
  thành `stale`: vector Gemini/KG cũ không dùng lại được với pipeline mới.
"""

import logging
from datetime import datetime, timezone

from pymongo import MongoClient

logger = logging.getLogger(__name__)

# Stages (thứ tự pipeline)
STAGE_PARSE = "parse"
STAGE_DESCRIBE = "describe"
STAGE_CHUNK = "chunk"
STAGE_EMBED = "embed"
STAGE_INDEXED = "indexed"
STAGES = (STAGE_PARSE, STAGE_DESCRIBE, STAGE_CHUNK, STAGE_EMBED, STAGE_INDEXED)

# Statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_STALE = "stale"
# Status của pipeline cũ (LightRAG + Gemini) trong registry trước migration
LEGACY_PROCESSED = "processed"


def effective_status(
    row: dict,
    current_embed_model: str | None = None,
    current_chunker_version: str | None = None,
) -> str:
    """Trạng thái thực tế của một row khi so với cấu hình hiện tại.

    Lệch embed_model/chunker_version (hoặc legacy "processed" không có embed_model)
    -> `stale`; pending/failed giữ nguyên; còn lại theo row.
    """
    status = row.get("status") or STATUS_PENDING
    if status in (STATUS_PENDING, STATUS_FAILED):
        return status
    if status == LEGACY_PROCESSED:
        return STATUS_STALE
    if (
        current_embed_model
        and row.get("embed_model")
        and row["embed_model"] != current_embed_model
    ):
        return STATUS_STALE
    if (
        current_chunker_version
        and row.get("chunker_version")
        and row["chunker_version"] != current_chunker_version
    ):
        return STATUS_STALE
    return status


class DocStatusStore:
    """Trạng thái pipeline theo document (Mongo `multimodal_rag_documents`)."""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str = "multimodal_rag_documents",
    ):
        self._client: MongoClient = MongoClient(mongo_uri, tz_aware=True)
        self._col = self._client[db_name][collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Indexes cho `status` (status/retry queries); best effort (idempotent)."""
        try:
            self._col.create_index("status")
            self._col.create_index("stage")
        except Exception as exc:
            logger.warning("could not ensure %s indexes: %s", self._col.name, exc)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, doc_id: str) -> dict | None:
        return self._col.find_one({"_id": doc_id})

    def all_rows(self) -> list[dict]:
        return list(self._col.find())

    def counts(self) -> dict:
        return {doc["_id"] or "unknown": doc["count"] for doc in self._col.aggregate(
            [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        )}

    def failed_rows(self) -> list[dict]:
        return list(self._col.find({"status": STATUS_FAILED}))

    def stale_rows(self) -> list[dict]:
        """Bao gồm cả legacy `processed` (pipeline cũ -> stale)."""
        return list(self._col.find({"status": {"$in": [STATUS_STALE, LEGACY_PROCESSED]}}))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def ensure_pending(self, doc_id: str, source_path: str = "", content_hash: str = "") -> None:
        """Tạo bản ghi `pending` nếu chưa có (scan/quét dữ liệu cũ)."""
        now = self._now()
        self._col.update_one(
            {"_id": doc_id},
            {
                "$setOnInsert": {
                    "status": STATUS_PENDING,
                    "stage": "",
                    "source_path": source_path,
                    "content_hash": content_hash,
                    "error": "",
                    "error_stage": "",
                    "attempts": 0,
                    "chunk_count": 0,
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    def mark_stage(
        self,
        doc_id: str,
        stage: str,
        *,
        source_path: str | None = None,
        content_hash: str | None = None,
        chunk_count: int | None = None,
        embed_model: str | None = None,
        embed_dim: int | None = None,
        chunker_version: str | None = None,
        source: str | None = None,
        parser: str | None = None,
        file_path: str | None = None,
        counts: dict | None = None,
        page_count: int | None = None,
        assets: list | None = None,
        document_type: str | None = None,
        title: str | None = None,
        organization_unit_id=None,
        owner_id: str | None = None,
        meta: dict | None = None,
    ) -> None:
        """Chuyển doc sang `processing` ở `stage`; giữ nguyên các trường khác."""
        updates: dict = {"status": STATUS_PROCESSING, "stage": stage, "updated_at": self._now()}
        if source_path is not None:
            updates["source_path"] = source_path
        if content_hash is not None:
            updates["content_hash"] = content_hash
        if chunk_count is not None:
            updates["chunk_count"] = chunk_count
        if embed_model is not None:
            updates["embed_model"] = embed_model
        if embed_dim is not None:
            updates["embed_dim"] = embed_dim
        if chunker_version is not None:
            updates["chunker_version"] = chunker_version
        if source is not None:
            updates["source"] = source
        if parser is not None:
            updates["parser"] = parser
        if file_path is not None:
            updates["file_path"] = file_path
        if counts is not None:
            updates["counts"] = counts
        if page_count is not None:
            updates["page_count"] = page_count
        if assets is not None:
            updates["assets"] = assets
        if document_type is not None:
            updates["document_type"] = document_type
        if title is not None:
            updates["title"] = title
        if organization_unit_id is not None:
            updates["organization_unit_id"] = organization_unit_id
        if owner_id is not None:
            updates["owner_id"] = owner_id
        if meta:
            updates.update(meta)
        self._col.update_one({"_id": doc_id}, {"$set": updates}, upsert=True)

    def mark_indexed(
        self,
        doc_id: str,
        *,
        chunk_count: int = 0,
        embed_model: str = "",
        embed_dim: int = 0,
        chunker_version: str = "",
        source_hash: str | None = None,
        source: str | None = None,
        parser: str | None = None,
        file_path: str | None = None,
        counts: dict | None = None,
        page_count: int | None = None,
        assets: list | None = None,
        document_type: str | None = None,
        title: str | None = None,
        organization_unit_id=None,
        owner_id: str | None = None,
        meta: dict | None = None,
    ) -> None:
        updates: dict = {
            "status": STATUS_INDEXED,
            "stage": STAGE_INDEXED,
            "chunk_count": chunk_count,
            "embed_model": embed_model,
            "embed_dim": embed_dim,
            "chunker_version": chunker_version,
            "error": "",
            "error_stage": "",
            "attempts": 0,
            "updated_at": self._now(),
        }
        if source_hash is not None:
            updates["content_hash"] = source_hash
        if source is not None:
            updates["source"] = source
        if parser is not None:
            updates["parser"] = parser
        if file_path is not None:
            updates["file_path"] = file_path
        if counts is not None:
            updates["counts"] = counts
        if page_count is not None:
            updates["page_count"] = page_count
        if assets is not None:
            updates["assets"] = assets
        if document_type is not None:
            updates["document_type"] = document_type
        if title is not None:
            updates["title"] = title
        if organization_unit_id is not None:
            updates["organization_unit_id"] = organization_unit_id
        if owner_id is not None:
            updates["owner_id"] = owner_id
        if meta:
            updates.update(meta)
        self._col.update_one({"_id": doc_id}, {"$set": updates}, upsert=True)

    def begin_attempt(self, doc_id: str, content_hash: str | None = None, **stage_meta) -> int:
        """Đánh dấu doc bắt đầu xử lý (processing + stage parse) và +1 attempts."""
        self.mark_stage(doc_id, STAGE_PARSE, content_hash=content_hash, **stage_meta)
        self._col.update_one(
            {"_id": doc_id},
            {"$inc": {"attempts": 1}, "$setOnInsert": {"created_at": self._now()}},
            upsert=True,
        )
        row = self.get(doc_id)
        return int((row or {}).get("attempts", 1))

    def release_attempt(self, doc_id: str) -> None:
        """Huỷ lần bắt đầu khi doc không được xử lý (skipped/restored)."""
        self._col.update_one(
            {"_id": doc_id},
            {"$inc": {"attempts": -1}},
        )
        row = self.get(doc_id)
        if row and row.get("status") == STATUS_PROCESSING and not row.get("content_hash"):
            self._col.delete_one({"_id": doc_id, "status": STATUS_PROCESSING})

    def mark_failed(self, doc_id: str, error: str, stage: str, *, increment_attempt: bool = True) -> None:
        updates: dict = {
            "status": STATUS_FAILED,
            "error": error[:2000],
            "error_stage": stage,
            "updated_at": self._now(),
        }
        if increment_attempt:
            self._col.update_one(
                {"_id": doc_id},
                {"$set": updates, "$inc": {"attempts": 1}, "$setOnInsert": {"created_at": self._now()}},
                upsert=True,
            )
        else:
            self._col.update_one(
                {"_id": doc_id},
                {"$set": updates, "$setOnInsert": {"created_at": self._now(), "attempts": 0}},
                upsert=True,
            )

    def mark_stale(self, doc_ids: list[str] | None = None) -> int:
        """Đánh dấu lại các doc `indexed` thành `stale` (ép reindex)."""
        query: dict = {"status": {"$in": [STATUS_INDEXED, LEGACY_PROCESSED]}}
        if doc_ids is not None:
            query["_id"] = {"$in": doc_ids}
        return self._col.update_many(
            query, {"$set": {"status": STATUS_STALE, "updated_at": self._now()}}
        ).modified_count

    def delete(self, doc_id: str) -> None:
        self._col.delete_one({"_id": doc_id})

    def close(self) -> None:
        self._client.close()
