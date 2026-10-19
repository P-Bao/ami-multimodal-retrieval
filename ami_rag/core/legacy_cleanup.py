"""Dò + xoá collection Mongo/Qdrant rác của pipeline LightRAG + model embed cũ.

Pipeline vector thuần chỉ dùng:
- Mongo (`RAG_DB`): duy nhất registry `RAG_DOCUMENTS_COLLECTION` (DocStatusStore) —
  các dòng legacy `processed` của pipeline cũ vẫn được đọc là `stale` để `reindex --scan`.
- Qdrant: collection theo quy ước `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`
  (hai gạch dưới). Collection model/version hiện tại được giữ; các collection của
  embed model cũ (ví dụ bản Qwen sau khi đổi sang Nemotron) được `cleanup --stale-models`
  dò + xoá khi cần.

Còn lại là rác của LightRAG:
- Mongo: LightRAG đặt tên `{WORKSPACE}_{namespace}` (một gạch) — `full_docs`,
  `text_chunks`, `llm_response_cache`, `doc_status`, graph `chunk_entity_relation`...;
  RAGAnything cũ thêm `parse_cache`, `multimodal_status`.
- Qdrant: `{WORKSPACE}_chunks` / `_entities` / `_relationships` (một gạch).

Quy tắc nhận diện (an toàn cho Qdrant dùng chung):
- Mongo: tiền tố `{WORKSPACE}_` trừ registry.
- Qdrant legacy LightRAG: tiền tố `{WORKSPACE}_` nhưng KHÔNG phải `{WORKSPACE}__` (vector pipeline).
- Qdrant model cũ (`--stale-models`): `{WORKSPACE}__*` nhưng không phải collection hiện tại
  (tính từ EMBED_MODEL + CHUNKER_VERSION).
Collection nào không thuộc tiền tố workspace không bao giờ bị đụng tới.
"""

from dataclasses import dataclass


@dataclass
class LegacyTarget:
    """Một collection legacy sẽ bị xoá."""

    db: str  # "mongo" | "qdrant"
    name: str
    count: int  # số document/points (ước lượng); -1 = không đếm được nhưng vẫn xoá được
    kind: str = "legacy"  # "legacy" (LightRAG) | "stale-model" (embed model cũ)


def is_legacy_mongo_collection(name: str, workspace: str, keep: str) -> bool:
    """Mongo: `{WORKSPACE}_*` trừ registry (`keep`)."""
    return name != keep and name.startswith(f"{workspace}_")


def is_legacy_qdrant_collection(name: str, workspace: str) -> bool:
    """Qdrant: `{WORKSPACE}_*` nhưng không phải `{WORKSPACE}__*` của vector pipeline."""
    return name.startswith(f"{workspace}_") and not name.startswith(f"{workspace}__")


def find_legacy_mongo(db, workspace: str, keep: str) -> list[LegacyTarget]:
    """Liệt kê collection legacy trong `RAG_DB` kèm số document ước lượng."""
    targets: list[LegacyTarget] = []
    for name in sorted(db.list_collection_names()):
        if not is_legacy_mongo_collection(name, workspace, keep):
            continue
        try:
            count = int(db[name].estimated_document_count())
        except Exception:
            count = -1
        targets.append(LegacyTarget("mongo", name, count))
    return targets


def find_legacy_qdrant(client, workspace: str) -> list[LegacyTarget]:
    """Liệt kê collection Qdrant legacy kèm số points."""
    targets: list[LegacyTarget] = []
    names = [c.name for c in client.get_collections().collections]
    for name in sorted(names):
        if not is_legacy_qdrant_collection(name, workspace):
            continue
        try:
            count = int(client.count(collection_name=name, exact=True).count)
        except Exception:
            count = -1
        targets.append(LegacyTarget("qdrant", name, count))
    return targets


def is_stale_model_collection(name: str, workspace: str, current_name: str) -> bool:
    """Qdrant: `{WORKSPACE}__*` (vector pipeline) nhưng không phải collection hiện tại."""
    return name.startswith(f"{workspace}__") and name != current_name


def find_stale_model_collections(client, workspace: str, current_name: str) -> list[LegacyTarget]:
    """Liệt kê collection Qdrant của embed model/version cũ (không phải hiện tại)."""
    targets: list[LegacyTarget] = []
    names = [c.name for c in client.get_collections().collections]
    for name in sorted(names):
        if not is_stale_model_collection(name, workspace, current_name):
            continue
        try:
            count = int(client.count(collection_name=name, exact=True).count)
        except Exception:
            count = -1
        targets.append(LegacyTarget("qdrant", name, count, kind="stale-model"))
    return targets


def delete_mongo_collections(db, names: list[str]) -> tuple[list[str], list[str]]:
    """Xoá từng collection; trả về (đã xoá, lỗi). Lỗi một cái không dừng cả lô."""
    deleted: list[str] = []
    failed: list[str] = []
    for name in names:
        try:
            db.drop_collection(name)
            deleted.append(name)
        except Exception:
            failed.append(name)
    return deleted, failed


def delete_qdrant_collections(client, names: list[str]) -> tuple[list[str], list[str]]:
    """Xoá từng collection; trả về (đã xoá, lỗi). Lỗi một cái không dừng cả lô."""
    deleted: list[str] = []
    failed: list[str] = []
    for name in names:
        try:
            client.delete_collection(collection_name=name)
            deleted.append(name)
        except Exception:
            failed.append(name)
    return deleted, failed
