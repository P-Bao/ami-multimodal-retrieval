"""Đổi marker đầu chunk tiếng Anh -> tiếng Việt cho chunks.json ĐÃ LƯU trong MinIO.

Dùng khi index đã tồn tại và không muốn gọi lại API caption (Gemini) cho ảnh:
caption giữ nguyên, chỉ thay marker/nhãn trong `content` của chunk:

    [Image Content] -> [Hình ảnh]      [Table Content] -> [Bảng]
    [Equation]      -> [Công thức]     [Audio Content] -> [Âm thanh]
    [Video Content] -> [Video]         "Caption: x"    -> "Chú thích: x" (đầu dòng)

chunk_id giữ nguyên -> point Qdrant được thay đúng chỗ (embed stage xoá
theo doc trước khi upsert). Sau khi chạy thật, re-embed + index lại:

    ami-rag reindex --all --from-stage embed --yes

(Chỉ chunk có marker đổi text -> cache embed miss; chunk khác cache hit, chi phí
server embed là GPU local, KHÔNG tốn tiền API caption.)

Usage:
    python scripts/migrate_chunk_markers.py            # dry-run: thống kê, không ghi
    python scripts/migrate_chunk_markers.py --yes      # ghi MinIO
    python scripts/migrate_chunk_markers.py --doc <id> --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ami_rag.core.vector_pipeline import ArtifactPaths, migrate_chunk_markers


def _chunks(asset_store, doc_id: str) -> list[dict] | None:
    stored = asset_store.load_json(ArtifactPaths.chunks_key(asset_store, doc_id))
    return stored if isinstance(stored, list) else None


def run(args: argparse.Namespace) -> int:
    from ami_rag.core.factory import get_asset_store
    from ami_rag.settings import get_settings
    from ami_rag.storage.doc_status import DocStatusStore

    settings = get_settings()
    asset_store = get_asset_store()
    if args.doc:
        doc_ids = [args.doc]
    else:
        store = DocStatusStore(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.RAG_DB,
            collection_name=settings.RAG_DOCUMENTS_COLLECTION,
        )
        doc_ids = [row["_id"] for row in store.all_rows()]

    docs_changed = docs_scanned = 0
    chunks_changed = chunks_scanned = 0
    for doc_id in doc_ids:
        chunks = _chunks(asset_store, doc_id)
        if chunks is None:
            if args.doc:
                print(f"doc {doc_id}: không có chunks.json trong MinIO — bỏ qua")
            continue
        docs_scanned += 1
        changed = 0
        for chunk in chunks:
            content = chunk.get("content") or ""
            new = migrate_chunk_markers(content)
            if new != content:
                chunk["content"] = new
                changed += 1
        chunks_scanned += len(chunks)
        if not changed:
            continue
        docs_changed += 1
        chunks_changed += changed
        print(f"doc {doc_id}: {changed}/{len(chunks)} chunk đổi marker")
        if args.yes:
            asset_store.save_json(
                ArtifactPaths.chunks_key(asset_store, doc_id), chunks
            )

    mode = "ĐÃ GHI" if args.yes else "DRY-RUN (không ghi)"
    print(
        f"\n[{mode}] {docs_scanned} doc, {chunks_scanned} chunk; "
        f"đổi marker: {chunks_changed} chunk trong {docs_changed} doc"
    )
    if docs_changed and not args.yes:
        print("Chạy lại với --yes để ghi, sau đó:")
    elif docs_changed and args.yes:
        print("Bước tiếp theo (re-embed + index lại, không gọi API caption):")
    else:
        return 0
    print("    ami-rag reindex --all --from-stage embed --yes")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đổi marker tiếng Anh trong chunks.json cũ sang tiếng Việt"
    )
    parser.add_argument("--doc", help="chỉ chạy 1 doc id (mặc định: mọi doc)")
    parser.add_argument(
        "--yes", action="store_true", help="ghi MinIO (mặc định dry-run)"
    )
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
