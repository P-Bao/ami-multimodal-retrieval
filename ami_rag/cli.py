"""AMI RAG maintenance CLI - 4 lệnh: status, retry, reindex, cleanup.

- `status`: xem trạng thái doc (mặc định chỉ đọc local: 0 LLM, 0 embed, 0 model call).
- `retry`: chạy lại doc `failed`, tiếp tục từ stage lỗi bằng dữ liệu trung gian đã lưu.
- `reindex`: xử lý lại thủ công (chunk -> embed từ dữ liệu parse + mô tả modal đã lưu).
- `cleanup`: kiểm tra + xoá collection Mongo/Qdrant rác của pipeline LightRAG cũ;
  `--stale-models` xoá thêm collection embed model cũ (Qdrant, ví dụ bản Qwen);
  `--purge-cache-model` xoá cache embedding SQLite theo model.

Pipeline thật (chunk -> embed -> upsert Qdrant) được cắm qua `PipelineRunner`
(triển khai ở giai đoạn chuyển pipeline); CLI chỉ orchestrate và đọc DocStatusStore.
"""

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from ami_rag.settings import get_settings

LOCK_PATH = Path("./.ami_rag_cli.lock")
STAGES = ("parse", "describe", "chunk", "embed", "indexed")


def _build_docs_repo(settings):
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    return MongoDocumentRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.ORG_DB,
        collection_name=settings.DOC_COLLECTION,
    )


def _build_status_store(settings):
    from ami_rag.storage.doc_status import DocStatusStore

    return DocStatusStore(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.RAG_DB,
        collection_name=settings.RAG_DOCUMENTS_COLLECTION,
    )


def _build_runner(settings):
    from ami_rag.core.pipeline import create_runner

    return create_runner(settings)


def _build_embedder(settings):
    from ami_rag.core.factory import build_embedder

    return build_embedder(settings)


def _build_mongo_db(settings):
    from pymongo import MongoClient

    return MongoClient(settings.MONGO_URI, tz_aware=True)[settings.RAG_DB]


def _build_qdrant_client(settings):
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def _resolve_doc_ids(raw_ids: list[str], docs_repo, store) -> list[tuple[str, str]]:
    """Chuyển --doc <id|path> thành (doc_id, label). Path được resolve qua docs repo."""
    resolved = []
    for raw in raw_ids:
        row = store.get(raw)
        if row:
            resolved.append((raw, row.get("source_path") or raw))
            continue
        doc = docs_repo.find_by_id(raw) or docs_repo.find_by_file_path(raw)
        if doc:
            resolved.append((str(doc["_id"]), doc.get("file_path") or raw))
        else:
            resolved.append((raw, raw))
    return resolved


def _fmt_time(value) -> str:
    return str(value)[:19] if value else "-"


def _print_rows(rows: list[dict]) -> None:
    print(f"{'doc_id':<26}{'status':<12}{'stage':<10}{'attempts':>9}  {'updated_at':<20}error")
    for row in rows:
        error = (row.get("error") or "")[:80]
        print(
            f"{row['_id']:<26}{row.get('effective') or row.get('status', '-'):<12}"
            f"{row.get('stage') or '-':<10}{row.get('attempts', 0):>9}  "
            f"{_fmt_time(row.get('updated_at')):<20}{error}"
        )


def _detail_line(row: dict) -> str:
    lines = [
        f"doc_id:         {row['_id']}",
        (
            f"status:         {row.get('effective') or row.get('status', '-')}"
            f"  stage: {row.get('stage') or '-'}"
        ),
        f"source_path:    {row.get('source_path') or '-'}",
        f"content_hash:   {row.get('content_hash') or '-'}",
        f"attempts:       {row.get('attempts', 0)}",
        f"chunk_count:    {row.get('chunk_count', 0)}",
        f"embed_model:    {row.get('embed_model') or '-'}  dim: {row.get('embed_dim') or '-'}",
        f"chunker_version:{row.get('chunker_version') or '-'}",
        f"updated_at:     {_fmt_time(row.get('updated_at'))}",
    ]
    if row.get("error"):
        lines.append(f"error ({row.get('error_stage') or '-'}): {row['error'][:400]}")
    return "\n".join(lines)


def _apply_embed_overrides(args, settings) -> None:
    if getattr(args, "embed_server_url", None):
        settings.EMBED_SERVER_URL = args.embed_server_url
    if getattr(args, "embed_batch_size", None):
        settings.EMBED_BATCH_SIZE = args.embed_batch_size


def _confirm(prompt: str, *, yes: bool) -> bool:
    if yes:
        return True
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
async def cmd_status(args, *, settings=None, store=None, docs_repo=None, embedder=None) -> None:
    """Xem trạng thái doc. Mặc định chỉ đọc local: 0 LLM, 0 embed, 0 model call."""
    import datetime

    settings = settings or get_settings()
    store = store or _build_status_store(settings)
    docs_repo = docs_repo or _build_docs_repo(settings)

    rows = await asyncio.to_thread(store.all_rows)
    for row in rows:
        from ami_rag.storage.doc_status import effective_status

        row["effective"] = effective_status(
            row, settings.EMBED_MODEL, settings.CHUNKER_VERSION
        )
    counts = Counter(row["effective"] for row in rows)

    total_docs = await asyncio.to_thread(docs_repo.count)
    missing = max(total_docs - len(rows), 0)
    if missing:
        counts["pending (thiếu bản ghi)"] = missing

    from ami_rag.core.embedder import collection_name

    print(f"workspace: {settings.WORKSPACE}")
    print(
        f"config: embed_model={settings.EMBED_MODEL} dim={settings.EMBED_DIM} "
        f"chunker_version={settings.CHUNKER_VERSION}"
    )
    print(
        f"collection đang dùng: {collection_name(settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION)}"
    )
    print(f"documents trong org_db: {total_docs}")
    print(f"trạng thái: {dict(sorted(counts.items())) or '{}'}")

    # doc `processing` quá lâu (tiến trình chết giữa chừng) -> treo
    stuck_cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=settings.CLI_STUCK_PROCESSING_MINUTES
    )
    stuck = [
        row
        for row in rows
        if row.get("status") == "processing"
        and row.get("updated_at")
        and row["updated_at"] < stuck_cutoff
    ]
    if stuck:
        print(f"TREO: {len(stuck)} doc đang `processing` quá "
              f"{settings.CLI_STUCK_PROCESSING_MINUTES} phút (tiến trình chết giữa chừng?):")
        for row in stuck[:10]:
            print(f"  {row['_id']} stage={row.get('stage') or '-'} updated_at={_fmt_time(row.get('updated_at'))}")

    if args.doc:
        for doc_id, label in _resolve_doc_ids([args.doc], docs_repo, store):
            row = store.get(doc_id)
            if row is None:
                print(f"\n{label}: không có bản ghi (pending)")
            else:
                from ami_rag.storage.doc_status import effective_status as es

                row["effective"] = es(row, settings.EMBED_MODEL, settings.CHUNKER_VERSION)
                print(f"\n{label}:")
                print(_detail_line(row))
        return

    from ami_rag.storage.doc_status import STATUS_FAILED, STATUS_STALE

    if args.failed:
        failed = [row for row in rows if row["effective"] == STATUS_FAILED]
        print(f"\nfailed ({len(failed)}):")
        if failed:
            _print_rows(failed)
        return
    if args.stale:
        stale = [row for row in rows if row["effective"] == STATUS_STALE]
        print(f"\nstale ({len(stale)}):")
        if stale:
            _print_rows(stale)
        return

    if args.check_embed_server:
        await _check_embed_server(settings, embedder)


async def _check_embed_server(settings, embedder=None) -> None:
    import httpx

    embedder = embedder or _build_embedder(settings)
    health_line = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.EMBED_SERVER_URL.rstrip('/')}/health")
            try:
                body = resp.json() if resp.status_code == 200 else {}
            except ValueError:
                body = {}
            health_line = (
                f"embed server /health: {'ok' if resp.status_code == 200 else resp.status_code}"
                f"  model={body.get('model', '-')} device={body.get('device', '-')}"
            )
    except Exception as exc:
        health_line = f"embed server /health: unreachable ({settings.EMBED_SERVER_URL}): {exc}"
    print(health_line)
    # Dù /health unreachable, vẫn thử handshake để báo lỗi rõ hơn.
    try:
        await embedder.verify()
        print(
            f"embed server handshake: model={getattr(embedder, 'model', settings.EMBED_MODEL)} "
            f"dim={embedder.dim} "
            f"khớp config (EMBED_MODEL={settings.EMBED_MODEL}, EMBED_DIM={settings.EMBED_DIM})"
        )
    except Exception as exc:
        print(f"embed server handshake: LỆCH hoặc lỗi: {exc}")


# --------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------
async def cmd_retry(args, *, settings=None, store=None, docs_repo=None, runner=None) -> int:
    """Chạy lại doc `failed`, tiếp tục từ stage lỗi bằng dữ liệu trung gian đã lưu."""
    settings = settings or get_settings()
    store = store or _build_status_store(settings)
    docs_repo = docs_repo or _build_docs_repo(settings)
    runner = runner or _build_runner(settings)

    rows = await asyncio.to_thread(store.all_rows)
    failed = [row for row in rows if row.get("status") == "failed"]

    if args.doc:
        wanted = {doc_id for doc_id, _ in _resolve_doc_ids(args.doc, docs_repo, store)}
        selected = [row for row in failed if row["_id"] in wanted]
        not_failed = wanted - {row["_id"] for row in selected}
        for doc_id in sorted(not_failed):
            print(f"bỏ qua {doc_id}: không phải doc `failed`")
    elif args.all_failed:
        selected = failed
    else:
        raise ValueError("retry cần --doc hoặc --all-failed")

    skipped_attempts = [row for row in selected if row.get("attempts", 0) >= args.max_attempts]
    selected = [row for row in selected if row.get("attempts", 0) < args.max_attempts]
    for row in skipped_attempts:
        print(
            f"bỏ qua {row['_id']}: đã thử {row.get('attempts', 0)} lần >= "
            f"--max-attempts {args.max_attempts}"
        )

    if not selected:
        print("không có doc nào để retry")
        return 0

    print(f"retry {len(selected)} doc (từ stage lỗi"
          f"{f', ép --from-stage {args.from_stage}' if args.from_stage else ''})")
    if not _confirm("Chạy retry?", yes=args.yes):
        print("đã huỷ")
        return 0

    # Trước khi chạy (các doc sẽ qua stage embed): kiểm tra embed server.
    # Không với tới được -> dừng sớm, KHÔNG đánh fail từng doc.
    try:
        await runner.preflight_embed()
    except Exception as exc:
        print(f"STOPPED: embed server không sẵn sàng: {exc}")
        print("  không đánh fail doc nào; sửa server rồi chạy lại `ami-rag retry`")
        return 2

    ok = failed_n = 0
    for row in selected:
        doc_id = row["_id"]
        try:
            outcome = await runner.run(doc_id, from_stage=args.from_stage)
        except Exception as exc:
            failed_n += 1
            print(f"FAILED {doc_id}: {str(exc)[:200]}")
            continue
        if outcome.ok:
            ok += 1
            print(f"indexed {doc_id} chunks={outcome.chunk_count}")
        else:
            failed_n += 1
            print(f"FAILED {doc_id} (stage={outcome.stage}): {outcome.error[:200]}")

    print(f"retry xong: thành công={ok} thất bại={failed_n} bỏ qua={len(skipped_attempts)}")
    return 1 if failed_n else 0


# --------------------------------------------------------------------------
# reindex
# --------------------------------------------------------------------------
async def cmd_reindex(args, *, settings=None, store=None, runner=None, docs_repo=None) -> int:
    """Xử lý lại: chunk -> embed từ dữ liệu parse đã lưu; --scan quét docs cũ."""
    from ami_rag.core.lockfile import Lockfile, LockHeld
    from ami_rag.core.pipeline import validate_stage
    from ami_rag.storage.doc_status import STATUS_STALE, effective_status

    settings = settings or get_settings()
    store = store or _build_status_store(settings)
    runner = runner or _build_runner(settings)
    docs_repo = docs_repo or _build_docs_repo(settings)
    _apply_embed_overrides(args, settings)
    from_stage = validate_stage(args.from_stage)

    rows = await asyncio.to_thread(store.all_rows)
    for row in rows:
        row["effective"] = effective_status(
            row, settings.EMBED_MODEL, settings.CHUNKER_VERSION
        )

    if args.doc:
        resolved = _resolve_doc_ids(args.doc, docs_repo, store)
        by_id = {row["_id"]: row for row in rows}
        selected = [by_id[doc_id] for doc_id, _ in resolved if doc_id in by_id]
        missing = [doc_id for doc_id, _ in resolved if doc_id not in by_id]
        for doc_id in missing:
            print(f"bỏ qua {doc_id}: không có bản ghi trạng thái (pending)")
    elif args.scan:
        # Scan backend docs -> tạo pending records cho docs thiếu bản ghi,
        # rồi chọn pending + stale (bao gồm legacy `processed`).
        from ami_rag.core.scan import reindex_candidates, scan_pending

        summary = await scan_pending(store, docs_repo)
        print(
            f"scan: tạo mới {summary['created']} pending, "
            f"{summary['existing']} đã có bản ghi"
        )
        rows = await asyncio.to_thread(store.all_rows)
        for row in rows:
            row["effective"] = effective_status(
                row, settings.EMBED_MODEL, settings.CHUNKER_VERSION
            )
        selected = reindex_candidates(
            rows,
            embed_model=settings.EMBED_MODEL,
            chunker_version=settings.CHUNKER_VERSION,
        )
    elif args.all:
        selected = list(rows)
    else:  # --stale (mặc định)
        selected = [row for row in rows if row["effective"] == STATUS_STALE]

    if not selected:
        print("không có doc nào cần reindex")
        return 0

    if args.dry_run:
        total_chunks = total_cache_hits = 0
        for row in selected:
            outcome = await runner.run(
                row["_id"], from_stage=_from_stage_for(row, from_stage, runner), dry_run=True
            )
            total_chunks += outcome.chunk_count
            total_cache_hits += outcome.cache_hits
        print(
            f"dry-run: {len(selected)} doc, {total_chunks} chunk cần embed, "
            f"{total_cache_hits} trúng cache (không gọi embed, không ghi gì)"
        )
        return 0

    try:
        await runner.preflight_embed()
    except Exception as exc:
        print(f"STOPPED: embed server không sẵn sàng: {exc}")
        print("  không đánh fail doc nào; sửa server rồi chạy lại `ami-rag reindex`")
        return 2

    if not _confirm(f"Reindex {len(selected)} doc (chunk -> embed)?", yes=args.yes):
        print("đã huỷ")
        return 0

    lock = Lockfile(LOCK_PATH)
    try:
        lock.acquire()
    except LockHeld as exc:
        print(f"STOPPED: {exc}")
        return 2

    ok = failed_n = 0
    try:
        for row in selected:
            try:
                outcome = await runner.run(
                    row["_id"], from_stage=_from_stage_for(row, from_stage, runner)
                )
            except Exception as exc:
                failed_n += 1
                print(f"FAILED {row['_id']}: {str(exc)[:200]}")
                continue
            if outcome.ok:
                ok += 1
                print(f"indexed {row['_id']} chunks={outcome.chunk_count}")
            else:
                failed_n += 1
                print(f"FAILED {row['_id']} (stage={outcome.stage}): {outcome.error[:200]}")
    finally:
        lock.release()

    print(f"reindex xong: thành công={ok} thất bại={failed_n}")
    return 1 if failed_n else 0


def _from_stage_for(row: dict, from_stage: str | None, runner) -> str | None:
    """Per-doc from_stage: None (full parse) khi content_list thiếu trong MinIO;
    "chunk" khi đã có (reindex không gọi lại parse/LLM)."""
    if from_stage is not None:
        return from_stage
    try:
        if runner.asset_store.load_content_list(row["_id"]) is not None:
            return "chunk"
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------
async def cmd_cleanup(args, *, settings=None, mongo_db=None, qdrant_client=None) -> int:
    """Kiểm tra + xoá collection Mongo/Qdrant rác của pipeline LightRAG cũ.

    `--stale-models`: dò + xoá thêm collection Qdrant của embed model/version cũ
    (`{WORKSPACE}__*` không phải collection hiện tại — ví dụ bản Qwen sau khi đổi
    sang Nemotron). `--purge-cache-model NAME...`: xoá cache embedding SQLite theo
    model. Chỉ đọc để dò (list + count); xoá sau khi có xác nhận. Không đụng:
    registry `RAG_DOCUMENTS_COLLECTION`, collection Qdrant hiện tại
    (`{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`), collection ngoài workspace.
    """
    from ami_rag.core.embedder import collection_name
    from ami_rag.core.legacy_cleanup import (
        delete_mongo_collections,
        delete_qdrant_collections,
        find_legacy_mongo,
        find_legacy_qdrant,
        find_stale_model_collections,
    )

    settings = settings or get_settings()
    do_mongo = not args.qdrant_only
    do_qdrant = not args.mongo_only

    targets = []
    try:
        if do_mongo:
            mongo_db = mongo_db or _build_mongo_db(settings)
            targets.extend(
                await asyncio.to_thread(
                    find_legacy_mongo,
                    mongo_db,
                    settings.WORKSPACE,
                    settings.RAG_DOCUMENTS_COLLECTION,
                )
            )
        if do_qdrant:
            qdrant_client = qdrant_client or _build_qdrant_client(settings)
            targets.extend(
                await asyncio.to_thread(find_legacy_qdrant, qdrant_client, settings.WORKSPACE)
            )
            if getattr(args, "stale_models", False):
                current = collection_name(
                    settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION
                )
                targets.extend(
                    await asyncio.to_thread(
                        find_stale_model_collections,
                        qdrant_client,
                        settings.WORKSPACE,
                        current,
                    )
                )
    except Exception as exc:
        print(f"STOPPED: không dò được collection legacy: {exc}")
        return 2

    current = collection_name(settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION)
    print(f"workspace: {settings.WORKSPACE}")
    print(
        f"giữ lại: Mongo `{settings.RAG_DOCUMENTS_COLLECTION}` (registry/DocStatusStore), "
        f"Qdrant `{current}` (collection hiện tại)"
        + ("" if getattr(args, "stale_models", False) else f" và mọi `{settings.WORKSPACE}__*` (vector pipeline)")
    )
    if not targets:
        print("không có collection legacy nào cần xoá")
        return 0

    for t in targets:
        size = f"{t.count:,}" if t.count >= 0 else "?"
        print(f"  {t.kind} [{t.db}] {t.name}  ({size} documents/points)")
    if args.dry_run:
        print(f"dry-run: {len(targets)} collection legacy sẽ bị xoá (không ghi gì)")
        return 0

    if not _confirm(f"Xoá {len(targets)} collection legacy (KHÔNG THỂ HOÀN TÁC)?", yes=args.yes):
        print("đã huỷ")
        return 0

    deleted: list[str] = []
    failed: list[str] = []
    try:
        if do_mongo:
            names = [t.name for t in targets if t.db == "mongo"]
            ok, err = await asyncio.to_thread(delete_mongo_collections, mongo_db, names)
            deleted.extend(ok)
            failed.extend(err)
        if do_qdrant:
            names = [t.name for t in targets if t.db == "qdrant"]
            ok, err = await asyncio.to_thread(delete_qdrant_collections, qdrant_client, names)
            deleted.extend(ok)
            failed.extend(err)
    except Exception as exc:
        print(f"STOPPED: lỗi khi xoá: {exc}")
        return 2

    purge_models = getattr(args, "purge_cache_model", None) or []
    if purge_models:
        try:
            from ami_rag.core.embedding_cache import EmbeddingCache

            cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
            for model in purge_models:
                purged = await asyncio.to_thread(cache.delete_by_prefix, f"{model}|")
                print(f"cache: đã xoá {purged} entry của model `{model}`")
            await asyncio.to_thread(cache.close)
        except Exception as exc:
            print(f"LỖI purge cache: {exc}")
            failed.append(f"cache:{','.join(purge_models)}")

    for name in failed:
        print(f"LỖI xoá: {name}")
    print(f"cleanup xong: đã xoá {len(deleted)}/{len(targets)} collection legacy")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ami-rag",
        description="AMI RAG maintenance CLI: status / retry / reindex / cleanup",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser(
        "status", help="trạng thái doc (mặc định chỉ đọc local, 0 model call)"
    )
    p_status.add_argument("--failed", action="store_true", help="chỉ hiện doc failed")
    p_status.add_argument("--stale", action="store_true", help="chỉ hiện doc stale")
    p_status.add_argument("--doc", default=None, help="xem chi tiết một doc (id hoặc path)")
    p_status.add_argument(
        "--check-embed-server",
        action="store_true",
        help="gọi /health + handshake kiểm tra embed server (máy B) có sẵn sàng và đúng model",
    )

    p_retry = sub.add_parser("retry", help="chạy lại doc failed từ stage lỗi")
    p_retry.add_argument("--doc", nargs="*", default=[], help="id hoặc path cụ thể")
    p_retry.add_argument("--all-failed", action="store_true", help="chạy lại mọi doc failed")
    p_retry.add_argument(
        "--from-stage",
        choices=list(STAGES[:-1]),
        default=None,
        help="ép làm lại từ stage này (mặc định: tiếp tục từ stage lỗi)",
    )
    p_retry.add_argument("--max-attempts", type=int, default=3)
    p_retry.add_argument("--yes", action="store_true", help="bỏ qua xác nhận")
    p_retry.add_argument("--embed-server-url", default=None, help="override EMBED_SERVER_URL")
    p_retry.add_argument("--embed-batch-size", type=int, default=None, help="override EMBED_BATCH_SIZE")

    p_reindex = sub.add_parser(
        "reindex", help="xử lý lại thủ công: chunk -> embed từ dữ liệu đã lưu"
    )
    p_reindex.add_argument(
        "--stale",
        action="store_true",
        default=True,
        help="chỉ doc stale (mặc định)",
    )
    p_reindex.add_argument(
        "--scan",
        action="store_true",
        help="quét backend docs -> tạo pending records cho docs thiếu bản ghi, "
        "rồi reindex pending + stale (bao gồm legacy `processed` của pipeline cũ)",
    )
    p_reindex.add_argument("--doc", nargs="*", default=[], help="id hoặc path cụ thể")
    p_reindex.add_argument("--all", action="store_true", help="xử lý lại mọi doc có bản ghi")
    p_reindex.add_argument(
        "--from-stage",
        choices=list(STAGES[:-1]),
        default=None,
        help="ép làm lại từ stage này (mặc định: chunk nếu content_list đã có "
        "trong MinIO, full parse nếu thiếu)",
    )
    p_reindex.add_argument(
        "--dry-run",
        action="store_true",
        help="in số doc, số chunk cần embed, số chunk trúng cache; không gọi embed, không ghi",
    )
    p_reindex.add_argument("--yes", action="store_true", help="bỏ qua xác nhận")
    p_reindex.add_argument("--embed-server-url", default=None, help="override EMBED_SERVER_URL")
    p_reindex.add_argument("--embed-batch-size", type=int, default=None, help="override EMBED_BATCH_SIZE")

    p_cleanup = sub.add_parser(
        "cleanup",
        help="kiểm tra + xoá collection Mongo/Qdrant legacy của pipeline LightRAG cũ",
    )
    p_cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="chỉ liệt kê collection legacy (kèm số document/points), không xoá",
    )
    p_cleanup.add_argument("--yes", action="store_true", help="bỏ qua xác nhận")
    p_cleanup.add_argument(
        "--stale-models",
        action="store_true",
        help="xoá thêm collection Qdrant của embed model/version cũ "
        "(`{WORKSPACE}__*` không phải collection hiện tại, ví dụ bản Qwen)",
    )
    p_cleanup.add_argument(
        "--purge-cache-model",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="xoá cache embedding SQLite của model (ví dụ 'Qwen/Qwen3-VL-Embedding-2B')",
    )
    p_cleanup.add_argument(
        "--mongo-only",
        action="store_true",
        help="chỉ dò/xoá collection Mongo legacy (bỏ qua Qdrant)",
    )
    p_cleanup.add_argument(
        "--qdrant-only",
        action="store_true",
        help="chỉ dò/xoá collection Qdrant legacy (bỏ qua Mongo)",
    )

    args = parser.parse_args()
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "pymongo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "retry" and not (args.doc or args.all_failed):
        parser.error("retry cần --doc hoặc --all-failed")
    if args.command == "status":
        asyncio.run(cmd_status(args))
    elif args.command == "retry":
        rc = asyncio.run(cmd_retry(args))
        if rc:
            raise SystemExit(rc)
    elif args.command == "reindex":
        rc = asyncio.run(cmd_reindex(args))
        if rc:
            raise SystemExit(rc)
    elif args.command == "cleanup":
        rc = asyncio.run(cmd_cleanup(args))
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
