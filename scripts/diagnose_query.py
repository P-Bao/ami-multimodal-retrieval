"""Chẩn đoán 1 query end-to-end: vector search -> rerank -> top_k.

In ra, cho từng ứng viên: hạng + điểm vector, modality, doc_id, trang, và text
rút gọn; sau đó in điểm rerank raw và thứ hạng sau rerank để biết ảnh bị rơi ở
tầng nào:

    - image KHÔNG có trong ứng viên vector  -> vấn đề embed (vector recall)
    - image CÓ trong ứng viên nhưng rớt hạng sau rerank -> vấn đề rerank/fusion

Usage:
    python scripts/diagnose_query.py --query "sân bóng đá cơ sở Hà Đông"
    python scripts/diagnose_query.py --query "..." --candidates 80 --top-k 5
    python scripts/diagnose_query.py --query "..." --output diag.json

Không gọi LLM/Gemini: chỉ dùng embed + rerank server (máy B) và Qdrant, không
ghi gì. Cần `.env` (EMBED_SERVER_URL, RERANK_BASE_URL, QDRANT_URL).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ami_rag.core.rerank_client import (
    MULTIMODAL_RERANK_TYPES,
    build_rerank_documents,
    build_rerank_func,
    fuse_modality_scores,
    resolve_rerank_fusion,
)
from ami_rag.settings import get_settings, resolve_rerank_backend

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PREVIEW_CHARS = 110
_MODALITY_WIDTH = 8


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _print_header(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 78 - len(title)))


async def run_diagnose(args: argparse.Namespace) -> dict:
    from ami_rag.core.factory import get_pipeline

    settings = get_settings()
    pipeline = await get_pipeline()
    rerank_func = build_rerank_func(settings)
    backend = resolve_rerank_backend(settings)
    fusion = resolve_rerank_fusion(settings)
    multimodal = getattr(settings, "RERANK_MULTIMODAL", True) and backend == "vllm"

    print(f"collection : {pipeline.collection}")
    print(f"query      : {args.query!r}")
    print(f"embedder   : {type(pipeline.embedder).__name__} model={settings.EMBED_MODEL}")
    print(f"rerank     : backend={backend} fusion={fusion} multimodal={multimodal}")
    print(f"candidates : max({settings.RETRIEVAL_CHUNK_TOP_K}, {args.top_k}*{settings.RETRIEVAL_OVERFETCH}"
          f", {args.candidates}) = {args.candidates}")

    vector = await pipeline.embedder.embed_query(args.query)
    hits = await pipeline.vector_store.search(
        pipeline.collection, vector, args.candidates
    )
    chunks = [hit.payload for hit in hits]

    _print_header("ỨNG VIÊN VECTOR (theo điểm Qdrant giảm dần)")
    print(f"{'#':>3} {'vec':>7}  {'modality':<{_MODALITY_WIDTH}} {'doc_id':<26} {'page':>5}  text")
    candidates: list[dict] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        modality = payload.get("modality") or "text"
        candidates.append({"rank": rank, "vector_score": hit.score, "chunk": payload})
        page = payload.get("page")
        print(
            f"{rank:>3} {hit.score:>7.4f}  {modality:<{_MODALITY_WIDTH}} "
            f"{str(payload.get('doc_id') or '')[:26]:<26} {page!s:>5}  "
            f"{_preview(payload.get('content'))}"
        )

    visual_ranks = [
        c["rank"] for c in candidates if (c["chunk"].get("modality") or "text") == "image"
    ]

    documents = await build_rerank_documents(
        chunks, pipeline.asset_store, multimodal=multimodal
    )
    with_image = sum(1 for d in documents if isinstance(d, dict))
    _print_header("RERANK")
    print(f"{len(documents)} documents ({with_image} kèm ảnh)")

    raw: list[dict] = []
    try:
        results = await rerank_func(
            query=args.query, documents=documents, top_n=len(documents)
        )
        for item in results or []:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(chunks):
                raw.append(
                    {"idx": idx, "score": float(item.get("relevance_score") or 0.0)}
                )
    except Exception as exc:  # noqa: BLE001 - chẩn đoán phải in ra lỗi, không raise
        print(f"!! rerank lỗi: {exc}")

    raw.sort(key=lambda r: -r["score"])
    print(f"\n{'rank':>4} {'raw':>8}  {'modality':<{_MODALITY_WIDTH}} {'doc_id':<26} {'page':>5}  text")
    for rank, r in enumerate(raw, start=1):
        payload = chunks[r["idx"]]
        modality = payload.get("modality") or "text"
        print(
            f"{rank:>4} {r['score']:>8.4f}  {modality:<{_MODALITY_WIDTH}} "
            f"{str(payload.get('doc_id') or '')[:26]:<26} {payload.get('page')!s:>5}  "
            f"{_preview(payload.get('content'))}"
        )

    fused_order = list(raw)
    if fusion == "rrf" and raw:
        fused = fuse_modality_scores(
            [{"index": r["idx"], "relevance_score": r["score"]} for r in raw],
            chunks,
            rrf_k=settings.RERANK_RRF_K,
            visual_weight=settings.RERANK_VISUAL_WEIGHT,
            visual_floor=settings.RERANK_VISUAL_FLOOR,
            text_floor=settings.RERANK_TEXT_FLOOR,
        )
        by_idx = {f["index"]: f["fused_score"] for f in fused}
        fused_order = sorted(
            raw, key=lambda r: (-by_idx.get(r["idx"], r["score"] - 1.0), r["idx"])
        )

    final = fused_order[: args.top_k]
    _print_header(f"KẾT QUẢ TOP {args.top_k} (fusion={fusion})")
    for rank, r in enumerate(final, start=1):
        payload = chunks[r["idx"]]
        print(
            f"{rank:>4} raw={r['score']:>8.4f}  {payload.get('modality') or 'text':<{_MODALITY_WIDTH}} "
            f"{str(payload.get('doc_id') or '')[:26]:<26} page={payload.get('page')!s:>5}  "
            f"{_preview(payload.get('content'))}"
        )

    # --- Kết luận ---
    image_final = [
        (i, chunks[r["idx"]])
        for i, r in enumerate(final, start=1)
        if (chunks[r["idx"]].get("modality") or "text") == "image"
    ]
    all_image_final = [
        (i, r) for i, r in enumerate(final, start=1)
        if (chunks[r["idx"]].get("modality") or "text") in MULTIMODAL_RERANK_TYPES
    ]
    best_text_raw = next(
        (r["score"] for r in raw if (chunks[r["idx"]].get("modality") or "text") != "image"),
        None,
    )
    best_image_raw = next(
        (r["score"] for r in raw if (chunks[r["idx"]].get("modality") or "text") == "image"),
        None,
    )

    _print_header("KẾT LUẬN")
    print(f"ứng viên vector có ảnh : {len(visual_ranks)}" + (
        f" (hạng vector {visual_ranks})" if visual_ranks else " -> KHÔNG có ảnh nào trong ứng viên"
    ))
    print(f"best image raw score   : {best_image_raw}")
    print(f"best text  raw score   : {best_text_raw}")
    if best_image_raw is not None and best_text_raw is not None:
        gap = best_text_raw - best_image_raw
        print(f"chênh lệch text - image: {gap:+.4f}"
              + ("  -> image thua text (vấn đề RERANK)" if gap > 0 else "  -> image đang thắng text"))
    print(f"ảnh trong top {args.top_k}      : {len(image_final)}"
          f"{' (modal ' + str(MULTIMODAL_RERANK_TYPES) + ': ' + str(len(all_image_final)) + ')' if not image_final else ''}")
    if visual_ranks and not image_final:
        print("-> ẢNH CÓ trong ứng viên vector nhưng bị cắt ở rerank: "
              "cần sửa tầng rerank (RERANK_MIN_VISUAL / RERANK_FUSION / weight).")
    elif not visual_ranks:
        print("-> ẢNH KHÔNG vào được ứng viên vector: cần sửa tầng embed "
              "(vector recall) hoặc index chưa có chunk ảnh.")

    payload_out = {
        "query": args.query,
        "collection": pipeline.collection,
        "rerank_backend": backend,
        "fusion": fusion,
        "candidates": [
            {
                "rank": c["rank"],
                "vector_score": c["vector_score"],
                "modality": c["chunk"].get("modality") or "text",
                "doc_id": c["chunk"].get("doc_id"),
                "page": c["chunk"].get("page"),
                "chunk_id": c["chunk"].get("chunk_id"),
                "content": c["chunk"].get("content"),
            }
            for c in candidates
        ],
        "rerank": [
            {
                "rank": i,
                "idx": r["idx"],
                "raw": r["score"],
                "modality": chunks[r["idx"]].get("modality") or "text",
                "chunk_id": chunks[r["idx"]].get("chunk_id"),
                "doc_id": chunks[r["idx"]].get("doc_id"),
            }
            for i, r in enumerate(fused_order, start=1)
        ],
        "final_top_k": [
            {
                "rank": i,
                "raw": r["score"],
                "modality": chunks[r["idx"]].get("modality") or "text",
                "chunk_id": chunks[r["idx"]].get("chunk_id"),
                "doc_id": chunks[r["idx"]].get("doc_id"),
                "content": chunks[r["idx"]].get("content"),
            }
            for i, r in enumerate(final, start=1)
        ],
    }

    await pipeline.embedder.close()
    return payload_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Chẩn đoán retrieval 1 query")
    parser.add_argument("--query", required=True, help="câu hỏi cần chẩn đoán")
    parser.add_argument("--top-k", type=int, default=5, help="số doc cuối cùng (giống RAG request)")
    parser.add_argument("--candidates", type=int, default=0,
                        help="số ứng viên lấy từ Qdrant (0 = max(RETRIEVAL_CHUNK_TOP_K, top_k*2))")
    parser.add_argument("--output", help="ghi kết quả JSON ra file")
    args = parser.parse_args()
    settings = get_settings()
    if not args.candidates:
        args.candidates = max(
            settings.RETRIEVAL_CHUNK_TOP_K,
            args.top_k * 2,
        )
    result = asyncio.run(run_diagnose(args))
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nđã ghi {args.output}")


if __name__ == "__main__":
    main()