import asyncio
import logging
import time

from fastapi import APIRouter, Depends, Header, HTTPException

from ami_rag.api.schemas import (
    RAGRequest,
    RetrievalResponseV2,
    RetrievedDocV2,
)
from ami_rag.observability import (
    ARTIFACT_URLS_RETURNED_TOTAL,
    CHUNKS_RETRIEVED,
    EMBED_FAILURES_TOTAL,
    PRESIGN_FAILURES_TOTAL,
    RERANK_FALLBACK_TOTAL,
    RETRIEVAL_DOCS_BY_MODALITY_TOTAL,
    SEARCH_MODE_TOTAL,
    embed_failure_reason,
    observe_stage,
    track_retrieval,
)
from ami_rag.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


async def get_pipeline():
    from ami_rag.core.factory import get_pipeline as _get_pipeline

    return await _get_pipeline()


def get_asset_store():
    from ami_rag.core.factory import get_asset_store as _get_asset_store

    return _get_asset_store()


def get_rerank_func():
    from ami_rag.core.rerank_client import build_rerank_func

    return build_rerank_func(get_settings())


_resolver_instance = None


def get_resolver():
    global _resolver_instance
    if _resolver_instance is None:
        from ami_rag.api.resolver import DocResolver
        from ami_rag.storage.mongo_docs import MongoDocumentRepo

        settings = get_settings()
        _resolver_instance = DocResolver(
            docs_repo=MongoDocumentRepo(
                mongo_uri=settings.MONGO_URI,
                db_name=settings.ORG_DB,
                collection_name=settings.DOC_COLLECTION,
            ),
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_access_key=settings.MINIO_ACCESS_KEY,
            minio_secret_key=settings.MINIO_SECRET_KEY,
            minio_bucket=settings.MINIO_BUCKET,
            minio_secure=settings.MINIO_SECURE,
            presign_expires=settings.MINIO_PRESIGN_EXPIRES,
        )
    return _resolver_instance


async def verify_api_key(authorization: str = Header(default="")):
    settings = get_settings()
    if not settings.RAG_API_KEY:
        return
    token = authorization.replace("Bearer ", "").strip()
    if token != settings.RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _last_user_message(payload: RAGRequest) -> str:
    if not payload.messages:
        return ""
    for message in reversed(payload.messages):
        if message.role == "user":
            return message.content
    return payload.messages[-1].content


async def _rerank_chunks(
    rerank_func, query: str, chunks: list[dict], top_n: int, asset_store=None
) -> tuple[list[tuple[dict, float | None]], str]:
    """Rerank candidates. Trả ([(chunk, raw_score)], fusion) với fusion đã dùng.

    Fusion "rrf": xin full scores (top_n = toàn bộ documents, gateway chấm điểm
    mọi documents rồi mới cắt) rồi so hạng nội bộ 2 nhóm modality (image vs
    phần còn lại) qua ``fuse_modality_scores`` - điểm cross-encoder không so
    sánh được giữa image (thấp) và text (cao).
    """
    if not chunks:
        return [], "empty"
    settings = get_settings()
    from ami_rag.core.rerank_client import (
        build_rerank_documents,
        fuse_modality_scores,
        resolve_rerank_fusion,
    )
    from ami_rag.settings import resolve_rerank_backend

    # Multimodal chỉ với backend vllm (legacy BGE chỉ nhận documents text)
    backend = resolve_rerank_backend(settings)
    multimodal = getattr(settings, "RERANK_MULTIMODAL", True) and backend == "vllm"
    documents = await build_rerank_documents(chunks, asset_store, multimodal=multimodal)
    reason = "empty"
    # Fusion rrf: legacy giờ trả full scores (top_n = len) nên thay vì gate
    # theo backend vllm, chỉ cần fusion bật là xin đủ điểm rồi so hạng nội bộ.
    fusion = resolve_rerank_fusion(settings) == "rrf"
    try:
        with observe_stage("rerank"):
            results = await rerank_func(
                query=query,
                documents=documents,
                top_n=len(documents) if fusion else top_n,
            )
        scored = []
        for result in results or []:
            idx = result.get("index")
            if isinstance(idx, int) and 0 <= idx < len(chunks):
                scored.append((chunks[idx], result.get("relevance_score")))
        if fusion and scored:
            fused = fuse_modality_scores(
                results,
                chunks,
                rrf_k=settings.RERANK_RRF_K,
                visual_weight=settings.RERANK_VISUAL_WEIGHT,
                visual_floor=settings.RERANK_VISUAL_FLOOR,
                text_floor=settings.RERANK_TEXT_FLOOR,
            )
            return (
                [(chunks[f["index"]], f["relevance_score"]) for f in fused][:top_n],
                "rrf",
            )
        if scored:
            return scored, "raw"
    except Exception as exc:
        reason = "error"
        logger.warning("rerank service failed, falling back to unscored chunks: %s", exc)
    RERANK_FALLBACK_TOTAL.labels(reason=reason).inc()
    return [(chunk, None) for chunk in chunks[:top_n]], "raw"


async def _presign_asset(asset_store, asset_key: str | None) -> str | None:
    if not asset_key:
        return None
    url = None
    if asset_store is not None:
        try:
            url = await asyncio.to_thread(asset_store.presign, asset_key)
        except Exception:
            url = None
    if url is None:
        PRESIGN_FAILURES_TOTAL.inc()
    return url


async def _build_documents(scored, resolver, asset_store=None) -> list:
    """Build v2 documents: resolve mỗi chunk qua doc_id (payload mang doc_id trực tiếp)."""
    documents = []
    resolved_cache: dict[str, dict | None] = {}
    for chunk, score in scored:
        doc_id = chunk.get("doc_id") or ""
        if doc_id not in resolved_cache:
            try:
                resolved_cache[doc_id] = await resolver.resolve_doc_id(doc_id)
            except Exception:
                resolved_cache[doc_id] = None
        resolved = resolved_cache[doc_id]
        modality = chunk.get("modality") or "text"
        page = chunk.get("page")
        artifact_urls: list[str] = []
        if resolved is not None and asset_store is not None:
            # Danh sách ảnh của doc (crawl 'Hình ảnh' + ảnh pdf/table) presigned;
            # store thiếu method / lỗi mạng -> [] giữ shape key.
            try:
                artifact_urls = await asyncio.to_thread(
                    asset_store.presign_doc_images, doc_id
                )
            except Exception:
                artifact_urls = []
        ARTIFACT_URLS_RETURNED_TOTAL.inc(len(artifact_urls))
        metadata = {
            "source": chunk.get("source_path") or "",
            "page": page,
            "document_id": doc_id,
            "chunk_index": None,
            "global_id": chunk.get("chunk_id"),
        }
        documents.append(
            RetrievedDocV2(
                text=chunk.get("content") or "",
                score=score,
                metadata=metadata,
                reference_id=chunk.get("chunk_id"),
                doc=resolved,
                modality=modality,
                artifact_url=await _presign_asset(asset_store, chunk.get("asset_key")),
                artifact_urls=artifact_urls,
                table_body=chunk.get("table_body"),
                page=page,
                caption=chunk.get("caption"),
            )
        )
        RETRIEVAL_DOCS_BY_MODALITY_TOTAL.labels(modality=modality).inc()
    return documents


def _final_top_k(payload: RAGRequest, settings) -> int:
    """Number of documents returned to the caller (client `top_k`, else RERANK_TOP_K)."""
    return payload.top_k or settings.RERANK_TOP_K


async def _search(payload: RAGRequest, pipeline, rerank_func, resolver, asset_store=None) -> dict:
    settings = get_settings()
    query = _last_user_message(payload)
    top_k = _final_top_k(payload, settings)
    with track_retrieval(query, "vector", 2, top_k) as tracker:
        response = await _run_search(
            payload, pipeline, rerank_func, resolver, asset_store, query, top_k
        )
        tracker.set_documents(response["documents"])
        return response


async def _run_search(
    payload: RAGRequest,
    pipeline,
    rerank_func,
    resolver,
    asset_store,
    query: str,
    top_k: int,
) -> dict:
    """embed (máy B) -> vector search -> rerank. `top_k * RETRIEVAL_OVERFETCH`
    candidates để rerank luôn đủ `top_k` kết quả khi có."""
    settings = get_settings()
    started = time.perf_counter()
    candidates_wanted = top_k * max(1, settings.RETRIEVAL_OVERFETCH)
    with observe_stage("embed_query"):
        # Hybrid: embedder có embed_query_sparse -> lấy dense + sparse của query;
        # embedder legacy (chỉ embed_query) -> dense-only (sparse=None).
        embed_q = getattr(pipeline.embedder, "embed_query_sparse", None)
        try:
            if embed_q is not None:
                vector, sparse = await embed_q(query)
            else:
                vector = await pipeline.embedder.embed_query(query)
                sparse = None
        except Exception as exc:
            EMBED_FAILURES_TOTAL.labels(reason=embed_failure_reason(exc)).inc()
            raise
        # sparse dict rỗng (0 indices) -> dense-only
        if sparse is not None and not sparse.get("indices"):
            sparse = None
    # Mode thực tế dùng để search (dense khi không có sparse) - đếm theo lượt retrieval
    SEARCH_MODE_TOTAL.labels(mode="hybrid" if sparse is not None else "dense").inc()
    with observe_stage("vector_search"):
        hits = await pipeline.vector_store.search(
            pipeline.collection,
            vector,
            max(settings.RETRIEVAL_CHUNK_TOP_K, candidates_wanted),
            sparse=sparse,
        )
    chunks = [hit.payload for hit in hits]
    CHUNKS_RETRIEVED.observe(len(chunks))
    candidates = len(chunks)
    scored, fusion_used = await _rerank_chunks(rerank_func, query, chunks, top_k, asset_store)
    scored = scored[:top_k]
    documents = await _build_documents(scored, resolver, asset_store)
    references = []
    if payload.include_references:
        seen: set[str] = set()
        for doc in documents:
            doc_id = (doc.doc or {}).get("document_id")
            if doc.doc and doc_id and doc_id not in seen:
                seen.add(doc_id)
                references.append(doc.doc)
    return RetrievalResponseV2(
        query=query,
        documents=documents,
        references=references,
        meta={
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "requested_top_k": top_k,
            "returned": len(documents),
            "candidates": candidates,
            "fusion": fusion_used,
        },
    ).model_dump()


@router.post("/", dependencies=[Depends(verify_api_key)])
async def rag_search(
    payload: RAGRequest,
    pipeline=Depends(get_pipeline),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
):
    return await _search(payload, pipeline, rerank_func, resolver, asset_store)


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def rag_stream(
    payload: RAGRequest,
    pipeline=Depends(get_pipeline),
    rerank_func=Depends(get_rerank_func),
    resolver=Depends(get_resolver),
    asset_store=Depends(get_asset_store),
):
    from fastapi.responses import StreamingResponse

    async def generate():
        yield '{"status": "retrieving"}\n'
        response = await _search(payload, pipeline, rerank_func, resolver, asset_store)
        for doc in response["documents"]:
            metadata = {k: v for k, v in doc["metadata"].items() if k not in ("page", "source")}
            doc_payload = {"text": doc["text"], "metadata": metadata}
            if doc.get("modality") is not None:
                doc_payload["modality"] = doc["modality"]
                doc_payload["artifact_url"] = doc.get("artifact_url")
                doc_payload["artifact_urls"] = doc.get("artifact_urls")
            if doc.get("score") is not None:
                doc_payload["score"] = doc["score"]
            yield f'{{"documents": [{_dumps(doc_payload)}]}}\n'
        yield '{"status": "done"}\n'

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
