"""Rerank clients: legacy BGE (text-only) + vLLM Nemotron VL (multimodal).

- ``build_rerank_model_func``: legacy BGE reranker (POST {base}/rerank trả
  ``{scores, ranked_indices}``) - documents text-only.
- ``build_vllm_rerank_func``: vLLM serving ``nvidia/llama-nemotron-rerank-vl-1b-v2``
  (POST {base}/rerank trả ``{results: [{index, relevance_score}]}``) - documents
  dạng ``str`` hoặc ``{"content": [text / image_url parts]}`` (multimodal).
- ``build_rerank_documents``: chunk payload (Qdrant) -> documents cho rerank;
  chunk image/table/equation có ``asset_key`` kèm ảnh (data URI từ asset store);
  text vượt token budget bị head-truncate (``RERANK_MAX_INPUT_TOKENS``) để
  không vượt max_model_len của rerank server.
- ``build_rerank_func``: dispatcher theo ``RERANK_BACKEND``.
"""

import asyncio
import logging

import httpx

from ami_rag.core.embedder import clamp_embed_text
from ami_rag.core.openai_embedder import data_url_from_bytes
from ami_rag.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Số request fetch ảnh đồng thời khi build documents cho rerank
ASSET_FETCH_CONCURRENCY = 4

# Modalities được kèm ảnh khi rerank multimodal
MULTIMODAL_RERANK_TYPES = ("image", "table", "equation")


def build_rerank_model_func(settings: Settings | None = None):
    """Legacy BGE reranker (documents text-only).

    Wraps the external rerank service (POST {base}/rerank returning
    {scores, ranked_indices}) and translates the response into the index-based
    format: [{"index": i, "relevance_score": s}, ...].
    Exceptions propagate: callers catch them and fall back to original chunks.
    """
    settings = settings or get_settings()
    base_url = settings.RERANK_BASE_URL.rstrip("/")

    async def rerank_model_func(query: str, documents: list[str], top_n: int | None = None, **_):
        payload = {
            "query": query,
            "documents": documents,
            "top_k": top_n if top_n is not None else settings.RERANK_TOP_K,
        }
        async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
            resp = await client.post(f"{base_url}/rerank", json=payload)
            resp.raise_for_status()
            body = resp.json()

        scores = body.get("scores") or []
        indices = body.get("ranked_indices")
        if indices is None:
            indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        results = []
        for i in indices:
            idx = int(i)
            if 0 <= idx < len(scores):
                results.append({"index": idx, "relevance_score": float(scores[idx])})
        return results

    return rerank_model_func


def build_vllm_rerank_func(settings: Settings | None = None):
    """vLLM reranker (Nemotron VL) - documents text hoặc multimodal.

    Payload: {model, query, documents, top_n}; documents là ``str`` hoặc
    ``{"content": [{"type": "text"|"image_url", ...}]}`` (data URI ảnh).
    Response: {results: [{index, relevance_score, ...}]}.
    """
    settings = settings or get_settings()
    base_url = settings.RERANK_BASE_URL.rstrip("/")
    model = settings.RERANK_MODEL

    async def rerank_func(query: str, documents: list[str | dict], top_n: int | None = None, **_):
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else settings.RERANK_TOP_K,
        }
        async with httpx.AsyncClient(timeout=settings.RERANK_TIMEOUT) as client:
            resp = await client.post(f"{base_url}/rerank", json=payload)
            resp.raise_for_status()
            body = resp.json()

        results = []
        for item in body.get("results") or []:
            idx = item.get("index")
            if isinstance(idx, int):
                results.append(
                    {"index": idx, "relevance_score": float(item.get("relevance_score") or 0.0)}
                )
        return results

    return rerank_func


async def build_rerank_documents(
    chunks: list[dict],
    asset_store=None,
    *,
    multimodal: bool = True,
    max_input_tokens: int | None = None,
    image_token_reserve: int | None = None,
) -> list[str | dict]:
    """Chunk payload (Qdrant) -> documents cho rerank.

    - text chunk (hoặc multimodal tắt / thiếu asset store / thiếu asset_key): str.
    - chunk image/table/equation có asset_key: fetch ảnh từ MinIO (bounded
      concurrency, lỗi -> rơi về text) rồi trả
      ``{"content": [text part, image_url data URI part]}``.
    - text (cả 2 dạng) vượt token budget bị head-truncate: rerank server cùng
      vLLM giới hạn max_model_len; content đầy đủ vẫn nằm ở Qdrant payload.
    """
    if not chunks:
        return []

    settings = get_settings()
    max_tokens = max_input_tokens if max_input_tokens is not None else (
        getattr(settings, "RERANK_MAX_INPUT_TOKENS", 6000)
    )
    img_reserve = image_token_reserve if image_token_reserve is not None else (
        getattr(settings, "RERANK_IMAGE_TOKEN_RESERVE", 1792)
    )

    data_uris: dict[int, str] = {}
    if multimodal and asset_store is not None and hasattr(asset_store, "get_bytes"):
        wanted = [
            (i, c["asset_key"])
            for i, c in enumerate(chunks)
            if (c.get("modality") or "text") in MULTIMODAL_RERANK_TYPES
            and c.get("asset_key")
        ]
        if wanted:
            sem = asyncio.Semaphore(ASSET_FETCH_CONCURRENCY)

            async def _fetch(i: int, key: str):
                async with sem:
                    try:
                        data = await asyncio.to_thread(asset_store.get_bytes, key)
                    except Exception as exc:
                        logger.warning("fetch asset %s cho rerank lỗi: %s", key, exc)
                        data = None
                    return i, data

            for i, data in await asyncio.gather(*(_fetch(i, k) for i, k in wanted)):
                if data:
                    data_uris[i] = data_url_from_bytes(data)

    documents: list[str | dict] = []
    truncated = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("content") or ""
        uri = data_uris.get(i)
        clamped = clamp_embed_text(
            text, max_tokens, has_image=bool(uri), image_tokens=img_reserve
        )
        if clamped != text:
            truncated += 1
            logger.warning(
                "rerank document %d vượt budget: cắt %d -> %d chars (has_image=%s)",
                i,
                len(text),
                len(clamped),
                bool(uri),
            )
        if uri:
            documents.append(
                {
                    "content": [
                        {"type": "text", "text": clamped},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ]
                }
            )
        else:
            documents.append(clamped)
    if truncated:
        logger.warning(
            "build_rerank_documents: %d/%d document bị cắt theo RERANK_MAX_INPUT_TOKENS=%s",
            truncated,
            len(chunks),
            max_tokens,
        )
    return documents


def fuse_modality_scores(
    results: list[dict],
    chunks: list[dict],
    *,
    rrf_k: int = 10,
    visual_weight: float = 0.8,
    visual_floor: float = 0.01,
    text_floor: float = 0.05,
) -> list[dict]:
    """Fuse điểm rerank theo 2 nhóm modality (image vs phần còn lại).

    Điểm cross-encoder không so sánh được giữa image (thấp - chỉ suy luận qua
    visual tokens) và text/table (cao - khớp keyword trực tiếp), nên xếp hạng
    theo hạng nội bộ từng nhóm qua RRF thay vì so điểm tuyệt đối:

    - nhóm visual: chunk ``modality == "image"``; nhóm text: modality còn lại.
    - hạng trong nhóm tính theo raw score (desc, 1-based);
      ``fused = weight / (rrf_k + rank)`` với weight của nhóm mình.
    - sàn điểm: raw score dưới floor của nhóm bị chặn không được promote -
      ``fused = raw - 1`` (luôn xếp sau mọi item đạt floor, giữ thứ tự raw).

    Trả ``[{"index", "relevance_score", "fused_score"}]`` sorted theo fused desc.
    """
    scored: list[tuple[int, float, bool]] = []  # (idx, raw, is_visual)
    for item in results or []:
        idx = item.get("index")
        if not isinstance(idx, int) or not 0 <= idx < len(chunks):
            continue
        is_visual = (chunks[idx].get("modality") or "text") == "image"
        scored.append((idx, float(item.get("relevance_score") or 0.0), is_visual))

    visual = [(i, s) for i, s, v in scored if v]
    text = [(i, s) for i, s, v in scored if not v]

    fused: dict[int, float] = {}

    def _fuse(group: list[tuple[int, float]], weight: float, floor: float) -> None:
        ordered = sorted(group, key=lambda pair: -pair[1])
        for rank, (idx, raw) in enumerate(ordered, start=1):
            if raw < floor:
                fused[idx] = raw - 1.0
            else:
                fused[idx] = weight / (rrf_k + rank)

    _fuse(visual, visual_weight, visual_floor)
    _fuse(text, 1.0, text_floor)

    return [
        {"index": idx, "relevance_score": raw, "fused_score": fused[idx]}
        for idx, raw, _ in sorted(scored, key=lambda t: -fused[t[0]])
    ]


def resolve_rerank_fusion(settings) -> str:
    """Resolve RERANK_FUSION: "raw" (so điểm tuyệt đối) | "rrf" (fusion 2 nhóm)."""
    fusion = (getattr(settings, "RERANK_FUSION", "raw") or "raw").lower()
    if fusion not in ("raw", "rrf"):
        raise ValueError(f"RERANK_FUSION không hợp lệ: {fusion} (raw | rrf)")
    return fusion


def build_rerank_func(settings: Settings | None = None):
    """Dispatcher theo RERANK_BACKEND: legacy (BGE) | vllm (Nemotron VL)."""
    from ami_rag.settings import resolve_rerank_backend

    settings = settings or get_settings()
    if resolve_rerank_backend(settings) == "vllm":
        return build_vllm_rerank_func(settings)
    return build_rerank_model_func(settings)
