import base64
from collections.abc import Callable

from ami_rag.settings import Settings, get_settings

_pipeline_instance = None
_asset_store_instance = None
_remote_embedder_instance = None


def _build_llm_func(settings: Settings) -> Callable:
    """LLM callable (qwen-selfhost, OpenAI-compatible) - describe/answer."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.QWEN_LLM_BASE_URL,
        api_key=settings.QWEN_LLM_API_KEY or "EMPTY",
    )

    async def qwen_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=settings.QWEN_LLM_MODEL, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    return qwen_llm


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def _openai_style_messages(
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict] | None,
    image_data: str | None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    if image_data:
        mime = _image_mime(base64.b64decode(image_data))
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_data}"},
            },
        ]
    else:
        content = prompt
    messages.append({"role": "user", "content": content})
    return messages


def _build_vision_func(settings: Settings) -> Callable:
    """Vision (VLM) callable cho modal processors (describe stage).

    Gọi dạng `(prompt, system_prompt=..., image_data=<base64>)` hoặc
    `("", messages=[OpenAI-style messages with image_url parts])`.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.QWEN_LLM_BASE_URL,
        api_key=settings.QWEN_LLM_API_KEY or "EMPTY",
    )
    model = settings.QWEN_VLM_MODEL or settings.QWEN_LLM_MODEL

    async def qwen_vision(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages is None:
            messages = _openai_style_messages(
                prompt, system_prompt, history_messages, image_data
            )
        resp = await client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    return qwen_vision


def _build_caption_func(settings: Settings) -> Callable | None:
    """Caption ảnh bằng Gemini (OpenAI-compatible endpoint của Google).

    Trả ``None`` khi chưa đặt CAPTION_LLM_API_KEY (caller tự fallback về vision
    func của Qwen VLM). Closure nhận cùng chữ ký như vision func:
    ``(prompt, system_prompt=None, history_messages=None, image_data=<base64>,
    messages=None, **kwargs)`` - tái dùng _openai_style_messages để dựng data URI.
    """
    api_key = getattr(settings, "CAPTION_LLM_API_KEY", "") or ""
    if not api_key:
        return None
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=getattr(settings, "CAPTION_LLM_BASE_URL", ""),
        api_key=api_key,
        timeout=float(getattr(settings, "CAPTION_LLM_TIMEOUT", 120) or 120),
        max_retries=int(getattr(settings, "CAPTION_LLM_RETRIES", 2) or 0),
    )
    model = getattr(settings, "CAPTION_LLM_MODEL", "") or "gemini-2.5-flash-lite"

    async def gemini_caption(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages is None:
            messages = _openai_style_messages(
                prompt, system_prompt, history_messages, image_data
            )
        resp = await client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    return gemini_caption


def _build_modal_processors(settings: Settings, caption_func: Callable | None = None) -> dict:
    """Modal processors cho describe stage (lightrag=None tolerant).

    ``caption_func`` (inject cho test) thắng; không có thì Gemini caption
    (khi đặt CAPTION_LLM_API_KEY), cuối cùng mới là vision func Qwen VLM.
    """
    from raganything.modalprocessors import (
        EquationModalProcessor,
        GenericModalProcessor,
        ImageModalProcessor,
        TableModalProcessor,
    )

    vision = _build_vision_func(settings)
    llm = _build_llm_func(settings)
    image_caption = caption_func or _build_caption_func(settings) or vision
    common = {
        "lightrag": None,
        "global_config": {"chunk_token_size": settings.CHUNK_SIZE},
    }
    processors: dict = {
        "image": ImageModalProcessor(modal_caption_func=image_caption, **common),
        "table": TableModalProcessor(modal_caption_func=llm, **common),
        "equation": EquationModalProcessor(modal_caption_func=llm, **common),
        "generic": GenericModalProcessor(modal_caption_func=llm, **common),
    }
    try:
        from raganything.modalprocessors_audio import (
            AudioModalProcessor,
            audio_deps_available,
        )

        if audio_deps_available():
            processors["audio"] = AudioModalProcessor(
                modal_caption_func=llm,
                **common,
            )
    except Exception:
        pass
    try:
        from raganything.modalprocessors_video import (
            VideoModalProcessor,
            video_deps_available,
        )

        if video_deps_available() and "audio" in processors:
            processors["video"] = VideoModalProcessor(
                modal_caption_func=vision,
                audio_processor=processors["audio"],
                **common,
            )
    except Exception:
        pass
    return processors


def build_asset_store(settings: Settings | None = None):
    from ami_rag.storage.assets import MinioAssetStore

    settings = settings or get_settings()
    return MinioAssetStore(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket=settings.MINIO_BUCKET,
        prefix=settings.ASSET_PREFIX,
        secure=settings.MINIO_SECURE,
        presign_expires=settings.MINIO_PRESIGN_EXPIRES,
    )


def get_asset_store():
    """Cached MinioAssetStore built from settings."""
    global _asset_store_instance
    if _asset_store_instance is None:
        _asset_store_instance = build_asset_store()
    return _asset_store_instance


def build_embedder(settings: Settings):
    """Build embedder theo EMBED_BACKEND: custom (qwen server) | bge (BAAI/bge-m3) | openai (vLLM)."""
    from ami_rag.settings import resolve_embed_backend

    backend = resolve_embed_backend(settings)
    if backend == "custom":
        from ami_rag.core.remote_embedder import build_remote_embedder

        return build_remote_embedder(settings)
    if backend == "bge":
        from ami_rag.core.bge_embedder import build_bge_embedder

        return build_bge_embedder(settings)
    from ami_rag.core.openai_embedder import build_openai_embedder

    return build_openai_embedder(settings)


def get_embedder():
    """Cached embedder built from settings (handshake chạy lười ở lần embed đầu)."""
    global _remote_embedder_instance
    if _remote_embedder_instance is None:
        _remote_embedder_instance = build_embedder(get_settings())
    return _remote_embedder_instance


def build_pipeline(
    settings: Settings | None = None,
    *,
    embedder=None,
    vector_store=None,
    asset_store=None,
    store=None,
    docs_repo=None,
    modal_processors=None,
    llm_func=None,
    caption_func=None,
    image_worker=None,
):
    """Build VectorPipeline wired to AMI infrastructure (test paths can inject deps).

    ``caption_func`` (tuỳ chọn): inject caption func giả cho test; khi bỏ trống,
    image processor dùng Gemini (CAPTION_LLM_*) hoặc vision func như cũ.
    ``image_worker`` (tuỳ chọn): inject CrawlImageWorker giả cho test; bỏ trống
    thì pipeline lazy-build từ settings.
    """
    from ami_rag.core.vector_pipeline import VectorPipeline
    from ami_rag.core.vector_store import QdrantVectorStore
    from ami_rag.storage.doc_status import DocStatusStore
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    settings = settings or get_settings()
    return VectorPipeline(
        settings,
        embedder=embedder or build_embedder(settings),
        vector_store=vector_store
        or QdrantVectorStore(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            max_request_bytes=getattr(settings, "QDRANT_UPSERT_MAX_MB", 16)
            * 1024
            * 1024,
        ),
        asset_store=asset_store or get_asset_store(),
        store=store
        or DocStatusStore(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.RAG_DB,
            collection_name=settings.RAG_DOCUMENTS_COLLECTION,
        ),
        docs_repo=docs_repo
        or MongoDocumentRepo(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.ORG_DB,
            collection_name=settings.DOC_COLLECTION,
        ),
        modal_processors=modal_processors
        or _build_modal_processors(settings, caption_func=caption_func),
        llm_func=llm_func or _build_llm_func(settings),
        image_worker=image_worker,
    )


async def get_pipeline():
    """Async singleton: build once (preflight chạy lười ở lần run đầu)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = build_pipeline()
    return _pipeline_instance


async def close_pipeline() -> None:
    global _pipeline_instance, _asset_store_instance, _remote_embedder_instance
    _asset_store_instance = None
    if _pipeline_instance is not None:
        embedder = getattr(_pipeline_instance, "embedder", None)
        if embedder is not None:
            await embedder.close()
        store = getattr(_pipeline_instance, "store", None)
        if store is not None:
            store.close()
        docs_repo = getattr(_pipeline_instance, "docs_repo", None)
        if docs_repo is not None:
            docs_repo.close()
        _pipeline_instance = None
    if _remote_embedder_instance is not None:
        await _remote_embedder_instance.close()
        _remote_embedder_instance = None


async def close_rag() -> None:
    """Alias backward-compatible của close_pipeline."""
    await close_pipeline()
