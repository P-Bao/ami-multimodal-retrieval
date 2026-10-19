"""Embedder protocol + phân loại lỗi cho embedding server chạy trên máy GPU.

Máy A (pipeline/CLI) KHÔNG chứa torch/transformers - mọi embedding đi qua
``RemoteEmbedder`` gọi sang embedding server (máy B) qua HTTP.
"""

import re
from typing import Protocol

# Xấp xỉ token -> chars thận trọng cho tiếng Việt (máy A không có tokenizer);
# budget chars = (max_tokens - image token reserve nếu có) * CHARS_PER_TOKEN
CHARS_PER_TOKEN = 3
# Một ảnh Nemotron VL tốn tối đa ~1792 visual token (6 tile + thumbnail - model card)
DEFAULT_IMAGE_TOKEN_RESERVE = 1792


class EmbedderError(RuntimeError):
    """Lỗi cơ sở từ embedding server."""


class EmbedServerUnreachable(EmbedderError):
    """Không với tới server: mất kết nối, timeout, hoặc 5xx sau khi hết retry."""


class EmbedModelMismatch(EmbedderError):
    """model_name/dim của server lệch với config/handshake, hoặc server bị
    đổi model giữa chừng (response không khớp)."""


class EmbedServerOOM(EmbedderError):
    """Server hết GPU memory (507) - không giảm được batch."""


class EmbedInputTooLong(EmbedderError):
    """Input vượt max_input_tokens / payload (413)."""


class EmbedAuthError(EmbedderError):
    """Token bị từ chối (401/403)."""


class EmbedCircuitOpen(EmbedderError):
    """Circuit breaker mở: N lần liên tiếp không với tới server, dừng cả lô sớm."""


class Embedder(Protocol):
    """Interface embedding - backend có thể thay (hiện tại: RemoteEmbedder)."""

    model_name: str
    dim: int

    async def embed_documents(self, texts: list[str | dict]) -> list[list[float]]:
        """Embed danh sách văn bản / multimodal items (document embedding, có cache)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (query embedding, không cache)."""
        ...


def clamp_embed_text(
    text: str,
    max_tokens: int,
    *,
    has_image: bool = False,
    image_tokens: int = DEFAULT_IMAGE_TOKEN_RESERVE,
) -> str:
    """Head-truncate text vượt token budget cho input embed/rerank.

    Budget chars xấp xỉ = (max_tokens - image_tokens nếu kèm ảnh) * CHARS_PER_TOKEN.
    Content đầy đủ vẫn lưu ở Qdrant payload (LLM dùng lúc trả lời) - chỉ input gửi
    lên embed/rerank server bị cắt để không vượt max_model_len.
    """
    if not text:
        return text
    if max_tokens <= 0:
        return text  # <=0 = tắt guard
    budget = max_tokens - (image_tokens if has_image else 0)
    max_chars = max(256, budget * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_model_slug(model_name: str) -> str:
    """'Qwen/Qwen3-VL-Embedding-2B' -> 'qwen3-vl-embedding-2b'."""
    slug = model_name.rsplit("/", 1)[-1].strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", slug).strip("-")


def collection_name(prefix: str, embed_model: str, chunker_version: str) -> str:
    """Quy ước tên collection vector: {prefix}__{embed_model_slug}__{chunker_version}.

    Không bao giờ trộn vector của hai embed model trong một collection.
    """
    return f"{prefix}__{embed_model_slug(embed_model)}__{chunker_version}"
