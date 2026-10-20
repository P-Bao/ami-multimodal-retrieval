"""RemoteEmbedder - client embedding server (máy B) cho máy A.

Implements interface ``Embedder``. Đặc điểm:
- Handshake lúc khởi tạo sử dụng: gọi ``/info``, so ``model_name``/``dim`` với config;
  lệch thì dừng với lỗi rõ ràng, không bao giờ ghi vector khi chưa xác minh.
- Mỗi response của ``/embed`` chứa ``model_name``/``dim`` - kiểm tra lại từng response
  để phát hiện server bị đổi model giữa chừng.
- Retry/backoff, circuit breaker, cache, chia batch: dùng chung ``BaseRemoteEmbedder``.
- Item vượt token budget bị head-truncate trước khi gửi (``EMBED_MAX_INPUT_TOKENS``).
"""

import hashlib
import logging

import httpx

from ami_rag.core.base_embedder import BaseRemoteEmbedder
from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedModelMismatch,
    EmbedServerUnreachable,
    clamp_embed_text,
)

logger = logging.getLogger(__name__)

__all__ = ["RemoteEmbedder", "build_remote_embedder"]


class RemoteEmbedder(BaseRemoteEmbedder):
    """Client embedding server máy B (qwen-embedding-server, ``POST /embed``)."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "Qwen/Qwen3-VL-Embedding-2B",
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        batch_size: int = 32,
        max_concurrency: int = 4,
        max_payload_bytes: int = 32 * 1024 * 1024,
        max_input_tokens: int = 6000,
        image_token_reserve: int = 1792,
        retries: int = 3,
        backoff_base: float = 0.5,
        cache=None,
        http_client: httpx.AsyncClient | None = None,
    ):
        super().__init__(
            base_url,
            model=model,
            expected_dim=expected_dim,
            token=token,
            timeout=timeout,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            retries=retries,
            backoff_base=backoff_base,
            cache=cache,
            http_client=http_client,
        )
        self.max_payload_bytes = max_payload_bytes
        self.max_input_tokens = max_input_tokens
        self.image_token_reserve = image_token_reserve

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------
    async def verify(self) -> dict:
        """Gọi /info và so model_name/dim với config. Lệch -> dừng với lỗi rõ ràng."""
        try:
            resp = await self._client.get(f"{self.base_url}/info")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise EmbedServerUnreachable(
                f"embed server không với tới được ({self.base_url}): {exc}"
            ) from exc
        if resp.status_code in (401, 403):
            raise EmbedAuthError(f"embed server từ chối token: {resp.status_code}")
        if resp.status_code != 200:
            raise EmbedServerUnreachable(f"embed server /info trả {resp.status_code}")
        info = resp.json()

        server_model = info.get("model_name")
        if server_model != self.model:
            raise EmbedModelMismatch(
                f"embed server chạy model '{server_model}', config mong đợi "
                f"'{self.model}'. Không ghi vector khi model chưa xác minh."
            )
        server_dim = info.get("dim")
        if self.expected_dim is not None and server_dim != self.expected_dim:
            raise EmbedModelMismatch(
                f"embed server dim={server_dim}, config EMBED_DIM={self.expected_dim}. "
                "Đổi model/dim phía server thì cập nhật EMBED_DIM và dùng collection mới."
            )

        self.dim = int(server_dim)
        self._instruction_ns = hashlib.sha256(
            (
                str(info.get("query_instruction", ""))
                + "|"
                + str(info.get("document_instruction", ""))
            ).encode()
        ).hexdigest()[:16]
        self._verified = True
        logger.info(
            "embed server handshake ok: model=%s dim=%s server_version=%s",
            server_model,
            server_dim,
            info.get("server_version"),
        )
        return info

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (type=query, không cache)."""
        await self._ensure_verified()
        text = clamp_embed_text(text, self.max_input_tokens)
        result = await self._post_embed([{"type": "query", "text": text}])
        self._check_response(result)
        return result["vectors"][0]

    # ------------------------------------------------------------------
    # Hooks cho BaseRemoteEmbedder.embed_documents
    # ------------------------------------------------------------------
    async def _post_embed(self, items: list[dict]) -> dict:
        return await self._post(f"{self.base_url}/embed", {"items": items})

    @staticmethod
    def _to_embed_payload(item: str | dict) -> dict:
        if isinstance(item, dict):
            p = {"type": item.get("type") or "document"}
            if "text" in item:
                p["text"] = item["text"]
            if item.get("image_b64"):
                p["image_b64"] = item["image_b64"]
            return p
        return {"type": "document", "text": item}

    def _check_response(self, result: dict) -> None:
        """Mỗi response /embed phải khớp handshake - phát hiện server đổi model giữa chừng."""
        model_name = result.get("model_name")
        dim = result.get("dim")
        if model_name != self.model or (self.dim and dim != self.dim):
            raise EmbedModelMismatch(
                f"embed server trả model_name={model_name!r} dim={dim!r}, "
                f"không khớp handshake (model={self.model!r} dim={self.dim!r}). "
                "Server có thể bị đổi model giữa chừng."
            )

    def _fill_vectors(
        self, result: dict, batch_indices: list[int], vectors: list[list[float] | None]
    ) -> None:
        for idx, vec in zip(batch_indices, result["vectors"]):
            vectors[idx] = vec


def build_remote_embedder(settings) -> RemoteEmbedder:
    """Build RemoteEmbedder từ Settings (một nguồn cấu hình duy nhất với pipeline).

    EMBED_SERVER_TOKEN chỉ đọc từ env - không bao giờ ghi vào file config.
    """
    cache = None
    if getattr(settings, "EMBED_CACHE_ENABLED", True):
        from ami_rag.core.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
    return RemoteEmbedder(
        base_url=settings.EMBED_SERVER_URL,
        model=settings.EMBED_MODEL,
        expected_dim=settings.EMBED_DIM,
        token=settings.EMBED_SERVER_TOKEN,
        timeout=settings.EMBED_TIMEOUT,
        batch_size=settings.EMBED_BATCH_SIZE,
        max_concurrency=settings.EMBED_MAX_CONCURRENCY,
        max_input_tokens=getattr(settings, "EMBED_MAX_INPUT_TOKENS", 6000),
        image_token_reserve=getattr(settings, "EMBED_IMAGE_TOKEN_RESERVE", 1792),
        retries=getattr(settings, "EMBED_RETRIES", 3),
        cache=cache,
    )
