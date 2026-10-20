"""OpenAIEmbedder - client cho gateway Nemotron VL (máy B, ``nemotron-vl-vllm``).

Dùng cho ``nvidia/llama-nemotron-embed-vl-1b-v2``: máy B chạy gateway FastAPI
trước 2 vLLM (embed + rerank, repo ``nemotron-vl-vllm``). Contract gateway:
- POST ``/v1/embeddings``: ``{"input": [items], "input_type": "query"|"document"}``
  (model tự động theo gateway; item: ``str`` | ``{"text":..., "image": <URL |
  data URL | base64>}`` | ``{"content": [parts]}`` pass-through). Gateway fan-out
  song song xuống vLLM rồi ghép kết quả theo thứ tự input.
- GET ``/health``: 200 khi cả embed + rerank sẵn sàng (gateway không có
  ``/v1/models``) - handshake dùng /health + probe embed 1 lần để xác minh model
  (field ``model`` trong response) và lấy dim thực tế; lệch thì dừng.
- Retry/backoff, circuit breaker, cache, chia batch: dùng chung ``BaseRemoteEmbedder``.
- Item vượt token budget bị head-truncate trước khi gửi (``EMBED_MAX_INPUT_TOKENS``).
"""

import base64
import hashlib
import logging

import httpx

from ami_rag.core.base_embedder import BaseRemoteEmbedder
from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedderError,
    EmbedModelMismatch,
    EmbedServerUnreachable,
    clamp_embed_text,
    image_mime,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenAIEmbedder", "build_openai_embedder", "data_url_from_bytes"]


def data_url_from_bytes(data: bytes) -> str:
    """bytes ảnh -> data URI (dùng chung cho embed + rerank multimodal)."""
    return f"data:{image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


class OpenAIEmbedder(BaseRemoteEmbedder):
    """Client gateway Nemotron VL embed (OpenAI-style response, batch qua gateway)."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        batch_size: int = 32,
        max_payload_bytes: int = 32 * 1024 * 1024,
        max_input_tokens: int = 6000,
        image_token_reserve: int = 1792,
        max_concurrency: int = 4,
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
        """GET /health + probe embed 1 lần để xác minh model + dim. Lệch -> dừng."""
        try:
            resp = await self._client.get(f"{self.base_url}/health")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise EmbedServerUnreachable(
                f"embed server không với tới được ({self.base_url}): {exc}"
            ) from exc
        if resp.status_code in (401, 403):
            raise EmbedAuthError(f"embed server từ chối token: {resp.status_code}")
        if resp.status_code != 200:
            raise EmbedServerUnreachable(
                f"embed server /health trả {resp.status_code}"
            )
        try:
            info = resp.json()
        except ValueError:
            info = {}

        # Probe 1 lần để xác minh model + dim (gateway không khai báo qua /health)
        result = await self._post_embed([{"text": "probe"}])
        self._check_response(result)
        data = result.get("data") or []
        if not data or not data[0].get("embedding"):
            raise EmbedderError("embed server trả response /v1/embeddings rỗng")
        server_dim = len(data[0]["embedding"])
        if self.expected_dim is not None and server_dim != self.expected_dim:
            raise EmbedModelMismatch(
                f"embed server dim={server_dim}, config EMBED_DIM={self.expected_dim}. "
                "Đổi model/dim phía server thì cập nhật EMBED_DIM và dùng collection mới."
            )

        self.dim = int(server_dim)
        self._instruction_ns = hashlib.sha256(self.model.encode()).hexdigest()[:16]
        self._verified = True
        logger.info(
            "embed server handshake ok: model=%s dim=%s (gateway %s)",
            self.model,
            server_dim,
            self.base_url,
        )
        return info

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (input_type=query, không cache)."""
        await self._ensure_verified()
        text = clamp_embed_text(text, self.max_input_tokens)
        result = await self._post_embed([{"text": text}], input_type="query")
        self._check_response(result)
        data = result.get("data") or []
        if not data or not data[0].get("embedding"):
            raise EmbedderError("embed server trả response /v1/embeddings rỗng")
        return data[0]["embedding"]

    # ------------------------------------------------------------------
    # Hooks cho BaseRemoteEmbedder.embed_documents
    # ------------------------------------------------------------------
    async def _post_embed(self, items: list[dict], *, input_type: str = "document") -> dict:
        return await self._post(
            f"{self.base_url}/v1/embeddings",
            {"input": items, "input_type": input_type},
        )

    @staticmethod
    def _to_embed_payload(item: str | dict) -> str | dict:
        """Item nội bộ -> item gateway: {"text", "image"} (gateway tự sniff mime)."""
        if isinstance(item, dict):
            p: dict = {}
            if item.get("text"):
                p["text"] = item["text"]
            if item.get("image_b64"):
                p["image"] = item["image_b64"]
            if not p:
                raise EmbedderError("embed item rỗng (thiếu text/image_b64)")
            return p
        return item

    def _check_response(self, result: dict) -> None:
        """Mỗi response /v1/embeddings phải khớp model handshake."""
        model = result.get("model")
        if model != self.model:
            raise EmbedModelMismatch(
                f"embed server trả model={model!r}, không khớp handshake "
                f"(model={self.model!r}). Server có thể bị đổi model giữa chừng."
            )

    def _fill_vectors(
        self, result: dict, batch_indices: list[int], vectors: list[list[float] | None]
    ) -> None:
        data = result.get("data") or []
        if len(data) != len(batch_indices):
            raise EmbedderError(
                f"embed server trả {len(data)} vector cho "
                f"{len(batch_indices)} item trong batch"
            )
        for idx, item in zip(batch_indices, data):
            if not item.get("embedding"):
                raise EmbedderError(
                    "embed server trả response /v1/embeddings rỗng"
                )
            vectors[idx] = item["embedding"]


def build_openai_embedder(settings) -> OpenAIEmbedder:
    """Build OpenAIEmbedder từ Settings (một nguồn cấu hình duy nhất với pipeline).

    EMBED_SERVER_TOKEN chỉ đọc từ env - không bao giờ ghi vào file config.
    """
    cache = None
    if getattr(settings, "EMBED_CACHE_ENABLED", True):
        from ami_rag.core.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
    return OpenAIEmbedder(
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
