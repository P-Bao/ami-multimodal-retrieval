"""VinternEmbedder - client embedding server multi-vector (máy B).

Model ``5CD-AI/Vintern-Embedding-1B`` serve qua ``vintern-embedding-server``
(:8007). Contract server:
- GET ``/info``: ``{model_name, dim=128, max_input_tokens, multivector: true,
  normalize: false, query_instruction, document_instruction, server_version,
  device, dtype, native_dim}`` - handshake so model_name + dim.
- POST ``/embed`` ``{items: [...]}`` (tối đa 8 item/request):
  - query: **text-only** (``{"type": "query", "text": ...}``);
  - document: **text XOR image_b64** - gửi cả hai server trả 422, nên client
    luôn gửi image-only cho item ảnh (caption không trộn kèm).
  → ``{model_name, dim, vectors: [items][tokens][128], truncated: [int]}``
  (per-token vectors đã L2-normalize phía model; giữ thứ tự item).
- Kết quả là **multi-vector**: mỗi item trả list vector theo token
  (``list[list[float]]``); Qdrant so khớp bằng MaxSim (multivector_config).
- Auth: ``Authorization: Bearer <EMB_API_KEY>`` khi có.
- Retry/backoff, circuit breaker, cache, chia batch: dùng chung ``BaseRemoteEmbedder``.
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

__all__ = ["VinternEmbedder", "build_vintern_embedder"]


class VinternEmbedder(BaseRemoteEmbedder):
    """Client embedding server máy B (multi-vector, ``POST /embed``).

    Khác ``RemoteEmbedder`` (single-vector): ``embed_query``/``embed_documents``
    trả vector **per-token** (``list[list[float]]`` cho mỗi item). Không có
    ``embed_documents_sparse``/``embed_query_sparse`` - Qdrant chỉ dùng dense
    named vector với ``MultiVectorConfig(MAX_SIM)``.
    """

    # Embedder nhận ảnh bytes (image_b64) -> pipeline gắn ảnh thay vì caption-only
    supports_images: bool = True
    # Vector multi (per-token) -> collection Qdrant cần multivector_config
    is_multivector: bool = True

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "5CD-AI/Vintern-Embedding-1B",
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        batch_size: int = 8,
        max_concurrency: int = 2,
        max_payload_bytes: int = 32 * 1024 * 1024,
        max_input_tokens: int = 8192,
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
        if info.get("multivector") is False:
            raise EmbedModelMismatch(
                "embed server không bật multivector - client này chỉ chạy với "
                "server multi-vector (custom_text_proj per-token)."
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
            "embed server handshake ok: model=%s dim=%s multivector=%s server_version=%s",
            server_model,
            server_dim,
            info.get("multivector"),
            info.get("server_version"),
        )
        return info

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_query(self, text: str) -> list[list[float]]:
        """Embed một câu hỏi (type=query, text-only, không cache).

        Trả multi-vector: ``vectors[0]`` là list vector theo token (dim 128)."""
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
        """Item -> payload /embed. Ràng buộc server: document = text XOR image_b64.

        Item có image_b64 luôn gửi **image-only** (server 422 nếu kèm text);
        item không ảnh gửi text."""
        if isinstance(item, dict):
            p: dict = {"type": item.get("type") or "document"}
            if item.get("image_b64"):
                p["image_b64"] = item["image_b64"]
            else:
                p["text"] = item.get("text") or ""
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
        self,
        result: dict,
        batch_indices: list[int],
        vectors: list[list[float] | None],
    ) -> None:
        """Vector mỗi item là multi-vector (list token vectors) - gán nguyên xi."""
        for idx, vec in zip(batch_indices, result["vectors"]):
            vectors[idx] = vec


def build_vintern_embedder(settings) -> VinternEmbedder:
    """Build VinternEmbedder từ Settings (một nguồn cấu hình duy nhất với pipeline).

    EMBED_SERVER_TOKEN chỉ đọc từ env - không bao giờ ghi vào file config.
    """
    cache = None
    if getattr(settings, "EMBED_CACHE_ENABLED", True):
        from ami_rag.core.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
    return VinternEmbedder(
        base_url=settings.EMBED_SERVER_URL,
        model=settings.EMBED_MODEL,
        expected_dim=settings.EMBED_DIM,
        token=settings.EMBED_SERVER_TOKEN,
        timeout=settings.EMBED_TIMEOUT,
        batch_size=settings.EMBED_BATCH_SIZE,
        max_concurrency=settings.EMBED_MAX_CONCURRENCY,
        max_input_tokens=getattr(settings, "EMBED_MAX_INPUT_TOKENS", 8192),
        image_token_reserve=getattr(settings, "EMBED_IMAGE_TOKEN_RESERVE", 1792),
        retries=getattr(settings, "EMBED_RETRIES", 3),
        cache=cache,
    )
