"""BGEM3Embedder - client embedding-server BAAI/bge-m3 (máy B, ``embedding-server``).

Text-only (dense 1024 + sparse lexical weights). Contract server:
- GET ``/v1/models``: ``{"data": [{"id": "BAAI/bge-m3", ...}]}`` - handshake so model.
- POST ``/embed``: ``{"texts": [...], "return_dense": true, "return_sparse": bool}``
  → ``{"dense_vecs": [[float, ...]], "sparse_weights": [{"100": 0.25, ...}, ...]}``
  (sparse_weights là dict token_id dàng chuỗi → weight).
- Auth: ``Authorization: Bearer <BGE_API_KEY>`` khi có; không key → bỏ.
- Serve tối đa 64 text/request (``EMBED_BATCH_SIZE`` <= 64); 400 khi text
  rỗng/whitespace → client thay bằng ``"."``; 401 sai key; 413/422 payload lớn;
  500/503/504 transient (retry).
- Item ``image_b64`` BỎ (BGE-M3 text-only): chỉ lấy text, log warning, không raise.
- Cache embedding (tuỳ chọn) chỉ dense - sparse tính lại mỗi lần gọi server.
- Retry/backoff, circuit breaker, chia nhóm batch: dùng chung ``BaseRemoteEmbedder``.
"""

import asyncio
import hashlib
import logging

import httpx

from ami_rag.core.base_embedder import CIRCUIT_THRESHOLD, BaseRemoteEmbedder
from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedModelMismatch,
    EmbedServerUnreachable,
    clamp_embed_text,
)

logger = logging.getLogger(__name__)

# Serve tối đa 64 text/request (server đơn giản từ chối 400 theo số text)
MAX_TEXTS_PER_REQUEST = 64

__all__ = ["BGEM3Embedder", "build_bge_embedder"]


class BGEM3Embedder(BaseRemoteEmbedder):
    """Client embedding server machine B (BAAI/bge-m3, dense + sparse)."""

    # Server trả cả 413 lẫn 422 cho payload/text quá dài
    too_long_statuses = (413, 422)

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "BAAI/bge-m3",
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        batch_size: int = 32,
        max_concurrency: int = 4,
        max_input_tokens: int = 8192,
        image_token_reserve: int = 0,
        retries: int = 3,
        backoff_base: float = 0.5,
        cache=None,
        return_sparse: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ):
        super().__init__(
            base_url,
            model=model,
            expected_dim=expected_dim,
            token=token,
            timeout=timeout,
            batch_size=max(1, min(batch_size, MAX_TEXTS_PER_REQUEST)),
            max_concurrency=max_concurrency,
            retries=retries,
            backoff_base=backoff_base,
            cache=cache,
            http_client=http_client,
        )
        self.max_input_tokens = max_input_tokens
        self.image_token_reserve = image_token_reserve
        self.return_sparse = return_sparse

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------
    async def verify(self) -> dict:
        """GET /v1/models + probe /embed 1 lần để xác minh model + dim. Lệch -> dừng."""
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise EmbedServerUnreachable(
                f"embed server không với tới được ({self.base_url}): {exc}"
            ) from exc
        if resp.status_code in (401, 403):
            raise EmbedAuthError(f"embed server từ chối token: {resp.status_code}")
        if resp.status_code != 200:
            raise EmbedServerUnreachable(
                f"embed server /v1/models trả {resp.status_code}"
            )
        info = resp.json()
        ids = [str(m.get("id")) for m in (info.get("data") or [])]
        if self.model not in ids:
            raise EmbedModelMismatch(
                f"embed server chạy model {ids}, config mong đợi '{self.model}'. "
                "Không ghi vector khi model chưa xác minh."
            )

        # Probe 1 text để chốt dim thực tế (server quyết định, bỏ qua ensure tránh đệ quy)
        dense, _ = await self._flow(["probe"], use_cache=False, want_sparse=False, ensure=False)
        server_dim = len(dense[0])
        if self.expected_dim is not None and server_dim != self.expected_dim:
            raise EmbedModelMismatch(
                f"embed server dim={server_dim}, config EMBED_DIM={self.expected_dim}. "
                "Đổi model/dim phía server thì cập nhật EMBED_DIM và dùng collection mới."
            )

        self.dim = int(server_dim)
        self._instruction_ns = hashlib.sha256(self.model.encode()).hexdigest()[:16]
        self._verified = True
        logger.info(
            "embed server handshake ok: model=%s dim=%s (%s)",
            self.model,
            server_dim,
            self.base_url,
        )
        return info

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_documents(self, items: list[str | dict]) -> list[list[float]]:
        """Embed danh sách text / caption items (dense, có cache). Giữ thứ tự đầu vào."""
        dense, _ = await self._flow(items, use_cache=True, want_sparse=False)
        return dense

    async def embed_documents_sparse(
        self, items: list[str | dict]
    ) -> tuple[list[list[float]], list[dict]]:
        """Dense + sparse cùng lượt; sparse trúng cache bị tính lại từ server
        (shortcut: EmbeddingCache chỉ chứa dense, cần sparse cache thì thêm bảng riêng)."""
        dense, sparse = await self._flow(
            items, use_cache=False, want_sparse=self.return_sparse
        )
        return dense, sparse

    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (dense, không cache)."""
        dense, _ = await self._flow([text], use_cache=False, want_sparse=False)
        return dense[0]

    async def embed_query_sparse(self, text: str) -> tuple[list[float], dict]:
        """Embed một câu hỏi (dense + sparse, không cache)."""
        dense, sparse = await self._flow(
            [text], use_cache=False, want_sparse=self.return_sparse
        )
        return dense[0], sparse[0]

    # ------------------------------------------------------------------
    # Luồng chính (dùng chung cho dense / sparse / query / documents)
    # ------------------------------------------------------------------
    async def _flow(
        self, items: list[str | dict], *, use_cache: bool, want_sparse: bool,
        ensure: bool = True,
    ) -> tuple[list[list[float]], list[dict]]:
        if not items:
            return [], []
        if ensure:
            await self._ensure_verified()

        texts = self._normalize_items(items)

        vectors: list[list[float] | None] = [None] * len(texts)
        sparse_out: list[dict] = [{} for _ in range(len(texts))]
        keys: list[str] = [self._cache_key(t) for t in texts]
        pending: list[int] = list(range(len(texts)))
        if use_cache and self.cache is not None:
            cached = await asyncio.to_thread(self.cache.get_many, keys)
            pending = [i for i in range(len(texts)) if keys[i] not in cached]
            for i, key in enumerate(keys):
                if key in cached:
                    vectors[i] = cached[key]

        if pending:
            batches = [
                pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)
            ]
            unreachable: EmbedServerUnreachable | None = None
            for group in self._groups(batches, self.max_concurrency):
                results = await asyncio.gather(
                    *(
                        self._post_embed(
                            [texts[i] for i in b],
                            return_sparse=want_sparse and bool(self.return_sparse),
                        )
                        for b in group
                    ),
                    return_exceptions=True,
                )
                for batch_indices, result in zip(group, results):
                    if isinstance(result, EmbedServerUnreachable):
                        # Giữ lỗi đầu tiên; circuit breaker quyết định dừng hay không
                        unreachable = unreachable or result
                        continue
                    if isinstance(result, BaseException):
                        raise result
                    dense_vecs = result.get("dense_vecs")
                    if (
                        not isinstance(dense_vecs, list)
                        or len(dense_vecs) != len(batch_indices)
                    ):
                        raise EmbedderError(
                            f"embed server trả {len(dense_vecs or [])} vector cho "
                            f"{len(batch_indices)} item trong batch"
                        )
                    sparse_weights = (
                        result.get("sparse_weights") if want_sparse else None
                    )
                    if want_sparse and not isinstance(sparse_weights, list):
                        raise EmbedderError(
                            "embed server không trả sparse_weights khi return_sparse=True"
                        )
                    if sparse_weights is not None and len(sparse_weights) != len(
                        batch_indices
                    ):
                        raise EmbedderError(
                            f"embed server trả {len(sparse_weights)} vector sparse cho "
                            f"{len(batch_indices)} item trong batch"
                        )
                    for pos, idx in enumerate(batch_indices):
                        vec = dense_vecs[pos]
                        if self.dim and len(vec) != self.dim:
                            raise EmbedModelMismatch(
                                f"embed server trả dim={len(vec)}, không khớp handshake "
                                f"(dim={self.dim}). Server có thể bị đổi model giữa chừng."
                            )
                        vectors[idx] = vec
                        if sparse_weights is not None:
                            sparse_out[idx] = self._to_sparse(sparse_weights[pos])
                if self._consecutive_unreachable >= CIRCUIT_THRESHOLD:
                    raise EmbedCircuitOpen(
                        f"embed server không với tới {self._consecutive_unreachable} "
                        "lô liên tiếp; dừng lô sớm - các doc còn lại giữ nguyên trạng thái"
                    )
            if unreachable is not None:
                raise unreachable
            if use_cache and self.cache is not None:
                new_entries = {
                    keys[i]: vectors[i] for i in pending if vectors[i] is not None
                }
                if new_entries:
                    await asyncio.to_thread(self.cache.put_many, new_entries)
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise EmbedderError(
                f"embed_documents thiếu vector cho {len(missing)}/{len(texts)} item"
            )
        dense = [v for v in vectors]
        return dense, sparse_out

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _post_embed(
        self, texts: list[str], *, return_sparse: bool
    ) -> dict:
        return await self._post(
            f"{self.base_url}/embed",
            {"texts": texts, "return_dense": True, "return_sparse": return_sparse},
        )

    # ------------------------------------------------------------------
    # Prepare texts / sparse decode
    # ------------------------------------------------------------------
    def _normalize_items(self, items: list[str | dict]) -> list[str]:
        """Item -> text gửi server: bỏ image_b64 (caption-only), head-truncate
        theo EMBED_MAX_INPUT_TOKENS, text rỗng/whitespace -> "." (server 400)."""
        out: list[str] = []
        dropped = 0
        truncated = 0
        emptied = 0
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("text") or "")
                if item.get("image_b64"):
                    dropped += 1
            else:
                text = str(item or "")
            clamped = clamp_embed_text(text, self.max_input_tokens)
            if clamped != text:
                truncated += 1
            if not clamped.strip():
                clamped = "."
                emptied += 1
            out.append(clamped)
        if dropped:
            logger.warning(
                "bge caption-only: bỏ image_b64 %d/%d item", dropped, len(items)
            )
        if truncated:
            logger.warning(
                "bge: %d/%d item bị cắt theo EMBED_MAX_INPUT_TOKENS=%s",
                truncated,
                len(items),
                self.max_input_tokens,
            )
        if emptied:
            logger.warning(
                "bge: %d/%d item rỗng/whitespace -> gửi '.'", emptied, len(items)
            )
        return out

    @staticmethod
    def _to_sparse(sparse_weights: dict) -> dict:
        """Token_id (khoá str|int) -> weight: indices tang dần + values tương ứng."""
        pairs = sorted(
            (int(k), float(v)) for k, v in (sparse_weights or {}).items()
        )
        return {
            "indices": [i for i, _ in pairs],
            "values": [v for _, v in pairs],
        }


def build_bge_embedder(settings) -> BGEM3Embedder:
    """Build BGEM3Embedder từ Settings (một nguồn cấu hình duy nhất với pipeline).

    EMBED_SERVER_TOKEN chỉ đọc từ env - không bao giờ ghi vào file config.
    """
    cache = None
    if getattr(settings, "EMBED_CACHE_ENABLED", True):
        from ami_rag.core.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(settings.EMBED_CACHE_PATH)
    return BGEM3Embedder(
        base_url=settings.EMBED_SERVER_URL,
        model=settings.EMBED_MODEL,
        expected_dim=settings.EMBED_DIM,
        token=settings.EMBED_SERVER_TOKEN,
        timeout=settings.EMBED_TIMEOUT,
        batch_size=settings.EMBED_BATCH_SIZE,
        max_concurrency=settings.EMBED_MAX_CONCURRENCY,
        max_input_tokens=getattr(settings, "EMBED_MAX_INPUT_TOKENS", 8192),
        image_token_reserve=getattr(settings, "EMBED_IMAGE_TOKEN_RESERVE", 0),
        retries=getattr(settings, "EMBED_RETRIES", 3),
        return_sparse=getattr(settings, "EMBED_RETURN_SPARSE", True),
        cache=cache,
    )
