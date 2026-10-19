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
- Circuit breaker: N lần liên tiếp không với tới server -> dừng cả lô sớm.
"""

import asyncio
import hashlib
import logging
import random
from itertools import islice
from typing import Protocol

import httpx

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedInputTooLong,
    EmbedModelMismatch,
    EmbedServerOOM,
    EmbedServerUnreachable,
    clamp_embed_text,
)

logger = logging.getLogger(__name__)

# N lần liên tiếp không với tới server -> dừng cả lô sớm (thay vì đánh failed hàng loạt)
CIRCUIT_THRESHOLD = 5
# Serve tối đa 64 text/request (server đơn giản từ chối 400 theo số text)
MAX_TEXTS_PER_REQUEST = 64


class _Cache(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, list[float]]: ...
    def put_many(self, mapping: dict[str, list[float]]) -> None: ...


class BGEM3Embedder:
    """Client embedding server machine B (BAAI/bge-m3, dense + sparse)."""

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
        cache: _Cache | None = None,
        return_sparse: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.expected_dim = expected_dim
        self.dim = expected_dim if expected_dim is not None else 0
        self.timeout = timeout
        self.batch_size = max(1, min(batch_size, MAX_TEXTS_PER_REQUEST))
        self.max_concurrency = max_concurrency
        self.max_input_tokens = max_input_tokens
        self.image_token_reserve = image_token_reserve
        self.retries = retries
        self.backoff_base = backoff_base
        self.cache = cache
        self.return_sparse = return_sparse
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        self._semaphore: asyncio.Semaphore | None = None
        self._verified = False
        self._instruction_ns = ""
        self._consecutive_unreachable = 0

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

    async def _ensure_verified(self) -> None:
        if not self._verified:
            await self.verify()

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
    # Cache (không HTTP)
    # ------------------------------------------------------------------
    def count_cache_hits(self, items: list[str | dict]) -> int:
        """Đếm số item đã có trong cache (lookup SQLite, KHÔNG gọi HTTP).

        Yêu cầu đã verify (dim + instruction_ns từ handshake); cache tắt -> 0.
        """
        if not items or self.cache is None or not self._verified:
            return 0
        keys = [self._cache_key(it) for it in items]
        cached = self.cache.get_many(keys)
        return sum(1 for k in keys if k in cached)

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
        """Một POST /embed với retry cho lỗi tạm thời; 4xx raise ngay."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self._client.post(
                        f"{self.base_url}/embed",
                        json={
                            "texts": texts,
                            "return_dense": True,
                            "return_sparse": return_sparse,
                        },
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.retries:
                    # Hết retry: đếm lô này là 1 lần unreachable liên tiếp
                    # (circuit breaker ở _flow quyết định dừng lô)
                    self._consecutive_unreachable += 1
                    raise EmbedServerUnreachable(
                        f"embed server không với tới được sau {attempt + 1} lần thử: {exc}"
                    ) from exc
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue

            self._consecutive_unreachable = 0

            if resp.status_code == 200:
                return resp.json()

            detail = self._detail(resp)
            if resp.status_code in (401, 403):
                raise EmbedAuthError(f"embed server từ chối token: {detail}")
            if resp.status_code in (413, 422):
                raise EmbedInputTooLong(
                    f"input quá dài / payload vượt giới hạn: {detail}"
                )
            if resp.status_code == 507:
                raise EmbedServerOOM(f"embed server hết GPU memory: {detail}")
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.retries:
                    raise EmbedServerUnreachable(
                        f"embed server lỗi tạm thời {resp.status_code} sau "
                        f"{attempt + 1} lần thử: {detail}"
                    )
                attempt += 1
                await asyncio.sleep(self._backoff(attempt))
                continue
            raise EmbedderError(f"embed server trả {resp.status_code}: {detail}")

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            return str(resp.json().get("detail", resp.text))[:300]
        except ValueError:
            return resp.text[:300]

    # ------------------------------------------------------------------
    # Prepare texts / batching / cache key
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

    @staticmethod
    def _groups(batches: list[list[int]], size: int):
        iterator = iter(batches)
        while group := list(islice(iterator, size)):
            yield group

    def _cache_key(self, item: str) -> str:
        content_hash = hashlib.sha256(str(item).encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{self._instruction_ns}|{content_hash}"

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

    async def close(self) -> None:
        if self.cache is not None:
            await asyncio.to_thread(self.cache.close)
        await self._client.aclose()


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
