"""RemoteEmbedder - client embedding server (máy B) cho máy A.

Implements interface ``Embedder``. Đặc điểm:
- Handshake lúc khởi tạo sử dụng: gọi ``/info``, so ``model_name``/``dim`` với config;
  lệch thì dừng với lỗi rõ ràng, không bao giờ ghi vector khi chưa xác minh.
- Mỗi response của ``/embed`` chứa ``model_name``/``dim`` - kiểm tra lại từng response
  để phát hiện server bị đổi model giữa chừng.
- Retry exponential backoff + jitter CHỈ cho lỗi tạm thời (timeout, 5xx, 429, mất
  kết nối); KHÔNG retry lỗi 4xx.
- Chia batch theo số item (``EMBED_BATCH_SIZE``) và theo byte payload
  (``EMB_MAX_PAYLOAD_BYTES``); item vượt token budget bị head-truncate trước khi
  gửi (``EMBED_MAX_INPUT_TOKENS``); giới hạn số request đồng thời.
- Cache embedding (tuỳ chọn, bật mặc định) theo khoá (model, dim, instruction, hash text).
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


class _Cache(Protocol):
    def get_many(self, keys: list[str]) -> dict[str, list[float]]: ...
    def put_many(self, mapping: dict[str, list[float]]) -> None: ...


class RemoteEmbedder:
    """Client embedding server máy B."""

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
        cache: _Cache | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.expected_dim = expected_dim
        self.dim = expected_dim if expected_dim is not None else 0
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_payload_bytes = max_payload_bytes
        self.max_input_tokens = max_input_tokens
        self.image_token_reserve = image_token_reserve
        self.retries = retries
        self.backoff_base = backoff_base
        self.cache = cache
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
            raise EmbedServerUnreachable(
                f"embed server /info trả {resp.status_code}"
            )
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

    async def _ensure_verified(self) -> None:
        if not self._verified:
            await self.verify()

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_documents(self, items: list[str | dict]) -> list[list[float]]:
        """Embed danh sách văn bản / multimodal items (type=document, có cache). Giữ thứ tự đầu vào."""
        if not items:
            return []
        await self._ensure_verified()

        items = self._clamp_items(items)
        vectors: list[list[float] | None] = [None] * len(items)
        keys: list[str] = [self._cache_key(it) for it in items]
        pending: list[int] = list(range(len(items)))
        if self.cache is not None:
            cached = await asyncio.to_thread(self.cache.get_many, keys)
            pending = [i for i in range(len(items)) if keys[i] not in cached]
            for i, key in enumerate(keys):
                if key in cached:
                    vectors[i] = cached[key]

        if pending:
            batches = self._split_batches(items, pending)
            unreachable: EmbedServerUnreachable | None = None
            for group in self._groups(batches, self.max_concurrency):
                results = await asyncio.gather(
                    *(
                        self._post_embed([self._to_embed_payload(items[i]) for i in b])
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
                    self._check_response(result)
                    for idx, vec in zip(batch_indices, result["vectors"]):
                        vectors[idx] = vec
                if self._consecutive_unreachable >= CIRCUIT_THRESHOLD:
                    raise EmbedCircuitOpen(
                        f"embed server không với tới {self._consecutive_unreachable} "
                        "lô liên tiếp; dừng lô sớm - các doc còn lại giữ nguyên trạng thái"
                    )
            if unreachable is not None:
                raise unreachable
            if self.cache is not None:
                new_entries = {
                    keys[i]: vectors[i] for i in pending if vectors[i] is not None
                }
                if new_entries:
                    await asyncio.to_thread(self.cache.put_many, new_entries)
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:
            raise EmbedderError(
                f"embed_documents thiếu vector cho {len(missing)}/{len(items)} item"
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed một câu hỏi (type=query, không cache)."""
        await self._ensure_verified()
        text = clamp_embed_text(text, self.max_input_tokens)
        result = await self._post_embed([{"type": "query", "text": text}])
        self._check_response(result)
        return result["vectors"][0]

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
    # HTTP
    # ------------------------------------------------------------------
    async def _post_embed(self, items: list[dict]) -> dict:
        """Một POST /embed với retry cho lỗi tạm thời; 4xx raise ngay."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self._client.post(
                        f"{self.base_url}/embed", json={"items": items}
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.retries:
                    # Hết retry: đếm lô này là 1 lần unreachable liên tiếp
                    # (circuit breaker ở embed_documents quyết định dừng lô)
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
            if resp.status_code == 413:
                raise EmbedInputTooLong(f"input quá dài / payload vượt giới hạn: {detail}")
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

    # ------------------------------------------------------------------
    # Batching / cache key / payload builder
    # ------------------------------------------------------------------
    def _clamp_items(self, items: list[str | dict]) -> list[str | dict]:
        """Head-truncate item vượt token budget (trước cache key + batching)."""
        out: list[str | dict] = []
        truncated = 0
        for item in items:
            if isinstance(item, dict):
                text = item.get("text") or ""
                has_image = bool(item.get("image_b64"))
                clamped = clamp_embed_text(
                    text,
                    self.max_input_tokens,
                    has_image=has_image,
                    image_tokens=self.image_token_reserve,
                )
                if clamped != text:
                    truncated += 1
                    logger.warning(
                        "embed item text vượt budget: cắt %d -> %d chars (has_image=%s)",
                        len(text),
                        len(clamped),
                        has_image,
                    )
                    out.append({**item, "text": clamped})
                else:
                    out.append(item)
            else:
                clamped = clamp_embed_text(item, self.max_input_tokens)
                if clamped != item:
                    truncated += 1
                    logger.warning(
                        "embed text vượt budget: cắt %d -> %d chars",
                        len(item),
                        len(clamped),
                    )
                out.append(clamped)
        if truncated:
            logger.warning(
                "embed_documents: %d/%d item bị cắt theo EMBED_MAX_INPUT_TOKENS=%s",
                truncated,
                len(items),
                self.max_input_tokens,
            )
        return out

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

    def _split_batches(self, items: list[str | dict], pending: list[int]) -> list[list[int]]:
        """Chia batch theo số item VÀ theo byte payload (ảnh/text rất lớn tách riêng)."""
        batches: list[list[int]] = []
        current: list[int] = []
        current_bytes = 0
        for idx in pending:
            item = items[idx]
            if isinstance(item, dict):
                t_bytes = len((item.get("text") or "").encode("utf-8"))
                img_bytes = len((item.get("image_b64") or "").encode("ascii"))
                item_bytes = t_bytes + img_bytes + 128
            else:
                item_bytes = len(item.encode("utf-8")) + 128
            if current and (
                len(current) >= self.batch_size
                or current_bytes + item_bytes > self.max_payload_bytes
            ):
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(idx)
            current_bytes += item_bytes
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _groups(batches: list[list[int]], size: int):
        iterator = iter(batches)
        while group := list(islice(iterator, size)):
            yield group

    def _cache_key(self, item: str | dict) -> str:
        if isinstance(item, dict):
            t = item.get("text") or ""
            img = item.get("image_b64") or ""
            content_hash = hashlib.sha256(f"{t}|{img}".encode()).hexdigest()
        else:
            content_hash = hashlib.sha256(item.encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{self._instruction_ns}|{content_hash}"

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

    async def close(self) -> None:
        if self.cache is not None:
            await asyncio.to_thread(self.cache.close)
        await self._client.aclose()


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
