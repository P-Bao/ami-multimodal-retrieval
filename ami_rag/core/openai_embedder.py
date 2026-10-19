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
- Retry exponential backoff + jitter CHỈ cho lỗi tạm thời; KHÔNG retry 4xx.
- Chia batch theo số item (``EMBED_BATCH_SIZE``) và theo byte payload; item vượt
  token budget bị head-truncate trước khi gửi (``EMBED_MAX_INPUT_TOKENS``, trừ
  ``EMBED_IMAGE_TOKEN_RESERVE`` khi kèm ảnh) - không vượt max_model_len vLLM.
- Cache embedding theo khoá (model, dim, hash nội dung) - chia sẻ EmbeddingCache.
- Circuit breaker: N lần liên tiếp không với tới server -> dừng cả lô sớm.
"""

import asyncio
import base64
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


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def data_url_from_bytes(data: bytes) -> str:
    """bytes ảnh -> data URI (dùng chung cho embed + rerank multimodal)."""
    return f"data:{_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


class OpenAIEmbedder:
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
        cache: _Cache | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.expected_dim = expected_dim
        self.dim = expected_dim if expected_dim is not None else 0
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_payload_bytes = max_payload_bytes
        self.max_input_tokens = max_input_tokens
        self.image_token_reserve = image_token_reserve
        self.max_concurrency = max_concurrency
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
        result = await self._post_embed_batch([{"text": "probe"}])
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

    async def _ensure_verified(self) -> None:
        if not self._verified:
            await self.verify()

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    async def embed_documents(self, items: list[str | dict]) -> list[list[float]]:
        """Embed danh sách văn bản / multimodal items (input_type=document, có cache). Giữ thứ tự đầu vào."""
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
                        self._post_embed_batch([self._to_embed_payload(items[i]) for i in b])
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
        """Embed một câu hỏi (input_type=query, không cache)."""
        await self._ensure_verified()
        text = clamp_embed_text(text, self.max_input_tokens)
        result = await self._post_embed_batch([{"text": text}], input_type="query")
        self._check_response(result)
        data = result.get("data") or []
        if not data or not data[0].get("embedding"):
            raise EmbedderError("embed server trả response /v1/embeddings rỗng")
        return data[0]["embedding"]

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
    async def _post_embed_batch(
        self, items: list[str | dict], *, input_type: str = "document"
    ) -> dict:
        """Một POST /v1/embeddings (batch items) với retry cho lỗi tạm thời; 4xx raise ngay."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self._client.post(
                        f"{self.base_url}/v1/embeddings",
                        json={"input": items, "input_type": input_type},
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
        """Mỗi response /v1/embeddings phải khớp model handshake."""
        model = result.get("model")
        if model != self.model:
            raise EmbedModelMismatch(
                f"embed server trả model={model!r}, không khớp handshake "
                f"(model={self.model!r}). Server có thể bị đổi model giữa chừng."
            )

    # ------------------------------------------------------------------
    # Batching / cache key / payload builder
    # ------------------------------------------------------------------
    def _clamp_items(self, items: list[str | dict]) -> list[str | dict]:
        """Head-truncate item vượt token budget (trước cache key + batching).

        Text đầy đủ vẫn được caller lưu ở Qdrant payload - chỉ input gửi lên
        server bị cắt để không vượt max_model_len.
        """
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
