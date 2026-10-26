"""Base dùng chung cho 3 embedder client HTTP (máy B).

Gom phần trùng lặp giữa ``RemoteEmbedder`` (qwen-embedding-server),
``OpenAIEmbedder`` (gateway nemotron-vl-vllm) và ``BGEM3Embedder``
(embedding-server bge-m3): init client, retry exponential backoff + jitter
CHỈ cho lỗi tạm thời (timeout/5xx/429/connect — không bao giờ 4xx), circuit
breaker N lần liên tiếp, cache embedding, chia nhóm batch song song.
Subclass cung cấp handshake (``verify``), payload và cách đọc vector từ
response (``_check_response`` / ``_fill_vectors`` / ``_post_embed``).
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
    EmbedServerOOM,
    EmbedServerUnreachable,
    clamp_embed_text,
)

logger = logging.getLogger(__name__)

# N lần liên tiếp không với tới server -> dừng cả lô sớm (thay vì đánh failed hàng loạt)
CIRCUIT_THRESHOLD = 5


class _Cache(Protocol):
    def get_many(
        self, keys: list[str]
    ) -> dict[str, list[float] | list[list[float]]]: ...
    def put_many(
        self, mapping: dict[str, list[float] | list[list[float]]]
    ) -> None: ...


class BaseRemoteEmbedder:
    """HTTP plumbing dùng chung; ``model``/``dim`` được verify ở handshake."""

    # Mã HTTP coi là "input vượt budget" (bge server trả thêm 422)
    too_long_statuses: tuple[int, ...] = (413,)

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        expected_dim: int | None = None,
        token: str = "",
        timeout: float = 60,
        batch_size: int = 32,
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
    # Handshake / verify (subclass cung cấp)
    # ------------------------------------------------------------------
    async def verify(self) -> dict:
        raise NotImplementedError

    async def _ensure_verified(self) -> None:
        if not self._verified:
            await self.verify()

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
    # HTTP dùng chung
    # ------------------------------------------------------------------
    async def _post(self, url: str, payload: dict) -> dict:
        """Một POST với retry cho lỗi tạm thời; 4xx raise ngay."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    resp = await self._client.post(url, json=payload)
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
            if resp.status_code in self.too_long_statuses:
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

    @staticmethod
    def _groups(batches: list[list[int]], size: int):
        iterator = iter(batches)
        while group := list(islice(iterator, size)):
            yield group

    def _backoff(self, attempt: int) -> float:
        return self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

    # ------------------------------------------------------------------
    # Cache key / clamp / batching (dùng chung remote + openai; bge truyền str)
    # ------------------------------------------------------------------
    def _cache_key(self, item: str | dict) -> str:
        if isinstance(item, dict):
            t = item.get("text") or ""
            img = item.get("image_b64") or ""
            content_hash = hashlib.sha256(f"{t}|{img}".encode()).hexdigest()
        else:
            content_hash = hashlib.sha256(item.encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{self._instruction_ns}|{content_hash}"

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
                out.append(clamped)
        if truncated:
            logger.warning(
                "embed_documents: %d/%d item bị cắt theo EMBED_MAX_INPUT_TOKENS=%s",
                truncated,
                len(items),
                self.max_input_tokens,
            )
        return out

    def _split_batches(
        self, items: list[str | dict], pending: list[int]
    ) -> list[list[int]]:
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

    # ------------------------------------------------------------------
    # Luồng embed_documents dùng chung (remote + openai; bge override riêng
    # vì cần dense+sparse trong một lượt gọi)
    # ------------------------------------------------------------------
    async def embed_documents(self, items: list[str | dict]) -> list[list[float]]:
        """Embed danh sách văn bản / multimodal items (có cache). Giữ thứ tự đầu vào."""
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
                    self._fill_vectors(result, batch_indices, vectors)
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

    # Hooks cho subclass trong embed_documents
    async def _post_embed(self, items: list[dict]) -> dict:
        raise NotImplementedError

    @staticmethod
    def _to_embed_payload(item: str | dict) -> dict:
        raise NotImplementedError

    def _check_response(self, result: dict) -> None:
        """Response phải khớp handshake - phát hiện server đổi model giữa chừng."""
        raise NotImplementedError

    def _fill_vectors(
        self, result: dict, batch_indices: list[int], vectors: list[list[float] | None]
    ) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        if self.cache is not None:
            await asyncio.to_thread(self.cache.close)
        await self._client.aclose()
