"""VinterOcrClient - client OCR fallback (gateway ``vintern-vllm``, máy B :8090).

Đối tác: repo ``vintern-vllm`` (máy B) - gateway FastAPI trước vLLM serving
``5CD-AI/Vintern-1B-v3_5``. Contract gateway:
- POST ``/v1/ocr`` (multipart ``file`` + ``mode`` + ``pages`` + ``dpi``):
  - ``mode=text``  -> ``{"pages", "text", "page_texts": [{"page", "text"}]}``
  - ``mode=table`` -> ``{"pages", "tables": [{"page", "html", "markdown"}]}``
- GET ``/health`` -> ``{"status", "model", "vllm"}`` (503 khi vLLM chưa lên).

Chỉ được gọi khi OCR service chính (171.226.10.153:12006) lỗi/timeout/trả rỗng -
việc chuyển nhánh do ``BackendDataExtractor`` (raganything/backend_data_extract.py)
quyết định qua ``fallback_modes`` / ``fallback_on_empty``.

Pattern lỗi giống fix 26d23f8: connect timeout tách riêng để fail-fast, message
phân biệt ReadTimeout (service treo) / ConnectError (unreachable) / ConnectTimeout
(SYN drop). Retry exponential backoff + jitter CHỈ cho 429/5xx/transport; 4xx
raise ngay.
"""

import asyncio
import logging
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRY_BASE = 0.5  # giây; backoff = _RETRY_BASE * 2^n + jitter


class OcrFallbackError(Exception):
    """Lỗi gọi OCR fallback Vinter (không reach được / HTTP lỗi / không JSON / text rỗng)."""


def _describe_http_error(exc: httpx.HTTPError) -> str:
    """Phân biệt loại lỗi kết nối trong message (giống backend_data_extract)."""
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return "OCR fallback nhận kết nối nhưng không phản hồi (service treo/quá tải)"
    if isinstance(exc, httpx.ConnectError):
        return "Không kết nối được OCR fallback (host/port unreachable hoặc firewall)"
    if isinstance(exc, httpx.ConnectTimeout):
        return "Kết nối OCR fallback timeout (firewall drop SYN?)"
    return "Lỗi HTTP OCR fallback"


class VinterOcrClient:
    """Client gateway OCR Vinter. Không tự quản vòng đời client nào lâu dài
    (mỗi request 1 ``AsyncClient`` như ``BackendDataExtractor``), nên ``close()``
    là no-op; giữ trong signature để tương thích khi đổi sang persistent client."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        token: str = "",
        timeout: float = 300.0,
        retries: int = 1,
        dpi: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._base_url = (base_url or "").rstrip("/")
        self._model = model or ""
        self._token = token or ""
        self._timeout = httpx.Timeout(float(timeout), connect=min(10.0, float(timeout)))
        self._retries = max(int(retries), 0)
        self._dpi = dpi
        self._http_client = http_client  # inject cho test (MockTransport)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    async def _post_ocr(self, file_bytes: bytes, filename: str, mode: str, pages: Any) -> dict:
        url = f"{self._base_url}/v1/ocr"
        data: dict[str, str] = {"mode": mode}
        if pages:
            data["pages"] = str(pages)
        if self._dpi:
            data["dpi"] = str(self._dpi)
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        last_error: OcrFallbackError | None = None
        for attempt in range(self._retries + 1):
            try:
                if self._http_client is not None:
                    response = await self._http_client.post(url, files=files, data=data, headers=headers)
                else:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(url, files=files, data=data, headers=headers)
            except httpx.HTTPError as exc:
                last_error = OcrFallbackError(f"{_describe_http_error(exc)}: {type(exc).__name__}: {exc}")
                if attempt == self._retries:
                    raise last_error from exc
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise OcrFallbackError("OCR fallback trả response không phải JSON") from exc
                last_error = OcrFallbackError(
                    f"OCR fallback trả mã lỗi {response.status_code}: {response.text[:300]}"
                )
                # 4xx (trừ 429) là lỗi caller - không retry
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == self._retries:
                    raise last_error
            await asyncio.sleep(_RETRY_BASE * (2**attempt) + random.uniform(0, 0.25))
        raise last_error or OcrFallbackError("OCR fallback: unreachable code")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    async def health(self) -> dict:
        """GET /health - không raise, trả ``{}`` khi lỗi (chỉ để kiểm tra/log)."""
        try:
            if self._http_client is not None:
                response = await self._http_client.get(f"{self._base_url}/health", timeout=5)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=5)) as client:
                    response = await client.get(f"{self._base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, dict) else {}
        except Exception:  # health không bao giờ raise
            pass
        return {}

    async def ocr_text(self, file_bytes: bytes, filename: str = "document.pdf", *, pages=None) -> str:
        """OCR text toàn bộ file (PDF/ảnh). Gộp ``page_texts`` theo thứ tự trang,
        fallback về field ``text`` khi thiếu. Text rỗng -> ``OcrFallbackError``."""
        data = await self._post_ocr(file_bytes, filename, "text", pages)
        if not isinstance(data, dict):
            raise OcrFallbackError("OCR fallback trả response rỗng")
        text = ""
        page_texts = data.get("page_texts")
        if isinstance(page_texts, list):
            parts: list[str] = []
            for item in page_texts:
                if isinstance(item, dict):
                    page_text = str(item.get("text") or "").strip()
                    if page_text:
                        parts.append(page_text)
            text = "\n\n".join(parts).strip()
        if not text:
            raw = data.get("text")
            if isinstance(raw, str):
                text = raw.strip()
        if not text:
            raise OcrFallbackError("OCR fallback không trích xuất được text nào")
        return text

    async def ocr_tables(self, file_bytes: bytes, filename: str = "document.pdf", *, pages=None) -> list[dict]:
        """OCR bảng -> ``[{"page": int | None, "html": str}]`` (bỏ item html rỗng).

        Không có bảng / lỗi -> caller tự xử lý (extract_tables trả ``[]`` khi cả
        primary lẫn fallback đều thất bại)."""
        data = await self._post_ocr(file_bytes, filename, "table", pages)
        if not isinstance(data, dict):
            raise OcrFallbackError("OCR fallback trả response rỗng")
        tables: list[dict] = []
        raw_tables = data.get("tables")
        if isinstance(raw_tables, list):
            for item in raw_tables:
                if not isinstance(item, dict):
                    continue
                html = str(item.get("html") or "").strip()
                if not html:
                    continue
                page = item.get("page")
                try:
                    page = int(page) if page is not None else None
                except (TypeError, ValueError):
                    page = None
                tables.append({"page": page, "html": html})
        return tables

    async def close(self) -> None:
        """No-op: client dùng ``AsyncClient`` per-request (tự đóng sau mỗi call)."""


def build_vinter_ocr_client(settings) -> VinterOcrClient | None:
    """Tạo client OCR fallback từ Settings; ``None`` khi bị tắt hoặc URL rỗng."""
    if not getattr(settings, "OCR_FALLBACK_ENABLED", True):
        return None
    base_url = (getattr(settings, "OCR_FALLBACK_URL", "") or "").strip()
    if not base_url:
        return None
    model = getattr(settings, "OCR_FALLBACK_MODEL", "") or ""
    logger.info("OCR fallback Vinter: %s (model %s)", base_url, model or "?")
    return VinterOcrClient(
        base_url,
        model=model,
        token=getattr(settings, "OCR_FALLBACK_TOKEN", "") or "",
        timeout=float(getattr(settings, "OCR_FALLBACK_TIMEOUT", 300) or 300),
        retries=int(getattr(settings, "OCR_FALLBACK_RETRIES", 1)),
        dpi=getattr(settings, "OCR_FALLBACK_DPI", None) or None,
    )
