import asyncio
import hashlib
import logging
import re
from io import BytesIO
from urllib.parse import urlparse

import httpx

from ami_rag.storage.assets import minio_http_client
from ami_rag.textnorm import fold_text

logger = logging.getLogger(__name__)

_IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_IMAGE_BULLET_RE = re.compile(r"^\s*[-*]\s*(https?://\S+)\s*$", re.MULTILINE)
_IMAGE_PLAIN_RE = re.compile(r"^(https?://\S+)$")
_IMAGE_HEADING_STRIP_RE = re.compile(r"^[\s#>*_~`]+")

_IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
    ".avif",
)


# shortcut: tắt verify SSL cho 2 domain trường (chuỗi chứng chỉ gãy) - chấp nhận
# rủi ro MITM vì chỉ crawl 2 nguồn nội bộ này; chặt hơn thì pin issuer cert.
_TRUSTED_NO_SSL_VERIFY_DOMAINS = (
    "ptit.edu.vn",
    "ptithcm.edu.vn",
)


def _fold(line: str) -> str:
    """Lowercase + bỏ dấu tiếng Việt ('Hình ảnh' -> 'hinh anh')."""
    return fold_text(line)


def _is_image_heading(line: str) -> bool:
    """Dòng heading section ảnh: 'Hình ảnh', '**Hình ảnh:**', '## HÌNH ẢNH' ..."""
    return _fold(_IMAGE_HEADING_STRIP_RE.sub("", line)).startswith("hinh anh")


def _has_image_ext(url: str) -> bool:
    return urlparse(url).path.lower().endswith(_IMAGE_EXTS)


def _line_image_url(line: str) -> str | None:
    """URL ảnh của MỘT dòng (bullet / plain / markdown image); None nếu dòng
    không phải URL ảnh (bao gồm URL không có đuôi ảnh - coi là prose, dừng section)."""
    s = line.strip()
    m = _IMAGE_MARKDOWN_RE.search(s)
    if m:
        url = m.group(1)
    else:
        m = _IMAGE_BULLET_RE.match(s) or _IMAGE_PLAIN_RE.match(s)
        if not m:
            return None
        url = m.group(1)
    return url if _has_image_ext(url) else None


def split_image_section(text: str) -> tuple[list[str], str]:
    """Tách section ảnh của bài crawl: heading 'Hình ảnh' + các dòng URL ảnh.

    Backend crawl ghi phần ảnh cuối content dạng::

        **Hình ảnh:**

        - https://.../a.jpg
        - https://.../b.jpg

    Trả về ``(urls, text_mới)``:
    - urls: URL ảnh (lọc đuôi ảnh, dedupe, giữ thứ tự) từ mọi section
      'Hình ảnh' trong bài.
    - text_mới: text đã xoá heading + dòng URL (kèm blank line giữa chúng)
      của các section có ít nhất 1 URL ảnh; section heading mà không theo
      sau URL ảnh nào (gặp prose/URL không phải ảnh) thì giữ nguyên.

    Không tìm thấy section hợp lệ nào -> ``([], text_gốc)``.
    """
    if not text:
        return [], text
    lines = text.splitlines()
    urls: list[str] = []
    seen: set[str] = set()
    remove: set[int] = set()
    i, n = 0, len(lines)
    while i < n:
        if not _is_image_heading(lines[i]):
            i += 1
            continue
        j = i + 1
        section_urls: list[str] = []
        section_lines: set[int] = set()
        blanks: list[int] = []
        while j < n:
            line = lines[j]
            if not line.strip():
                blanks.append(j)
                j += 1
                continue
            if _is_image_heading(line):
                break
            url = _line_image_url(line)
            if not url:
                break
            section_urls.append(url)
            section_lines.update(blanks)
            section_lines.add(j)
            blanks = []
            j += 1
        if section_urls:
            for u in section_urls:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
            remove.add(i)
            remove.update(section_lines)
            i = j
        else:
            i += 1
    if not urls:
        return [], text
    kept = "\n".join(
        ln for k, ln in enumerate(lines) if k not in remove
    ).rstrip()
    return urls, kept


def _ext_from_url(url: str, content_type: str | None) -> str:
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".webm", ".mp4"):
        if ext in url.lower():
            return ext.lstrip(".")
    if content_type:
        subtype = content_type.rsplit("/", 1)[-1].split(";")[0]
        if subtype and len(subtype) <= 5:
            return subtype
    return "bin"


def _is_trusted_no_ssl_host(host: str | None) -> bool:
    """ptit.edu.vn / ptithcm.edu.vn (và subdomain) được tin cậy -> bỏ verify SSL."""
    if not host:
        return False
    host = host.lower()
    return any(
        host == domain or host.endswith("." + domain)
        for domain in _TRUSTED_NO_SSL_VERIFY_DOMAINS
    )


def _classify_crawl_error(exc: Exception) -> str:
    """Nhóm lỗi tải ảnh crawl cho metric ``crawl_image_failures_total{reason}``."""
    import ssl

    from minio.error import S3Error

    if isinstance(exc, ssl.SSLError) or isinstance(getattr(exc, "__cause__", None), ssl.SSLError):
        return "ssl"
    if isinstance(exc, httpx.TooManyRedirects):
        return "redirect"
    if isinstance(exc, httpx.HTTPStatusError):
        return "http"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "connect"
    if isinstance(exc, S3Error):
        return "minio"
    return "other"


class CrawlImageWorker:
    """Download crawl images into MinIO so multimodal (VLM) processing and
    /rag responses can reference durable objects instead of remote URLs.

    Always on (no env switch). Idempotent: an object key is
    derived from (document_id, sha256(url)) and existing objects are reused.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str,
        secure: bool = False,
        timeout: int = 60,
    ):
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._secure = secure
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                http_client=minio_http_client(),
            )
        return self._client

    def collect_image_urls(self, doc: dict) -> list[str]:
        urls: list[str] = []
        metadata = doc.get("metadata") or {}
        for item in metadata.get("images") or []:
            url = item.get("url") if isinstance(item, dict) else item
            if isinstance(url, str) and url.startswith("http"):
                urls.append(url)
        content = doc.get("content") or ""
        urls.extend(m.group(1) for m in _IMAGE_MARKDOWN_RE.finditer(content))
        urls.extend(m.group(1) for m in _IMAGE_BULLET_RE.finditer(content))
        seen: set[str] = set()
        unique = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        return unique

    def _download(self, url: str) -> tuple[bytes, str | None]:
        from urllib.parse import urljoin, urlparse

        current_url = url
        for _ in range(10):
            parsed = urlparse(current_url)
            trusted = (
                parsed.scheme == "https"
                and _is_trusted_no_ssl_host(parsed.hostname)
            )
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=False,
                verify=not trusted,
            ) as client:
                resp = client.get(current_url)
            status = getattr(resp, "status_code", 0)
            if isinstance(status, int) and status in (301, 302, 303, 307, 308):
                location = resp.headers.get("location")
                if not location:
                    resp.raise_for_status()
                current_url = urljoin(current_url, location)
                continue
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type")
        raise httpx.TooManyRedirects("exceeded redirect limit", request=None)

    def _upload(self, object_name: str, data: bytes, content_type: str | None) -> None:
        client = self._get_client()
        if not client.bucket_exists(self._bucket):
            client.make_bucket(self._bucket)
        client.put_object(
            self._bucket,
            object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def _exists(self, object_name: str) -> bool:
        from minio.error import S3Error

        try:
            self._get_client().stat_object(self._bucket, object_name)
            return True
        except S3Error:
            return False

    async def ensure_in_minio(self, doc_id: str, url: str) -> str | None:
        """Download one image and store it in MinIO; returns the object key."""
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        object_name = f"{self._prefix}/{doc_id}/{url_hash}.{_ext_from_url(url, None)}"
        try:
            if await asyncio.to_thread(self._exists, object_name):
                return object_name
            data, content_type = await asyncio.to_thread(self._download, url)
            await asyncio.to_thread(self._upload, object_name, data, content_type)
            return object_name
        except Exception as exc:
            from ami_rag.observability import INGEST_CRAWL_IMAGE_FAILURES_TOTAL

            INGEST_CRAWL_IMAGE_FAILURES_TOTAL.labels(
                reason=_classify_crawl_error(exc)
            ).inc()
            logger.warning("crawl image %s (doc=%s) lỗi: %r", url, doc_id, exc)
            return None
