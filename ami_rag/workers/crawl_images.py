"""Tách section 'Hình ảnh' khỏi text crawl.

URL ảnh gốc được giữ lại trong ``artifact_urls`` của text item (client tự
dùng URL nguồn); ảnh crawl KHÔNG còn được tải về MinIO (đã bỏ CrawlImageWorker
— ảnh cũ dưới MinIO dọn bằng `ami-rag cleanup --crawl-assets`).
"""

import re
from urllib.parse import urlparse

from ami_rag.textnorm import fold_text

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
    kept = "\n".join(ln for k, ln in enumerate(lines) if k not in remove).rstrip()
    return urls, kept
