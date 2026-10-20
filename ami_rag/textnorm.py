"""Helper chuẩn hoá text dùng chung (fold bỏ dấu, so khớp tiêu đề).

Tách riêng để `core` (chia chunk) và `workers` (nhận diện heading crawl) dùng
chung một cách bỏ dấu — trước đây mỗi nơi tự viết, dễ lệch hành vi.
"""

import unicodedata

# Ký tự markdown đứng đầu dòng heading ("# Tiêu đề", "**Tiêu đề:**" ...)
_MD_PREFIX = "#*>-= \t"


def fold_text(text: str) -> str:
    """Lowercase + bỏ dấu tiếng Việt ('Hình ảnh' -> 'hinh anh')."""
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def join_caption(value) -> str:
    """Caption/list caption -> chuỗi (None/rỗng -> ""); nơi cần None bọc thêm ``or None``."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def strip_markdown_prefix(text: str) -> str:
    """Bỏ ký tự markdown ở đầu chuỗi (heading/bullet/table pipe)."""
    return (text or "").lstrip(_MD_PREFIX)


def starts_with_title(content: str, title: str) -> bool:
    """``content`` đã bắt đầu bằng ``title`` (so khớp không dấu, không phân biệt hoa thường).

    Dùng để không thêm tiêu đề doc vào chunk khi chunk vốn đã có tiêu đề ở dòng
    đầu (thường gặp ở bài crawl: `# Tiêu đề` ngay đầu content).
    """
    t = fold_text(title).strip()
    if not t:
        return True
    head = fold_text(strip_markdown_prefix((content or "").lstrip()))
    return head.startswith(t)


def with_title(content: str, title: str) -> str:
    """Prefix tiêu đề doc vào đầu chunk (bỏ qua nếu chunk đã bắt đầu bằng tiêu đề).

    Chunk crawl hay là một dòng trần như "- Địa chỉ: Tầng 1 nhà A1, ..." — không
    rõ thuộc đơn vị nào. Prefix tiêu đề làm chunk tự mô tả được (rất tốt cho
    cả vector search lẫn câu trả lời của LLM).
    """
    t = (title or "").strip()
    if not t or not (content or "").strip():
        return content
    if starts_with_title(content, t):
        return content
    return f"{t}\n{content}"
