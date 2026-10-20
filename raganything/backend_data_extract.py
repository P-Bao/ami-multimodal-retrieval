"""Text/table extraction kiểu `documents-management/backend_data` cho ami_rag.

Port từ `ami_data/documents-management/backend_data/app/services/`:
- `ocr_pdf_service.py`: text PDF bằng pdfplumber (PyMuPDF fallback), OCR scan
  qua endpoint `/api/v1/hierarchy-parse`, bảng qua `/api/v1/raw_miner_ocr/qwen`.
- `docx_to_markdown_service.py`: paragraph/heading/list/table DOCX bằng python-docx,
  .doc convert sang .docx bằng LibreOffice headless.

Dùng trong `ami_rag` để thay phần text/table của MinerU (giữ nguyên MinerU cho
ảnh). Ảnh (`type == "image"`) từ content_list MinerU được merge riêng.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://171.226.10.153:12006"
DEFAULT_TABLE_ENDPOINT = "/api/v1/raw_miner_ocr/qwen"
DEFAULT_TEXT_ENDPOINT = "/api/v1/hierarchy-parse"


class BackendDataParseError(Exception):
    """Lỗi chung khi trích xuất text/bảng kiểu backend_data."""


class BackendDataExtractor:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        table_endpoint: str = DEFAULT_TABLE_ENDPOINT,
        text_endpoint: str = DEFAULT_TEXT_ENDPOINT,
        http_client=None,
        fallback=None,
        fallback_modes: frozenset = frozenset({"text", "table"}),
        fallback_on_empty: bool = True,
    ):
        import httpx

        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(timeout, connect=min(10.0, float(timeout)))
        self._table_endpoint = table_endpoint.lstrip("/")
        self._text_endpoint = text_endpoint.lstrip("/")
        self._http_client = http_client
        # OCR fallback Vinter (ami_rag/core/ocr_client.py); None = tắt.
        self._fallback = fallback
        self._fallback_modes = frozenset(fallback_modes or ())
        self._fallback_on_empty = bool(fallback_on_empty)

    # ------------------------------------------------------------------
    # Fallback Vinter (OCR service chính lỗi/timeout/trả rỗng)
    # ------------------------------------------------------------------
    def _fallback_enabled(self, kind: str) -> bool:
        return self._fallback is not None and kind in self._fallback_modes

    def _log_fallback(
        self, kind: str, reason: str, exc: BaseException | None = None
    ) -> None:
        detail = f": {exc}" if exc is not None else ""
        logger.warning(
            "OCR %s: %s -> chuyển sang fallback Vinter%s", kind, reason, detail
        )

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------
    async def _post_file(
        self, endpoint: str, file_bytes: bytes, filename: str, content_type: str
    ):
        import httpx

        files = {"file": (filename, io.BytesIO(file_bytes), content_type)}
        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    f"{self._base_url}/{endpoint}",
                    files=files,
                    headers={"Accept": "application/json"},
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/{endpoint}",
                        files=files,
                        headers={"Accept": "application/json"},
                    )
        except httpx.HTTPError as exc:
            if isinstance(
                exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)
            ):
                hint = "OCR service nhận kết nối nhưng không phản hồi (service treo/quá tải)"
            elif isinstance(exc, httpx.ConnectError):
                hint = "Không kết nối được OCR service (host/port unreachable hoặc firewall)"
            elif isinstance(exc, httpx.ConnectTimeout):
                hint = "Kết nối OCR service timeout (firewall drop SYN?)"
            else:
                hint = "Lỗi HTTP OCR service"
            raise BackendDataParseError(f"{hint}: {type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise BackendDataParseError(
                f"OCR service trả mã lỗi {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise BackendDataParseError(
                "OCR service trả response không phải JSON"
            ) from exc

    # ------------------------------------------------------------------
    # Text: pdfplumber -> PyMuPDF -> hierarchy-parse endpoint
    # ------------------------------------------------------------------
    def extract_pdf_blocks(self, file_bytes: bytes) -> list[dict]:
        """Text + vị trí bảng bằng pdfplumber. Trả về list block theo thứ tự đọc
        (paragraph + table xen kẽ theo bbox top trong từng trang)."""
        try:
            import pdfplumber
        except ImportError as exc:
            raise BackendDataParseError("Thiếu dependency pdfplumber") from exc

        blocks: list[dict] = []
        try:
            pdf_file = io.BytesIO(file_bytes)
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_blocks: list[dict] = []
                    tables = page.find_tables()
                    table_bboxes = []
                    for table in tables:
                        bbox = table.bbox
                        table_bboxes.append(bbox)
                        rows = table.extract()
                        if not rows:
                            continue
                        md_rows = []
                        for i, row in enumerate(rows):
                            cells = [
                                str(c or "").strip().replace("|", "\\|") for c in row
                            ]
                            md_rows.append("| " + " | ".join(cells) + " |")
                            if i == 0:
                                md_rows.append(
                                    "| " + " | ".join(["---"] * len(cells)) + " |"
                                )
                        page_blocks.append(
                            {
                                "type": "table",
                                "content": "\n".join(md_rows),
                                "html": self._table_rows_to_html(rows),
                                "page": page_num,
                                "bbox": {
                                    "x0": bbox[0],
                                    "top": bbox[1],
                                    "x1": bbox[2],
                                    "bottom": bbox[3],
                                },
                            }
                        )

                    words = page.extract_words()
                    non_table_words = []
                    for word in words:
                        word_bbox = (
                            word["x0"],
                            word["top"],
                            word["x1"],
                            word["bottom"],
                        )
                        in_table = False
                        for tbbox in table_bboxes:
                            if (
                                word_bbox[0] >= tbbox[0]
                                and word_bbox[2] <= tbbox[2]
                                and word_bbox[1] >= tbbox[1]
                                and word_bbox[3] <= tbbox[3]
                            ):
                                in_table = True
                                break
                        if not in_table:
                            non_table_words.append(word)

                    lines: dict[float, list] = {}
                    for word in non_table_words:
                        top = round(word["top"], 1)
                        lines.setdefault(top, []).append(word)

                    for top in sorted(lines.keys()):
                        line_words = sorted(lines[top], key=lambda w: w["x0"])
                        line_text = " ".join(w["text"] for w in line_words)
                        if line_text.strip():
                            page_blocks.append(
                                {
                                    "type": "paragraph",
                                    "content": line_text.strip(),
                                    "page": page_num,
                                    "bbox": {
                                        "x0": min(w["x0"] for w in line_words),
                                        "top": top,
                                        "x1": max(w["x1"] for w in line_words),
                                        "bottom": max(w["bottom"] for w in line_words),
                                    },
                                }
                            )

                    page_blocks.sort(key=lambda b: (b.get("bbox") or {}).get("top", 0))
                    blocks.extend(page_blocks)
        except Exception as exc:
            logger.warning("pdfplumber thất bại: %s", exc)
            return []

        if any(b["type"] == "paragraph" for b in blocks):
            return blocks

        # Fallback: PyMuPDF (nhanh hơn, chịu font/encoding lạ tốt hơn)
        pymupdf_blocks = self._extract_with_pymupdf_blocks(file_bytes)
        if pymupdf_blocks:
            return pymupdf_blocks + [b for b in blocks if b["type"] == "table"]
        return blocks

    def _extract_with_pymupdf_blocks(self, file_bytes: bytes) -> list[dict]:
        try:
            import fitz
        except ImportError:
            return []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            out = []
            for page_index in range(len(doc)):
                text = doc[page_index].get_text("text").strip()
                if text:
                    out.append(
                        {"type": "paragraph", "content": text, "page": page_index + 1}
                    )
            doc.close()
            return out
        except Exception:
            return []

    def _extract_with_pymupdf(self, file_bytes: bytes) -> str:
        blocks = self._extract_with_pymupdf_blocks(file_bytes)
        return "\n\n".join(b["content"] for b in blocks)

    @staticmethod
    def _flatten_hierarchy(node: dict, parts: list[str]) -> None:
        if not isinstance(node, dict):
            return
        text = node.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
        for child in node.get("children") or []:
            BackendDataExtractor._flatten_hierarchy(child, parts)

    async def extract_scan_text(
        self, file_bytes: bytes, filename: str = "document.pdf"
    ) -> str:
        primary_exc: BackendDataParseError | None = None
        fallback_exc: BaseException | None = None
        data = None
        try:
            data = await self._post_file(
                self._text_endpoint, file_bytes, filename, "application/pdf"
            )
        except BackendDataParseError as exc:
            primary_exc = exc
        parts: list[str] = []
        text = ""
        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, dict) and content:
                self._flatten_hierarchy(content, parts)
                text = "\n\n".join(parts).strip()
            if not text:
                text = (data.get("text") or "").strip()
        # Fallback Vinter: primary lỗi, hoặc primary trả rỗng (khi fallback_on_empty)
        if (
            not text
            and self._fallback_enabled("text")
            and (primary_exc is not None or self._fallback_on_empty)
        ):
            self._log_fallback(
                "text",
                "primary lỗi" if primary_exc is not None else "primary trả rỗng",
                primary_exc,
            )
            fallback_exc = None
            try:
                text = (
                    await self._fallback.ocr_text(file_bytes, filename=filename)
                ).strip()
            except Exception as exc:  # nuốt mọi lỗi fallback để còn lớp PyMuPDF
                fallback_exc = exc
                logger.warning("OCR text: fallback Vinter cũng lỗi: %s", exc)
        if text:
            return text
        # Lớp cuối: PyMuPDF local (giữ như cũ)
        pymupdf_text = self._extract_with_pymupdf(file_bytes)
        if pymupdf_text.strip():
            return pymupdf_text
        if primary_exc is not None:
            if fallback_exc is not None:
                raise BackendDataParseError(
                    f"OCR service chính lỗi: {primary_exc}; OCR fallback Vinter cũng lỗi: {fallback_exc}"
                ) from primary_exc
            raise primary_exc
        raise BackendDataParseError(
            "OCR service không trích xuất được nội dung từ PDF scan"
        )

    # ------------------------------------------------------------------
    # Bảng: Qwen table endpoint
    # ------------------------------------------------------------------
    async def extract_tables(
        self, file_bytes: bytes, filename: str = "document.pdf"
    ) -> list[dict]:
        primary_exc: BackendDataParseError | None = None
        tables: list[dict] = []
        try:
            data = await self._post_file(
                self._table_endpoint, file_bytes, filename, "application/pdf"
            )
            tables = self._parse_table_response(data)
        except BackendDataParseError as exc:
            primary_exc = exc
        if tables:
            return tables
        # Fallback Vinter: primary lỗi, hoặc primary trả rỗng (khi fallback_on_empty)
        if self._fallback_enabled("table") and (
            primary_exc is not None or self._fallback_on_empty
        ):
            self._log_fallback(
                "table",
                "primary lỗi" if primary_exc is not None else "primary trả rỗng",
                primary_exc,
            )
            try:
                return await self._fallback.ocr_tables(file_bytes, filename=filename)
            except Exception as exc:  # bảng có html pdfplumber làm lưới an toàn
                logger.warning(
                    "OCR table: fallback Vinter cũng lỗi, dùng html pdfplumber: %s", exc
                )
                return []
        # Không có fallback (tắt/không configured): giữ hành vi cũ
        if primary_exc is not None:
            raise primary_exc
        return []

    def _parse_table_response(self, data) -> list[dict]:
        tables = []
        if isinstance(data, str):
            tables.append({"html": data, "page": None, "raw_response": data})
        elif isinstance(data, dict):
            if "tables" in data and isinstance(data["tables"], list):
                for t in data["tables"]:
                    tables.append(
                        {
                            "html": t.get("html", ""),
                            "page": t.get("page"),
                            "raw_response": t,
                        }
                    )
            elif "html" in data:
                tables.append(
                    {
                        "html": data.get("html", ""),
                        "page": data.get("page"),
                        "raw_response": data,
                    }
                )
            else:
                tables.append(
                    {
                        "html": json.dumps(data, ensure_ascii=False),
                        "page": None,
                        "raw_response": data,
                    }
                )
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "html" in item:
                    tables.append(
                        {
                            "html": item.get("html", ""),
                            "page": item.get("page"),
                            "raw_response": item,
                        }
                    )
                elif isinstance(item, str):
                    tables.append({"html": item, "page": None, "raw_response": item})
        return tables

    @staticmethod
    def _table_rows_to_html(rows: list[list]) -> str:
        if not rows:
            return "<table></table>"
        html_parts = ["<table>"]
        for i, row in enumerate(rows):
            html_parts.append("  <tr>")
            tag = "th" if i == 0 else "td"
            for cell in row:
                cell_text = str(cell or "").strip().replace("\n", "<br>")
                html_parts.append(f"    <{tag}>{cell_text}</{tag}>")
            html_parts.append("  </tr>")
        html_parts.append("</table>")
        return "\n".join(html_parts)

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------
    def ensure_docx_bytes(self, file_bytes: bytes) -> bytes:
        try:
            from docx import Document

            Document(io.BytesIO(file_bytes))
            return file_bytes
        except Exception:
            return self.convert_doc_to_docx(file_bytes)

    def convert_doc_to_docx(self, file_bytes: bytes) -> bytes:
        import os

        with tempfile.TemporaryDirectory() as workdir:
            input_path = os.path.join(workdir, "input.doc")
            output_path = os.path.join(workdir, "input.docx")
            with open(input_path, "wb") as f:
                f.write(file_bytes)
            try:
                result = subprocess.run(
                    [
                        "soffice",
                        "--headless",
                        "--norestore",
                        "--convert-to",
                        "docx",
                        "--outdir",
                        workdir,
                        input_path,
                    ],
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
            except FileNotFoundError as e:
                raise BackendDataParseError(
                    "LibreOffice (soffice) chưa được cài đặt"
                ) from e
            except subprocess.TimeoutExpired as e:
                raise BackendDataParseError(
                    "Convert .doc -> .docx quá thời gian chờ"
                ) from e
            if result.returncode != 0 or not os.path.exists(output_path):
                raise BackendDataParseError(
                    f"Convert .doc -> .docx thất bại: {result.stderr.decode(errors='ignore')[:300]}"
                )
            with open(output_path, "rb") as f:
                return f.read()

    def extract_docx_blocks(self, file_bytes: bytes) -> list[dict]:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(io.BytesIO(file_bytes))
        blocks: list[dict] = []
        for element in doc.element.body:
            if isinstance(element, CT_P):
                block = self._process_docx_paragraph(Paragraph(element, doc))
                if block:
                    blocks.append(block)
            elif isinstance(element, CT_Tbl):
                block = self._process_docx_table(Table(element, doc))
                if block:
                    blocks.append(block)
        return blocks

    def _process_docx_paragraph(self, paragraph) -> dict | None:
        text = paragraph.text.strip()
        if not text:
            return None
        style_name = paragraph.style.name.lower() if paragraph.style else ""
        if style_name.startswith("heading"):
            try:
                level = int(style_name.replace("heading", "").strip())
                return {
                    "type": "heading",
                    "content": f"{'#' * level} {text}",
                    "level": level,
                }
            except ValueError:
                pass
        if paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None:
            return {"type": "list", "content": f"- {text}", "list_type": "bullet"}
        return {"type": "paragraph", "content": text}

    def _process_docx_table(self, table) -> dict | None:
        if not table.rows:
            return None
        markdown_rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
            markdown_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                markdown_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return {
            "type": "table",
            "content": "\n".join(markdown_rows),
            "html": self._docx_table_to_html(table),
        }

    @staticmethod
    def _docx_table_to_html(table) -> str:
        html_parts = ["<table>"]
        for i, row in enumerate(table.rows):
            html_parts.append("  <tr>")
            tag = "th" if i == 0 else "td"
            for cell in row.cells:
                cell_text = cell.text.strip().replace("\n", "<br>")
                html_parts.append(f"    <{tag}>{cell_text}</{tag}>")
            html_parts.append("  </tr>")
        html_parts.append("</table>")
        return "\n".join(html_parts)


def create_extractor(settings) -> BackendDataExtractor:
    # Lazy import: raganything không phụ thuộc ami_rag ở module level
    from ami_rag.core.ocr_client import build_vinter_ocr_client
    from ami_rag.settings import resolve_ocr_fallback_modes

    return BackendDataExtractor(
        base_url=getattr(settings, "OCR_SERVICE_URL", "") or DEFAULT_BASE_URL,
        timeout=float(getattr(settings, "OCR_SERVICE_TIMEOUT", 300) or 300),
        table_endpoint=getattr(settings, "OCR_TABLE_ENDPOINT", "")
        or DEFAULT_TABLE_ENDPOINT,
        text_endpoint=getattr(settings, "OCR_TEXT_ENDPOINT", "")
        or DEFAULT_TEXT_ENDPOINT,
        fallback=build_vinter_ocr_client(settings),
        fallback_modes=resolve_ocr_fallback_modes(settings),
        fallback_on_empty=bool(getattr(settings, "OCR_FALLBACK_ON_EMPTY", True)),
    )


# ----------------------------------------------------------------------
# Ghép content_list
# ----------------------------------------------------------------------
def _page_of(block: dict, default: int = 1) -> int:
    page = block.get("page")
    try:
        return int(page) if page is not None else default
    except (TypeError, ValueError):
        return default


def _group_qwen_tables(qwen_tables: list[dict], valid_pages: set | None = None) -> dict:
    """Nhóm bảng Qwen theo page (normalize về 1-based). Chọn offset (0 = endpoint
    trả 1-based, 1 = 0-based) sao cho nhiều bảng nhất rơi vào trang có trong blocks."""
    best_groups: dict = {}
    best_hit = -1
    total = len(qwen_tables or [])
    for offset in (0, 1):
        groups: dict[Any, list[dict]] = {}
        for t in qwen_tables or []:
            p = t.get("page")
            if p is None or isinstance(p, bool):
                groups.setdefault(None, []).append(t)
                continue
            try:
                p = int(float(p)) + offset
            except (TypeError, ValueError):
                p = None
            groups.setdefault(p, []).append(t)
        hit = sum(
            len(v)
            for k, v in groups.items()
            if isinstance(k, int) and (valid_pages is None or k in valid_pages)
        )
        if hit > best_hit:
            best_groups, best_hit = groups, hit
        if best_hit == total and best_hit > 0:
            break
    return best_groups


def build_pdf_content_blocks(blocks: list[dict], qwen_tables: list[dict]) -> list[dict]:
    """Interleave text/table theo thứ tự đọc (bbox top từng trang), thay html bảng
    bằng html Qwen khi khớp (theo page + thứ tự); thiếu thì dùng html pdfplumber."""
    pages: dict[int, list[dict]] = {}
    for b in blocks:
        pages.setdefault(_page_of(b), []).append(b)

    qwen_by_page = _group_qwen_tables(qwen_tables, valid_pages=set(pages))
    qwen_no_page: list[dict] = list(qwen_by_page.get(None, []))

    out: list[dict] = []
    for page in sorted(pages):
        page_blocks = pages[page]
        qwen_iter = iter(qwen_by_page.get(page, []))
        for b in page_blocks:
            page_idx = max(page - 1, 0)
            if b["type"] == "table":
                qwen = next(qwen_iter, None)
                html = (
                    (qwen or {}).get("html") or b.get("html") or b.get("content") or ""
                )
                out.append({"type": "table", "table_body": html, "page_idx": page_idx})
            else:
                out.append(
                    {"type": "text", "text": b.get("content", ""), "page_idx": page_idx}
                )
        for leftover in qwen_iter:
            if leftover.get("html"):
                out.append(
                    {
                        "type": "table",
                        "table_body": leftover["html"],
                        "page_idx": max(page - 1, 0),
                    }
                )

    for page in sorted(
        k for k in qwen_by_page if isinstance(k, int) and k not in pages
    ):
        for t in qwen_by_page[page]:
            if t.get("html"):
                out.append(
                    {
                        "type": "table",
                        "table_body": t["html"],
                        "page_idx": max(page - 1, 0),
                    }
                )

    for t in qwen_no_page:
        if t.get("html"):
            out.append({"type": "table", "table_body": t["html"], "page_idx": None})
    return out


def append_image_items(
    content_blocks: list[dict], image_items: list[dict]
) -> list[dict]:
    """Gộp ảnh MinerU vào content_list theo page: cùng trang thì ảnh xếp sau
    text/bảng; DOCX (page_idx None) thì toàn bộ ảnh xếp sau text/bảng."""
    block_pages: dict[Any, list[dict]] = {}
    for b in content_blocks:
        block_pages.setdefault(b.get("page_idx"), []).append(b)
    img_pages: dict[Any, list[dict]] = {}
    for it in image_items:
        img_pages.setdefault(it.get("page_idx"), []).append(it)

    def _key(p):
        return 0 if p is None else p + 1

    out: list[dict] = []
    for page in sorted(set(block_pages) | set(img_pages), key=_key):
        out.extend(block_pages.get(page, []))
        out.extend(img_pages.get(page, []))
    return out


async def rebuild_pdf_content_list(
    mineru_list: list[dict],
    file_bytes: bytes,
    extractor: BackendDataExtractor,
    filename: str = "document.pdf",
) -> list[dict]:
    image_items = [it for it in mineru_list or [] if it.get("type") == "image"]
    blocks = await asyncio.to_thread(extractor.extract_pdf_blocks, file_bytes)
    if not any(b["type"] == "paragraph" for b in blocks):
        try:
            ocr_text = (await extractor.extract_scan_text(file_bytes, filename)).strip()
        except Exception as exc:
            logger.warning("extract_scan_text thất bại: %s", exc)
            ocr_text = ""
        if ocr_text.strip():
            blocks = [{"type": "paragraph", "content": ocr_text, "page": 1}]
    if not blocks:
        raise BackendDataParseError("Không trích xuất được text/bảng từ PDF")

    try:
        qwen_tables = await extractor.extract_tables(file_bytes, filename)
    except Exception as exc:
        logger.warning("extract_tables (Qwen) lỗi, dùng html pdfplumber: %s", exc)
        qwen_tables = []

    content_blocks = build_pdf_content_blocks(blocks, qwen_tables)
    return append_image_items(content_blocks, image_items)


async def rebuild_docx_content_list(
    mineru_list: list[dict], file_bytes: bytes, extractor: BackendDataExtractor
) -> list[dict]:
    image_items = [it for it in mineru_list or [] if it.get("type") == "image"]
    docx_bytes = await asyncio.to_thread(extractor.ensure_docx_bytes, file_bytes)
    doc_blocks = await asyncio.to_thread(extractor.extract_docx_blocks, docx_bytes)
    if not doc_blocks:
        raise BackendDataParseError("Không trích xuất được nội dung DOCX")
    content_blocks: list[dict] = []
    for b in doc_blocks:
        if b["type"] == "table":
            content_blocks.append(
                {"type": "table", "table_body": b.get("html") or "", "page_idx": None}
            )
        else:
            content_blocks.append(
                {"type": "text", "text": b.get("content", ""), "page_idx": None}
            )
    return append_image_items(content_blocks, image_items)


async def rebuild_content_list(
    mineru_list: list[dict], file_path: str | Path, extractor: BackendDataExtractor
) -> list[dict] | None:
    """Trả về content_list mới (text/table kiểu backend_data + ảnh MinerU) cho
    .pdf/.docx/.doc; None nếu định dạng không áp dụng (giữ nguyên MinerU output)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    try:
        file_bytes = path.read_bytes()
    except Exception:
        return None
    if ext == ".pdf":
        return await rebuild_pdf_content_list(
            mineru_list, file_bytes, extractor, filename=path.name
        )
    if ext in (".docx", ".doc"):
        return await rebuild_docx_content_list(mineru_list, file_bytes, extractor)
    return None
