"""Tests OCR fallback Vinter: VinterOcrClient + inject vào BackendDataExtractor.

0 mạng: primary (OCR service 171.226) và fallback (gateway vintern-vllm :8090)
đều giả bằng ``httpx.MockTransport``. PDF test dựng bằng PyMuPDF (trang trắng,
không text layer) để lớp PyMuPDF local không can thiệp kết quả.
"""

import fitz
import httpx
import pytest

import ami_rag.core.ocr_client as ocr_client_module
from ami_rag.core.ocr_client import (
    OcrFallbackError,
    VinterOcrClient,
    build_vinter_ocr_client,
)
from ami_rag.settings import Settings, resolve_ocr_fallback_modes
from raganything.backend_data_extract import (
    BackendDataExtractor,
    BackendDataParseError,
    create_extractor,
)


def _blank_pdf() -> bytes:
    """PDF 1 trang trắng (không text layer) -> PyMuPDF extract ra rỗng."""
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


BLANK_PDF = _blank_pdf()


# ---------------------------------------------------------------------------
# Fake upstreams
# ---------------------------------------------------------------------------
def primary_error(status: int = 500):
    """OCR service chính giả: mọi endpoint đều trả lỗi HTTP."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"primary boom {status}")

    return handler


def primary_text(text: str):
    """hierarchy-parse giả: trả cây content với 1 node text (rỗng nếu ``text``="")."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"content": ({"text": text} if text else {}), "text": text}
        )

    return handler


def primary_tables(tables: list):
    """raw_miner_ocr/qwen giả: trả danh sách bảng (có thể rỗng)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tables": tables})

    return handler


VINTER_TABLE_HTML = "<table><tr><td>v</td></tr></table>"


class FakeVinter:
    """Gateway vintern-vllm giả: POST /v1/ocr (multipart), route theo ``mode``."""

    def __init__(
        self,
        *,
        text_body: dict | None = None,
        table_body: dict | None = None,
        status: int = 200,
    ):
        self.text_body = text_body or {
            "pages": 1,
            "text": "VINTER TEXT",
            "page_texts": [{"page": 1, "text": "VINTER TEXT"}],
        }
        self.table_body = table_body or {
            "pages": 1,
            "tables": [{"page": 1, "html": VINTER_TABLE_HTML, "markdown": "| v |"}],
        }
        self.status = status
        self.requests: list[bytes] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.content)
        body = request.content
        is_table = b'name="mode"' in body and b"\r\n\r\ntable\r\n" in body
        payload = self.table_body if is_table else self.text_body
        return httpx.Response(self.status, json=payload)


def make_extractor(primary_handler, vinter: FakeVinter | None = None, **kwargs):
    """Extractor với primary + fallback đều inject MockTransport (0 mạng).

    ``vinter=None`` -> không có fallback (như cũ). VinterOcrClient dùng
    ``retries=0`` để test không chờ backoff.
    """
    fallback_obj = None
    if vinter is not None:
        fallback_obj = VinterOcrClient(
            "http://vinter:8090",
            model="5CD-AI/Vintern-1B-v3_5",
            retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(vinter.handler)
            ),
        )
    return BackendDataExtractor(
        base_url="http://primary:12006",
        timeout=10,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)),
        fallback=fallback_obj,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# extract_scan_text (PLAN nhóm 1-3, 5)
# ---------------------------------------------------------------------------
async def test_scan_text_primary_error_uses_fallback():
    vinter = FakeVinter()
    extractor = make_extractor(primary_error(500), vinter)
    assert await extractor.extract_scan_text(BLANK_PDF) == "VINTER TEXT"
    assert len(vinter.requests) == 1
    # gửi đúng filename lên fallback
    assert b"document.pdf" in vinter.requests[0]


async def test_scan_text_primary_empty_fallback_on_empty_toggle():
    # on_empty=True (default): primary trả rỗng -> fallback
    vinter = FakeVinter()
    extractor = make_extractor(primary_text(""), vinter, fallback_on_empty=True)
    assert await extractor.extract_scan_text(BLANK_PDF) == "VINTER TEXT"
    assert len(vinter.requests) == 1

    # on_empty=False: primary trả rỗng -> KHÔNG gọi fallback, PyMuPDF rỗng -> raise như cũ
    vinter2 = FakeVinter()
    extractor2 = make_extractor(primary_text(""), vinter2, fallback_on_empty=False)
    with pytest.raises(BackendDataParseError, match="không trích xuất được"):
        await extractor2.extract_scan_text(BLANK_PDF)
    assert vinter2.requests == []


async def test_scan_text_primary_ok_skips_fallback():
    vinter = FakeVinter()
    extractor = make_extractor(primary_text("PRIMARY TEXT"), vinter)
    assert await extractor.extract_scan_text(BLANK_PDF) == "PRIMARY TEXT"
    assert vinter.requests == []


async def test_scan_text_primary_error_fallback_error_raises_both():
    vinter = FakeVinter(status=503)
    extractor = make_extractor(primary_error(500), vinter)
    with pytest.raises(BackendDataParseError, match="chính lỗi.*cũng lỗi"):
        await extractor.extract_scan_text(BLANK_PDF)
    assert len(vinter.requests) == 1


async def test_scan_text_primary_ok_empty_fallback_error_pymupdf_rescues():
    # primary rỗng, fallback lỗi, nhưng PyMuPDF vẫn trích được text -> trả text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "text trong pdf")
    pdf_with_text = doc.tobytes()
    doc.close()

    vinter = FakeVinter(status=500)
    extractor = make_extractor(primary_text(""), vinter)
    assert "text trong pdf" in await extractor.extract_scan_text(pdf_with_text)


# ---------------------------------------------------------------------------
# fallback_modes (PLAN nhóm 4)
# ---------------------------------------------------------------------------
async def test_fallback_modes_text_off_table_on():
    modes = frozenset({"table"})
    # text: primary lỗi -> KHÔNG fallback -> raise lỗi primary (PyMuPDF rỗng)
    vinter = FakeVinter()
    extractor = make_extractor(primary_error(500), vinter, fallback_modes=modes)
    with pytest.raises(BackendDataParseError, match="mã lỗi 500"):
        await extractor.extract_scan_text(BLANK_PDF)
    assert vinter.requests == []

    # table: primary lỗi -> fallback
    vinter2 = FakeVinter()
    extractor2 = make_extractor(primary_error(500), vinter2, fallback_modes=modes)
    tables = await extractor2.extract_tables(BLANK_PDF)
    assert tables == [{"page": 1, "html": VINTER_TABLE_HTML}]
    assert len(vinter2.requests) == 1


# ---------------------------------------------------------------------------
# extract_tables (PLAN nhóm 6)
# ---------------------------------------------------------------------------
async def test_tables_primary_empty_uses_fallback():
    vinter = FakeVinter()
    extractor = make_extractor(primary_tables([]), vinter)
    tables = await extractor.extract_tables(BLANK_PDF)
    assert tables == [{"page": 1, "html": VINTER_TABLE_HTML}]
    assert len(vinter.requests) == 1


async def test_tables_primary_ok_skips_fallback():
    vinter = FakeVinter()
    primary_tables_ok = [{"page": 1, "html": "<table><tr><td>q</td></tr></table>"}]
    extractor = make_extractor(primary_tables(primary_tables_ok), vinter)
    assert await extractor.extract_tables(BLANK_PDF) == [
        {
            "page": 1,
            "html": "<table><tr><td>q</td></tr></table>",
            "raw_response": primary_tables_ok[0],
        }
    ]
    assert vinter.requests == []


async def test_tables_both_error_returns_empty():
    vinter = FakeVinter(status=500)
    extractor = make_extractor(primary_error(500), vinter)
    assert await extractor.extract_tables(BLANK_PDF) == []


async def test_tables_fallback_disabled_raises_primary():
    # fallback tắt (None) -> giữ hành vi cũ: primary lỗi -> raise
    extractor = make_extractor(primary_error(500), None)
    with pytest.raises(BackendDataParseError, match="mã lỗi 500"):
        await extractor.extract_tables(BLANK_PDF)


# ---------------------------------------------------------------------------
# build_vinter_ocr_client + resolve_ocr_fallback_modes (PLAN nhóm 7-8)
# ---------------------------------------------------------------------------
def test_build_vinter_ocr_client_disabled_or_empty_url():
    assert build_vinter_ocr_client(Settings(OCR_FALLBACK_ENABLED=False)) is None
    assert build_vinter_ocr_client(Settings(OCR_FALLBACK_URL="")) is None
    assert build_vinter_ocr_client(Settings(OCR_FALLBACK_URL="   ")) is None

    client = build_vinter_ocr_client(
        Settings(
            OCR_FALLBACK_URL="http://may-b:8090",
            OCR_FALLBACK_TOKEN="secret",
            OCR_FALLBACK_RETRIES=2,
        )
    )
    assert isinstance(client, VinterOcrClient)
    assert client._base_url == "http://may-b:8090"
    assert client._token == "secret"
    assert client._retries == 2


def test_resolve_ocr_fallback_modes():
    assert resolve_ocr_fallback_modes(Settings()) == frozenset({"text", "table"})
    assert resolve_ocr_fallback_modes(Settings(OCR_FALLBACK_MODES="")) == frozenset()
    assert resolve_ocr_fallback_modes(
        Settings(OCR_FALLBACK_MODES=" table ")
    ) == frozenset({"table"})
    assert resolve_ocr_fallback_modes(
        Settings(OCR_FALLBACK_MODES="text;text")
    ) == frozenset({"text"})
    with pytest.raises(ValueError, match="bogus"):
        resolve_ocr_fallback_modes(Settings(OCR_FALLBACK_MODES="text,bogus"))


def test_create_extractor_injects_fallback():
    settings = Settings(OCR_FALLBACK_URL="http://may-b:8090")
    extractor = create_extractor(settings)
    assert isinstance(extractor._fallback, VinterOcrClient)
    assert extractor._fallback_modes == frozenset({"text", "table"})
    assert extractor._fallback_on_empty is True

    extractor_off = create_extractor(Settings(OCR_FALLBACK_ENABLED=False))
    assert extractor_off._fallback is None


# ---------------------------------------------------------------------------
# VinterOcrClient đơn lẻ (hợp nhất page_texts, chuẩn hoá tables, retry, health)
# ---------------------------------------------------------------------------
def _client(handler, retries: int = 0) -> VinterOcrClient:
    return VinterOcrClient(
        "http://vinter:8090",
        retries=retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_client_ocr_text_merges_page_texts():
    vinter = FakeVinter(
        text_body={
            "pages": 3,
            "text": "GỘP CẢ LÔ",
            "page_texts": [
                {"page": 1, "text": "P1"},
                {"page": 2, "text": " P2 "},
                {"page": 3, "text": ""},
            ],
        }
    )
    # giữ thứ tự trang do gateway trả (đã sort); trang rỗng bị bỏ, strip từng phần
    assert await _client(vinter.handler).ocr_text(BLANK_PDF) == "P1\n\nP2"


async def test_client_ocr_text_falls_back_to_text_field():
    vinter = FakeVinter(text_body={"pages": 1, "text": "TOÀN VĂN", "page_texts": []})
    assert await _client(vinter.handler).ocr_text(BLANK_PDF) == "TOÀN VĂN"


async def test_client_ocr_text_empty_raises():
    vinter = FakeVinter(text_body={"pages": 1, "text": "", "page_texts": []})
    with pytest.raises(OcrFallbackError, match="không trích xuất được text"):
        await _client(vinter.handler).ocr_text(BLANK_PDF)


async def test_client_ocr_tables_normalizes():
    vinter = FakeVinter(
        table_body={
            "pages": 2,
            "tables": [
                {"page": 1, "html": "<table>ok</table>", "markdown": "| ok |"},
                {"page": "2", "html": "   "},  # html rỗng -> bỏ
                {"html": "<table>x</table>"},  # page None
                {"page": "abc", "html": "<table>y</table>"},  # page lạ -> None
            ],
        }
    )
    tables = await _client(vinter.handler).ocr_tables(BLANK_PDF)
    assert tables == [
        {"page": 1, "html": "<table>ok</table>"},
        {"page": None, "html": "<table>x</table>"},
        {"page": None, "html": "<table>y</table>"},
    ]


async def test_client_retry_on_5xx_not_4xx(monkeypatch):
    monkeypatch.setattr(ocr_client_module, "_RETRY_BASE", 0.0)

    calls: list[int] = []

    def handler_5xx(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    with pytest.raises(OcrFallbackError, match="mã lỗi 503"):
        await _client(handler_5xx, retries=1).ocr_text(BLANK_PDF)
    assert len(calls) == 2  # 1 lần đầu + 1 retry

    calls.clear()

    def handler_4xx(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad mode")

    with pytest.raises(OcrFallbackError, match="mã lỗi 400"):
        await _client(handler_4xx, retries=3).ocr_text(BLANK_PDF)
    assert len(calls) == 1  # 4xx: raise ngay, không retry


async def test_client_transport_error_messages():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OcrFallbackError, match="unreachable"):
        await _client(boom).ocr_text(BLANK_PDF)

    def hang(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(OcrFallbackError, match="không phản hồi"):
        await _client(hang).ocr_text(BLANK_PDF)

    def syn_drop(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timeout", request=request)

    with pytest.raises(OcrFallbackError, match="Kết nối OCR fallback timeout"):
        await _client(syn_drop).ocr_text(BLANK_PDF)


async def test_client_health_never_raises():
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "model": "m", "vllm": True})

    assert (await _client(ok).health())["vllm"] is True

    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "vllm not up"})

    assert await _client(down).health() == {}

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _client(dead).health() == {}
