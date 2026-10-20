"""Test split_image_section + CrawlImageWorker.collect_image_urls (không mạng)."""

from unittest.mock import MagicMock, patch

import httpx

from ami_rag.workers.crawl_images import (
    CrawlImageWorker,
    split_image_section,
)

PTIT_CONTENT = (
    "PTIT tổ chức lễ tốt nghiệp\n\n"
    "Nội dung bài viết chính.\n\n"
    "**Hình ảnh:**\n\n"
    "- https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04608-scaled.jpg\n"
    "- https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04739-scaled.jpg\n"
    "- https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04485-scaled.jpg\n"
)

PTIT_URLS = [
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04608-scaled.jpg",
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04739-scaled.jpg",
    "https://ptit.edu.vn/wp-content/uploads/2026/05/NL_04485-scaled.jpg",
]


def test_split_example_ptit():
    urls, text = split_image_section(PTIT_CONTENT)

    assert urls == PTIT_URLS
    assert "Hình ảnh" not in text
    assert "NL_04608" not in text
    assert "PTIT tổ chức lễ tốt nghiệp" in text
    assert "Nội dung bài viết chính." in text


def test_split_heading_variants():
    for heading in (
        "**Hình ảnh:**",
        "## Hình ảnh",
        "HÌNH ẢNH:",
        "Hinh anh:",
        "> *Hình ảnh*",
    ):
        text = f"Intro\n\n{heading}\n\n- https://x.com/a.png\n"
        urls, stripped = split_image_section(text)
        assert urls == ["https://x.com/a.png"], heading
        assert "a.png" not in stripped, heading


def test_split_markdown_image_and_query_string():
    text = (
        "Intro\n\nHình ảnh:\n\n"
        "- ![alt](https://x.com/b.webp)\n"
        "- https://x.com/c.jpg?resize=1024\n"
        "https://x.com/d.png\n"
    )
    urls, _ = split_image_section(text)
    assert urls == [
        "https://x.com/b.webp",
        "https://x.com/c.jpg?resize=1024",
        "https://x.com/d.png",
    ]


def test_split_dedupe_keeps_order():
    text = "S\n\nHình ảnh:\n\n- https://x.com/a.jpg\n- https://x.com/a.jpg\n- https://x.com/b.jpg\n"
    urls, _ = split_image_section(text)
    assert urls == ["https://x.com/a.jpg", "https://x.com/b.jpg"]


def test_split_stops_at_non_image_url_and_keeps_it():
    text = (
        "S\n\nHình ảnh:\n\n"
        "- https://x.com/a.png\n"
        "- https://ptit.edu.vn/tin-tuc/khong-phai-anh\n\n"
        "Kết luận."
    )
    urls, out = split_image_section(text)
    assert urls == ["https://x.com/a.png"]
    assert "- https://ptit.edu.vn/tin-tuc/khong-phai-anh" in out
    assert "Kết luận." in out
    assert "a.png" not in out


def test_split_non_image_url_before_any_image_invalidates_section():
    text = "S\n\nHình ảnh:\n\n- https://ptit.edu.vn/tin-tuc/x\n"
    urls, out = split_image_section(text)
    assert urls == []
    assert out == text


def test_split_no_section_returns_original():
    text = "Chỉ text thường, không có section ảnh."
    assert split_image_section(text) == ([], text)


def test_split_heading_without_urls_keeps_text():
    text = "Hình ảnh trường rất đẹp qua các năm.\n\nĐoạn văn bình thường."
    urls, out = split_image_section(text)
    assert urls == []
    assert out == text


def test_split_multiple_sections():
    text = (
        "A\n\nHình ảnh:\n\n- https://x.com/1.png\n\n"
        "Giữa hai section.\n\n"
        "**Hình ảnh:**\n\n- https://x.com/2.png\n"
    )
    urls, out = split_image_section(text)
    assert urls == ["https://x.com/1.png", "https://x.com/2.png"]
    assert "Giữa hai section." in out
    assert "1.png" not in out
    assert "2.png" not in out


def test_split_only_section_leaves_empty_text():
    urls, out = split_image_section("**Hình ảnh:**\n\n- https://x.com/a.jpg\n")
    assert urls == ["https://x.com/a.jpg"]
    assert out == ""


def test_split_empty_text():
    assert split_image_section("") == ([], "")
    assert split_image_section(None) == ([], None)


def test_collect_image_urls_metadata_markdown_and_bullets():
    worker = CrawlImageWorker(
        endpoint="localhost:9000",
        access_key="a",
        secret_key="s",
        bucket="b",
        prefix="rag-assets",
    )
    doc = {
        "metadata": {"images": [{"url": "https://x.com/m.png"}, "https://x.com/n.jpg"]},
        "content": (
            "![inline](https://x.com/inline.png)\n\n" "- https://x.com/bullet.jpg\n"
        ),
    }
    urls = worker.collect_image_urls(doc)
    assert urls == [
        "https://x.com/m.png",
        "https://x.com/n.jpg",
        "https://x.com/inline.png",
        "https://x.com/bullet.jpg",
    ]


def _make_resp():
    r = MagicMock()
    r.status_code = 200
    r.content = b"fake-image"
    r.headers = {"content-type": "image/jpeg"}
    r.raise_for_status.return_value = None
    return r


def test_download_ptit_disables_ssl():
    worker = CrawlImageWorker(
        endpoint="localhost:9000",
        access_key="a",
        secret_key="s",
        bucket="b",
        prefix="rag-assets",
    )
    with patch("ami_rag.workers.crawl_images.httpx.Client") as client_cls:
        ctx = client_cls.return_value.__enter__.return_value
        ctx.get.return_value = _make_resp()
        worker._download("https://ptit.edu.vn/wp-content/uploads/x.jpg")
        assert client_cls.call_args.kwargs["verify"] is False


def test_download_unknown_keeps_ssl():
    worker = CrawlImageWorker(
        endpoint="localhost:9000",
        access_key="a",
        secret_key="s",
        bucket="b",
        prefix="rag-assets",
    )
    with patch("ami_rag.workers.crawl_images.httpx.Client") as client_cls:
        ctx = client_cls.return_value.__enter__.return_value
        ctx.get.return_value = _make_resp()
        worker._download("https://example.com/x.jpg")
        assert client_cls.call_args.kwargs["verify"] is True


def test_download_ptithcm_disables_ssl():
    worker = CrawlImageWorker(
        endpoint="localhost:9000",
        access_key="a",
        secret_key="s",
        bucket="b",
        prefix="rag-assets",
    )
    with patch("ami_rag.workers.crawl_images.httpx.Client") as client_cls:
        ctx = client_cls.return_value.__enter__.return_value
        ctx.get.return_value = _make_resp()
        worker._download("https://ptithcm.edu.vn/x.jpg")
        assert client_cls.call_args.kwargs["verify"] is False


def test_trusted_subdomain_disables_ssl():
    from ami_rag.workers.crawl_images import _is_trusted_no_ssl_host

    assert _is_trusted_no_ssl_host("cdn.ptit.edu.vn") is True
    assert _is_trusted_no_ssl_host("images.ptithcm.edu.vn") is True


def test_untrusted_lookalike_keeps_ssl():
    from ami_rag.workers.crawl_images import _is_trusted_no_ssl_host

    assert _is_trusted_no_ssl_host("ptit.edu.vn.evil.com") is False
    assert _is_trusted_no_ssl_host("notptit.edu.vn") is False
    assert _is_trusted_no_ssl_host("evil-ptit.edu.vn") is False


def _crawl_fail_val(reason: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value(
            "multimodal_rag_ingest_crawl_image_failures_total", {"reason": reason}
        )
        or 0.0
    )


def _make_worker() -> CrawlImageWorker:
    return CrawlImageWorker(
        endpoint="localhost:9000",
        access_key="a",
        secret_key="s",
        bucket="b",
        prefix="rag-assets",
    )


async def test_ensure_in_minio_failure_increments_counter_connect():
    worker = _make_worker()
    before = _crawl_fail_val("connect")
    with (
        patch.object(worker, "_exists", return_value=False),
        patch.object(worker, "_download", side_effect=httpx.ConnectError("refused")),
    ):
        assert await worker.ensure_in_minio("doc1", "https://x.com/a.jpg") is None
    assert _crawl_fail_val("connect") == before + 1


async def test_ensure_in_minio_failure_increments_counter_redirect():
    worker = _make_worker()
    before = _crawl_fail_val("redirect")
    boom = httpx.TooManyRedirects("exceeded", request=None)
    with (
        patch.object(worker, "_exists", return_value=False),
        patch.object(worker, "_download", side_effect=boom),
    ):
        assert await worker.ensure_in_minio("doc1", "https://x.com/a.jpg") is None
    assert _crawl_fail_val("redirect") == before + 1


def test_classify_crawl_error_reasons():
    import ssl

    from minio.error import S3Error

    from ami_rag.workers.crawl_images import _classify_crawl_error

    request = httpx.Request("GET", "https://x.com/a.jpg")
    response = httpx.Response(404, request=request)
    assert (
        _classify_crawl_error(
            httpx.HTTPStatusError("404", request=request, response=response)
        )
        == "http"
    )
    assert (
        _classify_crawl_error(httpx.TooManyRedirects("loop", request=None))
        == "redirect"
    )
    assert _classify_crawl_error(httpx.ConnectError("refused")) == "connect"
    assert _classify_crawl_error(httpx.ConnectTimeout("timed out")) == "connect"
    assert _classify_crawl_error(ssl.SSLError("cert")) == "ssl"
    assert _classify_crawl_error(S3Error.__new__(S3Error)) == "minio"
    assert _classify_crawl_error(ValueError("other")) == "other"
