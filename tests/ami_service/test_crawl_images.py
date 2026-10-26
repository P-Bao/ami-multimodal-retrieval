"""Test split_image_section (không mạng)."""

from ami_rag.workers.crawl_images import split_image_section

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
