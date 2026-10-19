from ami_rag.sources import (
    SOURCE_MINIO_PARSE,
    SOURCE_MONGO_TEXT,
    resolve_file_path,
    select_source,
    source_hash,
)

DOC_ID = "64b0000000000000000000a1"


def test_select_source_empty_or_txt_uses_mongo_text():
    assert select_source({"original_file_name": None}) == SOURCE_MONGO_TEXT
    assert select_source({}) == SOURCE_MONGO_TEXT
    assert select_source({"original_file_name": "  "}) == SOURCE_MONGO_TEXT
    assert select_source({"original_file_name": "Quy-dinh.TXT"}) == SOURCE_MONGO_TEXT


def test_select_source_office_and_pdf_use_minio_parse():
    for name in ("a.pdf", "b.docx", "c.DOC", "so tay.PDF"):
        assert select_source({"original_file_name": name}) == SOURCE_MINIO_PARSE


def test_source_hash_mongo_text_ignores_whitespace_padding():
    a = {"original_file_name": None, "content": "xin chào"}
    b = {"original_file_name": None, "content": "  xin chào \n"}
    c = {"original_file_name": None, "content": "khác"}
    assert source_hash(a) == source_hash(b)
    assert source_hash(a) != source_hash(c)


def test_source_hash_minio_parse_depends_on_object_key_only():
    base = {"original_file_name": "a.pdf", "file_path": "documents/HV/x_a.pdf"}
    edited = {**base, "content": "admin edited", "content_hash": "different"}
    other_file = {**base, "file_path": "documents/HV/y_a.pdf"}
    assert source_hash(base) == source_hash(edited)
    assert source_hash(base) != source_hash(other_file)


def test_resolve_file_path_upload_crawl_and_fallback():
    upload = {"_id": DOC_ID, "file_path": "documents/HV/abc_file.pdf"}
    crawl = {"_id": DOC_ID, "metadata": {"source_url": "https://ptit.edu.vn/tin/thong-bao-1"}}
    text = {"_id": DOC_ID, "document_type": "text"}
    assert resolve_file_path(upload) == f"{DOC_ID}_abc_file.pdf"
    assert resolve_file_path(crawl) == f"{DOC_ID}_thong-bao-1"
    assert resolve_file_path(text) == f"{DOC_ID}_text"
    assert resolve_file_path({}) is None


def test_doc_link_fields_keep_mongo_types_for_lookup():
    from bson import ObjectId

    from ami_rag.sources import doc_link_fields

    oid, org = ObjectId(), ObjectId()
    fields = doc_link_fields(
        {
            "_id": oid,
            "organization_unit_id": org,
            "owner_id": "sub-123",
            "document_type": "pdf",
            "title": "Sổ tay",
            "content": "ignored",
        }
    )
    assert fields == {
        "document_oid": oid,
        "organization_unit_id": org,
        "owner_id": "sub-123",
        "document_type": "pdf",
        "title": "Sổ tay",
    }


def test_doc_link_fields_string_id_and_missing_values():
    from bson import ObjectId

    from ami_rag.sources import doc_link_fields

    assert doc_link_fields({"_id": DOC_ID}) == {"document_oid": ObjectId(DOC_ID)}
    assert doc_link_fields({"_id": "not-an-objectid", "title": None}) == {}
