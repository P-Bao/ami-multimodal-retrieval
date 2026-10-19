import hashlib
from pathlib import Path
from urllib.parse import urlparse

SOURCE_MONGO_TEXT = "mongo_text"
SOURCE_MINIO_PARSE = "minio_parse"


def select_source(doc: dict) -> str:
    """Decide where the document content comes from.

    - `original_file_name` empty or `.txt` (text, crawl, seed data): the text stored
      in Mongo `content` is the source of truth.
    - otherwise (docx/doc/pdf uploads): the original file in MinIO is re-parsed by
      the multimodal parser; Mongo text/tables are ignored.
    """
    name = (doc.get("original_file_name") or "").strip().lower()
    if not name or name.endswith(".txt"):
        return SOURCE_MONGO_TEXT
    return SOURCE_MINIO_PARSE


def source_hash(doc: dict, source: str | None = None) -> str:
    """Stable fingerprint of what the pipeline ingests for this document.

    mongo_text: sha256 of the stripped text (same recipe as the backend `content_hash`
    for text/crawl). minio_parse: MinIO object keys are unique per upload, so the key
    identifies the file; Mongo `content_hash` is deliberately ignored (admin edits of
    the stored text do not change what is parsed).
    """
    source = source or select_source(doc)
    if source == SOURCE_MINIO_PARSE:
        raw = f"minio|{doc.get('file_path') or ''}"
    else:
        raw = (doc.get("content") or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_basename(doc: dict) -> str:
    """Human-readable source basename: MinIO object basename (uuid-prefixed, unique),
    last URL path segment for crawl, else the document_type."""
    file_path = doc.get("file_path")
    if file_path:
        return Path(file_path).name
    source_url = (doc.get("metadata") or {}).get("source_url")
    if source_url:
        return Path(urlparse(source_url).path).name or "crawl"
    return doc.get("document_type") or "unknown"


def resolve_file_path(doc: dict) -> str | None:
    """LightRAG file_path for citation: "{doc_id}_{basename}".

    LightRAG canonicalizes file_path to its basename at write time, so the value must
    be unique at basename level; the 24-hex ObjectId prefix has no underscores, which
    lets the /v2/rag resolver recover the document id directly.
    """
    doc_id = str(doc.get("_id") or "")
    if not doc_id:
        return None
    return f"{doc_id}_{_source_basename(doc)}"


def doc_link_fields(doc: dict) -> dict:
    """Fields copied from the backend document into the RAG registry row.

    Values keep their Mongo types: `document_oid` (ObjectId) and `organization_unit_id`
    allow `$lookup` into `documents` / `organization_units` (same database), and
    `owner_id` (the OIDC identity: users.sub / preferred_username / email, not users._id)
    into `users`. Missing values are omitted.
    """
    from bson import ObjectId

    fields: dict = {}
    doc_id = doc.get("_id")
    if doc_id is not None and ObjectId.is_valid(str(doc_id)):
        fields["document_oid"] = doc_id if isinstance(doc_id, ObjectId) else ObjectId(str(doc_id))
    for key in ("organization_unit_id", "owner_id", "document_type", "title"):
        if doc.get(key) is not None:
            fields[key] = doc[key]
    return fields
