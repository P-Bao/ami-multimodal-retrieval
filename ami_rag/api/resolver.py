import asyncio
import re
from datetime import timedelta

from bson import ObjectId

from ami_rag.observability import PRESIGN_FAILURES_TOTAL

_OBJECTID_PREFIX_RE = re.compile(r"^([0-9a-f]{24})_")

_MINIO_HTTP_TIMEOUT_SECONDS = 3
_MINIO_POOL_MAXSIZE = 16


def _minio_http_client():
    import urllib3

    return urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=_MINIO_HTTP_TIMEOUT_SECONDS, read=_MINIO_HTTP_TIMEOUT_SECONDS
        ),
        retries=urllib3.Retry(total=1, backoff_factor=0),
        maxsize=_MINIO_POOL_MAXSIZE,
    )


class DocResolver:
    """Resolve LightRAG file_paths back to Mongo documents.

    LightRAG reference lists carry `file_path`, which the ingest worker set to
    the MinIO object key (uploads) or metadata.source_url (crawl). Resolution
    results (title, document_type, org, presigned MinIO URL, source URL) are
    cached per file_path for the process lifetime.
    """

    def __init__(
        self,
        docs_repo,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_bucket: str,
        minio_secure: bool = False,
        presign_expires: int = 3600,
    ):
        self.docs_repo = docs_repo
        self._minio_endpoint = minio_endpoint
        self._minio_access_key = minio_access_key
        self._minio_secret_key = minio_secret_key
        self._minio_bucket = minio_bucket
        self._minio_secure = minio_secure
        self._presign_expires = presign_expires
        self._cache: dict[str, dict | None] = {}
        self._minio_client = None

    def _find_doc(self, file_path: str) -> dict | None:
        match = _OBJECTID_PREFIX_RE.match(file_path)
        if match and ObjectId.is_valid(match.group(1)):
            doc = self.docs_repo.find_by_id(match.group(1))
            if doc:
                return doc
        doc = self.docs_repo.find_by_file_path(file_path)
        if doc:
            return doc
        return self.docs_repo.find_by_source_url(file_path)

    def _get_minio_client(self):
        if self._minio_client is None:
            from minio import Minio

            self._minio_client = Minio(
                self._minio_endpoint,
                access_key=self._minio_access_key,
                secret_key=self._minio_secret_key,
                secure=self._minio_secure,
                http_client=_minio_http_client(),
            )
        return self._minio_client

    def _presign(self, object_name: str) -> str | None:
        try:
            return self._get_minio_client().presigned_get_object(
                self._minio_bucket,
                object_name,
                expires=timedelta(seconds=self._presign_expires),
            )
        except Exception:
            return None

    def _build_payload(self, doc: dict, file_url: str | None) -> dict:
        return {
            "document_id": str(doc["_id"]),
            "title": doc.get("title"),
            "document_type": doc.get("document_type"),
            "organization_unit_id": str(doc["organization_unit_id"])
            if doc.get("organization_unit_id")
            else None,
            "file_url": file_url,
            "source_url": (doc.get("metadata") or {}).get("source_url"),
        }

    async def _presign_async(self, object_name: str) -> str | None:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._presign, object_name),
                timeout=_MINIO_HTTP_TIMEOUT_SECONDS + 2,
            )
        except Exception:
            PRESIGN_FAILURES_TOTAL.inc()
            return None

    async def resolve(self, file_path: str) -> dict | None:
        if not file_path:
            return None
        if file_path in self._cache:
            return self._cache[file_path]
        try:
            doc = await asyncio.to_thread(self._find_doc, file_path)
        except Exception:
            doc = None
        if doc is None:
            self._cache[file_path] = None
            return None
        file_url = None
        if doc.get("file_path"):
            file_url = await self._presign_async(doc["file_path"])
        payload = self._build_payload(doc, file_url)
        self._cache[file_path] = payload
        return payload

    async def resolve_doc_id(self, doc_id: str) -> dict | None:
        """Resolve a Mongo document id directly (vector chunks carry doc_id)."""
        if not doc_id:
            return None
        cache_key = f"id:{doc_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            doc = await asyncio.to_thread(self.docs_repo.find_by_id, doc_id)
        except Exception:
            doc = None
        if doc is None:
            self._cache[cache_key] = None
            return None
        file_url = None
        if doc.get("file_path"):
            file_url = await self._presign_async(doc["file_path"])
        payload = self._build_payload(doc, file_url)
        self._cache[cache_key] = payload
        return payload

    def invalidate(self, file_path: str) -> None:
        self._cache.pop(file_path, None)
