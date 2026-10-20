import asyncio
from datetime import timedelta

from ami_rag.observability import PRESIGN_FAILURES_TOTAL
from ami_rag.sources import is_crawl_doc
from ami_rag.storage.assets import minio_http_client

_MINIO_HTTP_TIMEOUT_SECONDS = 3


class DocResolver:
    """Resolve doc_id (vector chunks carry doc_id) về Mongo document.

    Payload trả về (title, document_type, org, presigned MinIO URL, source URL);
    field Mongo được cache theo doc_id trong vòng đời process, còn file_url
    presign lại mỗi call (URL chỉ sống ``presign_expires`` giây).
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

    def _get_minio_client(self):
        if self._minio_client is None:
            from minio import Minio

            self._minio_client = Minio(
                self._minio_endpoint,
                access_key=self._minio_access_key,
                secret_key=self._minio_secret_key,
                secure=self._minio_secure,
                http_client=minio_http_client(read_timeout=_MINIO_HTTP_TIMEOUT_SECONDS),
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

    async def _get_doc(self, doc_id: str) -> dict | None:
        if not doc_id:
            return None
        cache_key = f"id:{doc_id}"
        if cache_key not in self._cache:
            try:
                doc = await asyncio.to_thread(self.docs_repo.find_by_id, doc_id)
            except Exception:
                doc = None
            self._cache[cache_key] = doc
        return self._cache[cache_key]

    async def is_crawl_doc_id(self, doc_id: str) -> bool:
        """Doc crawl (Mongo text + source_url); đọc cache của ``resolve_doc_id``."""
        return is_crawl_doc(await self._get_doc(doc_id))

    async def resolve_doc_id(self, doc_id: str) -> dict | None:
        """Resolve a Mongo document id directly (vector chunks carry doc_id).

        Cache chỉ giữ field Mongo (ít đổi); file_url presigned chỉ sống
        ``presign_expires`` giây nên presign lại mỗi call - không cache URL.
        """
        doc = await self._get_doc(doc_id)
        if doc is None:
            return None
        file_url = None
        if doc.get("file_path"):
            file_url = await self._presign_async(doc["file_path"])
        return self._build_payload(doc, file_url)
