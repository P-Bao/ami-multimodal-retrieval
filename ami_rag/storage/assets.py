import hashlib
import json
import mimetypes
from datetime import timedelta
from io import BytesIO
from pathlib import Path

ASSET_TYPES = ("image", "table", "equation")
CONTENT_LIST_NAME = "content_list.json"

_MINIO_POOL_MAXSIZE = 16


class MinioAssetStore:
    """MinIO access for the RAG pipeline (single bucket, `ASSET_PREFIX/{doc_id}/...`).

    - fetch(): download the original upload (pdf/docx) for parsing.
    - upload_content_list_assets(): store extracted images (incl. table/equation
      renders) and stamp each item with `asset_key` so chunks keep a durable reference
      instead of a temp local path.
    - save/load_content_list(): the parsed content_list kept as JSON, served by
      /admin/documents/{id}/content.
    - presign(): short-lived GET URL for `/v2/rag` responses.
    - delete_doc_assets(): purge everything under the document prefix.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str = "rag-assets",
        secure: bool = False,
        presign_expires: int = 3600,
        client=None,
    ):
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._secure = secure
        self._presign_expires = presign_expires
        self._client = client

    def _get_client(self):
        if self._client is None:
            import urllib3
            from minio import Minio

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=3, read=60),
                retries=urllib3.Retry(total=1, backoff_factor=0),
                maxsize=_MINIO_POOL_MAXSIZE,
            )
            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
                http_client=http_client,
            )
        return self._client

    def doc_prefix(self, doc_id: str) -> str:
        return f"{self._prefix}/{doc_id}/"

    def fetch(self, object_name: str, dest_dir: Path) -> Path:
        """Download an object (original upload) into dest_dir, keeping its extension."""
        dest = Path(dest_dir) / Path(object_name).name
        self._get_client().fget_object(self._bucket, object_name, str(dest))
        return dest

    def upload_content_list_assets(self, doc_id: str, content_list: list[dict]) -> list[str]:
        """Upload local image files referenced by `img_path`; set `item["asset_key"]`.

        Returns the uploaded object keys. Items without a readable `img_path` are left
        untouched (text-only table/equation).
        """
        client = self._get_client()
        keys: list[str] = []
        for item in content_list:
            if item.get("type") not in ASSET_TYPES:
                continue
            img_path = item.get("img_path")
            if not img_path:
                continue
            path = Path(img_path)
            if not path.is_file():
                continue
            data = path.read_bytes()
            ext = path.suffix.lower() or ".png"
            key = f"{self.doc_prefix(doc_id)}{hashlib.sha256(data).hexdigest()[:16]}{ext}"
            if key not in keys:
                client.put_object(
                    self._bucket,
                    key,
                    BytesIO(data),
                    length=len(data),
                    content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                keys.append(key)
            item["asset_key"] = key
        return keys

    def save_content_list(self, doc_id: str, content_list: list[dict]) -> str:
        key = f"{self.doc_prefix(doc_id)}{CONTENT_LIST_NAME}"
        data = json.dumps(content_list, ensure_ascii=False, default=str).encode("utf-8")
        self._get_client().put_object(
            self._bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
        return key

    def load_content_list(self, doc_id: str) -> list[dict] | None:
        key = f"{self.doc_prefix(doc_id)}{CONTENT_LIST_NAME}"
        try:
            response = self._get_client().get_object(self._bucket, key)
        except Exception:
            return None
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def save_json(self, key: str, data) -> None:
        raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self._get_client().put_object(
            self._bucket,
            key,
            BytesIO(raw),
            length=len(raw),
            content_type="application/json",
        )

    def load_json(self, key: str):
        try:
            response = self._get_client().get_object(self._bucket, key)
        except Exception:
            return None
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def presign(self, key: str | None) -> str | None:
        if not key:
            return None
        try:
            return self._get_client().presigned_get_object(
                self._bucket, key, expires=timedelta(seconds=self._presign_expires)
            )
        except Exception:
            return None

    def get_bytes(self, key: str | None) -> bytes | None:
        """Đọc raw bytes của một asset từ MinIO."""
        if not key:
            return None
        try:
            response = self._get_client().get_object(self._bucket, key)
        except Exception:
            return None
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_doc_assets(self, doc_id: str) -> int:
        client = self._get_client()
        removed = 0
        for obj in client.list_objects(
            self._bucket, prefix=self.doc_prefix(doc_id), recursive=True
        ):
            client.remove_object(self._bucket, obj.object_name)
            removed += 1
        return removed

