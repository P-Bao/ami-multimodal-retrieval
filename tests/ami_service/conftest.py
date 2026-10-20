from pathlib import Path

import pytest

from ami_rag.queue.events import RagEvent


@pytest.fixture(autouse=True)
def _isolate_dotenv_settings(monkeypatch):
    """`.env` của dev (vd. RERANK_FUSION=rrf) không được làm lệch default mà test giả định.

    Biến môi trường thắng dotenv trong pydantic-settings, nên ghim các giá trị
    test phụ thuộc; `get_settings` là lru_cache nên reset trước/sau mỗi test.
    """
    from ami_rag.settings import get_settings

    monkeypatch.setenv("RERANK_FUSION", "raw")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeDocsRepo:
    def __init__(self, docs=None):
        self.docs = docs or {}

    def find_by_id(self, document_id):
        return self.docs.get(document_id)

    def find_by_file_path(self, file_path):
        for doc in self.docs.values():
            if doc.get("file_path") == file_path:
                return doc
        return None

    def count(self):
        return len(self.docs)


class FakeAssetStore:
    """In-memory stand-in for MinioAssetStore (đủ API pipeline dùng)."""

    def __init__(self, fail_presign=False):
        self.calls = []
        self.fetched = []
        self.content_lists = {}
        self.json_objects = {}
        self.deleted_assets = []
        self.fail_presign = fail_presign

    def doc_prefix(self, doc_id: str) -> str:
        return f"rag-assets/{doc_id}/"

    def fetch(self, object_name, dest_dir):
        self.calls.append(("fetch", object_name))
        dest = Path(dest_dir) / Path(object_name).name
        dest.write_bytes(b"%PDF-fake")
        self.fetched.append(object_name)
        return dest

    def upload_content_list_assets(self, doc_id, content_list):
        self.calls.append(("upload_assets", doc_id))
        keys = []
        for i, item in enumerate(content_list):
            if item.get("type") == "image" and item.get("img_path"):
                key = f"rag-assets/{doc_id}/img{i}.png"
                item["asset_key"] = key
                keys.append(key)
        return keys

    def save_content_list(self, doc_id, content_list):
        self.calls.append(("save_content_list", doc_id))
        self.content_lists[doc_id] = [dict(i) for i in content_list]
        return f"rag-assets/{doc_id}/content_list.json"

    def load_content_list(self, doc_id):
        return self.content_lists.get(doc_id)

    def save_json(self, key, data):
        self.calls.append(("save_json", key))
        self.json_objects[key] = data
        return key

    def load_json(self, key):
        return self.json_objects.get(key)

    def presign(self, key):
        if not key or self.fail_presign:
            return None
        return f"https://minio/presigned/{key}"

    def get_bytes(self, key):
        if not key or key in self.deleted_assets:
            return None
        return b"\x89PNG-fake-bytes"

    def delete_doc_assets(self, doc_id):
        self.calls.append(("delete_assets", doc_id))
        self.deleted_assets.append(doc_id)
        self.content_lists.pop(doc_id, None)
        return 1


class FakeEmbedder:
    """Stand-in for RemoteEmbedder (0 mạng)."""

    def __init__(self, dim=8):
        self.dim = dim
        self.embedded_texts: list[list[str]] = []
        self.embedded_queries: list[str] = []

    async def verify(self) -> dict:
        return {"model_name": "Qwen/Qwen3-VL-Embedding-2B", "dim": self.dim}

    async def embed_documents(self, texts):
        self.embedded_texts.append(list(texts))
        return [[0.1] * self.dim for _ in texts]

    async def embed_query(self, text):
        self.embedded_queries.append(text)
        return [0.2] * self.dim

    @staticmethod
    def _sparse_from_text(text: str) -> dict:
        """Deterministic sparse: indices từ md5, values đều 0.5."""
        import hashlib

        digest = hashlib.md5(text.encode("utf-8")).digest()
        indices = sorted(int(x) % 100000 for x in digest)
        return {"indices": indices, "values": [0.5] * len(indices)}

    async def embed_documents_sparse(self, texts):
        self.embedded_texts.append(list(texts))
        return (
            [[0.1] * self.dim for _ in texts],
            [self._sparse_from_text(t) for t in texts],
        )

    async def embed_query_sparse(self, text):
        self.embedded_queries.append(text)
        return ([0.2] * self.dim, self._sparse_from_text(text))

    def count_cache_hits(self, texts):
        return 0

    async def close(self):
        pass


class FakeVectorStore:
    """Stand-in for VectorStore (in-memory)."""

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.collections: dict[str, int] = {}
        self.sparse_collections: set[str] = set()
        self.search_limits: list[int] = []

    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = True
    ) -> None:
        self.collections[name] = dim
        # fake không có khái niệm legacy: chỉ ghi nhận sparse đã bật
        if sparse:
            self.sparse_collections.add(name)

    async def upsert(self, collection, records):
        for r in records:
            self.points[r.id] = dict(r.payload)

    async def search(self, collection, vector, top_k, *, sparse=None):
        from ami_rag.core.vector_store import SearchHit

        self.search_limits.append(top_k)
        return [
            SearchHit(id=k, score=0.9, payload=dict(p))
            for k, p in list(self.points.items())[:top_k]
        ]

    async def delete_by_doc(self, collection, doc_id):
        n = 0
        for k in list(self.points):
            if self.points[k].get("doc_id") == doc_id:
                del self.points[k]
                n += 1
        return n

    async def count(self, collection, doc_id=None):
        if doc_id is None:
            return len(self.points)
        return sum(1 for p in self.points.values() if p.get("doc_id") == doc_id)


class FakeModalProcessor:
    """Stand-in cho modal processor (describe stage - trả mô tả giả)."""

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    async def generate_chunk_sections(
        self, modal_content, content_type, item_info=None, entity_name=None
    ):
        self.calls.append((content_type, dict(modal_content)))
        return [
            {
                "description": f"mô tả {content_type}",
                "entity_info": {
                    "entity_name": "x",
                    "entity_type": content_type,
                    "summary": "x",
                },
                "window_meta": None,
            }
        ]


class FakeParser:
    """Stand-in cho raganything.parser.Parser (sync parse_document)."""

    def __init__(self, content_list=None):
        self.content_list = content_list
        self.calls = []

    def parse_document(self, file_path, method="auto", output_dir=None, **kwargs):
        self.calls.append(("parse", str(file_path), method))
        if self.content_list is not None:
            return [dict(i) for i in self.content_list]
        img = Path(output_dir) / "img.png"
        img.write_bytes(b"\x89PNG-fake")
        return [
            {"type": "text", "text": "Giới thiệu", "page_idx": 0},
            {
                "type": "image",
                "img_path": str(img),
                "image_caption": ["Sơ đồ"],
                "page_idx": 1,
            },
            {
                "type": "table",
                "table_body": "| a | b |\n|---|---|\n| 1 | 2 |",
                "page_idx": 2,
            },
        ]


class FakeImageWorker:
    """Stand-in cho CrawlImageWorker (tải ảnh crawl, không mạng)."""

    def __init__(self, fail_urls=()):
        self.calls = []
        self.fail_urls = set(fail_urls)

    async def ensure_in_minio(self, doc_id, url):
        self.calls.append((doc_id, url))
        if url in self.fail_urls:
            return None
        return f"rag-assets/{doc_id}/img{len(self.calls)}.jpg"


class FakeQueue:
    def __init__(self):
        self.messages = []
        self.acked = []
        self.published = []
        self._consumer = "fake-consumer"

    async def ensure_group(self):
        pass

    async def read_batch(self, count=10, block_ms=5000):
        batch = self.messages[:count]
        self.messages = self.messages[count:]
        return batch

    async def ack(self, message_id):
        self.acked.append(message_id)

    async def publish(self, event: RagEvent):
        self.published.append(event)
        return "1-1"

    async def pending_count(self):
        return len(self.messages)


class FakeRerank:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def __call__(self, query, documents, top_n=None, **kwargs):
        self.calls.append((query, len(documents), top_n))
        if self.fail:
            raise ConnectionError("rerank down")
        n = min(top_n or len(documents), len(documents))
        return [
            {"index": i, "relevance_score": round(0.9 - i * 0.1, 2)} for i in range(n)
        ]


@pytest.fixture
def fake_docs_repo():
    return FakeDocsRepo(
        docs={
            "64b000000000000000000001": {
                "_id": "64b000000000000000000001",
                "title": "Doc A",
                "content": "nội dung tài liệu A",
                "content_hash": "hash-a",
                "document_type": "pdf",
                "original_file_name": "x.pdf",
                "file_path": "documents/HV/a.pdf",
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000005": {
                "_id": "64b000000000000000000005",
                "title": "Text E",
                "content": "nội dung text E",
                "content_hash": "hash-e",
                "document_type": "text",
                "original_file_name": None,
                "file_path": None,
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000006": {
                "_id": "64b000000000000000000006",
                "title": "Txt F",
                "content": "nội dung file txt F",
                "content_hash": "hash-f",
                "document_type": "text",
                "original_file_name": "a.txt",
                "file_path": "documents/HV/a.txt",
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {},
            },
            "64b000000000000000000007": {
                "_id": "64b000000000000000000007",
                "title": "Crawl G",
                "content": "nội dung bài crawl G",
                "content_hash": "hash-g",
                "document_type": "news",
                "original_file_name": None,
                "file_path": None,
                "organization_unit_id": "64b000000000000000000002",
                "metadata": {"source_url": "https://ptit.edu.vn/tin/crawl-g"},
            },
        }
    )


@pytest.fixture
def fake_asset_store():
    return FakeAssetStore()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_vector_store():
    return FakeVectorStore()


@pytest.fixture
def fake_status_store():
    from ami_rag.storage.doc_status import DocStatusStore
    from tests.ami_service.fakes import FakeMongoCollection

    store = DocStatusStore.__new__(DocStatusStore)
    store._col = FakeMongoCollection()
    return store


@pytest.fixture
def fake_modal_processors():
    calls = []

    class _P(FakeModalProcessor):
        async def generate_chunk_sections(
            self, modal_content, content_type, item_info=None, entity_name=None
        ):
            calls.append((content_type, dict(modal_content)))
            return [
                {
                    "description": f"mô tả {content_type}",
                    "entity_info": {},
                    "window_meta": None,
                }
            ]

    procs = {
        t: _P(calls)
        for t in ("image", "table", "equation", "audio", "video", "generic")
    }
    return procs


@pytest.fixture
def fake_queue():
    return FakeQueue()


@pytest.fixture
def fake_pipeline(
    fake_embedder,
    fake_vector_store,
    fake_asset_store,
    fake_status_store,
    fake_docs_repo,
):
    from ami_rag.core.vector_pipeline import VectorPipeline
    from ami_rag.settings import Settings

    settings = Settings(
        _env_file=None,
        WORKSPACE="test",
        EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        EMBED_DIM=8,
        CHUNKER_VERSION="v1",
        CHUNK_SIZE=64,
        CHUNK_OVERLAP=8,
        PARSER="mineru",
        PARSE_METHOD="auto",
        MINERU_BACKEND="pipeline",
        PARSE_TEXT_SOURCE="mineru",
    )
    return VectorPipeline(
        settings,
        embedder=fake_embedder,
        vector_store=fake_vector_store,
        asset_store=fake_asset_store,
        store=fake_status_store,
        docs_repo=fake_docs_repo,
    )
