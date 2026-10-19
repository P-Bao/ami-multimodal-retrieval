from collections.abc import Iterator

from bson import ObjectId
from pymongo import MongoClient


class MongoDocumentRepo:
    """Read-only access to the backend docs Mongo collection (organization_db.documents).

    The RAG worker never writes to this collection; document lifecycle writes
    belong to the backend docs service alone.
    """

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str = "documents"):
        self._client: MongoClient = MongoClient(mongo_uri, tz_aware=True)
        self._col = self._client[db_name][collection_name]

    def find_by_id(self, document_id: str) -> dict | None:
        if not ObjectId.is_valid(document_id):
            return None
        return self._col.find_one({"_id": ObjectId(document_id)})

    def find_by_file_path(self, file_path: str) -> dict | None:
        return self._col.find_one({"file_path": file_path})

    def find_by_source_url(self, source_url: str) -> dict | None:
        return self._col.find_one({"metadata.source_url": source_url})

    def iter_all(
        self, batch_size: int = 100, document_types: list[str] | None = None
    ) -> Iterator[dict]:
        query: dict = {"status": "active"}
        if document_types:
            query["document_type"] = {"$in": document_types}
        cursor = self._col.find(query, batch_size=batch_size)
        yield from cursor

    def count(self) -> int:
        return self._col.count_documents({"status": "active"})

    def close(self) -> None:
        self._client.close()
