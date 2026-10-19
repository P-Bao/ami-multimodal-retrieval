"""Fake collection (in-memory) cho DocStatusStore tests - không cần Mongo."""

from pymongo.results import DeleteResult, UpdateResult


class FakeMongoCollection:
    """Đủ API pymongo mà DocStatusStore dùng: find/find_one/update_one/
    update_many/aggregate/delete_one/create_index."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    def create_index(self, *_args, **_kwargs):
        return "idx"

    def insert(self, doc_id: str, doc: dict):
        """Seed trực tiếp một document (không qua update)."""
        self._docs[doc_id] = {"_id": doc_id, **doc}

    def find_one(self, query: dict):
        return self._docs.get(query.get("_id"))

    def find(self, query: dict | None = None, projection: dict | None = None):
        docs = list(self._docs.values())
        status_q = (query or {}).get("status")
        if isinstance(status_q, dict) and "$in" in status_q:
            docs = [d for d in docs if d.get("status") in status_q["$in"]]
        elif status_q:
            docs = [d for d in docs if d.get("status") == status_q]
        return list(docs)

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        doc_id = query.get("_id")
        existing = self._docs.get(doc_id)
        if existing is None:
            if not upsert:
                return UpdateResult({"n": 0, "nModified": 0}, False)
            self._docs[doc_id] = {"_id": doc_id, **update.get("$setOnInsert", {})}
            existing = self._docs[doc_id]
            existing.update(update.get("$set", {}))
            for key, delta in update.get("$inc", {}).items():
                existing[key] = existing.get(key, 0) + delta
            return UpdateResult({"n": 1, "nModified": 1}, True)
        existing.update(update.get("$set", {}))
        for key, delta in update.get("$inc", {}).items():
            existing[key] = existing.get(key, 0) + delta
        return UpdateResult({"n": 1, "nModified": 1}, True)

    def update_many(self, query: dict, update: dict):
        id_q = query.get("_id")
        status_q = query.get("status")
        matched = 0
        for doc_id, doc in self._docs.items():
            if isinstance(id_q, dict) and "$in" in id_q:
                if doc_id not in id_q["$in"]:
                    continue
            elif id_q is not None and doc_id != id_q:
                continue
            if isinstance(status_q, dict) and "$in" in status_q:
                if doc.get("status") not in status_q["$in"]:
                    continue
            elif status_q and doc.get("status") != status_q:
                continue
            doc.update(update.get("$set", {}))
            matched += 1
        return UpdateResult({"n": matched, "nModified": matched}, True)

    def delete_one(self, query: dict):
        doc_id = query.get("_id")
        if doc_id in self._docs:
            self._docs.pop(doc_id)
            return DeleteResult(1, True)
        return DeleteResult(0, False)

    def aggregate(self, pipeline: list[dict]):
        counts: dict[str, int] = {}
        for doc in self._docs.values():
            key = doc.get("status") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return [{"_id": status, "count": count} for status, count in counts.items()]
