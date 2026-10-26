"""VectorStore interface và backend Qdrant mặc định.

Lớp persistence tách khỏi vector: kết quả parse + mô tả modal lưu riêng
(MinIO), chỉ chunk/vector mới vào đây. Metadata mỗi chunk (payload):
doc_id, page, type, source_path, chunk_id, chunker_version, embed_model,
embed_dim, modality/asset_key/table_body/caption.

Quy ước collection: {prefix}__{embed_model_slug}__{chunker_version}
(không trộn hai embed model trong một collection).
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Lỗi liên quan vector store (collection lệch cấu hình, mạng, v.v.)."""


def _point_estimate(p) -> int:
    """Ước lượng byte JSON của một point khi gửi Qdrant (payload + vector + overhead).

    Vector serialize dạng JSON float: ~32 bytes/chiều (số thập phân dài) —
    ước lượng thận trọng hơn thực tế để không vượt giới hạn request.
    Named vectors: đếm cả dense (kể cả multi-vector: tổng số float qua mọi
    token vector) và sparse (indices+values).
    """
    vector = p.vector
    if isinstance(vector, dict):
        dense = vector.get("dense") or []
        if dense and isinstance(dense[0], (list, tuple)):
            # multi-vector: list token vector -> đếm tổng số float
            vec_len = sum(len(t) for t in dense)
        else:
            vec_len = len(dense)
        sparse = vector.get("sparse")
        if sparse is not None:
            indices = getattr(sparse, "indices", None) or (
                sparse.get("indices") if isinstance(sparse, dict) else None
            )
            vec_len += 2 * len(indices or [])
    else:
        vec_len = len(vector) if isinstance(vector, (list, tuple)) else 0
    payload_bytes = len(json.dumps(p.payload or {}, ensure_ascii=False).encode("utf-8"))
    return payload_bytes + vec_len * 32 + 256


@dataclass
class ChunkRecord:
    """Một chunk (id + vector + payload) chuẩn bị upsert.

    ``vector``: dense single-vector (``list[float]``) hoặc multi-vector
    per-token (``list[list[float]]`` - embedder ``is_multivector``).
    """

    id: str  # chunk_id
    vector: list[float] | list[list[float]]
    payload: dict = field(default_factory=dict)
    # sparse đặt SAU payload để giữ tương thích positional ChunkRecord(id, vector, payload)
    sparse: dict | None = None  # {"indices": [int, ...], "values": [float, ...]}


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = True, multivector: bool = False
    ) -> None: ...
    async def upsert(self, collection: str, chunks: list[ChunkRecord]) -> None: ...
    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        *,
        sparse: dict | None = None,
    ) -> list[SearchHit]: ...
    async def delete_by_doc(self, collection: str, doc_id: str) -> int: ...
    async def count(self, collection: str, doc_id: str | None = None) -> int: ...


class QdrantVectorStore:
    """Backend mặc định: Qdrant (lệnh kế thừa từ RAG-Anything/AMI hiện có)."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        max_request_bytes: int = 16 * 1024 * 1024,
        *,
        client=None,
    ):
        from qdrant_client import QdrantClient

        self._url = url
        # cache: collection đã named-vectors ("dense"+"sparse") hay legacy vô danh
        self._named: dict[str, bool] = {}
        self._client = (
            client
            if client is not None
            else QdrantClient(url=url, api_key=api_key or None)
        )
        # Giới hạn byte ước lượng mỗi request upsert (server mặc định 32 MB);
        # batch được đóng gói theo byte VÀ số điểm để không vượt giới hạn.
        self.max_request_bytes = max_request_bytes

    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = True, multivector: bool = False
    ) -> None:
        """Tạo collection nếu thiếu; kiểm tra tương thích dim với collection có sẵn.

        multivector=True (embedder multi-vector, vd Vintern): collection mới dùng
        named vector "dense" (COSINE) kèm ``MultiVectorConfig(MAX_SIM)``, KHÔNG
        sparse; query multi-vector so khớp MaxSim từng token. Collection có sẵn
        thiếu multivector_config → VectorStoreError (không bổ sung được — phải
        tạo collection mới, reindex).

        sparse=True (hybrid single-vector): named vectors "dense" (COSINE)
        + "sparse"; collection legacy (1 vector vô danh) không thể bổ sung
        sparse → phải tạo collection mới (reindex).
        """
        from qdrant_client.http import models

        try:
            existing = await asyncio.to_thread(
                lambda: next(
                    (
                        c.name
                        for c in self._client.get_collections().collections
                        if c.name == name
                    ),
                    None,
                )
            )
            if existing is not None:
                info = await asyncio.to_thread(self._client.get_collection, name)
                vectors = info.config.params.vectors
                if isinstance(vectors, dict):
                    # named vectors ("dense" tối thiểu)
                    dense = vectors.get("dense")
                    size = dense.size if dense is not None else None
                    if size is not None and size != dim:
                        raise VectorStoreError(
                            f"collection '{name}' có dim={size} nhưng cấu hình là {dim}. "
                            "Đổi EMBED_DIM/EMBED_MODEL hoặc chuyển sang collection mới."
                        )
                    if multivector:
                        # thiếu multivector_config ở "dense" -> query MaxSim không chạy được
                        if (
                            dense is None
                            or getattr(dense, "multivector_config", None) is None
                        ):
                            raise VectorStoreError(
                                f"collection '{name}' thiếu multivector_config ở vector "
                                "'dense' — reindex cần tạo collection mới."
                            )
                    elif (
                        sparse
                        and (info.config.params.sparse_vectors or {}).get("sparse")
                        is None
                    ):
                        raise VectorStoreError(
                            f"collection '{name}' thiếu sparse vector 'sparse' "
                            "— reindex cần tạo collection mới."
                        )
                    self._named[name] = True
                    return
                # legacy: 1 vector vô danh
                if multivector or sparse:
                    reason = (
                        "multivector" if multivector else "sparse"
                    )
                    raise VectorStoreError(
                        f"collection '{name}' legacy (1 vector vô danh) nhưng cần "
                        f"{reason} — reindex cần tạo collection mới."
                    )
                size = vectors.size if vectors is not None else None
                if size is not None and size != dim:
                    raise VectorStoreError(
                        f"collection '{name}' có dim={size} nhưng cấu hình là {dim}. "
                        "Đổi EMBED_DIM/EMBED_MODEL hoặc chuyển sang collection mới."
                    )
                self._named[name] = False
                return
            if multivector:
                await asyncio.to_thread(
                    self._client.create_collection,
                    collection_name=name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=dim,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                        )
                    },
                )
                self._named[name] = True
            else:
                self._named[name] = sparse
                if sparse:
                    await asyncio.to_thread(
                        self._client.create_collection,
                        collection_name=name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=dim, distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={"sparse": models.SparseVectorParams()},
                    )
                else:
                    await asyncio.to_thread(
                        self._client.create_collection,
                        collection_name=name,
                        vectors_config=models.VectorParams(
                            size=dim, distance=models.Distance.COSINE
                        ),
                    )
            logger.info(
                "tạo Qdrant collection %s (dim=%s, multivector=%s, sparse=%s)",
                name,
                dim,
                multivector,
                sparse and not multivector,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant {self._url}: {exc}") from exc

    async def upsert(
        self,
        collection: str,
        chunks: list[ChunkRecord],
        batch_size: int = 64,
        max_request_bytes: int | None = None,
    ) -> None:
        from qdrant_client.http.models import PointStruct, SparseVector

        if not chunks:
            return
        limit = (
            max_request_bytes
            if max_request_bytes is not None
            else self.max_request_bytes
        )
        points = []
        for c in chunks:
            vector: dict = {"dense": c.vector}
            if c.sparse:
                vector["sparse"] = SparseVector(
                    indices=c.sparse["indices"], values=c.sparse["values"]
                )
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                    vector=vector,
                    payload={"chunk_id": c.id, **c.payload},
                )
            )
        batches = self._pack_batches(points, batch_size, limit)
        try:
            for batch in batches:
                await asyncio.to_thread(
                    self._client.upsert,
                    collection_name=collection,
                    points=batch,
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant {self._url}: {exc}") from exc

    def _pack_batches(self, points: list, batch_size: int, limit: int) -> list[list]:
        """Đóng gói batch theo số điểm VÀ byte ước lượng (payload JSON + vector).

        Point đơn vượt limit vẫn được gửi riêng (Qdrant giới hạn theo request,
        không thể chia một point) - chỉ log cảnh báo.
        """
        batches: list[list] = []
        current: list = []
        current_bytes = 0
        for p in points:
            est = _point_estimate(p)
            if current and (len(current) >= batch_size or current_bytes + est > limit):
                batches.append(current)
                current = []
                current_bytes = 0
            if est > limit:
                logger.warning(
                    "point %s ước ~%.1f MB vượt giới hạn upsert %.1f MB - gửi riêng",
                    p.id,
                    est / 1048576,
                    limit / 1048576,
                )
                batches.append([p])
                continue
            current.append(p)
            current_bytes += est
        if current:
            batches.append(current)
        return batches

    async def _is_named(self, collection: str) -> bool:
        """Collection có named vectors (query phải dùng using='dense')?"""
        if collection in self._named:
            return self._named[collection]
        info = await asyncio.to_thread(self._client.get_collection, collection)
        vectors = info.config.params.vectors
        named = isinstance(vectors, dict)
        self._named[collection] = named
        return named

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        *,
        sparse: dict | None = None,
    ) -> list[SearchHit]:
        from qdrant_client.http import models

        # sparse=None: dense-only. Legacy (vector vô danh) query nguyên vị;
        # named collection phải chọn vector "dense". Query multi-vector
        # (list token vectors) chạy qua đúng path này - Qdrant so khớp MaxSim
        # khi "dense" có multivector_config.
        if sparse is None:
            using = "dense" if await self._is_named(collection) else None
            results = await asyncio.to_thread(
                self._client.query_points,
                collection_name=collection,
                query=vector,
                **({"using": using} if using else {}),
                limit=top_k,
                with_payload=True,
            )
            return [
                SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
                for p in results.points
            ]
        # hybrid: dense + sparse prefetch fuse RRF
        results = await asyncio.to_thread(
            self._client.query_points,
            collection_name=collection,
            prefetch=[
                models.Prefetch(query=vector, using="dense", limit=top_k),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(p.id), score=p.score, payload=p.payload or {})
            for p in results.points
        ]

    async def delete_by_doc(self, collection: str, doc_id: str) -> int:
        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        filter_ = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        await asyncio.to_thread(
            self._client.delete,
            collection_name=collection,
            points_selector=FilterSelector(filter=filter_),
        )
        return 0  # Qdrant không trả số điểm đã xoá; 0 để batch>=1 dạng int hợp lệ

    async def count(self, collection: str, doc_id: str | None = None) -> int:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        count_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            if doc_id
            else None
        )
        result = await asyncio.to_thread(
            self._client.count,
            collection_name=collection,
            count_filter=count_filter,
            exact=True,
        )
        return int(result.count)
