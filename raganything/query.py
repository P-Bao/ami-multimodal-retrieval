"""Query functionality for RAGAnything (vector pipeline thuần - không LightRAG).

- aquery_data: embed (RemoteEmbedder) + search (VectorStore) - không entity/graph.
- aquery: answer_func (LLM) thuần văn bản; retrieval do caller làm qua
  aquery_data + rerank.
- Đã bỏ aquery_vlm_enhanced/aquery_with_multimodal và mọi LightRAG paths.
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class QueryMixin:
    """QueryMixin (không LightRAG)."""

    async def aquery(self, query: str, system_prompt: str | None = None, **kwargs) -> str:
        """Pure text query - gọi answer_func (llm_model_func) trực tiếp."""
        llm = getattr(self, "llm_model_func", None)
        if llm is None:
            raise ValueError(
                "llm_model_func must be provided to answer queries"
            )
        callback_manager = getattr(self, "callback_manager", None)
        query_start_time = time.time()
        if callback_manager is not None:
            callback_manager.dispatch("on_query_start", query=query, mode="vector")

        self.logger.info(f"Executing text query: {query[:100]}...")
        try:
            result = await llm(query, system_prompt=system_prompt)
        except Exception as exc:
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_query_error", query=query, mode="vector", error=exc
                )
            raise

        if callback_manager is not None:
            result_len = len(result) if isinstance(result, str) else 0
            callback_manager.dispatch(
                "on_query_complete",
                query=query,
                mode="vector",
                duration_seconds=time.time() - query_start_time,
                result_length=result_len,
            )
        return result

    async def aquery_data(
        self, query: str, top_k: int | None = None, **kwargs
    ) -> Dict[str, Any]:
        """embed + vector search (không entity/graph, không LLM).

        Yêu cầu RAGAnything có embedder + vector_store (+ collection).
        """
        embedder = getattr(self, "embedder", None)
        vector_store = getattr(self, "vector_store", None)
        collection = getattr(self, "collection", "")
        if embedder is None or vector_store is None:
            raise ValueError(
                "embedder + vector_store must be provided for aquery_data"
            )
        vector = await embedder.embed_query(query)
        hits = await vector_store.search(collection, vector, top_k or 5)
        return {
            "status": "success",
            "message": "",
            "data": {"chunks": [hit.payload for hit in hits]},
            "metadata": {"mode": "vector", "top_k": top_k or 5},
        }
