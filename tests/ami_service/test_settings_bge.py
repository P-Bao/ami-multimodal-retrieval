"""Test default settings BGE (BGE-M3 embed + BGE reranker) + resolve backend.

Contract mặc định mới:
- embed BGE-M3 qua HTTP server :8008 (backend "bge", dense 1024 + sparse);
- rerank BGE text-only :8010 (backend "legacy", RERANK_MULTIMODAL=False).
"""

import pytest

from ami_rag.settings import Settings, resolve_embed_backend, resolve_rerank_backend


class NS:
    def __init__(self, **kw):
        self.EMBED_BACKEND = kw.get("EMBED_BACKEND", "auto")
        self.EMBED_MODEL = kw.get("EMBED_MODEL", "")
        self.RERANK_BACKEND = kw.get("RERANK_BACKEND", "auto")
        self.RERANK_MODEL = kw.get("RERANK_MODEL", "")


def test_default_settings_bge():
    s = Settings(_env_file=None)
    assert s.EMBED_BACKEND == "bge"
    assert s.EMBED_SERVER_URL == "http://localhost:8008"
    assert s.EMBED_MODEL == "BAAI/bge-m3"
    assert s.EMBED_DIM == 1024
    assert s.EMBED_MAX_INPUT_TOKENS == 8192
    assert s.EMBED_IMAGE_TOKEN_RESERVE == 0
    assert s.EMBED_RETURN_SPARSE is True
    assert s.EMBED_BATCH_SIZE == 32
    assert s.RERANK_BACKEND == "legacy"
    assert s.RERANK_BASE_URL == "http://localhost:8010"
    assert s.RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
    assert s.RERANK_MULTIMODAL is False


def test_resolve_embed_backend_explicit_bge():
    assert resolve_embed_backend(NS(EMBED_BACKEND="bge")) == "bge"


def test_resolve_embed_backend_auto_bge_by_model():
    assert resolve_embed_backend(NS(EMBED_MODEL="BAAI/bge-m3")) == "bge"


def test_resolve_embed_backend_auto_qwen_custom():
    assert (
        resolve_embed_backend(NS(EMBED_MODEL="Qwen/Qwen3-VL-Embedding-2B")) == "custom"
    )


def test_resolve_embed_backend_auto_other_openai():
    assert (
        resolve_embed_backend(NS(EMBED_MODEL="nvidia/llama-nemotron-embed-vl-1b-v2"))
        == "openai"
    )


def test_resolve_embed_backend_invalid_raises():
    with pytest.raises(ValueError):
        resolve_embed_backend(NS(EMBED_BACKEND="ollama"))


def test_resolve_rerank_backend_auto_nemotron_vllm():
    assert (
        resolve_rerank_backend(
            NS(RERANK_MODEL="nvidia/llama-nemotron-rerank-vl-1b-v2")
        )
        == "vllm"
    )


def test_resolve_rerank_backend_auto_bge_legacy():
    assert resolve_rerank_backend(NS(RERANK_MODEL="BAAI/bge-reranker-v2-m3")) == "legacy"


def test_resolve_rerank_backend_explicit_legacy():
    assert resolve_rerank_backend(NS(RERANK_BACKEND="legacy")) == "legacy"


def test_resolve_rerank_backend_invalid_raises():
    with pytest.raises(ValueError):
        resolve_rerank_backend(NS(RERANK_BACKEND="cohere"))
