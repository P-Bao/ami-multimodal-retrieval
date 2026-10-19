"""Test rerank clients (legacy BGE + vLLM Nemotron) với server giả - không gọi mạng thật.

Phủ: parse response hai backend, payload (model/top_n), dispatcher theo
RERANK_BACKEND, build_rerank_documents multimodal (ảnh từ asset store, fallback text).
"""
import json

import httpx
import pytest

from ami_rag.core.rerank_client import (
    build_rerank_documents,
    build_rerank_func,
    build_rerank_model_func,
    build_vllm_rerank_func,
    fuse_modality_scores,
    resolve_rerank_fusion,
)
from ami_rag.settings import Settings

NEMOTRON = "nvidia/llama-nemotron-rerank-vl-1b-v2"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 6


def _settings(**kwargs) -> Settings:
    defaults = {
        "RERANK_BASE_URL": "http://testserver",
        "RERANK_TIMEOUT": 5,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class FakeAssetStore:
    """Asset store giả: key 'ok' trả PNG bytes, key 'boom' raise, còn lại None."""

    def __init__(self):
        self.fetched: list[str] = []

    def get_bytes(self, key):
        self.fetched.append(key)
        if key == "ok":
            return PNG_BYTES
        if key == "boom":
            raise RuntimeError("minio down")
        return None


class MockRerankServer:
    """Server giả: legacy trả {scores, ranked_indices}; vllm trả {results}."""

    def __init__(self, mode: str = "vllm"):
        self.mode = mode
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        self.requests.append(body)
        if self.mode == "legacy":
            scores = [0.9, 0.5, 0.1]
            ranked = [0, 1, 2][: body.get("top_k") or 3]
            return httpx.Response(
                200, json={"scores": scores, "ranked_indices": ranked}
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": -1.25},
                    {"index": 0, "relevance_score": 2.5},
                ]
            },
        )


def _patch_async_client(monkeypatch, server: MockRerankServer) -> None:
    """httpx.AsyncClient trong rerank funcs -> client với MockTransport."""
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        return original(transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_legacy_func_maps_scores_and_indices(monkeypatch):
    server = MockRerankServer(mode="legacy")
    _patch_async_client(monkeypatch, server)
    func = build_rerank_model_func(_settings())

    results = await func("q", ["a", "b", "c"], top_n=2)
    assert results == [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.5},
    ]
    assert server.requests[0]["top_k"] == 2
    assert server.requests[0]["documents"] == ["a", "b", "c"]


async def test_vllm_func_maps_results(monkeypatch):
    server = MockRerankServer(mode="vllm")
    _patch_async_client(monkeypatch, server)
    func = build_vllm_rerank_func(_settings())

    results = await func("q", ["a", "b", "c"], top_n=2)
    assert results == [
        {"index": 2, "relevance_score": -1.25},
        {"index": 0, "relevance_score": 2.5},
    ]
    assert server.requests[0]["model"] == NEMOTRON
    assert server.requests[0]["top_n"] == 2
    assert server.requests[0]["documents"] == ["a", "b", "c"]


async def test_dispatcher_by_backend(monkeypatch):
    server = MockRerankServer(mode="vllm")
    _patch_async_client(monkeypatch, server)

    # RERANK_MODEL nemotron -> vllm func (payload có model + top_n)
    func = build_rerank_func(_settings(RERANK_MODEL=NEMOTRON))
    await func("q", ["a"], top_n=1)
    assert server.requests[0]["model"] == NEMOTRON
    assert server.requests[0]["top_n"] == 1

    # RERANK_MODEL BGE -> legacy func (payload top_k, không model)
    server.requests.clear()
    func = build_rerank_func(_settings(RERANK_MODEL="BAAI/bge-reranker-v2-m3"))
    await func("q", ["a"], top_n=1)
    assert server.requests[0] == {"query": "q", "documents": ["a"], "top_k": 1}


async def test_dispatcher_default_settings_is_vllm(monkeypatch):
    # Settings default (RERANK_MODEL nemotron) -> dispatcher chọn vllm func
    import ami_rag.core.rerank_client as rc

    called = {}

    def fake_build_vllm(settings):
        async def f(query, documents, top_n=None, **_):
            called["backend"] = "vllm"
            return []

        return f

    monkeypatch.setattr(rc, "build_vllm_rerank_func", fake_build_vllm)
    func = rc.build_rerank_func(_settings())
    await func("q", ["a"])
    assert called["backend"] == "vllm"


async def test_build_rerank_documents_text_only():
    chunks = [
        {"content": "chunk 1", "modality": "text"},
        {"content": "chunk 2"},
    ]
    docs = await build_rerank_documents(chunks, None, multimodal=True)
    assert docs == ["chunk 1", "chunk 2"]


async def test_build_rerank_documents_multimodal_with_assets():
    store = FakeAssetStore()
    chunks = [
        {"content": "text chunk", "modality": "text"},
        {"content": "image chunk", "modality": "image", "asset_key": "ok"},
        {"content": "table chunk", "modality": "table", "asset_key": "missing"},
        {"content": "broken chunk", "modality": "image", "asset_key": "boom"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=True)
    assert docs[0] == "text chunk"
    assert isinstance(docs[1], dict)
    parts = docs[1]["content"]
    assert parts[0] == {"type": "text", "text": "image chunk"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # thiếu asset / fetch lỗi -> fallback text
    assert docs[2] == "table chunk"
    assert docs[3] == "broken chunk"
    # chunk text không được fetch asset
    assert store.fetched == ["ok", "missing", "boom"]


async def test_build_rerank_documents_multimodal_disabled():
    store = FakeAssetStore()
    chunks = [
        {"content": "image chunk", "modality": "image", "asset_key": "ok"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=False)
    assert docs == ["image chunk"]
    assert store.fetched == []


async def test_build_rerank_documents_equation_included():
    store = FakeAssetStore()
    chunks = [
        {"content": "eq", "modality": "equation", "asset_key": "ok"},
    ]
    docs = await build_rerank_documents(chunks, store, multimodal=True)
    assert isinstance(docs[0], dict)


async def test_build_rerank_documents_truncates_oversize_text():
    chunks = [
        {"content": "a" * 1000, "modality": "text"},
        {"content": "b" * 1000, "modality": "image", "asset_key": "ok"},
        {"content": "short", "modality": "text"},
    ]
    docs = await build_rerank_documents(
        chunks,
        FakeAssetStore(),
        multimodal=True,
        max_input_tokens=300,
        image_token_reserve=100,
    )
    # text-only: budget 300 token * 3 chars
    assert docs[0] == "a" * 900
    # kèm ảnh: budget (300 - 100) * 3 chars, ảnh giữ nguyên
    assert isinstance(docs[1], dict)
    assert docs[1]["content"][0] == {"type": "text", "text": "b" * 600}
    assert docs[1]["content"][1]["type"] == "image_url"
    assert docs[2] == "short"


# --- fuse_modality_scores ---


def _chunks_for_fusion() -> list[dict]:
    return [
        {"modality": "text"},   # idx 0
        {"modality": "image"},  # idx 1
        {"modality": "table"},  # idx 2 (cùng nhóm text)
        {"modality": "image"},  # idx 3
        {"modality": "text"},   # idx 4
        {},                     # idx 5 (thiếu modality -> text)
    ]


async def test_fuse_promotes_top_visual_over_low_text():
    # image idx1 score 0.14 (đạt floor, hạng 1 nhóm visual -> 0.8/11);
    # 4 text trên nó (hạng 1-4 nhóm text). Image thắng text hạng 4 (1/14).
    chunks = _chunks_for_fusion()
    results = [
        {"index": 0, "relevance_score": 0.6},
        {"index": 4, "relevance_score": 0.5},
        {"index": 5, "relevance_score": 0.4},
        {"index": 2, "relevance_score": 0.14},
        {"index": 1, "relevance_score": 0.14},
    ]
    fused = fuse_modality_scores(results, chunks)

    assert [f["index"] for f in fused] == [0, 4, 5, 1, 2]
    assert fused[0]["fused_score"] == pytest.approx(1.0 / 11)
    assert fused[1]["fused_score"] == pytest.approx(1.0 / 12)
    assert fused[2]["fused_score"] == pytest.approx(1.0 / 13)
    assert fused[3]["fused_score"] == pytest.approx(0.8 / 11)
    assert fused[4]["fused_score"] == pytest.approx(1.0 / 14)
    # relevance_score giữ nguyên raw
    assert fused[0]["relevance_score"] == 0.6


async def test_fuse_floor_blocks_promotion():
    # image idx1 score 0.005 < visual_floor 0.01 -> bị chặn, xếp cuối (raw - 1)
    chunks = _chunks_for_fusion()
    results = [
        {"index": 0, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.005},
    ]
    fused = fuse_modality_scores(results, chunks)

    assert [f["index"] for f in fused] == [0, 1]
    assert fused[0]["fused_score"] == pytest.approx(1.0 / 11)
    assert fused[1]["fused_score"] == pytest.approx(0.005 - 1.0)


async def test_fuse_floor_keeps_relative_order_within_blocked_group():
    # 2 image dưới floor: raw cao hơn -> fused cao hơn (gần 0 hơn)
    chunks = _chunks_for_fusion()
    results = [
        {"index": 1, "relevance_score": 0.002},
        {"index": 3, "relevance_score": 0.008},
    ]
    fused = fuse_modality_scores(results, chunks)

    assert [f["index"] for f in fused] == [3, 1]
    assert fused[0]["fused_score"] == pytest.approx(0.008 - 1.0)
    assert fused[1]["fused_score"] == pytest.approx(0.002 - 1.0)


async def test_fuse_table_equation_same_group_as_text():
    # table/equation cùng nhóm text với text chunk - chỉ image riêng nhóm
    chunks = _chunks_for_fusion()
    results = [
        {"index": 2, "relevance_score": 0.3},  # table
        {"index": 5, "relevance_score": 0.2},  # thiếu modality -> text
        {"index": 1, "relevance_score": 0.14},  # image
    ]
    fused = fuse_modality_scores(results, chunks)

    # text nhóm: table hạng 1 (1/11), no-modality hạng 2 (1/12) > image (0.8/11)
    assert [f["index"] for f in fused] == [2, 5, 1]
    assert fused[0]["fused_score"] == pytest.approx(1.0 / 11)
    assert fused[1]["fused_score"] == pytest.approx(1.0 / 12)
    assert fused[2]["fused_score"] == pytest.approx(0.8 / 11)


async def test_fuse_visual_weight_affects_ranking():
    # visual_weight thấp: image đạt floor vẫn thua text cùng khoảng hạng
    chunks = _chunks_for_fusion()
    results = [
        {"index": 4, "relevance_score": 0.5},
        {"index": 1, "relevance_score": 0.14},
    ]
    fused = fuse_modality_scores(results, chunks, visual_weight=0.3)
    assert [f["index"] for f in fused] == [4, 1]
    assert fused[1]["fused_score"] == pytest.approx(0.3 / 11)


async def test_fuse_empty_and_invalid_results():
    chunks = _chunks_for_fusion()
    assert fuse_modality_scores([], chunks) == []
    # idx ngoài phạm vi bị bỏ qua
    fused = fuse_modality_scores([{"index": 99, "relevance_score": 0.9}], chunks)
    assert fused == []
    # results None
    assert fuse_modality_scores(None, chunks) == []


async def test_fuse_tie_break_stable():
    # 2 text cùng điểm: giữ thứ tự xuất hiện trong results (stable sort)
    chunks = _chunks_for_fusion()
    results = [
        {"index": 4, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.5},
    ]
    fused = fuse_modality_scores(results, chunks)
    assert [f["index"] for f in fused] == [4, 0]


def test_resolve_rerank_fusion():
    assert resolve_rerank_fusion(Settings(RERANK_FUSION="raw")) == "raw"
    assert resolve_rerank_fusion(Settings(RERANK_FUSION="rrf")) == "rrf"
    assert resolve_rerank_fusion(Settings(RERANK_FUSION="")) == "raw"

    class _NoFusionAttr:  # settings thiếu attr -> getattr default
        pass

    assert resolve_rerank_fusion(_NoFusionAttr()) == "raw"
    with pytest.raises(ValueError):
        resolve_rerank_fusion(Settings(RERANK_FUSION="bogus"))
