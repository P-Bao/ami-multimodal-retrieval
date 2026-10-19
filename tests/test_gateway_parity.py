"""Parity test: clients của ami_rag ↔ gateway nemotron-vl-vllm (mã gateway thật).

Chạy gateway app THẬT (import từ repo máy B ``nemotron-vl-vllm/gateway/app.py``)
qua httpx ASGI transport (không cần mạng/GPU), upstream vLLM phía sau gateway
giả bằng MockTransport. Flow đối chiếu ``examples/client_example.py``:
batch embed (text / bảng / ảnh+text) -> embed query -> cosine -> rerank.

Skip nếu repo gateway không có trên máy này (máy dev/CI khác).
"""
import asyncio
import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from ami_rag.core.openai_embedder import OpenAIEmbedder
from ami_rag.core.rerank_client import build_rerank_documents, build_vllm_rerank_func
from ami_rag.settings import Settings

GATEWAY_APP = Path(r"D:\Code\Python\Ami\RAG\nemotron-vl-vllm\gateway\app.py")

pytestmark = pytest.mark.skipif(
    not GATEWAY_APP.is_file(), reason="repo nemotron-vl-vllm không có trên máy này"
)

MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
DIM = 8
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 6


# ---------------------------------------------------------------------------
# Upstream vLLM giả phía sau gateway
# ---------------------------------------------------------------------------
class FakeUpstream:
    """vLLM giả: /health, /v1/embeddings (messages), /rerank."""

    def __init__(self):
        self.embed_requests: list[dict] = []
        self.rerank_requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/embeddings":
            body = json.loads(request.content.decode())
            self.embed_requests.append(body)
            vec = [float((len(self.embed_requests) + i) % 5) / 5 for i in range(DIM)]
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": vec}],
                    "model": MODEL,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        if path == "/rerank":
            body = json.loads(request.content.decode())
            self.rerank_requests.append(body)
            # điểm giả theo nội dung: doc chứa "Doanh thu" liên quan nhất
            scores = []
            for doc in body["documents"]:
                parts = doc["content"] if isinstance(doc, dict) else [{"type": "text", "text": doc}]
                text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
                scores.append(0.9 if "doanh thu" in text.lower() else 0.1)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            return httpx.Response(
                200,
                json={"results": [
                    {"index": i, "relevance_score": scores[i]} for i in ranked
                ]},
            )
        return httpx.Response(404, json={"detail": "not found"})


def _load_gateway_module():
    os.environ.setdefault("EMBED_URL", "http://upstream-embed")
    os.environ.setdefault("RERANK_URL", "http://upstream-rerank")
    os.environ.setdefault("EMBED_MODEL_NAME", MODEL)
    os.environ.setdefault("RERANK_MODEL_NAME", RERANK_MODEL)
    os.environ.setdefault("GATEWAY_API_KEY", "")
    os.environ.setdefault("EMBED_NORMALIZE", "true")
    spec = importlib.util.spec_from_file_location("nemotron_gateway_app", GATEWAY_APP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nemotron_gateway_app"] = mod  # pydantic cần resolve annotations
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gateway(monkeypatch):
    """Gateway app thật qua ASGI + upstream giả; trả (module, http client)."""
    mod = _load_gateway_module()
    upstream = FakeUpstream()
    mod.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream.handler)
    )
    mod.state.embed_sem = asyncio.Semaphore(8)
    mod.state.rerank_sem = asyncio.Semaphore(4)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mod.app), base_url="http://gateway"
    )
    try:
        yield upstream, client
    finally:
        pass


# ---------------------------------------------------------------------------
# Tests theo flow client_example.py
# ---------------------------------------------------------------------------
async def test_handshake_and_embed_batch_like_example(gateway):
    upstream, client = gateway
    emb = OpenAIEmbedder(
        base_url="http://gateway",
        model=MODEL,
        expected_dim=DIM,
        batch_size=32,
        retries=1,
        backoff_base=0.0,
        http_client=client,
    )
    b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    docs = [
        "Hà Nội là thủ đô của Việt Nam.",                 # text
        {"text": "Caption: Kết quả kinh doanh 2025\n\n| Quý | Doanh thu |\n| --- | --- |\n| Q1 | 120 |"},  # bảng (text content)
        {"text": "Trang báo cáo tài chính", "image_b64": b64},  # ảnh + text
    ]
    vectors = await emb.embed_documents(docs)
    assert emb.dim == DIM
    assert len(vectors) == 3
    assert all(len(v) == DIM for v in vectors)
    # gateway fan-out: probe của verify + mỗi item 1 request (messages 1 conversation)
    assert len(upstream.embed_requests) == 4
    # ảnh -> image_url data URI part (gateway sniff mime từ base64)
    img_parts = upstream.embed_requests[3]["messages"][0]["content"]
    assert img_parts[0]["type"] == "image_url"
    assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert img_parts[1] == {"type": "text", "text": "Trang báo cáo tài chính"}


async def test_embed_query_and_cosine_like_example(gateway):
    _, client = gateway
    emb = OpenAIEmbedder(
        base_url="http://gateway",
        model=MODEL,
        expected_dim=DIM,
        retries=1,
        backoff_base=0.0,
        http_client=client,
    )
    await emb.verify()
    q_vec = await emb.embed_query("Lợi nhuận quý 3 là bao nhiêu?")
    assert len(q_vec) == DIM
    # gateway L2-normalize -> dot = cosine; norm ~1
    norm = sum(x * x for x in q_vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3


async def test_rerank_like_example(gateway, monkeypatch):
    upstream, client = gateway
    settings = Settings(RERANK_BASE_URL="http://gateway", RERANK_TIMEOUT=5)
    func = build_vllm_rerank_func(settings)

    def _factory(*_args, **_kwargs):
        return client

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    results = await func(
        "Lợi nhuận quý 3 là bao nhiêu?",
        [
            "Hà Nội là thủ đô của Việt Nam.",
            {"content": [
                {"type": "text", "text": "Bảng doanh thu quý"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ]},
        ],
        top_n=2,
    )

    assert results[0]["index"] == 1  # doc "Doanh thu" liên quan nhất
    assert results[0]["relevance_score"] == 0.9
    # upstream nhận đúng format vLLM: dict docs được gateway wrap qua pass-through
    sent_docs = upstream.rerank_requests[0]["documents"]
    assert isinstance(sent_docs[0], dict)
    assert sent_docs[1]["content"][0]["type"] == "text"


async def test_build_rerank_documents_through_gateway(gateway, monkeypatch):
    upstream, client = gateway
    settings = Settings(RERANK_BASE_URL="http://gateway", RERANK_TIMEOUT=5)
    func = build_vllm_rerank_func(settings)

    class Store:
        def get_bytes(self, key):
            return PNG_BYTES if key == "ok" else None

    chunks = [
        {"content": "text chunk", "modality": "text"},
        {"content": "image chunk", "modality": "image", "asset_key": "ok"},
    ]
    docs = await build_rerank_documents(chunks, Store(), multimodal=True)

    def _factory(*_args, **_kwargs):
        return client

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    results = await func("q", docs, top_n=2)
    assert len(results) == 2
    # upstream nhận đúng format vLLM: gateway wrap MỌI doc thành {"content": parts}
    # khi có ít nhất một dict doc (all_plain_text=False)
    sent_docs = upstream.rerank_requests[0]["documents"]
    assert sent_docs[0]["content"][0] == {"type": "text", "text": "text chunk"}
    img_part = sent_docs[1]["content"][1]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")
