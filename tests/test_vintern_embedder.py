"""Test VinternEmbedder (server multi-vector 5CD-AI/Vintern-Embedding-1B) với server giả.

Không gọi mạng thật. Phủ: handshake (model/dim/multivector lệch -> dừng),
embed_query multi shape, embed_documents ảnh image-only (server 422 nếu cả
text+image -> client không bao giờ gửi cả hai), retry transient, batch theo
EMBED_BATCH_SIZE, kiểm tra từng response (đổi model giữa chừng).
"""

import json

import httpx
import pytest

from ami_rag.core.embedder import (
    EmbedModelMismatch,
    EmbedServerUnreachable,
)
from ami_rag.core.vintern_embedder import VinternEmbedder

MODEL = "5CD-AI/Vintern-Embedding-1B"
DIM = 128


def _tokens(n_tokens: int, dim: int = DIM) -> list[list[float]]:
    return [[0.1 * ((i + j) % 7) for j in range(dim)] for i in range(n_tokens)]


def _embed_body(items: list[dict], *, model_name: str = MODEL, dim: int = DIM) -> dict:
    # server trả per-token vectors: giả lập 2 token/item
    return {
        "model_name": model_name,
        "dim": dim,
        "vectors": [_tokens(2) for _ in items],
        "truncated": [],
    }


class FakeServer:
    """Server giả: /info + /embed, bắt payload để kiểm tra ràng buộc item."""

    def __init__(self, dim: int = DIM, model_name: str = MODEL, multivector=True):
        self.requests: list[list[dict]] = []
        self.dim = dim
        self.model_name = model_name
        self.multivector = multivector
        self.fail_next: list[int | Exception] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/info":
            return httpx.Response(
                200,
                json={
                    "model_name": self.model_name,
                    "dim": self.dim,
                    "max_input_tokens": 8192,
                    "multivector": self.multivector,
                    "normalize": False,
                    "query_instruction": "Instruct: query.",
                    "document_instruction": "Instruct: document.",
                    "server_version": "test",
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "native_dim": 896,
                },
            )
        if path == "/embed":
            body = json.loads(request.content.decode())
            self.requests.append(body["items"])
            if self.fail_next:
                first = self.fail_next.pop(0)
                if isinstance(first, Exception):
                    raise first
                return httpx.Response(first, json={"detail": f"fake {first}"})
            return httpx.Response(200, json=_embed_body(body["items"], dim=self.dim))
        return httpx.Response(404, json={"detail": "not found"})


def _make_embedder(server: FakeServer, **kwargs) -> VinternEmbedder:
    defaults = {
        "base_url": "http://testserver",
        "model": MODEL,
        "expected_dim": DIM,
        "timeout": 5,
        "retries": 2,
        "backoff_base": 0.0,
    }
    defaults.update(kwargs)
    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return VinternEmbedder(http_client=client, **defaults)


# ------------------------------------------------------------------ handshake


async def test_handshake_ok_sets_model_and_dim():
    server = FakeServer()
    embedder = _make_embedder(server)
    info = await embedder.verify()
    assert info["model_name"] == MODEL
    assert embedder.dim == DIM
    assert embedder.is_multivector is True
    assert embedder.supports_images is True


async def test_handshake_model_mismatch_stops():
    embedder = _make_embedder(FakeServer(model_name="other-model"))
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_handshake_dim_mismatch_stops():
    embedder = _make_embedder(FakeServer(dim=1024))
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_handshake_non_multivector_server_stops():
    embedder = _make_embedder(FakeServer(multivector=False))
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_unreachable_on_info():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = VinternEmbedder(
        base_url="http://testserver",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        retries=1,
        backoff_base=0.0,
    )
    with pytest.raises(EmbedServerUnreachable):
        await embedder.verify()


# ------------------------------------------------------------------ embed_query


async def test_embed_query_returns_multi_vector():
    server = FakeServer()
    embedder = _make_embedder(server)
    vec = await embedder.embed_query("câu hỏi")
    # multi-vector: list token vector, mỗi token dim 128
    assert isinstance(vec, list) and vec
    assert all(isinstance(t, list) and len(t) == DIM for t in vec)
    # query luôn text-only
    assert server.requests == [[{"type": "query", "text": "câu hỏi"}]]


# ------------------------------------------------------------------ embed_documents


async def test_embed_documents_image_item_is_image_only():
    """Server 422 khi item có cả text + image_b64 -> client phải gửi image-only."""
    server = FakeServer()
    embedder = _make_embedder(server)
    await embedder.embed_documents(
        [
            {"text": "caption", "image_b64": "aW1n"},
            "text thuần",
        ]
    )
    batch = server.requests[0]
    assert len(batch) == 2
    img_item, text_item = batch
    assert img_item["type"] == "document"
    assert img_item["image_b64"] == "aW1n"
    assert "text" not in img_item  # tuyệt đối không gửi cả hai
    assert text_item == {"type": "document", "text": "text thuần"}


async def test_embed_documents_preserves_order_and_returns_multivector():
    server = FakeServer()
    embedder = _make_embedder(server)
    vecs = await embedder.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 2 and len(v[0]) == DIM for v in vecs)
    assert [it["text"] for it in server.requests[0]] == ["a", "b", "c"]


async def test_embed_documents_respects_batch_size():
    server = FakeServer()
    embedder = _make_embedder(server, batch_size=2)
    await embedder.embed_documents([f"t{i}" for i in range(5)])
    assert [len(b) for b in server.requests] == [2, 2, 1]


async def test_embed_documents_retry_transient():
    server = FakeServer()
    server.fail_next = [503]
    embedder = _make_embedder(server, retries=2)
    vecs = await embedder.embed_documents(["doc"])
    assert len(vecs) == 1
    assert len(server.requests) == 2  # lần 1 lỗi 503, lần 2 thành công


async def test_check_response_model_changed_midstream():
    server = FakeServer()
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder._check_response({"model_name": "other", "dim": DIM})
    with pytest.raises(EmbedModelMismatch):
        await embedder._check_response({"model_name": MODEL, "dim": 896})
