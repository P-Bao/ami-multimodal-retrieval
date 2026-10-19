"""Test BGEM3Embedder (embedding-server BAAI/bge-m3) với server giả - không gọi mạng thật.

Phủ: handshake (/v1/models + probe), batch giữ thứ tự, sparse convert (indices
tăng dần int + values float), caption-only bỏ image_b64, text rỗng -> ".", retry
đúng lỗi tạm thời / không retry 4xx, circuit breaker, cache (chỉ dense),
return_sparse=False → sparse {}.
"""
import base64
import json

import httpx
import pytest

from ami_rag.core.bge_embedder import BGEM3Embedder
from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedModelMismatch,
    EmbedServerUnreachable,
)
from ami_rag.core.embedding_cache import EmbeddingCache

MODEL = "BAAI/bge-m3"


def _vec(i: int, dim: int = 8) -> list[float]:
    return [float((i + j) % 7) / 7 for j in range(dim)]


def _embed_body(n_items: int, dim: int = 8, *, sparse: bool = True) -> dict:
    body: dict = {"dense_vecs": [_vec(i, dim) for i in range(n_items)]}
    if sparse:
        # token_id dạng khoá chuỗi (như server trả) — thứ tự xoá trộn để test sort
        body["sparse_weights"] = [
            {"200": 0.3, "100": 0.25} for _ in range(n_items)
        ]
    return body


class FakeServer:
    """Server giả bge (/v1/models + /embed); cấu hình được hành vi lỗi."""

    def __init__(self, dim: int = 8, model_name: str = MODEL):
        self.requests: list[list] = []  # danh sách texts mỗi request /embed
        self.dim = dim
        self.model_name = model_name
        self.fail_next: list[int | Exception] = []
        self.sparse_enabled = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": self.model_name, "object": "model"}]}
            )
        if path == "/embed":
            body = json.loads(request.content.decode())
            texts = body["texts"]
            if texts == ["probe"]:
                return httpx.Response(
                    200,
                    json=_embed_body(1, self.dim, sparse=self.sparse_enabled),
                )
            self.requests.append(texts)
            if self.fail_next:
                first = self.fail_next.pop(0)
                if isinstance(first, Exception):
                    raise first
                return httpx.Response(first, json={"detail": f"fake {first}"})
            return httpx.Response(
                200,
                json=_embed_body(len(texts), self.dim, sparse=self.sparse_enabled),
            )
        return httpx.Response(404, json={"detail": "not found"})


def _make_embedder(server: FakeServer, **kwargs) -> BGEM3Embedder:
    defaults = {
        "base_url": "http://testserver",
        "model": MODEL,
        "expected_dim": 8,
        "timeout": 5,
        "retries": 2,
        "backoff_base": 0.0,
    }
    defaults.update(kwargs)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler),
        headers={"Authorization": "Bearer t"} if kwargs.get("token") else {},
    )
    return BGEM3Embedder(http_client=client, **defaults)


async def test_handshake_ok_sets_model_dim_and_instruction_ns():
    server = FakeServer()
    embedder = _make_embedder(server)
    info = await embedder.verify()
    assert info["data"][0]["id"] == MODEL
    assert embedder.model_name == MODEL
    assert embedder.dim == 8
    assert embedder._verified is True
    # instruction_ns = sha256(model)[:16]
    import hashlib

    assert (
        embedder._instruction_ns
        == hashlib.sha256(MODEL.encode()).hexdigest()[:16]
    )


async def test_handshake_model_mismatch_stops():
    server = FakeServer(model_name="other-model")
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_handshake_dim_mismatch_stops():
    server = FakeServer(dim=1024)
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_unreachable_on_models():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = BGEM3Embedder(
        base_url="http://testserver",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        retries=1,
        backoff_base=0.0,
    )
    with pytest.raises(EmbedServerUnreachable):
        await embedder.verify()


async def test_embed_documents_batch_preserves_order_and_sparse_pairs():
    server = FakeServer()
    embedder = _make_embedder(server, batch_size=2)
    texts = [f"text {i}" for i in range(5)]
    dense, sparse = await embedder.embed_documents_sparse(texts)
    assert len(dense) == 5 and len(sparse) == 5
    assert len(server.requests) == 3  # 2 + 2 + 1
    flat = [t for batch in server.requests for t in batch]
    assert flat == texts
    # sparse theo thứ tự input, indices tăng dần int + values float
    assert sparse[0] == {"indices": [100, 200], "values": [0.25, 0.3]}
    # batch đầu đủ 4 vec theo i (batch sau sinh lại _vec(0..) — giả server theo batch)
    assert dense[0] == _vec(0)
    assert dense[1] == _vec(1)


async def test_no_sparse_when_return_sparse_false():
    server = FakeServer()
    server.sparse_enabled = False
    embedder = _make_embedder(server, return_sparse=False)
    dense, sparse = await embedder.embed_documents_sparse(["a", "b"])
    assert len(dense) == 2
    assert sparse == [{}, {}]


async def test_sparse_methods_bypass_cache(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)
    await embedder.embed_documents_sparse(["doc a"])
    assert len(server.requests) == 1
    server.requests.clear()
    # shortcut: cache chỉ dense → sparse flow gọi server lại cả item cũ
    await embedder.embed_documents_sparse(["doc a"])
    assert len(server.requests) == 1


async def test_dense_documents_use_cache(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 1

    server.requests.clear()
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 0  # cả hai trúng cache

    await embedder.embed_documents(["doc a", "doc c"])
    assert server.requests[0] == ["doc c"]


async def test_count_cache_hits_needs_verified(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(FakeServer(), cache=cache)
    assert embedder.count_cache_hits(["x"]) == 0  # chưa verify
    await embedder.verify()
    assert embedder.count_cache_hits([]) == 0
    await embedder.embed_documents(["x"])
    assert embedder.count_cache_hits(["x"]) == 1
    assert embedder.count_cache_hits(["y"]) == 0


async def test_caption_only_drops_image_b64(caplog):
    server = FakeServer()
    embedder = _make_embedder(server)
    png_b64 = base64.b64encode(b"fakenpng").decode("ascii")
    items = [
        "plain text",
        {"text": "caption ảnh", "image_b64": png_b64},
    ]
    vectors = await embedder.embed_documents(items)
    assert len(vectors) == 2  # không raise
    assert server.requests[0] == ["plain text", "caption ảnh"]  # chỉ text
    assert "bge caption-only: bỏ image_b64 1/2 item" in caplog.text


async def test_empty_text_replaced_with_dot(caplog):
    server = FakeServer()
    embedder = _make_embedder(server)
    vectors = await embedder.embed_documents(["   ", "", {"text": "  "}])
    assert vectors[0] == _vec(0)
    assert server.requests[0] == [".", ".", "."]
    assert "3/3 item rỗng/whitespace" in caplog.text


async def test_embed_query_dense_and_sparse():
    server = FakeServer()
    embedder = _make_embedder(server)
    dense = await embedder.embed_query("câu hỏi")
    assert dense == _vec(0)
    assert server.requests[0] == ["câu hỏi"]
    dense, sparse = await embedder.embed_query_sparse("câu hỏi")
    assert dense == _vec(0)
    assert sparse == {"indices": [100, 200], "values": [0.25, 0.3]}


async def test_request_body_has_return_sparse_flags():
    captured: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=_embed_body(1, 8, sparse=False))

    client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    embedder = BGEM3Embedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=client,
    )
    await embedder.verify()  # probe /embed
    await embedder.embed_query("q")
    assert captured[0]["return_dense"] is True
    assert captured[0]["return_sparse"] is False
    assert captured[1]["texts"] == ["q"]
    assert captured[1]["return_sparse"] is False


async def test_retry_on_5xx_then_success():
    server = FakeServer()
    server.fail_next = [500, 503]
    embedder = _make_embedder(server)
    vectors = await embedder.embed_documents(["hello"])
    assert len(vectors) == 1
    assert len(server.requests) == 3  # 2 fail + 1 success


async def test_no_retry_on_4xx():
    server = FakeServer()
    server.fail_next = [400]
    embedder = _make_embedder(server)
    with pytest.raises(Exception) as exc_info:
        await embedder.embed_documents(["hello"])
    assert "embed server trả 400" in str(exc_info.value)
    assert len(server.requests) == 1  # không retry


async def test_auth_error_no_retry():
    server = FakeServer()
    server.fail_next = [401]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedAuthError):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1


async def test_422_payload_too_long():
    from ami_rag.core.embedder import EmbedInputTooLong

    server = FakeServer()
    server.fail_next = [422]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedInputTooLong):
        await embedder.embed_documents(["hello"])


async def test_circuit_breaker_stops_batch_early():
    server = FakeServer()

    def dead_embed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return server.handler(request)
        body = json.loads(request.content.decode())
        if body["texts"] == ["probe"]:
            return httpx.Response(200, json=_embed_body(1, 8))
        raise httpx.ConnectError("refused")

    embedder = BGEM3Embedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead_embed)),
        batch_size=1,
        max_concurrency=1,
        retries=0,
        backoff_base=0.0,
    )
    await embedder.verify()
    with pytest.raises(EmbedCircuitOpen):
        await embedder.embed_documents([f"t{i}" for i in range(10)])
    # 5 lần liên tiếp không với tới -> dừng cả lô sớm, không thử 10 batch


async def test_embed_documents_truncates_oversize_text():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=100)
    vectors = await embedder.embed_documents(["a" * 1000, "short"])
    assert len(vectors) == 2
    sent = server.requests[0]
    # budget 100 token * 3 chars/token
    assert sent[0] == "a" * 300
    assert sent[1] == "short"


async def test_response_dim_change_detected_after_handshake():
    """Server đổi model giữa chừng: /embed trả vector dim lệch handshake."""
    state = {"dim": 8}

    def switching(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        if request.url.path == "/embed":
            body = json.loads(request.content.decode())
            texts = body["texts"]
            if texts == ["probe"]:
                return httpx.Response(200, json=_embed_body(1, 8))
            if state["dim"] != 8:  # server đã đổi model -> dim khác
                return httpx.Response(200, json=_embed_body(len(texts), state["dim"]))
            return httpx.Response(200, json=_embed_body(len(texts), 8))
        return httpx.Response(404)

    # Handshake dim=8 ok, sau đó server đổi model -> dim 16
    state["dim"] = 16
    embedder = BGEM3Embedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(switching)),
        retries=0,
        backoff_base=0.0,
    )
    await embedder.verify()  # probe dùng handler → 200 dim 8 (state ban đầu)
    with pytest.raises(EmbedModelMismatch):
        await embedder.embed_documents(["a", "b"])
