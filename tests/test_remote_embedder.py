"""Test RemoteEmbedder với server giả (httpx.MockTransport) - không gọi mạng thật.

Phủ: handshake (model/dim lệch -> dừng), retry đúng lỗi tạm thời / không retry 4xx,
chia batch theo số item và byte payload, kiểm tra từng response (đổi model giữa
chừng), circuit breaker, cache, timeout/mất kết nối.
"""
import json

import httpx
import pytest

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedInputTooLong,
    EmbedModelMismatch,
    EmbedServerUnreachable,
)
from ami_rag.core.embedding_cache import EmbeddingCache
from ami_rag.core.remote_embedder import RemoteEmbedder

MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def _vec(i: int, dim: int = 8) -> list[float]:
    return [float((i + j) % 7) / 7 for j in range(dim)]


def _embed_body(items: list[dict], *, model_name: str = MODEL, dim: int = 8) -> dict:
    return {
        "model_name": model_name,
        "dim": dim,
        "vectors": [_vec(i) for i in range(len(items))],
        "truncated": [],
    }


class FakeServer:
    """Server giả đếm request; cấu hình được hành vi lỗi."""

    def __init__(self, dim: int = 8, model_name: str = MODEL):
        self.requests: list[list[dict]] = []
        self.dim = dim
        self.model_name = model_name
        # mapping: thứ tự request -> kết quả ("ok", status code, hoặc exception)
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
                    "normalize": True,
                    "query_instruction": "Q instruction.",
                    "document_instruction": "D instruction.",
                    "server_version": "test",
                    "device": "cpu",
                    "dtype": "bfloat16",
                    "native_dim": 2048,
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


def _make_embedder(server: FakeServer, **kwargs) -> RemoteEmbedder:
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
    return RemoteEmbedder(http_client=client, **defaults)


async def test_handshake_ok_sets_model_and_dim():
    server = FakeServer()
    embedder = _make_embedder(server)
    info = await embedder.verify()
    assert info["model_name"] == MODEL
    assert embedder.dim == 8


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


async def test_unreachable_on_info():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = RemoteEmbedder(
        base_url="http://testserver",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        retries=1,
        backoff_base=0.0,
    )
    with pytest.raises(EmbedServerUnreachable):
        await embedder.verify()


async def test_embed_documents_preserves_order_across_batches():
    server = FakeServer()
    embedder = _make_embedder(server, batch_size=2)
    texts = [f"text {i}" for i in range(5)]
    vectors = await embedder.embed_documents(texts)
    assert len(vectors) == 5
    # Thứ tự giữ nguyên: text 0 -> vector 0 (server trả vector theo index trong batch)
    assert len(server.requests) == 3  # 2 + 2 + 1
    flat = [item["text"] for batch in server.requests for item in batch]
    assert flat == texts


async def test_batching_by_payload_bytes():
    server = FakeServer()
    # 1 text ~1000 byte + overhead -> mỗi batch chỉ chứa 1 text
    embedder = _make_embedder(server, batch_size=10, max_payload_bytes=2000)
    texts = ["x" * 1000 for _ in range(3)]
    await embedder.embed_documents(texts)
    assert len(server.requests) == 3


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
    with pytest.raises(EmbedderError):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1  # không retry


async def test_auth_error_no_retry():
    server = FakeServer()
    server.fail_next = [401]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedAuthError):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1


async def test_input_too_long_no_retry():
    server = FakeServer()
    server.fail_next = [413]
    embedder = _make_embedder(server)
    with pytest.raises(EmbedInputTooLong):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 1


async def test_timeout_retried_then_unreachable():
    server = FakeServer()
    server.fail_next = [
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
    ]
    embedder = _make_embedder(server, retries=2)
    with pytest.raises(EmbedServerUnreachable):
        await embedder.embed_documents(["hello"])
    assert len(server.requests) == 3  # retries=2 -> 3 lần thử


async def test_circuit_breaker_stops_batch_early():
    server = FakeServer()

    def dead_embed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return server.handler(request)
        raise httpx.ConnectError("refused")

    embedder = RemoteEmbedder(
        base_url="http://testserver",
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


async def test_response_model_change_detected_mid_stream():
    server = FakeServer()

    def switching(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/embed":
            if len(server.requests) == 0:
                server.requests.append([])
                return httpx.Response(200, json=_embed_body([{"text": "a"}], model_name=MODEL))
            # Các request sau: server đã bị đổi model
            return httpx.Response(
                200, json=_embed_body([{"text": "b"}], model_name="other-model")
            )
        return server.handler(request)

    embedder = RemoteEmbedder(
        base_url="http://testserver",
        model=MODEL,
        expected_dim=8,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(switching)),
        batch_size=1,
        retries=0,
        backoff_base=0.0,
    )
    await embedder.verify()
    with pytest.raises(EmbedModelMismatch):
        await embedder.embed_documents(["a", "b"])


async def test_cache_only_sends_misses(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 1  # 1 batch, 2 items

    # Lần 2: cả hai trúng cache -> 0 request mới
    server.requests.clear()
    await embedder.embed_documents(["doc a", "doc b"])
    assert len(server.requests) == 0

    # Lần 3: 1 mới 1 cũ -> chỉ gửi miss
    server.requests.clear()
    await embedder.embed_documents(["doc a", "doc c"])
    assert len(server.requests) == 1
    assert [i["text"] for i in server.requests[0]] == ["doc c"]


async def test_cache_invalidated_by_instruction_change(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)
    await embedder.embed_documents(["doc a"])
    server.requests.clear()

    # Server đổi instruction -> instruction_ns đổi -> cache miss
    original = server.handler

    def new_instruction(request: httpx.Request) -> httpx.Response:
        resp = original(request)
        if request.url.path == "/info":
            body = resp.json()
            body["query_instruction"] = "Changed instruction."
            return httpx.Response(200, json=body)
        return resp

    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(new_instruction))
    embedder._verified = False
    await embedder.verify()
    await embedder.embed_documents(["doc a"])
    assert len(server.requests) == 1  # cache cũ hết hiệu lực


async def test_embed_query_uses_query_type():
    server = FakeServer()
    embedder = _make_embedder(server)
    await embedder.embed_query("câu hỏi")
    assert server.requests[0][0]["type"] == "query"


async def test_embed_multimodal_sends_image_b64():
    server = FakeServer()
    embedder = _make_embedder(server)
    items = [
        "plain text",
        {"text": "caption", "image_b64": "aW1hZ2UtZGF0YQ=="},
    ]
    vectors = await embedder.embed_documents(items)
    assert len(vectors) == 2
    sent = server.requests[0]
    assert sent[0] == {"type": "document", "text": "plain text"}
    assert sent[1] == {
        "type": "document",
        "text": "caption",
        "image_b64": "aW1hZ2UtZGF0YQ==",
    }


async def test_cache_isolates_text_with_and_without_image(tmp_path):
    server = FakeServer()
    cache = EmbeddingCache(tmp_path / "cache.db")
    embedder = _make_embedder(server, cache=cache)

    # Embed text without image
    await embedder.embed_documents(["same text"])
    assert len(server.requests) == 1
    server.requests.clear()

    # Embed same text with image -> must be a cache miss because image differs
    await embedder.embed_documents([{"text": "same text", "image_b64": "aW1n"}])
    assert len(server.requests) == 1
    server.requests.clear()

    # Re-embed both -> both hit cache, 0 network requests
    hits = embedder.count_cache_hits([
        "same text",
        {"text": "same text", "image_b64": "aW1n"},
    ])
    assert hits == 2
    await embedder.embed_documents([
        "same text",
        {"text": "same text", "image_b64": "aW1n"},
    ])
    assert len(server.requests) == 0


async def test_embed_documents_truncates_oversize_text():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=100, image_token_reserve=50)
    vectors = await embedder.embed_documents(["a" * 1000, "short"])
    assert len(vectors) == 2
    sent = server.requests[0]
    # budget 100 token * 3 chars/token; item str -> {"type": "document", "text"}
    assert sent[0] == {"type": "document", "text": "a" * 300}
    assert sent[1] == {"type": "document", "text": "short"}


async def test_embed_documents_truncates_multimodal_with_image_reserve():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=300, image_token_reserve=100)
    await embedder.embed_documents([{"text": "b" * 1000, "image_b64": "aW1n"}])
    item = server.requests[0][0]
    # budget (300 - 100) * 3 chars/token; ảnh giữ nguyên
    assert item["image_b64"] == "aW1n"
    assert item["text"] == "b" * 600

