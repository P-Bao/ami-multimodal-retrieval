"""Test OpenAIEmbedder (gateway nemotron-vl-vllm) với server giả - không gọi mạng thật.

Phủ: handshake (/health + probe model/dim), batch input giữ thứ tự, payload item
{"text", "image"}, retry đúng lỗi tạm thời / không retry 4xx, kiểm tra response
(đổi model giữa chừng), circuit breaker, cache, input_type query.
"""
import base64

import httpx
import pytest

from ami_rag.core.embedder import (
    EmbedAuthError,
    EmbedCircuitOpen,
    EmbedderError,
    EmbedModelMismatch,
    EmbedServerUnreachable,
)
from ami_rag.core.embedding_cache import EmbeddingCache
from ami_rag.core.openai_embedder import OpenAIEmbedder, data_url_from_bytes

MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 6
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _vec(i: int, dim: int = 8) -> list[float]:
    return [float((i + j) % 7) / 7 for j in range(dim)]


def _embed_body(n_items: int, *, model_name: str = MODEL, dim: int = 8) -> dict:
    return {
        "object": "list",
        "model": model_name,
        "data": [
            {"object": "embedding", "index": i, "embedding": _vec(i, dim)}
            for i in range(n_items)
        ],
        "usage": {"prompt_tokens": n_items, "total_tokens": n_items},
    }


class FakeServer:
    """Server giả gateway (health + embeddings); cấu hình được hành vi lỗi."""

    def __init__(self, dim: int = 8, model_name: str = MODEL):
        self.requests: list[list] = []  # danh sách input items mỗi request
        self.dim = dim
        self.model_name = model_name
        self.fail_next: list[int | Exception] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={"status": "ok", "embed": True, "rerank": True},
            )
        if path == "/v1/embeddings":
            body = __import__("json").loads(request.content.decode())
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            # probe của verify(): không ghi, không tiêu fail_next
            if (
                len(items) == 1
                and isinstance(items[0], dict)
                and items[0].get("text") == "probe"
            ):
                return httpx.Response(
                    200, json=_embed_body(1, model_name=self.model_name, dim=self.dim)
                )
            self.requests.append(items)
            if self.fail_next:
                first = self.fail_next.pop(0)
                if isinstance(first, Exception):
                    raise first
                return httpx.Response(first, json={"detail": f"fake {first}"})
            return httpx.Response(
                200, json=_embed_body(len(items), model_name=self.model_name, dim=self.dim)
            )
        return httpx.Response(404, json={"detail": "not found"})


def _make_embedder(server: FakeServer, **kwargs) -> OpenAIEmbedder:
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
    return OpenAIEmbedder(http_client=client, **defaults)


async def test_handshake_ok_sets_model_and_dim():
    server = FakeServer()
    embedder = _make_embedder(server)
    info = await embedder.verify()
    assert info["status"] == "ok"
    assert embedder.dim == 8
    # probe của verify() không được ghi vào requests
    assert len(server.requests) == 0


async def test_handshake_probe_model_mismatch_stops():
    server = FakeServer(model_name="other-model")
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_handshake_dim_mismatch_stops():
    server = FakeServer(dim=1024)
    embedder = _make_embedder(server)
    with pytest.raises(EmbedModelMismatch):
        await embedder.verify()


async def test_unreachable_on_health():
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = OpenAIEmbedder(
        base_url="http://testserver",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(dead)),
        retries=1,
        backoff_base=0.0,
    )
    with pytest.raises(EmbedServerUnreachable):
        await embedder.verify()


async def test_embed_documents_batch_preserves_order():
    server = FakeServer()
    embedder = _make_embedder(server, batch_size=2)
    texts = [f"text {i}" for i in range(5)]
    vectors = await embedder.embed_documents(texts)
    assert len(vectors) == 5
    # Thứ tự giữ nguyên: gateway trả data theo index trong batch
    assert len(server.requests) == 3  # 2 + 2 + 1
    flat = [
        it if isinstance(it, str) else it["text"]
        for batch in server.requests
        for it in batch
    ]
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
        if request.url.path == "/health":
            return server.handler(request)
        body = __import__("json").loads(request.content.decode())
        items = body["input"] if isinstance(body["input"], list) else [body["input"]]
        if (
            len(items) == 1
            and isinstance(items[0], dict)
            and items[0].get("text") == "probe"
        ):
            return httpx.Response(200, json=_embed_body(1, dim=8))
        raise httpx.ConnectError("refused")

    embedder = OpenAIEmbedder(
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


async def test_response_model_change_detected_mid_stream():
    server = FakeServer()

    def switching(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings" and len(server.requests) >= 1:
            # Các request sau request đầu: server đã bị đổi model
            return httpx.Response(
                200, json=_embed_body(1, model_name="other-model")
            )
        return server.handler(request)

    embedder = OpenAIEmbedder(
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
    assert [i if isinstance(i, str) else i["text"] for i in server.requests[0]] == ["doc c"]


async def test_embed_query_uses_query_input_type():
    server = FakeServer()
    embedder = _make_embedder(server)
    await embedder.embed_query("câu hỏi")
    assert server.requests[0][0]["text"] == "câu hỏi"


async def test_embed_multimodal_sends_image_b64():
    server = FakeServer()
    embedder = _make_embedder(server)
    items = [
        "plain text",
        {"text": "caption", "image_b64": PNG_B64},
    ]
    vectors = await embedder.embed_documents(items)
    assert len(vectors) == 2
    sent = server.requests[0]
    assert sent[0] == "plain text"
    # gateway item: {"text", "image"} (base64, gateway tự sniff mime)
    assert sent[1] == {"text": "caption", "image": PNG_B64}


async def test_data_url_mime_detection():
    assert data_url_from_bytes(PNG_BYTES).startswith("data:image/png;base64,")
    jpeg_b64 = b"\xff\xd8\xff\xe0rest"
    assert data_url_from_bytes(jpeg_b64).startswith("data:image/jpeg;base64,")


async def test_embed_documents_truncates_oversize_text():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=100, image_token_reserve=50)
    vectors = await embedder.embed_documents(["a" * 1000, "short"])
    assert len(vectors) == 2
    sent = server.requests[0]
    # budget 100 token * 3 chars/token
    assert sent[0] == "a" * 300
    assert sent[1] == "short"


async def test_embed_documents_truncates_multimodal_with_image_reserve():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=300, image_token_reserve=100)
    await embedder.embed_documents([{"text": "b" * 1000, "image_b64": PNG_B64}])
    item = server.requests[0][0]
    # budget (300 - 100) * 3 chars/token; ảnh giữ nguyên
    assert item["image"] == PNG_B64
    assert item["text"] == "b" * 600


async def test_embed_documents_keeps_items_within_budget_untouched():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=100)
    text = "a" * 300  # == budget: không cắt
    await embedder.embed_documents([text])
    assert server.requests[0][0] == text


async def test_embed_query_truncates_oversize_query():
    server = FakeServer()
    embedder = _make_embedder(server, max_input_tokens=100)
    await embedder.embed_query("q" * 500)
    assert server.requests[0][0]["text"] == "q" * 300


async def test_clamp_floor_prevents_degenerate_short_text():
    from ami_rag.core.embedder import clamp_embed_text

    # budget rất nhỏ bị floor 256 chars (chống text degenerate khi embed)
    assert clamp_embed_text("a" * 1000, 10) == "a" * 256
    # kèm ảnh trừ hết budget -> vẫn floor 256
    assert clamp_embed_text("a" * 1000, 50, has_image=True, image_tokens=50) == "a" * 256
