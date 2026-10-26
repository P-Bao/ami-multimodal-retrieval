"""Parity test: VinternEmbedder của ami_rag ↔ vintern-embedding-server (app thật).

Chạy server app THẬT (import từ repo máy B ``vintern-embedding-server/app/main.py``)
qua httpx ASGI transport (không cần mạng/GPU), model giả inject vào ``state``
(theo pattern ``tests/test_server.py`` của server). Flow: handshake /info ->
embed_query multi-vector -> embed_documents (text + image_b64 image-only) ->
chặn mismatch model/dim.

Skip nếu repo server không có trên máy này (máy dev/CI khác) hoặc thiếu deps.
"""

import asyncio
import base64
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import numpy as np
import pytest

from ami_rag.core.embedder import EmbedModelMismatch
from ami_rag.core.vintern_embedder import VinternEmbedder

SERVER_REPO = Path(r"D:\Code\Python\Ami\RAG\vintern-embedding-server")

MODEL = "5CD-AI/Vintern-Embedding-1B"
DIM = 128  # dim thật: custom_text_proj 896 -> 128


def _load_server():
    """Import app thật; trả None nếu repo/deps thiếu (skip thay vì error)."""
    if not (SERVER_REPO / "app" / "main.py").is_file():
        return None
    try:
        sys.path.insert(0, str(SERVER_REPO))
        return importlib.import_module("app.main")
    except Exception:  # noqa: BLE001 - bất kỳ lỗi import nào -> skip, không error
        if str(SERVER_REPO) in sys.path:
            sys.path.remove(str(SERVER_REPO))
        return None


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[0] * (len(text.split()) * 2))


class FakeModel:
    """Model giả multi-vector: mỗi item -> mảng [n_tokens, dim]."""

    TOKENS: ClassVar[dict[str, int]] = {"query": 3, "doc_text": 4, "image": 5}

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.processor = SimpleNamespace(tokenizer=FakeTokenizer())
        self.seen_groups: list[tuple[str, int]] = []

    def encode_batch(self, kind: str, items: list):
        self.seen_groups.append((kind, len(items)))
        return [
            np.full((self.TOKENS[kind], self.dim), 0.5, dtype=np.float32)
            for _ in items
        ]


def _tiny_png_b64() -> str:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), color=(200, 100, 50)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run_server_case():
    server = _load_server()
    if server is None:
        pytest.skip("repo vintern-embedding-server không nạp được trên máy này")

    model = FakeModel()
    server.state.model = model
    server.state.ready = True
    server.state.dim = model.dim
    server.settings.api_key = None
    try:
        asyncio.run(_case(server, model))
    finally:
        server.state.model = None
        server.state.ready = False


async def _case(server, model: FakeModel) -> None:
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        # 1. Handshake: model + dim + flag multivector
        emb = VinternEmbedder(
            "http://testserver",
            model=MODEL,
            expected_dim=DIM,
            http_client=client,
            batch_size=8,
        )
        info = await emb.verify()
        assert info["model_name"] == MODEL
        assert info["dim"] == DIM
        assert info["multivector"] is True

        # 2. embed_query -> multi-vector [n_tokens, dim]
        q = await emb.embed_query("Cảng Hải Phòng ở đâu ?")
        assert len(q) == FakeModel.TOKENS["query"]
        assert len(q[0]) == DIM

        # 3. embed_documents: text + image + plain str, giữ thứ tự
        docs = await emb.embed_documents(
            [
                {"type": "document", "text": "chunk văn bản"},
                {"type": "document", "image_b64": _tiny_png_b64()},
                "plain text item",
            ]
        )
        assert len(docs) == 3
        assert len(docs[0]) == FakeModel.TOKENS["doc_text"] and len(docs[0][0]) == DIM
        assert len(docs[1]) == FakeModel.TOKENS["image"]
        assert len(docs[2]) == FakeModel.TOKENS["doc_text"]
        kinds = [kind for kind, _ in model.seen_groups]
        assert "image" in kinds and "doc_text" in kinds

        # 4. Mismatch model -> chặn trước khi ghi vector
        bad = VinternEmbedder(
            "http://testserver",
            model="Qwen/Qwen3-VL-Embedding-2B",
            http_client=client,
        )
        with pytest.raises(EmbedModelMismatch):
            await bad.verify()

        # 5. Mismatch dim -> chặn
        bad_dim = VinternEmbedder(
            "http://testserver", expected_dim=64, http_client=client
        )
        with pytest.raises(EmbedModelMismatch):
            await bad_dim.verify()


def test_parity_vintern_embedder_with_real_server():
    _run_server_case()
