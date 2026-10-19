"""VectorPipeline - pipeline vector thuần thay thế LightRAG.

Các stage (ghi DocStatusStore ở MỖI stage):
    parse -> describe -> chunk -> embed -> indexed

- parse: giữ nguyên parser (MinerU/...) qua `get_parser().parse_document`;
  assets (ảnh/table/equation render) + content_list lưu MinIO.
- describe: với item multimodal dùng modal processor sinh mô tả; lưu mô tả vào
  MinIO (`{doc_id}/descriptions.json`). Đây là chỗ duy nhất có thể gọi LLM
  (modal processors, tuỳ chọn tắt).
- chunk: text chunk bằng plain splitter; multimodal chunk bằng template
  (Nội dung = mô tả đã lưu, không bao giờ gọi LLM ở stage này); lưu
  `{doc_id}/chunks.json`.
- embed: RemoteEmbedder/OpenAIEmbedder (máy B, chọn theo EMBED_BACKEND) với
  cache; luôn delete_by_doc trước upsert (idempotent). KHÔNG tạo entity,
  KHÔNG ghi vào graph.
- indexed: mark_indexed + meta. Verify: count vector trong Qdrant ==
  số chunk đã lưu.

Data giữa các stage nằm trong MinIO: `{doc_id}/content_list.json`,
`{doc_id}/descriptions.json`, `{doc_id}/chunks.json`.
- `dry_run`: handshake (GET /info) + đếm chunk/cache hits; KHÔNG gọi /embed,
  KHÔNG ghi gì.
"""

import asyncio
import hashlib
import itertools
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ami_rag.core.embedder import collection_name
from ami_rag.core.pipeline import PipelineRunner, StageOutcome
from ami_rag.core.vector_store import ChunkRecord, VectorStore, VectorStoreError
from ami_rag.observability import (
    INGEST_ITEMS_TOTAL,
    INGEST_PAGES_TOTAL,
    INGEST_PARSE_FAILURES_TOTAL,
)
from ami_rag.settings import parser_kwargs, resolve_image_describe_mode
from ami_rag.sources import (
    SOURCE_MINIO_PARSE,
    resolve_file_path,
    select_source,
    source_hash,
)
from ami_rag.storage.doc_status import (
    STAGE_CHUNK,
    STAGE_DESCRIBE,
    STAGE_EMBED,
    STAGE_INDEXED,
    STAGE_PARSE,
    DocStatusStore,
)
from ami_rag.textnorm import with_title

logger = logging.getLogger(__name__)

# Alias stage hiển thị dev-facing (có indexed là kết thúc)
PROCESSING_STAGES = ["parse", "describe", "chunk", "embed"]


class ArtifactPaths:
    """Path helper cho artifacts trung gian trong MinIO (theo key convention của asset store)."""

    @staticmethod
    def descriptions_key(asset_store, doc_id: str) -> str:
        return f"{asset_store.doc_prefix(doc_id)}descriptions.json"

    @staticmethod
    def chunks_key(asset_store, doc_id: str) -> str:
        return f"{asset_store.doc_prefix(doc_id)}chunks.json"


def _count_by_type(content_list: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in content_list or []:
        t = item.get("type") or "other"
        counts[t] = counts.get(t, 0) + 1
    return counts


def _page_count(content_list: list[dict]) -> int:
    pages = [item.get("page_idx", 0) for item in content_list if item.get("page_idx") is not None]
    return max(pages) + 1 if pages else 0


def _normalize_item_type(t: str | None) -> str:
    return t if t in ("text", "image", "table", "equation", "audio", "video") else "generic"


@dataclass
class Chunk:
    """Chunk đã build (trước embed)."""

    id: str  # chunk_id
    content: str
    modality: str  # text|image|table|equation|audio|video|generic
    page_idx: int | None
    is_multimodal: bool
    source_path: str = ""
    asset_key: str | None = None
    table_body: str | None = None
    caption: str | None = None
    extra: dict = field(default_factory=dict)

    def payload(self, doc_id: str, *, source_path: str = "") -> dict:
        p = {
            "doc_id": doc_id,
            "modality": self.modality,
            "page": self.page_idx,
            "source_path": source_path or self.source_path,
            "chunk_id": self.id,
            "is_multimodal": self.is_multimodal,
            "content": self.content,
        }
        if self.asset_key:
            p["asset_key"] = self.asset_key
        if self.table_body:
            p["table_body"] = self.table_body
        if self.caption:
            p["caption"] = self.caption
        p.update(self.extra)
        return p

    def to_json(self, doc_id: str, source_path: str = "") -> dict:
        return {
            "chunk_id": self.id,
            "content": self.content,
            "modality": self.modality,
            "page_idx": self.page_idx,
            "is_multimodal": self.is_multimodal,
            "source_path": source_path or self.source_path,
            "asset_key": self.asset_key,
            "table_body": self.table_body,
            "caption": self.caption,
            "doc_id": doc_id,
            "extra": self.extra,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Chunk":
        return cls(
            id=data["chunk_id"],
            content=data["content"],
            modality=data.get("modality", "text"),
            page_idx=data.get("page_idx"),
            is_multimodal=data.get("is_multimodal", False),
            source_path=data.get("source_path", ""),
            asset_key=data.get("asset_key"),
            table_body=data.get("table_body"),
            caption=data.get("caption"),
            extra=data.get("extra", {}),
        )


def _chunk_id(doc_id: str, content: str, index: int | None = None) -> str:
    seed = f"{doc_id}#{index}#{content}" if index is not None else f"{doc_id}{content}"
    return f"chunk-{hashlib.md5(seed.encode()).hexdigest()[:24]}"


def _modal_text(item: dict, type_: str) -> str:
    """Text dùng được của item modal (bảng/công thức) — rỗng nghĩa là không index."""
    if type_ == "table":
        return _join_caption(
            item.get("table_body") or item.get("text") or item.get("table_caption") or ""
        ).strip()
    if type_ == "equation":
        return _join_caption(item.get("text") or item.get("latex") or "").strip()
    return ""


def _format_modal_chunk(type_: str, item: dict, description: str) -> str:
    """Template chunk cho item multimodal (không ghi entity, không gọi LLM)."""
    if type_ == "image":
        captions = _join_caption(item.get("image_caption") or item.get("img_caption"))
        footnotes = _join_caption(item.get("image_footnote") or item.get("img_footnote"))
        parts = [f"[Image Content]\n{description}"]
        # Không ghi "Tóm tắt: Không có" - ảnh crawl không có caption nên dòng này
        # chỉ là nhiễu cho mọi chunk ảnh.
        if captions:
            parts.append(f"Tóm tắt: {captions}")
        if footnotes:
            parts.append(footnotes)
        return "\n".join(parts).strip()
    if type_ == "table":
        caption = _join_caption(item.get("table_caption"))
        # Không rơi về str(item): item MinerU thiếu table_body sẽ in thẳng dict-repr
        # (kèm bbox/path tạm) vào chunk. Chunk không có body thì bị bỏ ở
        # `_build_chunks` (xem `_modal_text`).
        body = _join_caption(item.get("table_body") or item.get("text") or "")
        parts = ["[Table Content]", f"Caption: {caption or 'Không có'}"]
        # Mô tả rỗng (LLM không với được) thì bỏ hẳn dòng, không để "Mô tả:" trơ trọng.
        if description.strip():
            parts.append(f"Mô tả: {description}")
        parts.append(body)
        return "\n".join(parts).strip()
    if type_ == "equation":
        text = _join_caption(item.get("text") or item.get("latex") or "")
        head = "[Equation]"
        if description.strip():
            head += f"\nMô tả: {description}"
        return f"{head}\nCông thức: {text}".strip()
    if type_ == "audio":
        return f"[Audio Content]\n{description}"
    if type_ == "video":
        return f"[Video Content]\n{description}"
    return f"[{_normalize_item_type(type_).title()}]\n{description}"


def _join_caption(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


# Chunk ngắn hơn ngưỡng này bị gộp vào chunk kề. Doc crawl viết mỗi trường một
# dòng ("- Địa chỉ: Tầng 1 nhà A1, ...") và backend_data tách PDF theo từng
# dòng, nên trước đây xuất hiện nhiều chunk mồ côi đúng một dòng: khớp mọi query
# "địa chỉ / cơ sở Hà Đông" nhưng không rõ thuộc đơn vị nào.
MIN_CHUNK_CHARS = 200

# Overlap chỉ gắn khi còn đủ chỗ cho phần thân chunk (>= ngưỡng này), tránh cắt
# nội dung chỉ để nhồi overlap.
_MIN_OVERLAP_ROOM = 200


def _split_units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Tách text thành các đơn vị <= max_chars.

    Trả ``[(nội dung, sep_giữa_với_đơn_vị_trước), ...]`` với ``sep`` là
    ``"\\n\\n"`` (hết đoạn) hoặc ``"\\n"`` (vẫn trong đoạn) để giữ nguyên ranh giới
    đoạn khi gom lại.

    - Đoạn (tách bằng dòng trống) vừa budget thì giữ nguyên làm một đơn vị.
    - Đoạn quá dài thì gom NHIỀU dòng kế tiếp tới budget (bản cũ cắt từng dòng
      một — nguyên nhân gốc của chunk mồ côi).
    - Chỉ dòng đơn dài hơn cả budget mới bị cắt cứng.
    """
    units: list[tuple[str, str]] = []

    def _add(content: str, sep: str) -> None:
        units.append((content, sep if units else ""))

    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            _add(block, "\n\n")
            continue
        current: list[str] = []
        size = 0

        def _flush_lines() -> None:
            nonlocal current, size
            if current:
                _add("\n".join(current), "\n")
                current, size = [], 0

        for line in (ln.strip() for ln in block.splitlines()):
            if not line:
                continue
            if len(line) > max_chars:
                _flush_lines()
                for i in range(0, len(line), max_chars):
                    _add(line[i : i + max_chars], "\n")
                continue
            if size + len(line) + 1 > max_chars and current:
                _flush_lines()
            current.append(line)
            size += len(line) + 1
        _flush_lines()
    return units


def _pack_units(units: list[tuple[str, str]], max_chars: int) -> list[str]:
    """Gom các đơn vị vào chunk tới budget (đơn vị không bị cắt đôi)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for content, sep in units:
        if current and size + len(content) + len(sep) > max_chars:
            chunks.append("".join(current))
            current, size = [], 0
            sep = "\n\n" if not sep else sep
        current.append(sep + content)
        size += len(sep) + len(content)
    if current:
        chunks.append("".join(current))
    return chunks


def _apply_overlap(chunks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Chèn đuôi chunk trước vào đầu chunk sau (rồi cắt phần thân cho vừa budget)."""
    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in itertools.pairwise(chunks):
        tail = prev[-overlap_chars:].lstrip()
        room = max_chars - len(tail) - 1
        if not tail or room < _MIN_OVERLAP_ROOM:
            out.append(cur)
            continue
        body = cur if len(cur) <= room else cur[:room]
        out.append((tail + "\n" + body).strip())
    return out


def _merge_small(chunks: list[str], max_chars: int, min_chars: int) -> list[str]:
    """Gộp chunk ngắn (< min_chars) vào chunk trước khi vẫn còn chỗ."""
    if min_chars <= 0 or min_chars >= max_chars or len(chunks) < 2:
        return chunks
    out: list[str] = []
    pending: str | None = None
    for cur in chunks:
        if pending is None:
            pending = cur
            continue
        if len(pending) < min_chars and len(pending) + 1 + len(cur) <= max_chars:
            pending = f"{pending}\n{cur}"
            continue
        out.append(pending)
        pending = cur
    if pending is not None:
        out.append(pending)
    return out


def chunk_text(
    text: str,
    max_tokens: int = 1200,
    overlap_tokens: int = 100,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Chia text thành chunk theo ngân sách token xấp xỉ (~4 chars/token).

    V1 (nay): tách theo DÒNG trống, đoạn quá dài gom nhiều dòng, và mọi chunk
    ngắn hơn ``min_chars`` được gộp vào chunk kề — để không sinh chunk mồ côi
    kiểu "- Địa chỉ: Tầng 1 nhà A1" (khớp query địa chỉ nhưng vô nghĩa).
    """
    max_chars = max(512, max_tokens * 4)
    overlap_chars = overlap_tokens * 4
    if not text or not text.strip():
        return [text]
    chunks = _apply_overlap(
        _pack_units(_split_units(text, max_chars), max_chars), max_chars, overlap_chars
    )
    chunks = _merge_small(chunks, max_chars, min_chars)
    return chunks or [text]


@dataclass
class IntermediateDoc:
    """Artifacts trung gian của một doc ở từng stage."""

    doc_id: str
    content_list: list[dict] | None = None
    file_path: str = ""


class VectorPipeline(PipelineRunner):
    """Pipeline vector thuần (không torch/transformers phía máy A)."""

    def __init__(
        self,
        settings,
        *,
        embedder,
        vector_store: VectorStore,
        asset_store,
        store: DocStatusStore,
        docs_repo=None,
        parser=None,
        modal_processors=None,
        llm_func=None,
        image_worker=None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.asset_store = asset_store
        self.store = store
        self.docs_repo = docs_repo
        self.parser = parser
        self.modal_processors = modal_processors or {}
        # LLM text-only (table/equation/generic processors); ảnh caption qua
        # Gemini (CAPTION_LLM_*) hoặc vision func ngay trong modal processors.
        self.llm_func = llm_func
        # CrawlImageWorker: tải ảnh section 'Hình ảnh' của doc crawl về MinIO
        self.image_worker = image_worker
        self._bd_extractor = None
        # Giới hạn song song cho caption ảnh (fetch MinIO + gọi caption func)
        self._caption_sem: asyncio.Semaphore | None = None
        # Giới hạn song song tải ảnh crawl (băng thông + MinIO, độc lập caption)
        self._crawl_dl_sem: asyncio.Semaphore | None = None
        self.collection = collection_name(
            settings.WORKSPACE, settings.EMBED_MODEL, settings.CHUNKER_VERSION
        )

    # ------------------------------------------------------------------
    # Doc / JSON I/O (qua asset store)
    # ------------------------------------------------------------------
    def _load_doc(self, doc_id: str) -> dict | None:
        if self.docs_repo is None:
            return None
        try:
            return self.docs_repo.find_by_id(doc_id)
        except Exception as exc:
            logger.warning("load doc %s lỗi: %s", doc_id, exc)
            return None

    def _load_json(self, key: str) -> Any:
        return self.asset_store.load_json(key)

    def _save_json(self, key: str, data: Any) -> None:
        self.asset_store.save_json(key, data)

    def _load_descriptions(self, doc_id: str) -> list[dict] | None:
        stored = self._load_json(ArtifactPaths.descriptions_key(self.asset_store, doc_id))
        return stored if isinstance(stored, list) else None

    # ------------------------------------------------------------------
    # Stage: parse (giữ nguyên parser đã dùng)
    # ------------------------------------------------------------------
    async def _stage_parse(self, doc_id: str, doc: dict) -> IntermediateDoc:
        if doc is None:
            raise ValueError(
                f"doc {doc_id} không có bản ghi trong documents "
                "(docs_repo thiếu hoặc document đã bị xoá); không thể parse"
            )
        if self.parser is None:
            from raganything.parser import get_parser

            self.parser = get_parser(self.settings.PARSER)

        source = select_source(doc)
        content_list: list[dict]
        file_path = ""
        assets: list[str] = []

        if source == SOURCE_MINIO_PARSE:
            file_path = doc.get("file_path") or ""
            with tempfile.TemporaryDirectory() as tmp_dir:
                local = await asyncio.to_thread(
                    self.asset_store.fetch, file_path, Path(tmp_dir)
                )
                content_list = await asyncio.to_thread(
                    self.parser.parse_document,
                    str(local),
                    self.settings.PARSE_METHOD,
                    tmp_dir,
                    **parser_kwargs(self.settings),
                )
                if (
                    self.settings.PARSER == "mineru"
                    and getattr(self.settings, "PARSE_TEXT_SOURCE", "mineru") == "backend_data"
                ):
                    content_list = await self._rebuild_backend_data(
                        doc_id, content_list, local
                    )
                assets = await asyncio.to_thread(
                    self.asset_store.upload_content_list_assets, doc_id, content_list
                )
        else:
            text = doc.get("content") or ""
            content_list = await self._crawl_text_content_list(doc_id, text)
            file_path = resolve_file_path(doc) or ""

        counts = _count_by_type(content_list)
        page_count = _page_count(content_list)
        for item_type, n in counts.items():
            INGEST_ITEMS_TOTAL.labels(modality=_normalize_item_type(item_type)).inc(n)
        INGEST_PAGES_TOTAL.inc(page_count)
        await asyncio.to_thread(self.asset_store.save_content_list, doc_id, content_list)
        self.store.mark_stage(
            doc_id,
            STAGE_PARSE,
            source_path=file_path or doc_id,
            content_hash=source_hash(doc, source),
            source=source,
            parser=self.settings.PARSER,
            file_path=file_path,
            counts=counts,
            page_count=page_count,
            assets=assets,
            document_type=doc.get("document_type"),
            title=doc.get("title"),
            organization_unit_id=doc.get("organization_unit_id"),
            owner_id=doc.get("owner_id"),
        )
        return IntermediateDoc(doc_id=doc_id, content_list=content_list, file_path=file_path)

    def _get_bd_extractor(self):
        from raganything.backend_data_extract import create_extractor

        if self._bd_extractor is None:
            self._bd_extractor = create_extractor(self.settings)
        return self._bd_extractor

    # ------------------------------------------------------------------
    # Crawl text: 1 text item + ảnh section 'Hình ảnh' (tuỳ chọn)
    # ------------------------------------------------------------------
    _CRAWL_DOWNLOAD_CONCURRENCY = 4

    def _get_image_worker(self):
        """CrawlImageWorker (luôn bật - không có env tắt).

        Chưa inject thì lazy-build từ settings. Prefix = ASSET_PREFIX để
        asset_key nằm dưới doc prefix: ``delete_doc_assets`` dọn theo doc.
        """
        if self.image_worker is None:
            from ami_rag.workers.crawl_images import CrawlImageWorker

            self.image_worker = CrawlImageWorker(
                endpoint=self.settings.MINIO_ENDPOINT,
                access_key=self.settings.MINIO_ACCESS_KEY,
                secret_key=self.settings.MINIO_SECRET_KEY,
                bucket=self.settings.MINIO_BUCKET,
                prefix=self.settings.ASSET_PREFIX,
                secure=self.settings.MINIO_SECURE,
            )
        return self.image_worker

    def _get_crawl_dl_semaphore(self) -> asyncio.Semaphore:
        if self._crawl_dl_sem is None:
            self._crawl_dl_sem = asyncio.Semaphore(self._CRAWL_DOWNLOAD_CONCURRENCY)
        return self._crawl_dl_sem

    async def _crawl_text_content_list(self, doc_id: str, text: str) -> list[dict]:
        """Content_list doc mongo_text (text/crawl): 1 text item + ảnh crawl.

        Tách section 'Hình ảnh' khỏi text (hàm ``split_image_section``),
        tải các ảnh về MinIO (asset_key nằm dưới doc prefix) và append
        item image — ảnh crawl đi tiếp qua describe/chunk/embed như ảnh
        của doc/pdf.

        - Text mới là text đã strip section; strip rỗng thì bỏ text item.
        - Không có section -> đúng hành vi cũ: 1 text item (không dựng
          worker, không tải gì).
        - Lỗi tải từng ảnh bỏ qua (không crash parse); không tải được ảnh
          nào -> giữ nguyên text gốc (không mất URL khỏi index).
        """
        from ami_rag.workers.crawl_images import split_image_section

        urls, stripped = split_image_section(text)
        if not urls:
            return [{"type": "text", "text": text, "page_idx": 0}]
        worker = self._get_image_worker()
        sem = self._get_crawl_dl_semaphore()

        async def _one(url: str) -> tuple[str, str | None]:
            async with sem:
                return url, await worker.ensure_in_minio(doc_id, url)

        results = await asyncio.gather(*(_one(u) for u in urls))
        ok = [(u, key) for u, key in results if key]
        if not ok:
            logger.warning(
                "crawl doc=%s: không tải được ảnh nào (%d URL, giữ text gốc)",
                doc_id,
                len(urls),
            )
            return [{"type": "text", "text": text, "page_idx": 0}]
        if len(ok) < len(urls):
            logger.warning(
                "crawl doc=%s: tải được %d/%d ảnh, %d URL lỗi bị bỏ qua",
                doc_id,
                len(ok),
                len(urls),
                len(urls) - len(ok),
            )
        content_list: list[dict] = []
        if stripped.strip():
            content_list.append({"type": "text", "text": stripped, "page_idx": 0})
        content_list.extend(
            {
                "type": "image",
                "asset_key": key,
                "image_caption": [],
                "image_footnote": [],
                "page_idx": 0,
            }
            for _, key in ok
        )
        logger.info(
            "crawl doc=%s: %d ảnh tải về MinIO (%d text item)",
            doc_id,
            len(ok),
            len(content_list) - len(ok),
        )
        return content_list

    async def _rebuild_backend_data(
        self, doc_id: str, content_list: list[dict], local: Path
    ) -> list[dict]:
        """Thay text/table của MinerU bằng pdfplumber/fitz + OCR endpoint + bảng
        Qwen endpoint (phong cách backend_data); ảnh MinerU giữ nguyên.
        Thất bại -> fallback MinerU content_list."""
        from raganything.backend_data_extract import rebuild_content_list

        try:
            rebuilt = await rebuild_content_list(
                content_list, local, self._get_bd_extractor()
            )
        except Exception as exc:
            logger.warning(
                "backend_data text/table parse doc=%s lỗi, giữ MinerU content_list: %s",
                doc_id,
                exc,
            )
            return content_list
        if rebuilt is None:
            return content_list
        logger.info(
            "backend_data parse doc=%s: %d items (MinerU) -> %d items (text/table backend_data + ảnh MinerU)",
            doc_id,
            len(content_list),
            len(rebuilt),
        )
        return rebuilt

    # ------------------------------------------------------------------
    # Stage: describe (LLM – duy nhất chỗ mô tả modal)
    # ------------------------------------------------------------------
    async def _stage_describe(
        self, doc_id: str, content_list: list[dict], doc_title: str = ""
    ) -> list[dict]:
        multimodal_indices = [
            i
            for i, item in enumerate(content_list)
            if item.get("type") in ("image", "table", "equation", "audio", "video", "generic")
        ]
        if not multimodal_indices:
            self._save_json(
                ArtifactPaths.descriptions_key(self.asset_store, doc_id), []
            )
            self.store.mark_stage(doc_id, STAGE_DESCRIBE)
            return []

        # Context source cho processor: trích ngữ cảnh xung quanh item (cùng trang)
        for processor in self.modal_processors.values():
            set_source = getattr(processor, "set_content_source", None)
            if callable(set_source):
                set_source(content_list, "minerU")

        # Materialize ảnh MinIO -> file tạm (img_path) cho caption Gemini/vision.
        # describe_items là shallow copy; content_list gốc KHÔNG bị mutate
        # (stage chunk + save_content_list dùng bản gốc).
        describe_items, temp_dir = await self._materialize_images(
            content_list, multimodal_indices
        )
        try:
            async def _one(i: int) -> dict:
                item = describe_items[i]
                type_ = _normalize_item_type(item.get("type"))
                try:
                    desc_text = await self._describe_item(
                        type_, item, content_list, i, doc_id, doc_title
                    )
                except Exception as exc:
                    logger.warning("describe doc=%s item=%d lỗi: %s", doc_id, i, exc)
                    desc_text = ""
                return {"index": i, "type": type_, "description": desc_text}

            descriptions = list(
                await asyncio.gather(*(_one(i) for i in multimodal_indices))
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
        self._save_json(
            ArtifactPaths.descriptions_key(self.asset_store, doc_id), descriptions
        )
        self.store.mark_stage(doc_id, STAGE_DESCRIBE)
        return descriptions

    # ------------------------------------------------------------------
    # Materialize ảnh MinIO -> file tạm cho caption (Gemini / vision)
    # ------------------------------------------------------------------
    def _get_caption_semaphore(self) -> asyncio.Semaphore:
        if self._caption_sem is None:
            limit = max(
                1, int(getattr(self.settings, "CAPTION_LLM_MAX_CONCURRENCY", 4))
            )
            self._caption_sem = asyncio.Semaphore(limit)
        return self._caption_sem

    @staticmethod
    def _image_ext(data: bytes, asset_key: str) -> str:
        """Ext file ảnh từ magic bytes; fallback suffix asset_key, cuối cùng .png."""
        if data.startswith(b"\x89PNG"):
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:3] == b"GIF":
            return ".gif"
        if data.startswith(b"\xff\xd8"):
            return ".jpg"
        suffix = Path(asset_key or "").suffix
        return suffix if suffix else ".png"

    async def _materialize_images(
        self, content_list: list[dict], indices: list[int]
    ) -> tuple[list[dict], Any]:
        """Tải ảnh image-item từ MinIO về temp dir, gán ``img_path`` cho describe.

        Trả ``(describe_items, temp_dir)``:
        - mode none (hoặc không có ảnh/asset_key): ``(content_list, None)``
          - dùng thẳng bản gốc, không copy, không temp dir.
        - ngược lại: describe_items là shallow copy từng item; image-item có
          asset_key được gán img_path = file tạm (item thiếu asset_key giữ
          img_path gốc). Caller PHẢI ``temp_dir.cleanup()`` sau khi describe xong.

        Fetch song song theo semaphore CAPTION_LLM_MAX_CONCURRENCY; lỗi/bytes
        rỗng từng ảnh -> bỏ qua ảnh đó (không crash cả stage).
        """
        mode = resolve_image_describe_mode(self.settings)
        if mode not in ("gemini", "vision") or not hasattr(self.asset_store, "get_bytes"):
            return content_list, None

        image_indices = [
            i for i in indices if content_list[i].get("type") == "image"
        ]
        keys: list[str] = []
        seen: set[str] = set()
        for i in image_indices:
            key = content_list[i].get("asset_key")
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        if not keys:
            return content_list, None

        temp_dir = tempfile.TemporaryDirectory(prefix="ami-rag-img-")
        sem = self._get_caption_semaphore()

        async def _fetch(key: str) -> tuple[str, str | None]:
            async with sem:
                try:
                    data = await asyncio.to_thread(self.asset_store.get_bytes, key)
                except Exception as exc:
                    logger.warning("fetch ảnh %s lỗi (bỏ qua materialize): %s", key, exc)
                    return key, None
            if not data:
                return key, None
            name = hashlib.sha1(key.encode()).hexdigest()[:12] + self._image_ext(data, key)
            path = Path(temp_dir.name) / name
            try:
                await asyncio.to_thread(path.write_bytes, data)
            except Exception as exc:
                logger.warning("ghi ảnh %s ra temp lỗi (bỏ qua): %s", key, exc)
                return key, None
            return key, str(path)

        key_to_path: dict[str, str] = {}
        for key, path in await asyncio.gather(*(_fetch(k) for k in keys)):
            if path:
                key_to_path[key] = path
        if not key_to_path:
            temp_dir.cleanup()
            return content_list, None

        describe_items = [dict(it) for it in content_list]
        for i in image_indices:
            path = key_to_path.get(content_list[i].get("asset_key"))
            if path:
                describe_items[i]["img_path"] = path
        return describe_items, temp_dir

    # ------------------------------------------------------------------
    # Describe 1 item
    # ------------------------------------------------------------------
    def _describe_item_info(
        self, item: dict, index: int, type_: str, doc_title: str = ""
    ) -> dict:
        """``item_info`` cho modal processor.

        ``caption_language`` + ``doc_title`` là tùy chọn (thiếu thì processor bỏ
        qua): buộc mô tả ảnh theo ngôn ngữ corpus/truy vấn và có tiêu đề tài
        liệu làm ngữ cảnh, đồng thời nhắc ghi nguyên văn mọi chữ trong ảnh.
        """
        info = {
            "page_idx": item.get("page_idx", 0),
            "index": index,
            "type": type_,
        }
        language = (getattr(self.settings, "CAPTION_LLM_LANGUAGE", "") or "").strip()
        if language:
            info["caption_language"] = language
        if doc_title:
            info["doc_title"] = doc_title
        return info

    async def _describe_item(
        self,
        type_: str,
        item: dict,
        content_list: list[dict],
        index: int = 0,
        doc_id: str = "",
        doc_title: str = "",
    ) -> str:
        if type_ == "image":
            mode = resolve_image_describe_mode(self.settings)
            if mode == "none":
                return self._describe_image_raw(item, content_list, index)
            processor = (
                self.modal_processors.get("image") or self.modal_processors.get("generic")
            )
            if processor is None:
                return self._describe_image_raw(item, content_list, index)
            desc = ""
            try:
                async with self._get_caption_semaphore():
                    sections = await processor.generate_chunk_sections(
                        modal_content=item,
                        content_type=type_,
                        item_info=self._describe_item_info(item, index, type_, doc_title),
                        entity_name=None,
                    )
                desc = str(sections[0]["description"]) if sections else ""
            except Exception as exc:
                logger.warning(
                    "describe ảnh doc=%s item=%d lỗi (%s) -> text thô",
                    doc_id,
                    index,
                    type(exc).__name__,
                )
                desc = ""
            # ImageModalProcessor nuốt lỗi nội bộ và trả str(modal_content)
            # (dict repr) - phát hiện bằng so khớp để rơi về text thô.
            if not desc.strip() or desc.strip() == str(item).strip():
                logger.warning(
                    "describe ảnh doc=%s item=%d: caption func thất bại/rỗng -> text thô",
                    doc_id,
                    index,
                )
                return self._describe_image_raw(item, content_list, index)
            return desc.strip()

        processor = self.modal_processors.get(type_) or self.modal_processors.get("generic")
        if processor is None:
            return _join_caption(item.get("text") or item.get("table_body") or "")
        try:
            sections = await processor.generate_chunk_sections(
                modal_content=item,
                content_type=type_,
                item_info=self._describe_item_info(item, index, type_, doc_title),
                entity_name=None,
            )
            desc = str(sections[0]["description"]) if sections else ""
        except Exception as exc:
            logger.warning(
                "describe %s doc=%s item=%d lỗi (%s) -> text thô",
                type_,
                doc_id,
                index,
                type(exc).__name__,
            )
            return _join_caption(item.get("text") or item.get("table_body") or "")
        # Processor nuốt lỗi nội bộ rồi trả str(modal_content) (dict-repr của item)
        # khi caption func chết (LLM không với được) — nhận ra qua so khớp, không
        # ghi dict-repr vào descriptions.json (lộ path tạm, rác cho vector).
        if desc.strip() == str(item).strip():
            logger.warning(
                "describe %s doc=%s item=%d: caption func thất bại/rỗng -> text thô",
                type_,
                doc_id,
                index,
            )
            return _join_caption(item.get("text") or item.get("table_body") or "")
        return desc.strip()

    def _describe_image_raw(
        self, item: dict, content_list: list[dict], index: int
    ) -> str:
        """Mô tả ảnh bằng text thô (không LLM): caption + footnotes + ngữ cảnh."""
        caption = _join_caption(item.get("image_caption") or item.get("img_caption"))
        footnotes = _join_caption(item.get("image_footnote") or item.get("img_footnote"))
        context = self._extract_image_context(content_list, index)
        return "\n".join(p for p in (caption, footnotes, context) if p).strip()

    def _extract_image_context(
        self, content_list: list[dict], index: int, window: int = 1
    ) -> str:
        """Ngữ cảnh text xung quanh ảnh (cùng trang, ±window trang)."""
        processor = self.modal_processors.get("image") or self.modal_processors.get("generic")
        item_info = {
            "page_idx": content_list[index].get("page_idx", 0),
            "index": index,
            "type": "image",
        }
        getter = getattr(processor, "_get_context_for_item", None)
        if callable(getter):
            try:
                return getter(item_info)
            except Exception:
                pass
        # Fallback: tự trích text các trang lân cận khi processor không có context extractor
        page = content_list[index].get("page_idx") or 0
        parts = []
        for it in content_list:
            if it.get("type") != "text":
                continue
            it_page = it.get("page_idx") or 0
            if abs(it_page - page) > window:
                continue
            text = (it.get("text") or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts)[:4000]

    def _build_chunks(
        self,
        doc_id: str,
        content_list: list[dict],
        descriptions: list[dict] | None,
        *,
        source_path: str = "",
        doc_title: str = "",
    ) -> list[Chunk]:
        """Ghép content_list thành chunk (text gom theo dải liên tiếp).

        Với `PARSE_TEXT_SOURCE=backend_data`, content_list của PDF chứa **mỗi
        dòng một item** (``backend_data_extract.build_pdf_content_blocks``), và
        doc crawl viết mỗi trường một dòng. Bản cũ chunk từng item riêng nên mỗi
        dòng thành một chunk mồ côi không rõ thuộc đơn vị nào; nay gom các item
        text liên tiếp (nối bằng "\\n") rồi mới chia theo ngân sách. Buffer chỉ
        flush khi gặp item modal (image/table/...) để ảnh vẫn nằm cạnh text của nó.

        ``doc_title`` được prefix vào content chunk (trừ khi chunk đã bắt đầu bằng
        tiêu đề) — chunk kiểu "- Địa chỉ: Tầng 1 nhà A1" trở nên tự mô tả được.
        """
        desc_by_index = {}
        for d in descriptions or []:
            if d.get("description"):
                desc_by_index.setdefault(d["index"], []).append(d["description"])

        title = (doc_title or "").strip()
        chunks: list[Chunk] = []
        chunk_idx = 0
        buf: list[str] = []
        buf_page: Any = None
        buf_last_page: Any = None

        def _page(value: Any) -> Any:
            return 0 if value is None else value

        def _flush() -> None:
            nonlocal chunk_idx, buf, buf_page, buf_last_page
            if not buf:
                return
            text = "\n".join(buf)
            page = _page(buf_page)
            extra: dict = {"title": title} if title else {}
            if buf_last_page is not None and _page(buf_last_page) != page:
                extra["page_end"] = _page(buf_last_page)
            for piece in chunk_text(
                text,
                max_tokens=self.settings.CHUNK_SIZE,
                overlap_tokens=self.settings.CHUNK_OVERLAP,
                min_chars=getattr(self.settings, "CHUNK_MIN_CHARS", MIN_CHUNK_CHARS),
            ):
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc_id, piece, chunk_idx),
                        content=with_title(piece, title),
                        modality="text",
                        page_idx=page,
                        is_multimodal=False,
                        source_path=source_path,
                        extra=dict(extra),
                    )
                )
                chunk_idx += 1
            buf, buf_page, buf_last_page = [], None, None

        for i, item in enumerate(content_list):
            t = item.get("type") or "text"
            if t == "text":
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                if not buf:
                    buf_page = item.get("page_idx", 0)
                buf.append(text)
                buf_last_page = item.get("page_idx", 0)
                continue

            _flush()
            # multimodal: description from describe stage if present, else raw caption
            descs = desc_by_index.get(i) or []
            description = descs[0] if descs else ""
            if not description.strip() or description.strip() == str(item).strip():
                # Processor nuốt lỗi nội bộ và trả str(modal_content) (dict-repr của
                # item) khi caption func chết — không được để dict-repr lọt vào
                # chunk (lộ path tạm /tmp/... và làm loãng text embed). Rơi về
                # caption/text thô; `table_body` vẫn do _format_modal_chunk ghi.
                description = _join_caption(
                    item.get("image_caption")
                    or item.get("table_caption")
                    or item.get("latex")
                    or item.get("text")
                    or ""
                )
            content = _format_modal_chunk(t, item, description)
            # Item modal không có text nào dùng được (vd bảng MinerU thiếu
            # table_body) thì chunk chỉ còn tiêu đề + nhãn template -> bỏ hẳn,
            # tránh rác trong ứng viên rerank.
            if t in ("table", "equation") and not _modal_text(item, t) and not description.strip():
                logger.info(
                    "chunk: bỏ item %s không có nội dung text (page=%s)",
                    t,
                    item.get("page_idx"),
                )
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(doc_id, content, chunk_idx),
                    content=with_title(content, title),
                    modality=_normalize_item_type(t),
                    page_idx=item.get("page_idx", 0),
                    is_multimodal=True,
                    source_path=source_path,
                    asset_key=item.get("asset_key"),
                    table_body=item.get("table_body"),
                    caption=_join_caption(item.get("image_caption") or item.get("table_caption")) or None,
                    extra={"title": title} if title else {},
                )
            )
            chunk_idx += 1
        _flush()
        return chunks

    async def _stage_chunk(
        self,
        doc_id: str,
        content_list: list[dict],
        descriptions: list[dict] | None,
        *,
        source_path: str = "",
        doc_title: str = "",
    ) -> list[Chunk]:
        chunks = self._build_chunks(
            doc_id, content_list, descriptions, source_path=source_path, doc_title=doc_title
        )
        self._save_json(
            ArtifactPaths.chunks_key(self.asset_store, doc_id),
            [c.to_json(doc_id) for c in chunks],
        )
        self.store.mark_stage(
            doc_id, STAGE_CHUNK, chunk_count=len(chunks), source_path=source_path
        )
        return chunks

    async def _load_or_build_chunks(
        self,
        doc_id: str,
        content_list: list[dict] | None,
        descriptions: list[dict] | None,
        source_path: str,
        doc_title: str = "",
    ) -> list[Chunk]:
        """Resume từ stage embed: chunks.json nếu có, else build lại từ MinIO."""
        stored = await asyncio.to_thread(
            self._load_json, ArtifactPaths.chunks_key(self.asset_store, doc_id)
        )
        if isinstance(stored, list) and stored:
            return [Chunk.from_json(c) for c in stored]
        if content_list is None:
            content_list = await asyncio.to_thread(self.asset_store.load_content_list, doc_id)
        if content_list is None:
            raise FileNotFoundError(
                f"doc {doc_id}: thiếu chunks.json + content_list trong MinIO - cần parse lại"
            )
        if descriptions is None:
            descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
        return self._build_chunks(
            doc_id, content_list, descriptions, source_path=source_path, doc_title=doc_title
        )

    # ------------------------------------------------------------------
    async def _prepare_embed_items(self, chunks: list[Chunk]) -> list[str]:
        """Caption-only: luôn trả list[str] (c.content) — không nạp ảnh asset cho embed.

        (BGE-M3 không nhận ảnh; ảnh đi qua caption describe stage.)"""
        return [c.content for c in chunks]

    # ------------------------------------------------------------------
    # Stage: embed (máy B qua RemoteEmbedder) + verify (idempotent)
    # ------------------------------------------------------------------
    async def _stage_embed(
        self, doc_id: str, chunks: list[Chunk], *, source_path: str = ""
    ) -> None:
        if not chunks:
            raise RuntimeError(f"doc {doc_id} không có chunk nào để embed")
        seen_ids: set[str] = set()
        for idx, c in enumerate(chunks):
            if c.id in seen_ids:
                c.id = f"{c.id}-{idx}"
            seen_ids.add(c.id)
        # idempotency: xoá vector cũ của doc trước khi upsert
        await self.vector_store.delete_by_doc(self.collection, doc_id)
        items = await self._prepare_embed_items(chunks)
        if getattr(self.embedder, "embed_documents_sparse", None):
            dense, sparse = await self.embedder.embed_documents_sparse(items)  # không cache
        else:  # legacy embedder không có sparse
            dense = await self.embedder.embed_documents(items)
            sparse = [None] * len(items)
        if len(dense) != len(chunks):
            raise RuntimeError("embed server trả số vector không khớp số chunk")
        records = [
            ChunkRecord(
                id=c.id,
                vector=dense[i],
                # sparse rỗng ({}) cũng như None: không ghi sparse point
                sparse=sparse[i] if sparse[i] else None,
                payload=c.payload(doc_id, source_path=source_path),
            )
            for i, c in enumerate(chunks)
        ]
        await self.vector_store.upsert(self.collection, records)
        actual = await self.vector_store.count(self.collection, doc_id)
        if actual < len(chunks):
            raise RuntimeError(
                f"doc {doc_id}: verify count={actual} < chunk đã lưu={len(chunks)}"
            )
        self.store.mark_stage(doc_id, STAGE_EMBED, chunk_count=len(chunks))

    # ------------------------------------------------------------------
    # Run / dry_run / preflight
    # ------------------------------------------------------------------
    async def preflight_embed(self) -> None:
        await self.embedder.verify()
        if getattr(self.embedder, "embed_query_sparse", None):
            await self.embedder.embed_query_sparse("probe")
        await self.vector_store.ensure_collection(self.collection, self.embedder.dim)

    async def run(
        self, doc_id: str, *, from_stage: str | None = None, dry_run: bool = False
    ) -> StageOutcome:
        row = self.store.get(doc_id)
        if row is None:
            raise ValueError(f"doc {doc_id} không có bản ghi trạng thái (pending)")
        effective_start = from_stage or row.get("stage") or STAGE_PARSE
        if effective_start not in PROCESSING_STAGES:
            effective_start = STAGE_PARSE
        start_idx = PROCESSING_STAGES.index(effective_start)

        if dry_run:
            return await self._dry_run(doc_id, row, from_stage)

        try:
            # Indexed rồi: không làm lại (retry/reindex không chọn doc indexed;
            # branch này chỉ là bảo hiểm khi from_stage=None nhưng stage=indexed)
            if from_stage is None and row.get("stage") == STAGE_INDEXED:
                return StageOutcome(
                    doc_id=doc_id,
                    stage=STAGE_INDEXED,
                    ok=True,
                    chunk_count=int(row.get("chunk_count") or 0),
                )

            doc = await asyncio.to_thread(self._load_doc, doc_id)
            source_path = row.get("source_path") or ""
            # Tiêu đề doc prefix vào mỗi chunk (trừ khi chunk đã có tiêu đề ở dòng đầu)
            doc_title = (doc or {}).get("title") or row.get("title") or ""
            content_list: list[dict] | None = None
            descriptions: list[dict] | None = None
            chunks: list[Chunk] | None = None

            if start_idx <= 0:
                try:
                    intermediate = await self._stage_parse(doc_id, doc)
                except Exception:
                    document_type = (doc or {}).get("document_type") or "unknown"
                    INGEST_PARSE_FAILURES_TOTAL.labels(document_type=document_type).inc()
                    raise
                content_list = intermediate.content_list
                source_path = intermediate.file_path or doc_id

            if start_idx <= 1:
                if content_list is None:
                    content_list = await asyncio.to_thread(
                        self.asset_store.load_content_list, doc_id
                    )
                if content_list is None:
                    raise FileNotFoundError(
                        f"doc {doc_id}: thiếu content_list trong MinIO - cần parse lại"
                    )
                descriptions = await self._stage_describe(doc_id, content_list, doc_title)

            if start_idx <= 2:
                if content_list is None:
                    content_list = await asyncio.to_thread(
                        self.asset_store.load_content_list, doc_id
                    )
                if content_list is None:
                    raise FileNotFoundError(
                        f"doc {doc_id}: thiếu content_list trong MinIO - cần parse lại"
                    )
                if descriptions is None:
                    descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
                chunks = await self._stage_chunk(
                    doc_id,
                    content_list,
                    descriptions,
                    source_path=source_path,
                    doc_title=doc_title,
                )

            if start_idx <= 3:
                if chunks is None:
                    chunks = await self._load_or_build_chunks(
                        doc_id, content_list, descriptions, source_path, doc_title
                    )
                await self._stage_embed(doc_id, chunks, source_path=source_path)
                # Refresh row: parse/describe/chunk đã ghi meta (counts, assets, ...)
                row = self.store.get(doc_id) or row
                self.store.mark_indexed(
                    doc_id,
                    chunk_count=len(chunks),
                    embed_model=self.settings.EMBED_MODEL,
                    embed_dim=self.embedder.dim,
                    chunker_version=self.settings.CHUNKER_VERSION,
                    source_hash=row.get("content_hash"),
                    source=row.get("source"),
                    parser=row.get("parser") or self.settings.PARSER,
                    file_path=row.get("file_path"),
                    counts=row.get("counts"),
                    page_count=row.get("page_count"),
                    assets=row.get("assets"),
                    document_type=(doc or {}).get("document_type"),
                    title=(doc or {}).get("title"),
                    organization_unit_id=(doc or {}).get("organization_unit_id"),
                    owner_id=(doc or {}).get("owner_id"),
                    meta={"index_stats": {"chunks": len(chunks), "vectors": len(chunks)}},
                )

            return StageOutcome(
                doc_id=doc_id,
                stage=STAGE_INDEXED,
                ok=True,
                chunk_count=len(chunks) if chunks is not None else 0,
            )
        except VectorStoreError as exc:
            self.store.mark_failed(doc_id, f"vector store: {exc}", effective_start)
            return StageOutcome(doc_id=doc_id, stage=effective_start, ok=False, error=str(exc))
        except Exception as exc:
            self.store.mark_failed(doc_id, f"{type(exc).__name__}: {exc}", effective_start)
            return StageOutcome(doc_id=doc_id, stage=effective_start, ok=False, error=str(exc))

    async def _dry_run(self, doc_id: str, row: dict, from_stage: str | None) -> StageOutcome:
        """Handshake (GET /info) + đếm chunk / cache hits. Không POST /embed, không ghi."""
        await self.embedder.verify()
        await self.vector_store.ensure_collection(self.collection, self.embedder.dim)

        content_list = await asyncio.to_thread(self.asset_store.load_content_list, doc_id)
        descriptions = await asyncio.to_thread(self._load_descriptions, doc_id)
        source_path = row.get("source_path") or ""
        stored = await asyncio.to_thread(
            self._load_json, ArtifactPaths.chunks_key(self.asset_store, doc_id)
        )
        if isinstance(stored, list) and stored:
            chunks = [Chunk.from_json(c) for c in stored]
        elif content_list is not None:
            doc = await asyncio.to_thread(self._load_doc, doc_id)
            chunks = self._build_chunks(
                doc_id,
                content_list,
                descriptions,
                source_path=source_path,
                doc_title=(doc or {}).get("title") or row.get("title") or "",
            )
        else:
            logger.info("doc %s: không có trung gian (content_list/chunks), dry_run trả 0", doc_id)
            chunks = []
        items = await self._prepare_embed_items(chunks)
        return StageOutcome(
            doc_id=doc_id,
            stage=from_stage or "chunk",
            ok=True,
            chunk_count=len(chunks),
            cache_hits=self.embedder.count_cache_hits(items),
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete_doc(self, doc_id: str) -> None:
        await self.vector_store.delete_by_doc(self.collection, doc_id)
        try:
            await asyncio.to_thread(self.asset_store.delete_doc_assets, doc_id)
        except Exception:
            pass
        self.store.delete(doc_id)
