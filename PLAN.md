# PLAN — Vinter vLLM OCR fallback (máy B) + caption ảnh bằng Gemini 2.5 Flash Lite

> Cập nhật: 10/10/2026. **HOÀN TẤT 4/4 phase** (commit `45f43c8` Phase 2, `2ae2500` Phase 3; Phase 1 là repo `vintern-vllm` ngoài git, Phase 4 là docs). Quy tắc làm việc + hiện trạng hệ thống: xem `AGENTS.md`.

## Nền tảng đã hoàn tất (không đụng lại)
- Migration LightRAG → vector pipeline thuần: ✅ Phase 0-6 (`7221759`..`b2f8828`, `787ff0d`).
- Nemotron VL embed + rerank multimodal qua gateway máy B :8080: ✅ (`9ab5ce5`, `bd5ad49`, `1179c41`, `a9a2e55`).
- OCR bridge ảnh + describe text-only: ✅ (`2fee86b`, `27ddaee`) — **sẽ bị XOÁ ở Phase 3** (thay bằng Gemini caption).
- Fix MinIO pool maxsize + httpx connect timeout + phân biệt loại lỗi OCR: ✅ (`26d23f8`) — giữ nguyên, áp dụng pattern này cho client Vinter.

## Objective
1. **OCR text + bảng**: giữ service chính `http://171.226.10.153:12006` (`/api/v1/hierarchy-parse`, `/api/v1/raw_miner_ocr/qwen`), thêm **Vinter vLLM (máy B) làm fallback** khi service chính lỗi/treo/rỗng. Bố cục (vị trí bảng/đoạn chữ/ảnh, `bbox`, `page_idx`) **vẫn do MinerU + pdfplumber xử lý** — không đổi tầng layout.
2. **Caption ảnh**: chuyển HOÀN TOÀN sang **Gemini `gemini-2.5-flash-lite`** (OpenAI-compatible), gửi thẳng ảnh lấy từ MinIO. Xoá path text-only OCR-ảnh cũ.
3. **Repo mới ngoài RAG-Anything**: `D:\Code\Python\Ami\RAG\vintern-vllm\` (tham khảo `../nemotron-vl-vllm`), gateway FastAPI :8090 trước vLLM serving `5CD-AI/Vintern-1B-v3_5`.

## Quyết định đã duyệt (10/10/2026)
| Câu hỏi | Chốt |
|---|---|
| Thứ tự OCR | 171.226 **chính** → Vinter **dự phòng** (lỗi/timeout/rỗng) |
| Phạm vi fallback | cả `extract_scan_text` (text PDF scan) và `extract_tables` (bảng) |
| Tầng layout | giữ `PARSE_TEXT_SOURCE=backend_data` như cũ, chỉ thêm fallback |
| Caption ảnh | Gemini `gemini-2.5-flash-lite`, gửi thẳng ảnh, tái dùng `ImageModalProcessor` (materialize ảnh MinIO → file tạm) |
| Gemini không khả dụng | fallback **text thô** (caption + footnotes + ngữ cảnh), KHÔNG crash; **xoá** path OCR-ảnh text-only cũ |
| Port Vinter gateway máy B | **8090** (8080 = nemotron gateway, 8007 = qwen-embedding-server) |
| Model Vinter | **`5CD-AI/Vintern-1B-v3_5`** (mặc định) |

## Bối cảnh đã xác minh (đọc code, không phải suy đoán)
- `raganything/backend_data_extract.py`: `BackendDataExtractor` (httpx, `_post_file`, connect timeout tách riêng sau fix `26d23f8`), `extract_scan_text` (hierarchy-parse → `_flatten_hierarchy` → fallback PyMuPDF → raise nếu rỗng), `extract_tables` (`raw_miner_ocr/qwen` → `_parse_table_response`), `create_extractor(settings)`, `rebuild_content_list` (pdf/docx). `extract_image_ocr_text` + `image_to_pdf_bytes` + `_flatten_hierarchy_headings` chỉ dùng bởi path OCR-ảnh (sẽ xoá).
- `ami_rag/core/vector_pipeline.py`: `_stage_describe` (gather mọi item multimodal, `set_source(content_list, "minerU")`), `_describe_item`, `_describe_image_text_only` + `_ocr_image_text` + `_image_ocr_enabled` + `_get_ocr_semaphore` (sẽ xoá), `_extract_image_context`, `_parse_image_description` (chỉ path text-only dùng → xoá cùng), `_build_chunks`, `_prepare_embed_items`.
- **Bug đã biết**: nhánh `vision` gọi `ImageModalProcessor.generate_description_only` → đọc `content_data["img_path"]` và `Path.exists()` (`raganything/modalprocessors.py:772-790`), nhưng `img_path` trỏ vào `tempfile.TemporaryDirectory()` đã bị xoá ngay khi parse xong (`vector_pipeline.py:324-344`). ⇒ vision describe **chưa từng chạy được**, kể cả `reindex --from-stage describe`. Phase 3 sửa luôn bằng bước materialize.
- Ảnh sau parse chỉ còn ở MinIO theo `asset_key`; `MinioAssetStore.get_bytes(key)` có sẵn (`ami_rag/storage/assets.py:162`), pool `maxsize=16` (fix `26d23f8`).
- `ami_rag/core/factory.py`: `_build_llm_func`, `_image_mime`, `_openai_style_messages(prompt, system_prompt, history, image_data)` (đã build `image_url` data URI), `_build_vision_func`, `_build_modal_processors` (image→vision, table/equation/generic→llm, audio/video gate theo deps).
- OpenAPI service 171.226 (đã fetch `/openapi.json`): `/api/v1/hierarchy-parse` trả `{status, pages, content(cây), summary, key_points, document_type, ...}`; `/api/v1/raw_miner_ocr/qwen` trả bảng HTML; `/health`, `/` có sẵn.
- Repo tham khảo `../nemotron-vl-vllm` (**KHÔNG phải git repo**): `Dockerfile` (ARG `VLLM_VERSION=v0.19.0`, base `vllm/vllm-openai`), `entrypoint.sh` (tính `GPU_VRAM_CAP_GB × VRAM_SHARE − VRAM_RESERVE_GB` → `--gpu-memory-utilization`, `--limit-mm-per-prompt`, `--enforce-eager`), `docker-compose.yml` (vllm bind `127.0.0.1`, gateway bind `0.0.0.0`), `gateway/app.py` (FastAPI, `require_auth`, `_post_retry` + `_is_oom` markers, `maybe_clear`, `track()`, `/health`, `/admin/stats`), `.env.example`, `examples/client_example.py`, README tiếng Việt.
- Test infra: `tests/ami_service/conftest.py` (`FakeAssetStore.get_bytes` → `b"\x89PNG-fake-bytes"`, `FakeParser`, `FakeModalProcessor`, `fake_modal_processors`), `tests/ami_service/test_vector_pipeline.py` (nhóm test OCR-ảnh dòng 472-727 → sẽ thay), `tests/test_backend_data_extract.py` (5 test OCR-ảnh → xoá), `tests/test_gateway_parity.py` (pattern chạy gateway app thật qua `httpx.ASGITransport` + upstream `MockTransport` — **tái dùng cho repo vintern-vllm**), `Makefile`: `make test` = `tests/ami_service/`, `make test-lib` = phần còn lại.

---

## Phase 1 — Repo mới `D:\Code\Python\Ami\RAG\vintern-vllm\` (máy B)

```
vintern-vllm/
├── Dockerfile              # ARG VLLM_VERSION=v0.19.0; COPY gateway/ entrypoint.sh; EXPOSE 8000 8090
├── entrypoint.sh           # vllm serve $MODEL (ngân sách VRAM như nemotron, --limit-mm-per-prompt image=1, --enforce-eager)
├── docker-compose.yml      # vllm (127.0.0.1:8001) + gateway (0.0.0.0:8090), depends_on healthcheck
├── .env.example            # GPU_ID, GPU_VRAM_CAP_GB, VRAM_RESERVE_GB, MODEL, SERVED_NAME, OCR_PORT, OCR_DPI, OCR_CONCURRENCY, GATEWAY_API_KEY, REQUEST_TIMEOUT, LOG_LEVEL
├── gateway/
│   ├── __init__.py
│   ├── app.py              # POST /v1/ocr, GET /health, GET /admin/stats
│   └── prompts.py          # PROMPT_OCR_TEXT / PROMPT_OCR_TABLE (tiếng Việt)
├── examples/client_example.py
├── tests/test_gateway.py   # upstream vLLM giả bằng MockTransport + gateway qua ASGITransport
├── requirements.txt        # chỉ cho test local (fastapi, httpx, pytest, pymupdf)
└── README.md               # tiếng Việt: kiến trúc, VRAM, API, quy trình RAG, xử lý sự cố
```

**Contract `POST /v1/ocr`** (multipart/form-data — khớp style client ami_rag đang dùng với 171.226):

| Field | Bắt buộc | Ý nghĩa |
|---|---|---|
| `file` | ✅ | ảnh (png/jpg/webp) **hoặc** PDF |
| `mode` | ❌ (`text`) | `text` hoặc `table` |
| `pages` | ❌ | PDF nhiều trang: `1-5` hoặc `1,3,7` (1-based; rỗng = tất cả) |
| `dpi` | ❌ | default `OCR_DPI=200` |

Xử lý trong gateway: PyMuPDF mở file → nếu là ảnh thì coi như 1 trang → render `get_pixmap(dpi)` → PNG bytes → data URI → `POST {VLLM_URL}/v1/chat/completions` với `messages=[{role:"user", content:[{type:"image_url",...},{type:"text",text:PROMPT_*}]}]`, `max_tokens=OCR_MAX_TOKENS(2048)`, `temperature=0`. Semaphore `OCR_CONCURRENCY=4`, retry qua `_post_retry` + `_is_oom` (copy pattern nemotron gateway), auth `Authorization: Bearer` nếu đặt `GATEWAY_API_KEY`.

Response:
- `mode=text` → `{"model", "filename", "pages", "text", "page_texts": [{"page": 1, "text": "..."}]}`
- `mode=table` → `{"model", "filename", "pages", "tables": [{"page": 1, "html": "<table>...</table>", "markdown": "| a | b |"}]}`
- `GET /health` → `{"status":"ok","model":"5CD-AI/Vintern-1B-v3_5","vllm":true}` (503 khi vLLM chưa lên)
- Lỗi: 400 (file sai định dạng / mode lạ / trang không tồn tại), 502 (vLLM unreachable), 503 (OOM sau retry)

**Prompt** (`gateway/prompts.py`): `PROMPT_OCR_TEXT` = "Trích xuất TOÀN BỘ chữ trong ảnh theo đúng thứ tự đọc, giữ tiêu đề/mục; chỉ trả text thuần, không bình luận." `PROMPT_OCR_TABLE` = "Trích mọi bảng trong ảnh thành HTML `<table>` hợp lệ (kèm `<th>` cho header); nhiều bảng thì trả lần lượt; chỉ trả HTML, không giải thích." — cần test thử trên GPU thật và tinh chỉnh (README ghi rõ đây là điểm phải kiểm chứng).

**Tests repo mới** (`tests/test_gateway.py`, không cần GPU):
- render PDF giả (PyMuPDF) → MockTransport upstream trả text cố định → assert `page_texts` đúng số trang + thứ tự.
- `mode=table` → upstream trả `<table>` → assert `tables[0].html`/`markdown`.
- `pages=2` với PDF 3 trang → chỉ 1 request upstream.
- upstream 500 chứa "out of memory" → retry (`OOM_RETRIES`) → 503.
- `GATEWAY_API_KEY` đặt → thiếu header = 401.
- `/health` → 200/503 theo ping upstream.

**Rủi ro ghi trong README**: (a) vLLM bản cũ có thể không nhận architecture Vinter → bump `VLLM_VERSION` / `docker compose build --no-cache`; (b) chất lượng OCR bảng của Vinter-1B chưa kiểm chứng trên tài liệu tiếng Việt thật — fallback chỉ dùng khi 171.226 chết, và bảng còn có html pdfplumber làm lưới an toàn thứ 2; (c) chưa chạy thử GPU thật (giống nemotron repo lúc đầu).

---

## Phase 2 — Fallback OCR trong ami_rag (client)

**Mới `ami_rag/core/ocr_client.py`**
```python
class OcrFallbackError(Exception): ...

class VinterOcrClient:
    def __init__(self, base_url, *, model="", token="", timeout=300, retries=1,
                 dpi=None, http_client=None): ...
    async def health(self) -> dict                      # GET /health (không raise, trả {})
    async def ocr_text(self, file_bytes, filename="document.pdf", *, pages=None) -> str
    async def ocr_tables(self, file_bytes, filename="document.pdf", *, pages=None) -> list[dict]
    async def close(self) -> None

def build_vinter_ocr_client(settings) -> VinterOcrClient | None   # None khi disabled/URL rỗng
```
- httpx `Timeout(timeout, connect=min(10, timeout))` (fail-fast như fix `26d23f8`), phân biệt `ConnectError` / `ReadTimeout` trong message.
- retry chỉ 429/5xx/transport, backoff `0.5 * 2^n + jitter`; 4xx raise ngay.
- `ocr_text` gộp `page_texts` theo thứ tự trang (fallback field `text`), strip; `ocr_tables` chuẩn hoá về `[{"page": int|None, "html": str}]` (bỏ item html rỗng).
- Response rỗng/không JSON → `OcrFallbackError`.

**Settings mới** (`ami_rag/settings.py`, ngay sau block `OCR_*`):
```python
OCR_FALLBACK_ENABLED: bool = True
OCR_FALLBACK_URL: str = "http://localhost:8090"       # vintern-vllm gateway máy B
OCR_FALLBACK_MODEL: str = "5CD-AI/Vintern-1B-v3_5"    # chỉ để log/verify
OCR_FALLBACK_TOKEN: str = ""                          # chỉ từ env
OCR_FALLBACK_TIMEOUT: int = 300
OCR_FALLBACK_RETRIES: int = 1
OCR_FALLBACK_ON_EMPTY: bool = True                    # primary trả rỗng cũng fallback
OCR_FALLBACK_MODES: str = "text,table"                # tắt riêng từng loại
```
+ helper `resolve_ocr_fallback_modes(settings) -> frozenset[str]` (validate `text|table`, raise ValueError khi lạ).

**Sửa `raganything/backend_data_extract.py`** (chạy `gitnexus_impact` trước cho `BackendDataExtractor`, `extract_scan_text`, `extract_tables`, `create_extractor`):
- `__init__(..., fallback=None, fallback_modes=frozenset({"text","table"}), fallback_on_empty=True)`; thuộc tính `self._fallback`, `self._fallback_modes`, `self._fallback_on_empty`.
- helper nội bộ `_fallback_enabled(kind)` + `_log_fallback(kind, reason, exc)`.
- `extract_scan_text`: bọc call primary trong try → `BackendDataParseError`/lỗi httpx hoặc text rỗng (và `fallback_on_empty`) → `await self._fallback.ocr_text(...)`; **giữ** fallback PyMuPDF local làm lớp cuối (như hiện tại); fallback Vinter cũng lỗi → raise lỗi gốc (caller `rebuild_pdf_content_list` đã bắt và log).
- `extract_tables`: primary lỗi hoặc trả `[]` → `await self._fallback.ocr_tables(...)`; cả hai lỗi → `[]` (giữ hành vi hiện tại: `rebuild_pdf_content_list` log warning rồi dùng html pdfplumber).
- `create_extractor(settings)`: build `VinterOcrClient` qua `build_vinter_ocr_client(settings)` (None khi `OCR_FALLBACK_ENABLED=False` hoặc URL rỗng) rồi inject.
- **KHÔNG xoá gì ở phase này** (`extract_image_ocr_text`/`image_to_pdf_bytes` xoá ở Phase 3 cùng commit với caller — xem ⚠️ dưới).

**Tests mới `tests/test_ocr_fallback.py`** (`httpx.MockTransport`, 0 mạng):
1. primary 500 → fallback được gọi, trả text fallback.
2. primary trả text rỗng + `fallback_on_empty=True` → fallback; `False` → raise như cũ.
3. primary OK → fallback KHÔNG được gọi.
4. `fallback_modes={"table"}` → text không fallback, table có.
5. fallback cũng lỗi → raise `BackendDataParseError` (message nêu cả 2 lỗi).
6. `extract_tables`: primary `[]` → fallback tables; fallback lỗi → `[]`.
7. `build_vinter_ocr_client`: `OCR_FALLBACK_ENABLED=False` / URL rỗng → `None`.
8. `resolve_ocr_fallback_modes`: hợp lệ / raise.

**Tuỳ chọn (nếu còn thời gian phase)**: `ami-rag status --check-ocr` in `/health` của 171.226 + Vinter (pattern `_check_embed_server` trong `ami_rag/cli.py:221`).

⚠️ **Thứ tự bắt buộc**: Phase 2 chỉ THÊM. Việc xoá `extract_image_ocr_text`/`image_to_pdf_bytes`/`_flatten_hierarchy_headings` (+ 5 test trong `tests/test_backend_data_extract.py`) nằm ở **Phase 3**, cùng commit với xoá caller `vector_pipeline._ocr_image_text`, nếu không sẽ vỡ runtime giữa 2 commit.

---

## Phase 3 — Caption ảnh bằng Gemini 2.5 Flash Lite

**Settings** (`ami_rag/settings.py`):
```python
CAPTION_LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
CAPTION_LLM_MODEL: str = "gemini-2.5-flash-lite"
CAPTION_LLM_API_KEY: str = ""          # chỉ từ env, không commit
CAPTION_LLM_TIMEOUT: int = 120
CAPTION_LLM_MAX_CONCURRENCY: int = 4
CAPTION_LLM_RETRIES: int = 2
IMAGE_DESCRIBE_MODE: str = "auto"      # auto | gemini | vision | none
```
- `resolve_image_describe_mode` mới: explicit (`gemini|vision|none`) validate; legacy `"text"` → log warning + `"none"`; `auto` → `gemini` nếu `CAPTION_LLM_API_KEY` else `vision` nếu `QWEN_VLM_MODEL` else `none`.
- **Xoá** `DESCRIBE_IMAGE_OCR`, `DESCRIBE_IMAGE_OCR_SKIP_IF_CAPTION`, `DESCRIBE_IMAGE_OCR_MAX_CONCURRENCY`, `DESCRIBE_IMAGE_OCR_TIMEOUT`.

**`ami_rag/core/factory.py`**:
- `_build_caption_func(settings) -> Callable | None`: `None` khi không có API key; ngược lại `AsyncOpenAI(base_url=CAPTION_LLM_BASE_URL, api_key=..., timeout=..., max_retries=...)` + closure nhận `(prompt, system_prompt=None, history_messages=None, image_data=None, messages=None, **kwargs)` — tái dùng `_openai_style_messages` (đã build data URI từ base64, `_image_mime` sniff png/webp/gif/jpeg).
- `_build_modal_processors`: `image_caption_func = _build_caption_func(settings) or _build_vision_func(settings)`; `ImageModalProcessor(modal_caption_func=image_caption_func, ...)`. Video giữ `vision`.
- `build_pipeline(..., caption_func=None)` passthrough (test inject), không đổi chữ ký bắt buộc.

**`ami_rag/core/vector_pipeline.py`**:
- `_stage_describe`: thêm bước **materialize ảnh** —
  1. gom `asset_key` của item `type == "image"` (dedupe), fetch song song có giới hạn (`asyncio.to_thread(asset_store.get_bytes)`, semaphore `CAPTION_LLM_MAX_CONCURRENCY`);
  2. ghi bytes vào `tempfile.TemporaryDirectory()` (tên `{sha1(asset_key)[:12]}{ext}` — ext suy từ magic bytes, tái dùng logic `_image_mime`/suffix asset_key);
  3. tạo `describe_items = [dict(item) ...]` shallow copy với `img_path` = đường dẫn file tạm (item không có asset_key giữ `img_path` gốc); **không mutate `content_list`** (stage chunk + `save_content_list` dùng bản gốc);
  4. `set_content_source(content_list, "minerU")` như cũ (context vẫn từ content_list gốc);
  5. gọi `_describe_item(type_, describe_items[i], content_list, i, doc_id)`;
  6. `finally`: xoá temp dir.
- `_describe_item`: `image` → theo mode: `none` → `_describe_image_raw(item, content_list, index)`; `gemini`/`vision` → processor (ảnh đã có `img_path` thật); processor raise hoặc trả rỗng → `_describe_image_raw` (log warning, không crash).
- Mới `_describe_image_raw(item, content_list, index) -> str`: `"\n".join(p for p in (caption, footnotes, context) if p).strip()` (caption/footnotes qua `_join_caption`, context qua `_extract_image_context`).
- Semaphore caption: `self._caption_sem` (lazy, `CAPTION_LLM_MAX_CONCURRENCY`) bọc call processor cho item image; **xoá** `self._ocr_sem` + `_get_ocr_semaphore`.
- **Xoá**: `_describe_image_text_only`, `_ocr_image_text`, `_image_ocr_enabled`, `_parse_image_description`. **GIỮ** `_get_bd_extractor` + `_extract_image_context` + `_join_caption`.
- **Xoá `raganything/backend_data_extract.py`**: `extract_image_ocr_text`, `image_to_pdf_bytes`, `_flatten_hierarchy_headings` + 5 test tương ứng trong `tests/test_backend_data_extract.py` (`test_image_to_pdf_bytes_wraps_image`, `test_extract_image_ocr_text_flattens_headings_and_summary`, `test_extract_image_ocr_text_empty_when_no_content`, `test_extract_image_ocr_text_error_raises`, `test_flatten_hierarchy_headings_dedupe_heading_vs_text`).
- **Xoá prompt**: `PROMPTS["IMAGE_TEXT_ONLY_SYSTEM"]`, `PROMPTS["image_text_only_prompt"]` (`raganything/prompt.py:164-191`) + bản dịch `PROMPTS_ZH` tương ứng (`raganything/prompts_zh.py:103-…`). Kiểm tra `tests/test_prompt_language.py` có assert số key/parity EN-ZH không → sửa theo.

**Tests** (`tests/ami_service/test_vector_pipeline.py`, thay nhóm dòng 472-727):
- caption func giả (`async def(prompt, system_prompt=None, image_data=None, **kw)`) → assert nhận `image_data` base64 của đúng bytes MinIO (`b"\x89PNG-fake-bytes"`), prompt chứa captions/context; description = JSON `detailed_description` được parse bởi `ImageModalProcessor._parse_response` (dùng `fake_modal_processors` cho đường pipeline, thêm 1 test dùng `ImageModalProcessor` THẬT với caption func giả để chứng minh `img_path` materialize hoạt động).
- `img_path` trong `describe_items` tồn tại lúc describe, và temp dir đã xoá sau `_stage_describe`; `content_list` gốc KHÔNG bị thêm `img_path` mới.
- mode `none` → text thô (caption + context), không gọi caption func.
- caption func raise → `_describe_image_raw`, không crash, `descriptions.json` vẫn ghi.
- ảnh thiếu `asset_key` → không materialize, không crash.
- concurrency ≤ `CAPTION_LLM_MAX_CONCURRENCY` (fake delay + đếm `max_active`).
- `test_resolve_image_describe_mode` viết lại: auto/gemini/vision/none + legacy `text` → `none` + `bogus` raise.

---

## Phase 4 — Docs + env + dọn dẹp

- `.env.ami.example`: thêm block `CAPTION_LLM_*` (kèm cảnh báo "API key chỉ từ env, không commit"), block `OCR_FALLBACK_*`; cập nhật `IMAGE_DESCRIBE_MODE` (auto|gemini|vision|none); **xoá** `DESCRIBE_IMAGE_OCR*` (dòng 10-22) và sửa comment `QWEN_VLM_MODEL` (dòng 6-8).
- `docs/ami_service.md`:
  - dòng 38: "Gemini không còn được hỗ trợ (đã gỡ)" → Gemini chỉ dùng cho **caption ảnh** qua OpenAI-compatible endpoint (`CAPTION_LLM_*`).
  - dòng 20 + 36: mô tả nhánh ảnh mới (gemini/vision/none).
  - bảng settings (dòng 143-153): thay `DESCRIBE_IMAGE_OCR*` bằng `CAPTION_LLM_*` + `IMAGE_DESCRIBE_MODE` mới; thêm `OCR_FALLBACK_*` cạnh `OCR_SERVICE_URL`.
  - viết lại §5.3 → "Caption ảnh (Gemini 2.5 Flash Lite) + materialize ảnh từ MinIO"; thêm §5.4 "OCR fallback Vinter (máy B)" kèm contract `/v1/ocr`, thứ tự ưu tiên, cách tắt (`OCR_FALLBACK_ENABLED=false` / `OCR_FALLBACK_MODES`).
  - mục reindex (dòng ~435-440): chi phí mới = 1 call Gemini/ảnh (không còn ~30s OCR/ảnh).
- `AGENTS.md`: cập nhật section task hiện hành (trạng thái phase + commit hash); phần "Hiện trạng hệ thống" thêm repo `vintern-vllm` :8090 đã chạy.
- `Makefile` / `docker-compose.ami.yml`: kiểm tra có pass-through env mới không (compose thường dùng `env_file: .env` → không cần sửa; verify).
- **Vận hành sau khi deploy**: mô tả ảnh đổi → chunk image đổi text → `ami-rag reindex --all --from-stage describe` (collection Qdrant KHÔNG đổi vì `EMBED_MODEL` giữ nguyên; chunk text không đổi vẫn trúng cache embed, chỉ chunk ảnh re-embed). Chạy `--dry-run` trước, `--doc <id>` để thử vài doc.

---

## Trạng thái phase
| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| 1 | Repo `vintern-vllm` (gateway `/v1/ocr` + Docker + tests + README) | ✅ xong — 22 test pass, không git init (quyết định user 11/10/2026, như nemotron) |
| 2 | `ami_rag/core/ocr_client.py` + settings `OCR_FALLBACK_*` + inject fallback vào `BackendDataExtractor` + `tests/test_ocr_fallback.py` | ✅ commit `45f43c8` (20 test) |
| 3 | Caption Gemini (`CAPTION_LLM_*`, `_build_caption_func`, materialize ảnh MinIO, mode `none` fallback, xoá path OCR-ảnh cũ + prompts text-only) | ✅ commit `2ae2500` (8 test describe mới + xoá 12 test cũ) |
| 4 | Docs (`docs/ami_service.md` §5.3/§5.4, `.env.ami.example`, AGENTS.md) + runbook reindex | ✅ commit này |

## Kiểm chứng cuối (trước khi báo hoàn tất)
- [x] `uv run pytest` cả 2 nhóm: 142 (service) + 406 passed, 29 skipped (reportlab pre-existing) — 0 fail.
- [x] `uv run ruff check ami_rag tests/ami_service` sạch.
- [x] repo `vintern-vllm`: `pytest` 22/22 sạch (không cần GPU).
- [x] `rg -n "DESCRIBE_IMAGE_OCR|extract_image_ocr_text|image_to_pdf_bytes|IMAGE_TEXT_ONLY" .` → 0 kết quả trong code (chỉ còn ở `docs/migration_inventory.md` + lịch sử git).
- [x] `gitnexus_detect_changes()` trước mỗi commit; HIGH/CRITICAL đã cảnh báo user (`create_extractor` Phase 2, `_build_modal_processors` Phase 3 — thay đổi additive).
- [ ] Chạy thật trên server (khi có GPU máy B + Gemini key): 1 doc có ảnh → `descriptions.json` có mô tả Gemini; tắt 171.226 (hoặc đặt `OCR_SERVICE_URL` sai) → log thấy "chuyển Vinter" và parse vẫn ra text/bảng.
