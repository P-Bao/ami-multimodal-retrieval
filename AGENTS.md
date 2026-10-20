<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ami-multimodal-retrieval** (5488 symbols, 10723 relationships, 296 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ami-multimodal-retrieval/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ami-multimodal-retrieval/clusters` | All functional areas |
| `gitnexus://repo/ami-multimodal-retrieval/processes` | All execution flows |
| `gitnexus://repo/ami-multimodal-retrieval/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

# Task hiện hành — BGE-M3 migration: hybrid Qdrant + `artifact_urls` + reindex ngắt quãng

Trạng thái: **code xong, chưa reindex** (branch `feature/backend-data-parser`, baseline `6a7366a`). Index Nemotron cũ (dim 2048, collection 1-vector vô danh) không dùng được với config mới — phải reindex sau khi deploy.

## Nội dung
1. **Embed BGE-M3** — `ami_rag/core/bge_embedder.py` (`BGEM3Embedder`): server `bge-m3-embed:8008` (repo `embedding-server`, auth `Bearer BGE_API_KEY`, tối đa 64 text/request, max_length 8192). `POST /embed {texts:[str], return_dense, return_sparse}` → `{dense_vecs, sparse_weights[{token_id: w}]}`; `_to_sparse` đổi `sparse_weights` → `SparseVector{indices, values}` (sort token_id). Handshake `GET /v1/models` + probe `/embed` chốt dim. Embed-only **caption-only**: `_prepare_embed_items` trả `list[str]` (không embed ảnh bytes); caption ảnh tiếng Việt do describe stage (Gemini) sinh.
2. **Hybrid Qdrant** — named vectors `dense` (1024 COSINE) + `sparse`; query `sparse=None` → dense-only; collection legacy (1 vector vô danh) + cần sparse → `VectorStoreError` (không bổ sung được — phải collection mới); `EMBED_MODEL` đổi ⇒ collection mới tự sinh `multimodal__bge-m3__v1` (slug từ `EMBED_MODEL`).
3. **Rerank legacy BGE** — `bge-reranker-v2-m3` :8010 (repo `rerank`, documents text-only, tối đa 100/request batch nội bộ), `RERANK_MULTIMODAL=False`; `RERANK_FUSION=rrf` giờ hoạt động cả legacy (legacy trả full scores). API trả `RetrievedDocV2.artifact_urls: list[str]` — presigned mọi ảnh **chỉ cho doc crawl** (`presign_doc_images`; crawl = `original_file_name` rỗng/`.txt` + không có `file_path` + `metadata.source_url`; doc khác/không resolve → `[]`), giữ `artifact_url`.
4. **CLI reindex ngắt quãng** — `--limit N` (0=all), `--sleep S` giây giữa doc (mặc định 0.2), `--batch B` (20), `--batch-sleep S` (2.0); `--from-stage embed` dùng lại `chunks.json` (không parse/describe).
5. Settings (nguồn sự thật `ami_rag/settings.py`): `EMBED_BACKEND=bge`, `EMBED_SERVER_URL=http://localhost:8008`, `EMBED_MODEL=BAAI/bge-m3`, `EMBED_DIM=1024`, `EMBED_MAX_INPUT_TOKENS=8192`, `EMBED_IMAGE_TOKEN_RESERVE=0`, `EMBED_RETURN_SPARSE=true`, `EMBED_BATCH_SIZE=32` (≤64); `RERANK_BACKEND=legacy`, `RERANK_BASE_URL=http://localhost:8010`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `RERANK_MULTIMODAL=false`; `resolve_embed_backend`: `auto` + model chứa `bge` → `bge`.
6. Server máy B: bảng dưới được thay — gateway `nemotron-vl-vllm` (8080) / `qwen-embedding-server` (8007) **không còn**; code backend `custom`/`openai` giữ lại cho legacy. Vintern :8090 vẫn là OCR fallback.

## Brief runbook (máy A)
```bash
ami-rag status --check-embed-server                          # :8008 + :8010 sẵn sàng
ami-rag reindex --stale --from-stage embed --dry-run         # scope + cache hits
ami-rag reindex --stale --from-stage embed --limit 20 --yes  # thử lô nhỏ trước
ami-rag reindex --stale --from-stage embed --batch 20 --yes  # chạy thật, ngắt quãng
```
- Resume/lặp vô hạn: doc xong embed được đánh `indexed` (kèm `embed_model` mới) trong `multimodal_rag_documents`; doc còn lại → effective `stale` ⇒ chạy lại lệnh `--stale` (mặc định) tự tiếp tục đến hết. Chỉ ép `--from-stage embed` khi doc đã có `chunks.json`; thiếu thì bỏ `--from-stage` (per-doc tự chọn `chunk`).
- Index mới ổn: `ami-rag cleanup --stale-models --purge-cache-model "nvidia/llama-nemotron-embed-vl-1b-v2" --yes`.
- Chi tiết: `docs/ami_service.md` — §5.2 (embed/rerank BGE), §5.6 (Qdrant hybrid), §7 (runbook reindex), §6 (`artifact_urls`).

---

# Task trước — Crawl images: tải ảnh section 'Hình ảnh' về MinIO (12/10/2026)

Trạng thái: **HOÀN TẤT** (5 commit `c361475`..docs). Quyết định user: **luôn bật, không env tắt** (đã xoá `CRAWL_IMAGES_ENABLED`/`CRAWL_IMAGES_PREFIX`); timeout tải 60s/ảnh; ảnh crawl lưu dưới `ASSET_PREFIX` (`rag-assets/{doc_id}/…`) nên `delete_doc_assets` dọn theo doc.

## Nội dung
1. `c361475` — `crawl_images.py::split_image_section`: tách section `Hình ảnh` khỏi text crawl (heading fold không dấu nhận `**Hình ảnh:**`/`## HÌNH ẢNH`/`Hinh anh`; URL bullet/plain/markdown, lọc đuôi ảnh, dedupe; heading không có URL ảnh → giữ nguyên) + 12 test.
2. `80036b7` — `_stage_parse` nhánh `mongo_text` → `_crawl_text_content_list`: split → tải ảnh (semaphore 4, lỗi từng ảnh bỏ qua) → append item `image` kèm `asset_key`; text strip section, strip rỗng → bỏ text item; fail toàn bộ → giữ text gốc (URL không mất khỏi index). Ảnh crawl đi describe/chunk/embed như ảnh pdf.
3. `aff2b87` — lazy-build `CrawlImageWorker` trong pipeline; gỡ plumbing chết `image_worker` khỏi `IngestWorker`/`_build_default_deps` (giờ 4-tuple)/`main.py`; `build_pipeline(image_worker=)` cho test inject.
4. `4c813ca` — luôn bật: xoá setting, split-first (doc không section không dựng worker).
5. commit này — docs (`docs/ami_service.md` §1/§2/§5.3/runbook, `.env.ami.example`, AGENTS.md).

## Verify ảnh doc/pdf (đã kiểm tra code — không phải sửa)
MinerU `img_path` tuyệt đối → `upload_content_list_assets` gắn `asset_key` → `_materialize_images` (describe) → chunk modality `image` → embed `image_b64`. Test phủ: `test_describe_image_via_real_image_processor`, `test_prepare_embed_items_attaches_image_b64`. Còn checklist runtime server: parse 1 PDF có ảnh thật → `counts.image > 0` + chunk Qdrant có `asset_key`.

## Deploy
Doc crawl đã index trước tính năng: `ami-rag reindex --all --from-stage parse` (parse lại cả PDF/DOCX bằng MinerU — ưu tiên `--doc <id>` từng bài crawl). Bài crawl sửa sau này tự chạy đúng luồng, không cần env.

---

# Hiện trạng hệ thống (đã hoàn tất — chỉ giữ phần còn giá trị)

Chi tiết lịch sử: `git log` (`7221759`..`4c813ca`), `docs/ami_service.md`.

## Kiến trúc
- Pipeline vector thuần, **không LightRAG** (không entity/graph/neo4j; chỉ còn param compat `lightrag=None` trong `modalprocessors`/`factory`). Stages: `parse → describe → chunk → embed → indexed`, artifacts trung gian (`content_list.json`/`descriptions.json`/`chunks.json`) lưu MinIO nên retry chạy tiếp từ stage lỗi.
- **describe là stage DUY NHẤT có thể gọi LLM** (tuỳ chọn). Máy A không cần GPU, không cài torch/transformers.
- Parse: MinerU (ảnh + layout) + `PARSE_TEXT_SOURCE=backend_data` (pdfplumber/fitz local dựng bố cục text/bảng, nội dung chữ/bảng từ OCR service chính, fallback Vinter máy B :8090 khi lỗi/treo/rỗng). Doc crawl (`mongo_text`): ảnh section `Hình ảnh` **luôn tải** về MinIO (`split_image_section`, item `image` kèm `asset_key`, lỗi bỏ qua) — không có env tắt.
- API v2-only: `POST /rag` + `/rag/stream` (`RAGRequest` = messages/top_k/include_references); query chỉ text. CLI `ami-rag` chỉ còn status/retry/reindex/cleanup.
- Collection Qdrant: `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` (`collection_name()` trong `ami_rag/core/embedder.py`). Đổi `EMBED_MODEL`/`CHUNKER_VERSION` → collection mới → `ami-rag reindex --scan`.

## Service máy B (HTTP)
| Port | Repo (ngoài RAG-Anything) | Vai trò |
|---|---|---|
| 8008 | `embedding-server` | `bge-m3-embed` — `BAAI/bge-m3`: dense 1024 + sparse lexical weights, auth `Bearer BGE_API_KEY`, tối đa 64 text/request, max_length 8192 |
| 8010 | `rerank` | BGE reranker `BAAI/bge-reranker-v2-m3`: documents text-only, tối đa 100/request |
| 8090 | `D:\Code\Python\Ami\RAG\vintern-vllm\` | OCR fallback `5CD-AI/Vintern-1B-v3_5` (đã build + 22 test; chưa git init; chạy thật chờ GPU máy B) |

- Contract embedding-server (client `BGEM3Embedder` đã đối chiếu khớp): `GET /v1/models` → `{"data": [{"id": "BAAI/bge-m3", ...}]}`; `POST /embed` với `{"texts": [...], "return_dense": true, "return_sparse": bool}` → `{"dense_vecs": [[float]], "sparse_weights": [{"<token_id>": w}]}` (token_id dạng chuỗi); auth `Authorization: Bearer` trùng `BGE_API_KEY` server; 400 khi >64 text hoặc text rỗng, 401 sai key. Rerank :8010 `POST /rerank` `{query, documents, top_n}` → `{results: [{index, relevance_score}]}`, documents text-only; `RERANK_FUSION=rrf` gửi `top_n=len` để nhận full scores (rrf hoạt động cả legacy).
- Contract gateway vintern (client `VinterOcrClient` đã đối chiếu khớp): `POST /v1/ocr` multipart `file` (PDF/ảnh) + `mode=text|table` + `pages` (`1-5`/`1,3,7`) + `dpi`; response `mode=text` → `{pages, text, page_texts: [{page, text}]}`, `mode=table` → `{pages, tables: [{page, html, markdown}]}`; `GET /health` → `{status, model, vllm}`; lỗi 400 (file/mode/trang), 401 (thiếu `Authorization: Bearer`), 502 (vLLM chết), 503 (OOM sau retry). Cần `templates/vintern-ocr.jinja` cho vLLM (tokenizer Vinter đóng template Qwen text-only, không sinh `<image>`).
- Backend dispatch: `EMBED_BACKEND` (`auto|custom|openai|bge`; auto: model chứa `bge` → bge, prefix `Qwen/` → custom) + `RERANK_BACKEND` (`auto|legacy|vllm`; auto: rỗng/chứa `bge` → legacy BGE text-only) + `RERANK_MULTIMODAL` (default False). Resolver: `settings.resolve_embed_backend`/`resolve_rerank_backend`; factory `build_embedder`; cli `_build_embedder`.
- Multimodal: `_prepare_embed_items` giờ **caption-only** — trả `list[str]`, không nạp ảnh bytes cho embed (caption VI do describe stage sinh). API trả thêm `artifact_urls` = presigned mọi ảnh **chỉ cho doc crawl** qua `presign_doc_images` (doc khác → `[]`). Dim 1024.
- URL default `EMBED_SERVER_URL` = `http://localhost:8008`, `RERANK_BASE_URL` = `http://localhost:8010`; trong Docker trỏ hostname service compose của bộ BGE.

## Vận hành
- `ami-rag status [--check-embed-server]` — `_check_embed_server` backend-agnostic (`/health` + `embedder.verify()`); doc lỗi nằm ở `failed` kèm `error`/`error_stage`.
- `ami-rag reindex --scan` (doc legacy `processed` → pending) / `--all --from-stage <stage>` / `--dry-run` / `--doc <id>`; from_stage per-doc tự chọn (`chunk` khi `content_list` đã có trong MinIO, full parse khi thiếu). Doc crawl index trước tính năng ảnh crawl → `--from-stage parse` (docs §7 runbook).
- `ami-rag cleanup --stale-models` (xoá Qdrant `{WORKSPACE}__*` không phải collection hiện tại) + `--purge-cache-model "<model>"` (xoá cache SQLite theo prefix `{MODEL}|`). Helpers: `legacy_cleanup.find_stale_model_collections`, `EmbeddingCache.delete_by_prefix`.
- Embed client: handshake + probe xác minh model/dim, batch `EMBED_BATCH_SIZE` + `EMBED_MAX_CONCURRENCY`, cache SQLite (`model|dim|instruction_ns|sha256`), retry CHỈ transient (timeout/5xx/429/connect — không bao giờ 4xx), circuit breaker 5 fail liên tiếp → `EmbedCircuitOpen`.
- OCR service `.153:12006` đã treo toàn cục (accept TCP nhưng ReadTimeout mọi endpoint) → đã có fallback Vinter (`OCR_FALLBACK_*`, `docs/ami_service.md` §5.4): primary lỗi/timeout/rỗng → `VinterOcrClient` :8090, cả hai lỗi → text dùng PyMuPDF local / bảng dùng html pdfplumber. Log warning `chuyển sang fallback Vinter` cho biết service chính đang chết.
- Caption ảnh (`docs/ami_service.md` §5.3): `IMAGE_DESCRIBE_MODE` `auto|gemini|vision|none`; ảnh được materialize từ MinIO sang temp dir (`img_path`) trước khi gọi processor — đã fix bug vision đọc temp dir đã xoá. Deploy máy A cần `CAPTION_LLM_API_KEY` (env) để bật Gemini; mô tả ảnh đổi → `ami-rag reindex --all --from-stage describe`.
- Chunk (docs §5.5): `chunk_text` v2 + `_build_chunks` gom item text liên tiếp + prefix tiêu đề doc; `CHUNK_MIN_CHARS` (200) gộp chunk mồ côi. `CHUNKER_VERSION` là **tên collection** — đổi ⇒ phải `reindex --all --from-stage chunk` (không gọi LLM) rồi `cleanup --stale-models`.
- Chẩn đoán retrieval: `python scripts/diagnose_query.py --query "..."` (chỉ đọc Qdrant + embed/rerank, không Gemini). Ngoài máy Docker phải override `QDRANT_URL=http://172.23.0.9:6333` (pl_data_qdrant) và `MINIO_ENDPOINT=172.23.0.12:9000` — tên service `qdrant`/`minio` không resolve từ host.

## Bẫy đã biết (bắt buộc nhớ)
- **`tests/__init__.py` phải tồn tại**: site-packages có package `tests` shadow local dir → `tests.ami_service` import lỗi nếu thiếu.
- GitNexus CLI phải dùng bản global 1.6.5 (`C:\Users\phanb\AppData\Local\pnpm\bin\gitnexus.CMD`), KHÔNG `npx` latest (DB mismatch v42 vs v40). Nếu GitNexus tool warns index STALE (thường sau khi đổi schema symbol) → chạy `gitnexus.CMD analyze` trước khi `gitnexus_impact`; nếu symbol vẫn không ra thì dùng `rg` thủ công và ghi rõ trong commit message.
- MinIO `PoolManager(maxsize=16)` ở cả 3 nơi tạo client (`storage/assets.py`, `api/resolver.py`, `workers/crawl_images.py`) — describe fetch nhiều ảnh song song; đừng hạ về 1.
- httpx gọi OCR: `connect` timeout tách riêng `min(10s, timeout)` để fail-fast; message lỗi phải phân biệt `ReadTimeout` (service treo) / `ConnectError` (unreachable-firewall) / `ConnectTimeout` (SYN drop).

# Quy tắc làm việc (bắt buộc)
- **MỖI phase commit riêng, STOP xin duyệt sau mỗi phase.** Trước commit: `gitnexus_detect_changes()`; trước khi sửa symbol: `gitnexus_impact({target, direction:"upstream", repo:"RAG-Anything"})`. HIGH/CRITICAL → cảnh báo user trước khi sửa.
- Test: `$env:PYTHONIOENCODING="utf-8"; uv run pytest -q tests/ami_service/` (service, `make test`) và `uv run pytest -q tests --ignore=tests/ami_service` (lib, `make test-lib`); lint `uv run ruff check ami_rag tests/ami_service` (ruff 0.16.0 global; `ami_rag/**` đã có per-file-ignores BLE001/S110/B008).
- File có tiếng Việt: chỉ dùng tool write/edit, KHÔNG `Add-Content`/`Set-Content` (cp1252 mojibake).
- Không gọi LLM/API ngoài ý muốn: test chỉ dùng fake + `httpx.MockTransport`/`ASGITransport`.
- Repo server máy B luôn nằm RIÊNG ngoài RAG-Anything (`embedding-server`, `rerank`, `qwen-embedding-server`, `nemotron-vl-vllm`, `vintern-vllm`). `nemotron-vl-vllm` KHÔNG phải git repo → hỏi user trước khi `git init` cho repo mới.
- Test infra sẵn có: `tests/ami_service/conftest.py` (`FakeAssetStore.get_bytes` → `b"\x89PNG-fake-bytes"`, `FakeParser`, `FakeModalProcessor`), `tests/test_gateway_parity.py` (pattern chạy gateway app THẬT qua ASGI + upstream `MockTransport`, skip nếu repo không có — copy cho `vintern-vllm`).
