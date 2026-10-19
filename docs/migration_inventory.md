# Migration inventory — RAG-Anything → vector thuần + Qwen3-VL-Embedding-2B

Kết quả kiểm kê Giai đoạn 0 (đã duyệt). Repo được index bằng GitNexus (v1.6.5) để xác minh call-graph.

## Kiến trúc mục tiêu

- Pipeline (parse, chunk, vector store, trả lời) chạy trên **máy A** (không GPU).
- Embedding `Qwen/Qwen3-VL-Embedding-2B` chạy trên **máy B** (GPU) qua embedding server riêng
  (`qwen-embedding-server`, nằm ngoài repo này, kiến trúc theo BGE-M3 embedding server đã có).
- Knowledge graph + LightRAG bị loại hoàn toàn; thay bằng pipeline vector thuần (Qdrant).
- CLI (`ami-rag`) chỉ phục vụ: xem trạng thái doc, chạy lại doc lỗi, reindex thủ công.

## Quyết định đã chốt

1. API: giữ `POST /rag` + `/rag/stream` (v2 response); bỏ v1, `filters`, `include_kg`, `mode`.
2. `describe_func`/`answer_func` dùng profile qwen-selfhost (OpenAI-compatible vLLM); xoá Gemini hoàn toàn.
3. Vector backend: **Qdrant**. DocStatusStore: **Mongo** (mở rộng `multimodal_rag_documents`).
4. Dim: **2048 mặc định, qua env** (MRL 64–2048). Cache embedding: **bật mặc định** (SQLite, máy A).
5. Server Qwen: thư mục riêng **ngoài** repo RAG-Anything, độc lập với `embedding-server` cũ.
6. `EMB_MAX_INPUT_TOKENS` mặc định **8192** (theo code tham chiếu; model hỗ trợ tới 32k).
7. `reproduce/` và các example dùng LightRAG: xoá.

## a) Luồng hiện tại (trước migration)

```
INDEX (worker, ami_rag/workers/ingest_worker.py:177 handle_event)
  ├─ download file gốc từ MinIO            ingest_worker.py:215
  ├─ PARSE (MinerU)                        raganything/processor.py:387 parse_document
  │    └─ parse cache (LightRAG KV storage) processor.py:270-385
  ├─ Lưu assets + content_list vào MinIO   ingest_worker.py:232,244
  └─ INSERT  raganything/processor.py:2224 insert_content_list
       ├─ Text:  ainsert → LightRAG        raganything/utils.py:390-467
       │    (chunk → embed → extract_entities → merge_nodes_and_edges → doc_status)
       └─ Multimodal: processor.py:630 → batch_type_aware :895
            1. sinh mô tả bằng LLM/VLM     modalprocessors.py:1016,1222,1417,1594 (video :360)
            2. chunk template (PROMPTS)      processor.py:1147 _apply_chunk_template
            3. text_chunks + chunks_vdb     processor.py:1263
            3.5 entities_vdb + graph + full_entities  processor.py:1280
            4. extract_entities (LLM/chunk)  processor.py:1434
            5. belongs_to relations         processor.py:1463
            6. merge_nodes_and_edges         processor.py:1527
            7. cập nhật doc_status          processor.py:1564
  VERIFY   ingest_worker.py:107 → index_check.py:146 inspect_document
  REGISTRY Mongo `multimodal_rag_documents` storage/rag_documents.py

QUERY (API /v2/rag, ami_rag/api/routes/rag.py:302)
  _run_search :234 → RAGAnything.aquery_data raganything/query.py:221
    → LightRAG.aquery_data (mode + keywords-LLM + entities/relations/chunks_vdb)
    → filter → rerank service (ami_rag/core/rerank_client.py:17)
  RAGAnything.aquery :128 → LightRAG.aquery (LLM answer) / aquery_vlm_enhanced :460 (VLM)
  aquery_with_multimodal :306 → modal_caption_func :629-691 → aquery
```

## b) Bảng phân loại XOÁ / THAY / GIỮ

| Vùng | Phân loại | Lý do |
|---|---|---|
| `lightrag` imports: `raganything.py:27,28`, `processor.py:31`, `query.py:13,14`, `modalprocessors.py:20-27`, `utils.py` (ainsert), `ami_rag/core/factory.py` (build_rag), `index_check.py:168-217`, `cli.py:153`, `ingest_worker.py:104,117`, `api/routes/admin.py:56` | XOÁ | Thay bằng pipeline vector thuần + DocStatusStore riêng |
| `ainsert` (utils.py:390-467) | XOÁ | Insert mới: chunk → embed → upsert vector |
| `extract_entities` / `merge_nodes_and_edges` (processor.py:859-886,1434-1461,1527-1562; modalprocessors.py:828-891; index_check.py:200-219) | XOÁ | Không còn KG |
| `entities_vdb`, `relationships_vdb`, `full_entities`, `full_relations`, `entity_chunks`, `relation_chunks`, `chunk_entity_relation_graph` | XOÁ | Không còn KG |
| `chunks_vdb` (processor.py:1272; modalprocessors.py:579,819; index_check.py:149,185,188; ingest_worker.py:117) | THAY | Interface `VectorStore` (upsert/search/delete_by_doc/count) trên Qdrant |
| `doc_status` của LightRAG (processor.py:134-229,687-716,781,913,1564-1665,1678-1732; index_check.py:142-143,174,203; admin.py:56) | THAY | `DocStatusStore` Mongo mở rộng `multimodal_rag_documents` theo stage |
| `QueryParam`, `mode=` (query.py; rag.py:224,253) | THAY | Query mới: embed → search → rerank tuỳ chọn → LLM 1 lần |
| `keywords` (processor.py:1507; modalprocessors.py:844-861; rag.py:292) | XOÁ | Sản phẩm của KG pipeline |
| `EmbeddingFunc` (factory.py:218-246) | THAY | Interface `Embedder` + `RemoteEmbedder` (client) → server máy B |
| `gemini`/`genai` (settings.py:14-19; factory.py:27-40,99-161,191-215,221-233; index_check.py:36; .env.ami.example:3-7,57) | XOÁ | Bỏ Gemini hoàn toàn |
| `rerank_client.py` + `RERANK_*` (settings.py:80-82; factory.py:278; rag.py:43-46,93-110; cli.py:474-478) | GIỮ | Tuỳ chọn độc lập, không liên quan KG/Gemini |
| Parser (raganything/parser.py + MINERU_*) | GIỮ | Giữ nguyên theo yêu cầu |
| Parse cache (processor.py:270-385) | THAY (nhẹ) | Chuyển backend khỏi LightRAG KV sang persistence lớp riêng |
| Modal processors (caption, build_modal_chunk_metadata, prompt templates) | GIỮ (đổi đích ghi) | Chỉ sinh mô tả + chunk, không ghi graph |
| `reproduce/*.py`, examples dùng LightRAG | XOÁ | Đã quyết định |
| Tests nhắc lightrag (~20 file) | THAY | Theo pipeline mới |

## c) Các điểm gọi LLM / API ngoài

**Index:**
1. `factory.py:28-40` `gemini_complete_if_cache` — XOÁ
2. `factory.py:42-55` `openai_complete_if_cache` (qwen) — THAY thành `describe_func`/`answer_func`
3. `factory.py:104-159` Gemini vision (`google-genai`) — XOÁ
4. `factory.py:163-186` qwen vision — THAY thành `describe_func`
5. `factory.py:194-215` `gemini_embed` — XOÁ, thay `RemoteEmbedder` → máy B
6. `factory.py:235-245` `openai_embed` (qwen vLLM) — XOÁ ở GĐ 4 (khi bỏ LightRAG)
7. `modalprocessors.py:1016,1222,1417,1594` + `modalprocessors_video.py:360` `modal_caption_func` — GIỮ (describe_func)
8. Ngầm trong lightrag (ainsert/aquery): extract entities, keywords, answer — XOÁ hết khi bỏ LightRAG
9. `index_check.py:85-86` preflight probe — THAY (probe embed server /health + /info)
10. `rerank_client.py:17-37` POST `/rerank` — GIỮ (tuỳ chọn)
11. `crawl_images.py:86` httpx tải ảnh crawl — GIỮ

**Query:**
12. `query.py:629,658,675,691` modal_caption_func mô tả content query — XOÁ (query mới chỉ embed câu hỏi)
13. `query.py:920,925` vision_model_func VLM — THAY thành `answer_func` đúng 1 lần
14. LightRAG nội bộ: keywords-LLM + answer-LLM — XOÁ
15. `cli.py:474-476` ping rerank /health trong `status` — GIỮ nhưng làm tuỳ chọn (`--check-*`)

**Còn lại sau migration:** describe_func (index, tuỳ chọn), answer_func (1 lần/query), embed server (máy B), rerank (tuỳ chọn), MinIO/Mongo/Redis (storage).

## d) CLI hiện tại

**Lệnh/flag:** `reindex` (--all, --doc-ids, --type, --limit, --batch-size, --force, --dry-run, --direct, --repair), `verify` (--doc-ids, --type, --limit, --batch-size, --all), `status`, `purge-doc` (--doc-id). Entry: `ami-rag = ami_rag.cli:main`.

**Hard-code Gemini:** `settings.py:14-19`, `factory.py` các nhánh gemini, `index_check.py:36` (hint "fix GEMINI_API_KEY"), `.env.ami.example:3-7,57`.

**Trạng thái doc (2 lớp):**
- LightRAG `doc_status` (`{WORKSPACE}_doc_status`): `chunks_list`, PROCESSED/FAILED → `verify`/`repair`/`purge-doc` phụ thuộc.
- `RagDocumentsRepo` (Mongo `multimodal_rag_documents`): processing/processed/failed/stale, attempts, error_code/stage, source_hash → `status`, `reindex`, skip-unchanged, mark_stale.

CLI mới đọc từ `DocStatusStore` (Mongo, mở rộng `RagDocumentsRepo`): thêm `stage`, `content_hash`, `embed_model`, `embed_dim`, `chunker_version`, `chunk_count`.

## e) Dependency

**Gỡ:** `lightrag-hku` (pyproject:25, requirements.txt:3), `google-genai` (service/all extras).
**Máy A giữ:** mineru[core], tqdm, fastapi, uvicorn, pydantic-settings, redis, pymongo, qdrant-client, httpx, minio, prometheus-client, opentelemetry, Pillow, reportlab.
**Máy A không được có:** torch, transformers, flash-attn (chỉ thuộc server máy B).
**Máy B (qwen-embedding-server):** torch (base image), transformers>=4.57, qwen-vl-utils>=0.0.14, pillow, accelerate, fastapi, uvicorn, pydantic.
