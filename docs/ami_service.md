# AMI multimodal RAG service (`ami_rag`)

Package `ami_rag/` trong fork RAG-Anything này là service multimodal RAG của hệ thống AMI: một ingest worker (parse tài liệu bằng MinerU, chunk + embed bằng vector pipeline thuần) và một API truy xuất `POST /v2/rag`. Nó thay thế hoàn toàn repo cũ `ami-multimodal-retrieval`. Pipeline đã chuyển sang **vector thuần**: không còn LightRAG (không entity, không knowledge graph, không neo4j).

## 1. Kiến trúc

```
ami_data (backend docs: MinIO + MongoDB organization_db.documents)
   │  XADD event mỏng (event / document_id / content_hash / document_type / org_id / ts)
   ▼
Redis Streams  rag:ingest  (consumer group ami-rag)
   ▼
ami_rag worker  (ami-rag-worker, hoặc nhúng trong API khi WORKER_ENABLED=true)
    ├─ select_source(doc)                                          (mục 2)
    │    ├─ mongo_text : text Mongo `content` → content_list = [text] (+ item image crawl khi có section 'Hình ảnh')
    │    └─ minio_parse: tải file gốc từ MinIO → MinerU parse → content_list (text/image/table/equation)
    │         └─ PARSE_TEXT_SOURCE=backend_data: text/bảng dựng lại bằng pdfplumber/fitz + OCR service
    │            (fallback Vinter, mục 5.4); ảnh vẫn lấy từ MinerU
    ├─ upload ảnh/bảng/công thức lên MinIO  rag-assets/{doc_id}/…  (gắn item["asset_key"])
    ├─ lưu content_list.json vào MinIO
    ├─ describe: mô tả modal (stage duy nhất có thể gọi LLM, tuỳ chọn) → descriptions.json
    │    └─ ảnh: caption theo IMAGE_DESCRIBE_MODE — Gemini (có CAPTION_LLM_API_KEY) / VLM / text thô (mục 5.3)
    ├─ chunk: gom item text liên tiếp + cắt theo ngân sách (CHUNK_SIZE/CHUNK_OVERLAP/CHUNK_MIN_CHARS),
    │    prefix tiêu đề doc → chunks.json                          (chunk v2, mục 5.5)
    ├─ embed: BGEM3Embedder (HTTP máy B, BAAI/bge-m3 — dense 1024 + sparse)
    │    → upsert Qdrant named vectors `dense` + `sparse` (delete_by_doc trước, idempotent)  (mục 5.6)
    └─ mark_indexed: ghi trạng thái vào DocStatusStore
         ▼
   Qdrant — collection `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`
            (theo `.env.ami.example`: multimodal__bge-m3__v1)
   MongoDB `organization_db.multimodal_rag_documents` (DocStatusStore, một dòng/doc theo stage)

Hệ thống AI ──POST /v2/rag──▶ ami_rag API: embed_query dense+sparse (máy B) → hybrid search Qdrant (RRF dense+sparse)
                               → rerank BGE → resolve doc Mongo → presign MinIO (artifact_url/artifact_urls, file_url)
                               → response
```

- Event mỏng: queue chỉ mang `event` (`created`/`updated`/`deleted`), `document_id`, `content_hash`, `document_type`, `org_id`, `ts` (`ami_rag/queue/events.py`). Worker đọc nội dung từ Mongo/MinIO.
- Doc id = MongoDB ObjectId. Mỗi chunk payload mang `doc_id` trực tiếp nên API tra ngược về Mongo (`ami_rag/api/resolver.py::resolve_doc_id`).
- `created`/`updated` → `ensure_pending` + chạy pipeline từ stage `parse`; `deleted` → xoá vector Qdrant + asset MinIO + dòng registry. Nguồn không đổi (`content_hash` trùng, status `indexed`, đúng `embed_model`/`chunker_version`) thì bị bỏ qua, cả ở worker lẫn ở CLI `reindex`.
- Pipeline (`ami_rag/core/vector_pipeline.py::VectorPipeline.run`): 5 stage `parse → describe → chunk → embed → indexed`, mỗi stage ghi trạng thái vào DocStatusStore. Kết quả trung gian lưu MinIO (`content_list.json`, `descriptions.json`, `chunks.json`) nên retry chạy tiếp từ stage lỗi, không cần parse lại; stage embed đọc lại `chunks.json` nếu có.
- Embed (tài liệu): stage duy nhất gọi mạng thật; luôn `delete_by_doc` trước khi upsert (idempotent) rồi verify count. Embedder trả **dense + sparse** cùng lượt (`embed_documents_sparse`); cache SQLite (`EMBED_CACHE_PATH`) chỉ giữ dense — sparse tính lại mỗi lần gọi server. Describe là stage duy nhất có thể gọi LLM (tuỳ chọn): ảnh được caption bằng **Gemini `gemini-2.5-flash-lite`** (OpenAI-compatible, `CAPTION_LLM_*`) khi có API key; không key mà có `QWEN_VLM_MODEL` thì gửi ảnh cho VLM (`vision`); còn lại `none` = text thô caption + footnotes + ngữ cảnh (`IMAGE_DESCRIBE_MODE` — xem mục 5.3). Bảng/công thức/generic vẫn do Qwen text-only tổng hợp (`QWEN_LLM_*`).
- Retry: `WORKER_MAX_DELIVERY` lần; quá ngưỡng thì dòng registry được đánh dấu `failed` và message được ack. Event lỗi chưa đủ ngưỡng được để pending và tự giao lại (XAUTOCLAIM, `ami_rag/queue/streams.py`) sau `WORKER_RETRY_IDLE_MS` (mặc định 10 phút; cần Redis ≥ 6.2) — đặt đủ lớn hơn thời gian parse MinerU dài nhất để worker khác không "cướp" message đang xử lý. Khi `failed`, `attempts` được reset nên `reprocess_failed`/`retry --all-failed` có đủ ngân sách thử lại.
- LLM (describe): profile `qwen-selfhost` (vLLM, OpenAI-compatible) qua `QWEN_LLM_*` cho bảng/công thức/generic; **caption ảnh dùng Gemini** `gemini-2.5-flash-lite` qua OpenAI-compatible endpoint của Google (`CAPTION_LLM_*`, mục 5.3). Rerank là service ngoài `POST {RERANK_BASE_URL}/rerank` (mặc định legacy BGE `BAAI/bge-reranker-v2-m3` :8010, documents text-only, client chia batch ≤ 100/request; `RERANK_MULTIMODAL=false`; lỗi rerank thì trả chunk không điểm (fallback)).
- Embedding: mặc định backend `bge` — **`BAAI/bge-m3`** serve qua `embedding-server` (máy B, repo riêng, port **8008**, auth `Bearer <EMBED_SERVER_TOKEN>`; `GET /v1/models` + `POST /embed` trả dense 1024 + sparse lexical weights, tối đa 64 text/request). Client `BGEM3Embedder` (`ami_rag/core/bge_embedder.py`). Gateway `nemotron-vl-vllm` (8080) và `qwen-embedding-server` (8007) **đã thay bằng bộ BGE**; chọn backend qua `EMBED_BACKEND` (`auto`: model chứa `bge` → bge, prefix `Qwen/` → custom, còn lại → openai). Ở truy vấn: `embed_query_sparse` trả dense + sparse cho query, Qdrant fuse 2 prefetch bằng RRF (mục 5.6). BGE-M3 text-only: chunk ảnh **không** kèm ảnh bytes — ảnh hiện diện trong vector qua caption tiếng Việt do stage describe (Gemini) sinh (mục 5.3). Máy A chỉ gọi HTTP — không cài torch/transformers, không cần GPU cho embed.

## 2. Quyết định nguồn nội dung

`ami_rag/sources.py::select_source`:

| Điều kiện trên `organization_db.documents` | Nguồn | Hành vi |
|---|---|---|
| `original_file_name` rỗng hoặc đuôi `.txt` (text, crawl, seed) | `mongo_text` | dùng text trong Mongo `content`; rỗng thì bỏ qua |
| còn lại (pdf, doc, docx, ...) | `minio_parse` | tải file `file_path` từ MinIO, parse lại bằng MinerU; text/bảng Mongo **không dùng** |

`source_hash` (dấu vân tay quyết định có nạp lại hay không):
- `mongo_text`: sha256 của `content.strip()`.
- `minio_parse`: sha256 của `"minio|{file_path}"` (key MinIO duy nhất cho mỗi lần upload; `content_hash` Mongo bị bỏ qua có chủ đích).

Doc `minio_parse` không có `file_path` thì lỗi (`ValueError`).

**Ảnh crawl** (luôn bật, không có env tắt): bài crawl có section cuối dạng `**Hình ảnh:**` + các dòng `- <url ảnh>`. Parse stage tách section này (`ami_rag/workers/crawl_images.py::split_image_section` — chỉ nhận URL có đuôi ảnh, heading biến thể `**Hình ảnh:**`/`## HÌNH ẢNH`/`Hinh anh` đều nhận qua fold không dấu), tải từng ảnh về MinIO `rag-assets/{doc_id}/…` (idempotent, key = hash URL, timeout 60s/ảnh, semaphore 4) rồi append item `image` kèm `asset_key` — ảnh crawl đi tiếp qua describe (caption Gemini/VLM như ảnh pdf/docx, mục 5.3), chunk modality `image`, embed caption text (BGE-M3 text-only, không nhúng ảnh bytes) và presign `/admin` như ảnh thường. Block URL bị xoá khỏi text chunk. Ảnh lỗi bỏ qua từng con; không tải được con nào thì giữ nguyên text gốc. Bài không có section → chỉ 1 text item như cũ (không tải gì).

## 3. Dữ liệu

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Mongo DB | `RAG_DB` = `organization_db` | dùng chung DB với backend ami_data |
| Registry / DocStatusStore | `organization_db.multimodal_rag_documents` (`RAG_DOCUMENTS_COLLECTION`) | một dòng/doc, `_id` = ObjectId dạng string, ghi bởi worker/CLI/admin |
| Qdrant | `QDRANT_URL` | vector chunk, collection `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`, named vectors `dense` + `sparse` (theo `.env.ami.example`: `multimodal__bge-m3__v1`) |
| MinIO asset | `{ASSET_PREFIX}/{doc_id}/` (`rag-assets/…`) trong `MINIO_BUCKET` | ảnh/bảng/công thức (tên `sha256[:16]` + ext), `content_list.json`, `descriptions.json`, `chunks.json` |
| Nguồn doc | `organization_db.documents` (`ORG_DB`/`DOC_COLLECTION`) | **chỉ đọc**, lọc `status: "active"` khi duyệt toàn bộ |

Quy ước collection vector (`ami_rag/core/embedder.py::collection_name`): `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` — đổi embed model hoặc `CHUNKER_VERSION` là sang collection mới, không bao giờ trộn vector hai embed model. Slug lấy từ phần sau `/` của `EMBED_MODEL` (`BAAI/bge-m3` → `bge-m3`). Giá trị `CHUNKER_VERSION` lấy từ env: `settings.py` mặc định **`v1`** và `.env.ami.example` cũng để **`v1`** (→ `multimodal__bge-m3__v1`). Collection mới dùng **named vectors** `dense` + `sparse` (hybrid, mục 5.6); collection legacy 1 vector vô danh không bổ sung được sparse (mục 5.6).

Ranh giới:

| Collection | Quyền của RAG |
|---|---|
| `documents` | chỉ ĐỌC (`ORG_DB`/`DOC_COLLECTION`) |
| `document_versions`, `organization_units`, `users` | không đọc trong code; chỉ dùng để `$lookup` khi truy vấn/báo cáo từ bên ngoài |
| `multimodal_rag_documents` | GHI (tạo collection, index, upsert/xoá) |
| Qdrant collection `multimodal__*` | GHI |

Lưu ý vận hành:
- Quyền Mongo: user của service RAG cần `createCollection` + `createIndex` + đọc/ghi trên `multimodal_rag_documents` trong `organization_db` (`_ensure_indexes` tạo index `status`, `stage`; lỗi chỉ log warning), và chỉ cần `find` trên `documents`.
- Backup/restore/drop `organization_db` ảnh hưởng cả backend lẫn RAG; khi chỉ muốn xoá dữ liệu RAG, drop collection `multimodal_rag_documents` + collection Qdrant + prefix MinIO `rag-assets/`, không drop database.

### Liên kết với organization_units / users

Mỗi dòng registry được ghi thêm các field liên kết khi `indexed` (`ami_rag/sources.py::doc_link_fields`, truyền qua `meta=` vào `mark_indexed`; dòng `pending`/`processing`/`failed` chưa có):

| Field registry | Kiểu | Liên kết tới |
|---|---|---|
| `_id` | string | `documents._id` dưới dạng chuỗi |
| `document_oid` | ObjectId | `documents._id` (dùng cho `$lookup`) |
| `organization_unit_id` | giữ kiểu gốc của `documents` (ObjectId) | `organization_units._id` |
| `owner_id` | string | định danh OIDC: khớp `users.sub` / `users.preferred_username` / `users.email`, KHÔNG phải `users._id` (xem `ami_data/.../app/repositories/user_repository.py`, `find_all_users_with_upload_count`) |
| `document_type`, `title` | string | sao chép từ `documents` |

`$lookup` chỉ chạy trong cùng một database, vì vậy RAG phải dùng chung `organization_db`. Ví dụ:

```javascript
db.multimodal_rag_documents.aggregate([
  { $match: { status: "indexed" } },
  { $lookup: { from: "documents", localField: "document_oid", foreignField: "_id", as: "doc" } },
  { $lookup: { from: "organization_units", localField: "organization_unit_id", foreignField: "_id", as: "unit" } },
  { $lookup: {
      from: "users",
      let: { owner: "$owner_id" },
      pipeline: [
        { $match: { $expr: { $and: [
            { $ne: ["$$owner", null] },
            { $or: [
              { $eq: ["$sub", "$$owner"] },
              { $eq: ["$preferred_username", "$$owner"] },
              { $eq: ["$email", "$$owner"] } ] } ] } } }
      ],
      as: "owner" } },
  { $project: { status: 1, title: 1, "unit.name": 1, "owner.preferred_username": 1, "doc.status": 1 } }
])
```

(Tên field hiển thị như `unit.name` tuỳ schema `organization_units` của backend; điều chỉnh cho khớp.)

### Field của `multimodal_rag_documents`

Các field (`ami_rag/storage/doc_status.py`):

| Field | Ý nghĩa |
|---|---|
| `_id` | document id |
| `document_oid`, `organization_unit_id`, `owner_id`, `document_type`, `title` | field liên kết (bảng trên), chỉ có sau lần `indexed` |
| `status` | `pending` / `processing` / `indexed` / `failed` / `stale`; legacy `processed` (pipeline cũ) đọc là `stale` |
| `stage` | stage hiện tại: `parse` / `describe` / `chunk` / `embed` / `indexed` |
| `source`, `content_hash` | nguồn (`mongo_text`/`minio_parse`) và vân tay nội dung |
| `source_path`, `file_path` | key MinIO của file gốc và đường dẫn citation |
| `parser` | giá trị `PARSER` lúc nạp |
| `assets` | danh sách key MinIO đã upload |
| `counts`, `page_count` | số item theo modality (`text/image/table/equation/other`), số trang |
| `chunk_count`, `embed_model`, `embed_dim`, `chunker_version` | thông tin vector (để kiểm tra tương thích khi reindex) |
| `attempts`, `error`, `error_stage` | theo dõi lần thử/lỗi (`error` cắt 2000 ký tự) |
| `created_at`, `updated_at` | thời điểm |

## 4. Biến môi trường

File mẫu: `.env.ami.example` (copy thành `.env`). Nguồn sự thật: `ami_rag/settings.py`.

| Nhóm | Biến | Mặc định | Ý nghĩa |
|---|---|---|---|
| LLM | `QWEN_LLM_BASE_URL` / `QWEN_LLM_MODEL` / `QWEN_LLM_API_KEY` | `http://vllm:8000/v1` / `Qwen/Qwen3-32B` / rỗng | qwen-selfhost (OpenAI-compatible), dùng cho describe bảng/công thức/generic |
| | `QWEN_VLM_MODEL` | rỗng | VLM cho caption ảnh ở chế độ `vision`. Rỗng = auto sẽ dùng Gemini (có key) hoặc text thô (mục 5.3) |
| Caption ảnh (Gemini) | `IMAGE_DESCRIBE_MODE` | `auto` | `auto`/`gemini`/`vision`/`none`. `auto`: có `CAPTION_LLM_API_KEY` → `gemini`; không key nhưng có `QWEN_VLM_MODEL` → `vision`; còn lại → `none` (text thô caption + footnotes + ngữ cảnh, không gọi LLM). Legacy `"text"` → warning + `none` |
| | `CAPTION_LLM_API_KEY` | rỗng | API key Gemini, **chỉ từ env** — không ghi vào file trong git. Có key là bật caption Gemini |
| | `CAPTION_LLM_BASE_URL` / `CAPTION_LLM_MODEL` | `https://generativelanguage.googleapis.com/v1beta/openai/` / `gemini-2.5-flash-lite` | OpenAI-compatible endpoint của Google AI Studio |
| | `CAPTION_LLM_TIMEOUT` | `120` | giây/request caption (mỗi ảnh 1 request) |
| | `CAPTION_LLM_MAX_CONCURRENCY` | `4` | số request caption song song (vừa fetch ảnh MinIO vừa gọi Gemini) |
| | `CAPTION_LLM_RETRIES` | `2` | retry do client OpenAI tự xử lý |
| | `CAPTION_LLM_LANGUAGE` | `vi` | ngôn ngữ mô tả ảnh (chữ trong ảnh vẫn ghi nguyên văn); rỗng = dùng prompt gốc. Xem mục 5.3 |
| OCR service (backend_data) | `PARSE_TEXT_SOURCE` | `backend_data` | nguồn text/table khi parse PDF/DOCX: `backend_data` (pdfplumber/fitz + OCR service) / `mineru` (giữ MinerU). Ảnh vẫn từ MinerU |
| | `OCR_SERVICE_URL` | `http://171.226.10.153:12006` | service chính: text hierarchy-parse + bảng Qwen |
| | `OCR_SERVICE_TIMEOUT` | `300` | giây |
| | `OCR_TABLE_ENDPOINT` / `OCR_TEXT_ENDPOINT` | `/api/v1/raw_miner_ocr/qwen` / `/api/v1/hierarchy-parse` | bảng Qwen / text + OCR |
| OCR fallback (Vinter, máy B) | `OCR_FALLBACK_ENABLED` | `true` | bật dự phòng khi service chính lỗi/timeout/trả rỗng (mục 5.4) |
| | `OCR_FALLBACK_URL` | `http://localhost:8090` | gateway `vintern-vllm` máy B — máy A phải trỏ IP thật qua env |
| | `OCR_FALLBACK_MODEL` | `5CD-AI/Vintern-1B-v3_5` | chỉ để log/verify, không gọi |
| | `OCR_FALLBACK_TOKEN` | rỗng | **chỉ từ env**, không ghi vào file trong git |
| | `OCR_FALLBACK_TIMEOUT` / `OCR_FALLBACK_RETRIES` | `300` / `1` | giây; retry chỉ 429/5xx/transport |
| | `OCR_FALLBACK_ON_EMPTY` | `true` | primary trả rỗng cũng chuyển fallback (`false` = chỉ fallback khi lỗi) |
| | `OCR_FALLBACK_MODES` | `text,table` | tắt riêng từng loại: `text` / `table` / `""` (tắt hết); giá trị lạ → lỗi lúc khởi động |
| Embed server | `EMBED_SERVER_URL` | `http://localhost:8008` | trỏ đúng server tương ứng `EMBED_BACKEND` (embedding-server BGE-M3: 8008) |
| | `EMBED_SERVER_TOKEN` | rỗng | **chỉ từ env** (key trùng `BGE_API_KEY` trong `.env` của server), không ghi vào file trong git |
| | `EMBED_MODEL` | `BAAI/bge-m3` | dùng để xác minh với server lúc handshake |
| | `EMBED_BACKEND` | `bge` | `auto` / `custom` (qwen-embedding-server: `/info` + `/embed`) / `openai` (gateway nemotron: `/health` + `/v1/embeddings`) / `bge` (embedding-server: `/v1/models` + `/embed`); auto: model chứa `bge` → bge, prefix `Qwen/` → custom, còn lại → openai |
| | `EMBED_DIM` | `1024` | để xác minh (dim thực tế do server quyết định) |
| | `EMBED_TIMEOUT` | `60` | giây |
| | `EMBED_BATCH_SIZE` | `32` | số item mỗi request — **không vượt 64** (giới hạn server BGE-M3) |
| | `EMBED_MAX_CONCURRENCY` | `4` | số request đồng thời |
| | `EMBED_RETRIES` | `3` | retry cho lỗi tạm thời (timeout/5xx/429) |
| | `EMBED_MAX_INPUT_TOKENS` | `8192` | budget token mỗi item embed (máy A không tokenizer, xấp xỉ ~3 chars/token); item vượt bị cắt đầu trước khi gửi — content đầy đủ vẫn lưu Qdrant. Default = max seq của BGE-M3; server chạy thấp hơn thì giảm qua env |
| | `EMBED_IMAGE_TOKEN_RESERVE` | `0` | BGE-M3 text-only không kèm ảnh → không trừ budget token ảnh |
| | `EMBED_RETURN_SPARSE` | `true` | xin thêm `sparse_weights` từ server cho hybrid Qdrant (mục 5.6); cache embedding chỉ giữ dense |
| | `EMBED_CACHE_ENABLED` | `true` | cache embedding SQLite ở máy A |
| | `EMBED_CACHE_PATH` | `./embed_cache.db` | |
| Mongo | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | |
| | `RAG_DB` | `organization_db` | dùng chung DB với backend |
| | `RAG_DOCUMENTS_COLLECTION` | `multimodal_rag_documents` | registry / DocStatusStore |
| | `ORG_DB` / `DOC_COLLECTION` | `organization_db` / `documents` | chỉ đọc |
| Qdrant | `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / rỗng | |
| | `QDRANT_UPSERT_MAX_MB` | `16` | giới hạn ước lượng byte mỗi request upsert (Qdrant server mặc định giới hạn request 32 MB); batch được đóng gói theo byte VÀ số điểm |
| Index | `WORKSPACE` | `multimodal` | tiền tố collection Qdrant `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` |
| Parser | `PARSER` | `mineru` | |
| | `PARSE_METHOD` | `auto` | |
| | `PARSER_OUTPUT_DIR` | `./output` | |
| MinerU (chỉ khi `PARSER=mineru`) | `MINERU_BACKEND` | `pipeline` | truyền `-b`; xem mục 5.1 vì sao không bỏ trống |
| | `MINERU_DEVICE` | rỗng | rỗng = tự nhận; `cuda` / `cuda:0` / `cpu`; map sang env `MINERU_DEVICE_MODE` của tiến trình mineru |
| | `MINERU_VIRTUAL_VRAM_SIZE` | `0` | GB; `0` = dùng VRAM thật; >0 map sang env cùng tên, MinerU chọn batch size như thể GPU có chừng đó VRAM |
| | `MINERU_LANG` | rỗng | gợi ý ngôn ngữ OCR (`ch`, `en`, `latin`, ...); rỗng = mặc định MinerU |
| | `MINERU_SOURCE` | rỗng | nguồn model: `huggingface` / `modelscope` / `local`; rỗng = mặc định |
| | `MINERU_TIMEOUT` | `1800` | giây/tài liệu; `0` = không giới hạn |
| Chunking | `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `100` | token xấp xỉ — `chunk_text` tính ký tự theo tỉ lệ ~4 ký tự/token (`max_chars = max(512, CHUNK_SIZE*4)`) |
| | `CHUNK_MIN_CHARS` | `200` | chunk ngắn hơn ngưỡng này bị gộp vào chunk kề (0 = tắt). Xem mục 5.5 |
| Queue | `REDIS_URL` | `redis://localhost:6379/0` | |
| | `RAG_STREAM` | `rag:ingest` | |
| | `RAG_CONSUMER_GROUP` | `ami-rag` | |
| | `RAG_CONSUMER_NAME` | rỗng | rỗng = tự sinh |
| Worker | `WORKER_ENABLED` | `true` | `true`: API nhúng worker; compose đặt `false` cho `ami-rag-api` và chạy `ami-rag-worker` riêng |
| | `WORKER_BATCH` | `10` | |
| | `WORKER_POLL_BLOCK_MS` | `5000` | |
| | `WORKER_MAX_DELIVERY` | `3` | |
| | `WORKER_RETRY_IDLE_MS` | `600000` | Sau bao lâu (ms) message pending lỗi được giao lại |
| | `WORKER_METRICS_PORT` | `9109` | `/metrics` của `ami-rag-worker` |
| Rerank | `RERANK_BASE_URL` | `http://localhost:8010` | service BGE reranker (repo `rerank`, text-only) |
| | `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | rỗng hoặc chứa `bge` → legacy BGE (documents text-only) |
| | `RERANK_BACKEND` | `legacy` | `legacy` (BGE `:8010`, text-only) / `vllm` (gateway `nemotron-vl-vllm` máy B, multimodal — bộ cũ, code giữ lại cho legacy) / `auto` (model rỗng hoặc chứa `bge` → legacy, còn lại → vllm) |
| | `RERANK_MULTIMODAL` | `false` | chỉ backend `vllm` dùng (chunk kèm ảnh data URI từ MinIO); legacy BGE text-only luôn tắt |
| | `RERANK_TOP_K` | `5` | số tài liệu trả về mặc định khi request không chỉ định `top_k` |
| | `RERANK_TIMEOUT` | `60` | giây |
| | `RERANK_MAX_INPUT_TOKENS` | `6000` | budget token text mỗi document rerank; vượt bị cắt đầu (kèm ảnh → trừ `RERANK_IMAGE_TOKEN_RESERVE`) — rerank server cùng vLLM giới hạn `max_model_len` |
| | `RERANK_IMAGE_TOKEN_RESERVE` | `1792` | như `EMBED_IMAGE_TOKEN_RESERVE` |
| | `RERANK_FUSION` | `raw` | `raw` (so điểm tuyệt đối - hành vi cũ) / `rrf` (so hạng nội bộ 2 nhóm modality qua RRF + sàn điểm) — `rrf` hoạt động cả legacy (legacy trả full scores) |
| | `RERANK_RRF_K` | `10` | hằng số RRF: `fused = weight / (rrf_k + rank)` |
| | `RERANK_VISUAL_WEIGHT` | `0.8` | weight nhóm image trong RRF (nhóm còn lại = 1.0, mã hoá prior text) |
| | `RERANK_VISUAL_FLOOR` | `0.01` | raw score dưới floor nhóm image bị chặn không được promote (xếp cuối theo raw) |
| | `RERANK_TEXT_FLOOR` | `0.05` | như trên cho nhóm không-image (text/table/equation) |
| Retrieval | `RETRIEVAL_CHUNK_TOP_K` | `40` | số chunk tối thiểu lấy từ Qdrant trước khi rerank; tổng ứng viên = `max(RETRIEVAL_CHUNK_TOP_K, top_k * 2)` |
| Pipeline/CLI | `CHUNKER_VERSION` | `v1` | một phần tên collection Qdrant + kiểm tra tương thích lúc reindex; đổi giá trị ⇒ collection mới + reindex (mục 5.5). `settings.py` mặc định `v1` và `.env.ami.example` cũng để `v1` |
| | `CLI_STUCK_PROCESSING_MINUTES` | `60` | doc `processing` quá lâu bị `status` báo là treo |
| MinIO | `MINIO_ENDPOINT` | `localhost:9000` | |
| | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / rỗng | |
| | `MINIO_BUCKET` | `ami-data-documents` | |
| | `MINIO_SECURE` | `false` | |
| | `ASSET_PREFIX` | `rag-assets` | cũng là nơi ảnh crawl được tải về (`{doc_id}/…`) |
| | `MINIO_PRESIGN_EXPIRES` | `3600` | giây, cho `artifact_url`/`file_url` |
| API | `API_HOST` / `API_PORT` | `0.0.0.0` / `8009` | |
| | `RAG_API_KEY` | rỗng | rỗng = không auth |
| Tracing | `OTEL_EXPORTER_OTLP_ENDPOINT` | rỗng | rỗng = tracing tắt; đọc từ `os.environ`. Chỉ API dùng; compose để trống cho worker. Trỏ tới OTLP/HTTP của Tempo riêng `tempo-multimodal-rag` (xem 9) |
| | `OTEL_SERVICE_NAME` | `multimodal-rag-retrieval` | |
| | `TRACE_OUTPUT_MAX_LEN` | `2000000` | giới hạn ký tự event `documents_json`; `0` = không cắt |
| | `TRACE_OUTPUT_DOCUMENTS` | bật | `0` = bỏ payload document trong span event + log (còn `num_docs`/`duration`/`trace_id`); dashboard panel "Output" đọc `event.documents_json` nên chỉ tắt khi không cần xem nội dung |

Lưu ý: `OTEL_*` và `TRACE_OUTPUT_MAX_LEN` do `observability.py` đọc trực tiếp từ `os.environ` (không qua `Settings`). Docker Compose nạp `.env` vào môi trường container nên hoạt động; khi chạy local bằng `make start_backend`, cần `export` các biến này trong shell.

Chênh lệch giữa default trong `settings.py` và `.env.ami.example` (file mẫu là gợi ý vận hành, `settings.py` là nguồn sự thật khi không set env):

| Biến | `settings.py` | `.env.ami.example` |
|---|---|---|
| `EMBED_SERVER_URL` | `http://localhost:8008` | `http://bge-m3-embed:8008` (hostname service trong Docker của bộ embed) |
| `MINERU_VIRTUAL_VRAM_SIZE` | `0` (= tự nhận VRAM thật) | `4` (giới hạn VRAM khi GPU dùng chung) |

## 5. Chạy local và Docker

Local (cần Redis, Mongo, Qdrant, MinIO, rerank service, embed server sẵn sàng):

```bash
make setup                    # uv sync --extra service
cp .env.ami.example .env      # điền EMBED_SERVER_TOKEN, MINIO_SECRET_KEY, hosts...
make start_backend            # API :8009 (uvicorn --reload)   | hoặc: ami-rag-api
make start_worker             # ingest worker                  | hoặc: ami-rag-worker
make test                     # pytest -v tests/ami_service/
make style / make lint        # ruff trên ami_rag và tests/ami_service
```

Docker (`docker-compose.ami.yml`, `Dockerfile.ami`, `python:3.12-slim`):

```bash
cp .env.ami.example .env      # host Mongo/Qdrant/Redis/MinIO/rerank/embed = tên container trên ami-network
make start_docker             # tạo network ami-network nếu thiếu, rồi up --build -d
make ps && make health        # /healthz + đếm series multimodal_rag_retrieval_* và multimodal_rag_ingest_*
make logs SERVICE=ami-rag-worker
make restart | make down | make docker-clean
```

- Build image chậm (~20 phút khi đổi code): `Dockerfile.ami` chép `ami_rag/` trước `pip install ".[service]"` nên mỗi lần sửa code `pip` tải lại toàn bộ gói.
- Hai service dùng chung image: `ami-rag-api` (port 8009) và `ami-rag-worker` (port 9109 cho `/metrics`). Mongo, Qdrant, Redis, MinIO, rerank chạy ngoài compose trên network external `ami-network`; trong container, host trong `.env` phải là tên container, không phải `localhost`.
- GPU: khối `deploy.resources.reservations.devices` (nvidia) của `ami-rag-worker` trong `docker-compose.ami.yml` **đang bị comment — compose mặc định chạy CPU**. Có GPU thì bỏ comment khối này (xem 5.1). Embed/rerank không cần GPU ở máy A.
- Cache model MinerU: volume `mineru-models` mount tại `/models` (`HF_HOME=/models/huggingface`, `MINERU_MODEL_SOURCE=huggingface`), tránh tải lại mỗi lần rebuild. Volume `rag-output` mount tại `/app/output`.
- LibreOffice (`libreoffice-writer`) đã cài trong image để chuyển doc/docx sang PDF trước khi MinerU parse; chạy local cần tự cài LibreOffice.
- Biến Make ghi đè: `PORT`, `WORKER_METRICS_PORT`, `COMPOSE_FILE`, `SERVICE`, `NETWORK`, `CLI_SERVICE`.

### 5.1 MinerU: backend, GPU/VRAM, chạy CPU

Stage parse của pipeline gọi `parser.parse_document(..., **parser_kwargs(settings))` (`ami_rag/core/vector_pipeline.py::_stage_parse`, kwargs dựng bởi `ami_rag/settings.py::parser_kwargs`). Hàm này đổi `MINERU_BACKEND/LANG/SOURCE/TIMEOUT` thành kwargs `backend/lang/source/timeout`, còn `MINERU_DEVICE` và `MINERU_VIRTUAL_VRAM_SIZE` thành `env` của tiến trình `mineru` (`MINERU_DEVICE_MODE`, `MINERU_VIRTUAL_VRAM_SIZE`). Biến rỗng/`0` thì không truyền. Parser khác `mineru` thì trả `{}`.

**Vì sao phải đặt `MINERU_BACKEND=pipeline`**: MinerU 3.4.x khi không có `-b` mặc định dùng `hybrid-engine`, chậm hơn ~6 lần và tốn VRAM gấp ~8 lần `pipeline`. Đừng để trống.

Số đo trên Colab T4, PDF 10 trang, MinerU 3.x:

| Cấu hình | Thời gian | VRAM đỉnh |
|---|---|---|
| `pipeline`, GPU auto (không đặt `MINERU_VIRTUAL_VRAM_SIZE`) | 50.5 s | 1821 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=4` | 57.0 s | 1135 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=8` | 49.8 s | 1817 MiB |
| `pipeline`, `MINERU_VIRTUAL_VRAM_SIZE=16` | 47.4 s | 3369 MiB |
| `hybrid-engine` (mặc định MinerU khi không có `-b`) | 286 s | 14539 MiB |
| Đường thư viện `parse_document` (GPU auto) | 46.4 s | 1821 MiB |
| CPU (3 trang) | 82.7 s (~27 s/trang) | n/a |
| GPU | ~5 s/trang | |

Khuyến nghị:
- `pipeline` dưới 4 GiB VRAM nên chạy GPU trực tiếp (`MINERU_DEVICE` rỗng = tự nhận). Local `make start_worker` thấy GPU là dùng ngay; Docker thì phải **bỏ comment khối `deploy` của `ami-rag-worker`** (mặc định compose là CPU).
- GPU dùng chung với service khác (reranker, vLLM, ...): đặt `MINERU_VIRTUAL_VRAM_SIZE=4` (đỉnh ~1.1 GiB, chậm hơn nhẹ; `.env.ami.example` đặt sẵn 4). Giá trị lớn hơn làm MinerU dùng batch lớn hơn nên tốn thêm VRAM, không nhanh hơn đáng kể.
- Chạy CPU: giữ khối `deploy` bị comment (mặc định compose) và đặt `MINERU_DEVICE=cpu`. Chậm ~5 lần (~27 s/trang): tăng `MINERU_TIMEOUT` cho file dài và `WORKER_RETRY_IDLE_MS` cho đủ lớn hơn thời gian parse.

### 5.2 Embed server (máy B) và Redis (đã gặp thực tế)

- Embedding + rerank chạy trên **server riêng** ở máy B. Bộ cũ gateway `nemotron-vl-vllm` (8080) + `qwen-embedding-server` (8007) **đã thay bằng bộ BGE** (2 service mới, chọn qua `EMBED_BACKEND`/`RERANK_BACKEND`):
  - **`bge` (mặc định)**: repo `embedding-server` (port **8008**), model **`BAAI/bge-m3`** — dense 1024 + sparse lexical weights, text-only, max_length 8192. Contract: `GET /v1/models` → `{"data": [{"id": "BAAI/bge-m3", ...}]}` (handshake so model); `POST /embed` với `{"texts": [str], "return_dense": true, "return_sparse": bool}` → `{"dense_vecs": [[float, ...]], "sparse_weights": [{"<token_id>": weight, ...}]}` (token_id dạng **chuỗi**); auth `Authorization: Bearer <EMBED_SERVER_TOKEN>` (trùng `BGE_API_KEY` trong `.env` của server; thiếu/sai → 401). Server nhận tối đa **64 text/request** (vượt → 400) nên `EMBED_BATCH_SIZE` không được đặt vượt 64; text rỗng/whitespace → server 400 (client thay bằng `"."`); 413/422 payload lớn, 429/5xx/504 transient (retry). Client `BGEM3Embedder` (`ami_rag/core/bge_embedder.py`): handshake = `/v1/models` + probe `/embed` 1 lần chốt dim, batch `EMBED_BATCH_SIZE` + `EMBED_MAX_CONCURRENCY`.
  - **Caption-only cho ảnh**: `_prepare_embed_items` (`vector_pipeline.py`) luôn trả **`list[str]`** (nội dung chunk) — **không nạp ảnh bytes** cho embed. BGE-M3 không nhận ảnh; ảnh hiện diện trong vector qua caption tiếng Việt do stage describe (Gemini) sinh (mục 5.3). Item còn mang `image_b64` (embedder fake/test) bị bỏ kèm log warning, không raise.
  - **Sparse → Qdrant**: server trả `sparse_weights` là dict `token_id → weight`; client `_to_sparse` sắp token_id tăng dần rồi gói `{"indices": [...], "values": [...]}` thành `SparseVector` khi upsert (mục 5.6). `EMBED_RETURN_SPARSE=true` mặc định; cache embedding chỉ chứa **dense** — luồng sparse luôn gọi server (`embed_documents_sparse` dùng `use_cache=False`), không cache sparse riêng.
- Rerank **legacy BGE `BAAI/bge-reranker-v2-m3`** trên port **8010** (repo `rerank`): documents **text-only**, tối đa **100 documents/request** (client batch nội bộ); `RERANK_MULTIMODAL=false` (không nhận ảnh). Score là logit (có thể âm), chỉ dùng so thứ hạng. Lỗi rerank → trả chunk không điểm (fallback, `rerank_fallback_total`).
- **Fusion 2 nhóm (`RERANK_FUSION`)**: điểm cross-encoder không so sánh được giữa chunk image (thấp, chỉ suy luận gián tiếp) và text/table (cao, khớp keyword trực tiếp) khi so điểm tuyệt đối trong 1 danh sách chung. `rrf`: rerank xin **full scores** (`top_n` = toàn bộ documents) rồi so **hạng nội bộ** từng nhóm qua RRF `fused = weight_m / (rrf_k + rank_m)`, kèm sàn điểm tuyệt đối — item dưới floor nhóm mình bị chặn không được promote. **`rrf` giờ hoạt động cả legacy**: legacy trả full scores nên không còn gate theo backend `vllm`. Response giữ nguyên `score` = raw relevance (không đổi ngữ nghĩa); thứ tự documents theo fused; `meta.fusion` = `"raw"|"rrf"` cho biết cấu hình đã áp dụng. Tune weight/floor qua `scripts/eval_rerank_fusion.py`.
- Handshake lúc khởi tạo: so model (và dim — probe embed 1 lần); lệch thì dừng với lỗi rõ ràng (không ghi vector khi chưa xác minh). Mỗi response được kiểm tra model/dim để phát hiện server bị đổi model giữa chừng.
- Retry chỉ cho lỗi tạm thời (timeout, 5xx, 429, mất kết nối), backoff + jitter; **không retry** 4xx (401 token sai, 400 input sai định dạng). Circuit breaker: 5 lô liên tiếp không với tới server thì dừng cả lô sớm — doc chưa xử lý không bị đánh fail hàng loạt. Lỗi `embed server unreachable` trong một doc không làm hỏng doc khác.
- Cache embedding SQLite ở máy A (`EMBED_CACHE_ENABLED`, key = `model|dim|instruction_ns|sha256(text)`); `reindex --dry-run` dùng `count_cache_hits` để báo trước số chunk trúng cache, không gọi HTTP.
- **Kích thước input gửi lên máy B**:
  - BGE-M3 giới hạn max_length **8192** — server phải chạy 8192, không thì giảm `EMBED_MAX_INPUT_TOKENS` qua env; client head-truncate item vượt budget trước khi gửi (`EMBED_IMAGE_TOKEN_RESERVE=0`). Rerank cắt qua `RERANK_MAX_INPUT_TOKENS`. Content đầy đủ vẫn lưu Qdrant payload nên LLM vẫn thấy toàn văn khi trả lời; chỉ vector/rerank tính trên phần đầu.
  - Qdrant giới hạn request (mặc định 32 MB): `QDRANT_UPSERT_MAX_MB=16` — upsert đóng gói batch theo byte ước lượng (payload + vector) và số điểm; doc nhiều bảng lớn (payload `content` + `table_body`) không còn gộp vượt giới hạn trong một request.
- Hết credit/quota hay model quá tải trên server (5xx sau retry) → doc ở stage `embed` thành `failed` với lý do rõ ràng; chạy lại `retry` sau khi server ổn định.
- Redis: `redis-py` ≥ 8 mặc định `socket_timeout=5s`, trùng với `XREADGROUP BLOCK` (`WORKER_POLL_BLOCK_MS`) nên worker báo `queue read failed: Timeout reading from redis`. Client của worker đặt `socket_timeout = WORKER_POLL_BLOCK_MS/1000 + 5` và `health_check_interval=30` (`ingest_worker.py::_build_default_deps`).

### 5.3 Caption ảnh (Gemini 2.5 Flash Lite) + materialize ảnh từ MinIO

Ảnh sau parse **chỉ còn ở MinIO** theo `asset_key` (file local của parse stage nằm trong temp dir đã bị xoá ngay sau parse). Trong khi đó `ImageModalProcessor` đọc ảnh từ `img_path` qua `Path.exists()` — trước đây `img_path` trỏ đúng temp dir đã xoá đó, nên **nhánh `vision` chưa từng chạy được** (kể cả `reindex --from-stage describe`); path OCR-ảnh text-only (OCR service ~30s/ảnh) cũng vậy. Giải pháp hiện tại: chuyển hoàn toàn sang **caption bằng Gemini `gemini-2.5-flash-lite`** (OpenAI-compatible endpoint của Google, `CAPTION_LLM_*`) kèm bước materialize ảnh:

1. **Materialize** (`vector_pipeline._materialize_images`): với mọi image-item có `asset_key`, fetch bytes từ MinIO **song song** theo semaphore `CAPTION_LLM_MAX_CONCURRENCY` → ghi vào `tempfile.TemporaryDirectory()` (tên `sha1(asset_key)[:12]` + ext suy từ magic bytes) → tạo `describe_items` là shallow copy gán `img_path` = file tạm (item thiếu `asset_key` giữ `img_path` gốc). **`content_list` gốc không bị mutate** (chunk + `save_content_list` dùng bản gốc); temp dir được xoá trong `finally` sau khi describe xong.
2. **Caption** (`factory._build_caption_func`): `AsyncOpenAI(base_url=CAPTION_LLM_BASE_URL, api_key=…)` — closure nhận `(prompt, system_prompt, image_data=<base64>, …)`, tái dùng `_openai_style_messages` dựng data URI (sniff mime png/webp/gif/jpeg). `ImageModalProcessor` vẫn là processor chạy mô tả (đọc file tạm, build prompt vision có caption/footnote/ngữ cảnh, parse JSON `detailed_description` + `entity_info`).

Ảnh crawl (mục 2) cũng chỉ có `asset_key` (không có file local) nên đi qua đúng bước materialize + caption này như ảnh parse từ pdf/docx.

Chế độ (`IMAGE_DESCRIBE_MODE`, xem bảng mục 4): `gemini` (có key) → caption Gemini; `vision` → gửi ảnh cho Qwen VLM (`QWEN_VLM_MODEL`); `none` → **text thô** `_describe_image_raw` = caption + footnotes + ngữ cảnh, không gọi LLM. `auto`: có `CAPTION_LLM_API_KEY` → `gemini`, elif có `QWEN_VLM_MODEL` → `vision`, else → `none`. Legacy `"text"` tự chuyển `"none"` kèm warning (path OCR-ảnh đã xoá).

**Ngôn ngữ mô tả (`CAPTION_LLM_LANGUAGE`, mặc định `vi`; để rỗng = prompt gốc, mô tả tiếng Anh).** Corpus và truy vấn là tiếng Việt nhưng prompt gốc ("comprehensive and detailed") sinh mô tả tiếng Anh dài dòng, làm vector ảnh lệch xa truy vấn. Đo thực tế trên index (cosine giữa vector truy vấn "sân bóng đá cơ sở Hà Đông" và vector chunk ảnh sơ đồ mặt bằng): **caption EN = 0.337, caption VI đầy đủ = 0.484**. Cơ chế: `ImageModalProcessor` nối thêm `vision_prompt_language_note` khi `item_info` có `caption_language`/`doc_title` (do `_stage_describe` truyền vào, lấy từ `CAPTION_LLM_LANGUAGE` và tiêu đề doc) — yêu cầu viết mô tả bằng ngôn ngữ đó, **ghi nguyên văn mọi chữ trong ảnh** (không dịch, không diễn giải) và liệt kê đầy đủ không bỏ sót; với sơ đồ/tổ chức/schematic phải nêu rõ từng phần tử nằm ở đâu so với nhau và mọi mũi tên theo hướng nào; đồng thời cấm bình luận kiểu "matching the context mentioned in the text" vốn làm nhèu nội dung. Đổi biến này cần `ami-rag reindex --all --from-stage describe` (chỉ gọi lại Gemini cho ảnh; chunk text giữ nguyên nên embed trúng cache).

**Kiểm chứng caption trước khi reindex cả doc**: `scripts/recaption_image.py` đi đúng đường caption của pipeline (cùng processor, prompt, ngữ cảnh, ngôn ngữ) nhưng **dừng trước khi ghi `descriptions.json`/chunk/embed** — thử vài ảnh với chi phí đúng bằng số ảnh đó:

```bash
python scripts/recaption_image.py --doc <doc_id> --list                    # 0 call: liệt kê ảnh
python scripts/recaption_image.py --doc <doc_id> --page 13 --prompt-only   # 0 call: in prompt
python scripts/recaption_image.py --doc <doc_id> --page 13                 # 1 call Gemini mỗi ảnh
```

Lệnh cuối in ra `description` sinh được và `content` chunk sẽ ghi vào Qdrant.

> **Bảng không có mô tả LLM khi Qwen không với được**: bảng/công thức/generic dùng `QWEN_LLM_BASE_URL` (mặc định `http://vllm:8000/v1`). Nếu host đó chết thì processor nuốt lỗi và trả `str(modal_content)` — dict-repr của item. Pipeline nay nhận ra (so khớp với `str(item)`) ở **cả** `_describe_item` lẫn `_build_chunks` nên dict-repr không còn lọt vào `descriptions.json`/chunk; thay bằng caption/text thô (bảng vẫn giữ nguyên `table_body`). Item bảng không có `table_body` lẫn caption thì bị bỏ khỏi index. Vì việc dọn nằm ở **lúc dựng chunk**, chạy `ami-rag reindex --all --from-stage chunk` là sạch được mà **không tốn call Gemini nào**. Khi ưng thì áp dụng cho doc: `ami-rag reindex --doc <doc_id> --from-stage describe --yes` (lưu ý: stage describe **ghi đè** `descriptions.json` không có điều kiện — nếu caption lỗi/hết quota thì rơi về text thô, nên xem log trước khi dọn collection cũ).

An toàn + chi phí:
- Mỗi ảnh đúng **1 call Gemini** (không còn ~30s OCR/ảnh); `CAPTION_LLM_MAX_CONCURRENCY` giới hạn số request song song (vừa fetch MinIO vừa gọi Gemini).
- Gemini lỗi (raise/trả rỗng/quota), ảnh thiếu `asset_key`, ảnh không có gì để nói → **không crash**: rơi về `_describe_image_raw` (text thô caption + footnotes + ngữ cảnh), log warning, `descriptions.json` vẫn ghi. `ImageModalProcessor` nuốt lỗi nội bộ trả dict-repr của item — pipeline phát hiện bằng so khớp để không lưu rác vào chunk.
- Không có Gemini key và không có VLM ⇒ mode `none`: chunk ảnh vẫn có caption MinerU + ngữ cảnh xung quanh làm text phân biệt cho vector search/rerank (giới hạn đã biết với ảnh thuần hình).

Doc đã index trước khi có tính năng này mang `descriptions.json` cũ — cần reindex từ stage `describe` (xem mục `reindex` bên dưới).

### 5.4 OCR fallback Vinter (máy B)

OCR service chính (`OCR_SERVICE_URL`, 171.226.10.153:12006) đã từng **treo toàn cục** (accept TCP nhưng ReadTimeout mọi endpoint). Từ nay có lớp dự phòng: gateway `vintern-vllm` trên máy B (repo riêng `D:\Code\Python\Ami\RAG\vintern-vllm`, FastAPI **:8090** trước vLLM serving `5CD-AI/Vintern-1B-v3_5` — model OCR tài liệu tiếng Việt ~1B, kiến trúc InternVL). Client: `ami_rag/core/ocr_client.py` (`VinterOcrClient`), inject vào `BackendDataExtractor` qua `create_extractor`.

Thứ tự ưu tiên (bố cục vẫn do MinerU + pdfplumber quyết định — fallback chỉ cung cấp **chữ**):

1. **Primary** (171.226): `extract_scan_text` → `hierarchy-parse`; `extract_tables` → `raw_miner_ocr/qwen`.
2. **Fallback Vinter** khi primary **lỗi/timeout** hoặc **trả rỗng** (kiểm tra riêng qua `OCR_FALLBACK_ON_EMPTY`): gọi `POST {OCR_FALLBACK_URL}/v1/ocr` (multipart `file` + `mode=text|table` + `pages` + `dpi`; gateway render PDF trang nào thành PNG và gửi ảnh vào VLM). Log warning `chuyển sang fallback Vinter` để vận hành biết service chính đang chết.
3. **Lớp cuối** (như cũ): text → PyMuPDF local; bảng → HTML dựng từ lưới pdfplumber.

Chi tiết hành vi:
- `extract_scan_text`: cả primary lẫn fallback đều lỗi → `BackendDataParseError` gộp **cả 2 lỗi** trong message (caller đã bắt và log); PyMuPDF vẫn là lớp cứu cuối trước khi raise.
- `extract_tables`: cả hai lỗi → trả `[]` (caller log rồi dùng html pdfplumber — lưới an toàn thứ hai của bảng).
- Retry chỉ 429/5xx/transport (backoff `0.5×2ⁿ` + jitter), 4xx raise ngay; connect timeout tách riêng `min(10s, timeout)` để fail-fast; message phân biệt `ReadTimeout` (service treo) / `ConnectError` (unreachable) / `ConnectTimeout` (SYN drop).
- Tắt: `OCR_FALLBACK_ENABLED=false` (toàn bộ), hoặc `OCR_FALLBACK_MODES=table` / `text` / `""` (tắt riêng từng loại / tắt hết). Chất lượng OCR bảng của Vintern-1B trên tài liệu tiếng Việt thật **chưa đo** — xem README repo `vintern-vllm` mục rủi ro.

### 5.5 Chunk (v2): gom dòng + prefix tiêu đề doc

Bản chunk cũ (index tạo trước khi đổi cách chunk — nhãn collection `CHUNKER_VERSION=v1`) chunk mỗi **content_list item** riêng. Với `PARSE_TEXT_SOURCE=backend_data` điều đó rất tệ: `raganything/backend_data_extract.py` gom word theo tọa độ `top` nên **mỗi dòng của PDF thành một item** (`build_pdf_content_blocks`), còn doc crawl viết mỗi trường một dòng. Doc `Sổ tay sinh viên 2026` (265 trang, 3372 item) thành **3378 chunk**, phần lớn là một dòng trơ trọi kiểu `- Địa chỉ: Tầng 1 nhà A1, Cơ sở đào tạo Hà Đông` — khớp mọi query "địa chỉ / cơ sở Hà Đông" nhưng không rõ thuộc đơn vị nào, và làm 40 ứng viên rerank gần như trùng lặp.

Ba thay đổi ở `ami_rag/core/vector_pipeline.py`:

1. **Gom item text liên tiếp** trong `_build_chunks` (nối bằng `\n`) rồi mới chia theo ngân sách; buffer chỉ flush khi gặp item modal (image/table/equation) nên ảnh vẫn nằm cạnh text của nó. Chunk lấy `page` của dòng đầu, thêm `page_end` trong payload khi trải nhiều trang. Doc nêu trên: **3378 → 825 chunk**.
2. **`chunk_text` v2**: tách đoạn theo dòng trống (giữ ranh giới đoạn khi gom lại), đoạn quá dài thì gom **nhiều dòng** tới ngân sách — chỉ dòng đơn dài hơn cả ngân sách mới bị cắt cứng. Chunk ngắn hơn `CHUNK_MIN_CHARS` (mặc định 200, `0` = tắt) được gộp vào chunk kề → hết chunk mồ côi.
3. **Prefix tiêu đề doc** vào `content` (bỏ qua nếu chunk đã bắt đầu bằng tiêu đề, hay gặp ở bài crawl): `- Địa chỉ: Tầng 1 nhà A1` → `Sổ tay sinh viên 2026 / VIII. PHÒNG KẾ HOẠCH ĐẦU TƯ - Địa chỉ: Nhà B14, ...`. Áp dụng cho **cả chunk ảnh** (ảnh không còn trạng thái "không rõ của doc nào"); tiêu đề cũng nằm trong payload để client hiển thị.

`CHUNKER_VERSION` chỉ là **nhãn tên collection** — code chunk theo logic trên bất kể giá trị; đặt giá trị mới (ví dụ `v2`, `settings.py` và `.env.ami.example` đều mặc định `v1`) để index mới không trộn chunk hai cách chunk khác nhau trong cùng collection.

Đổi `CHUNKER_VERSION` ⇒ ra collection Qdrant mới ⇒ phải reindex (xem mục `reindex`): `ami-rag reindex --all --from-stage chunk` (đọc `content_list.json` + `descriptions.json` trong MinIO, **không gọi LLM/Gemini**; chunk ảnh không đổi nên embed trúng cache).

> Chunking không tự kéo ảnh lên top-k. Đo trên index thật (query "sân bóng đá cơ sở Hà Đông", `scripts/diagnose_query.py`): sau chunk v2, chunk ảnh sơ đồ vẫn xếp hạng 246/1123 ở tầng vector, và **không** phải vì rerank — khi ảnh đã có trong ứng viên thì `raw` fusion xếp nó hạng 1 (0.0158 > 0.0129 của text đầu). Thủ phạm là caption tiếng Anh: xem mục 5.3.

### 5.6 Qdrant hybrid: named vectors `dense` + `sparse`

Collection mới (backend `bge`) do `ensure_collection(sparse=True)` tạo với **named vectors**: `"dense"` (COSINE, `EMBED_DIM`=1024) + `"sparse"` (`SparseVectorParams`) (`ami_rag/core/vector_store.py`). Upsert gói `{"dense": vector, "sparse": SparseVector(indices, values)}`; chunk sparse rỗng (`{}`) chỉ ghi dense.

Search (`ami_rag/api/routes/rag.py::_run_search`):
- Embedder có `embed_query_sparse` (BGE) → query lấy dense + sparse; sparse query rỗng (0 indices) hoặc embedder legacy (chỉ `embed_query`) → `sparse=None` ⇒ **dense-only**.
- `sparse` có giá trị ⇒ hybrid: 2 prefetch (`dense` limit k + `sparse` limit k) fuse bằng `FusionQuery(RRF)` ngay tại Qdrant — đây là tầng chọn ứng viên; tầng rerank `RERANK_FUSION=rrf` xếp hạng trên ứng viên đó, hai RRF không trùng nhau.

Collection legacy (pipeline Nemotron/Qwen cũ — 1 vector vô danh, không có sparse config): `ensure_collection(sparse=True)` gặp nó thì lỗi `VectorStoreError` "reindex cần tạo collection mới" — sparse vector không bổ sung được vào collection cũ. Đổi `EMBED_MODEL` ⇒ slug đổi (`bge-m3`) ⇒ collection mới tự sinh (ví dụ `multimodal__bge-m3__v1`; tên thật theo giá trị env); dọn collection cũ bằng `ami-rag cleanup --stale-models`.

## 6. API

Nguồn sự thật: `ami_rag/api/main.py`, `ami_rag/api/routes/rag.py`, `ami_rag/api/routes/admin.py`, `ami_rag/api/schemas.py`. Mọi route `/v2/rag/*` và `/admin/*` yêu cầu `Authorization: Bearer <RAG_API_KEY>` khi `RAG_API_KEY` khác rỗng (rỗng = mở; nên chỉ dùng trong mạng nội bộ); sai key trả `401` `{"detail": "Invalid API key"}`. `/`, `/healthz`, `/readyz`, `/metrics` không yêu cầu auth; chặn `/metrics` ở reverse proxy nếu cần.

Chỉ còn **v2**: request chỉ gồm `messages`, `top_k`, `include_references` (`RAGRequest`). Đã bỏ v1, bỏ `version`/`mode`/`filters`/`include_kg`. Response (`RetrievalResponseV2`) **không có field `answer`** — service chỉ truy xuất chunk, không sinh câu trả lời.

### POST /v2/rag/

Request (`RAGRequest`): `messages` là list `{role, content}` (nullable — query lấy nội dung message `role: "user"` cuối cùng; không có message user thì lấy message cuối cùng), `top_k` (`1..200`, nullable — mặc định `RERANK_TOP_K` = 5), `include_references` (mặc định `true`). `top_k` ngoài `1..200` → `422` (Pydantic).

```json
{
  "messages": [{"role": "user", "content": "bảng học phí các ngành"}],
  "top_k": 10,
  "include_references": true
}
```

Response `200` (`RetrievalResponseV2` = `query`, `documents`, `references`, `meta`; ví dụ rút gọn — mỗi document có đủ các field, giá trị không áp dụng là `null`):

```json
{
  "query": "bảng học phí các ngành",
  "documents": [
    { "text": "Đoạn văn mô tả học phí ...", "score": 0.93, "modality": "text",
      "artifact_url": null, "artifact_urls": [], "table_body": null, "page": null, "caption": null,
      "reference_id": "<chunk_id>",
      "metadata": {"source": "665f..._hocphi.pdf", "page": null, "document_id": "665f...", "chunk_index": null, "global_id": "<chunk_id>"},
      "doc": {"document_id": "665f...", "title": "Học phí 2026", "document_type": "pdf",
              "organization_unit_id": "...", "file_url": "https://minio/presigned/...", "source_url": null} },
    { "text": "... Asset: rag-assets/665f.../ab12cd34ef567890.png ...", "score": 0.81, "modality": "image",
      "artifact_url": "https://minio/presigned/rag-assets/665f.../ab12cd34ef567890.png",
      "artifact_urls": ["https://minio/presigned/rag-assets/665f.../ab12cd34ef567890.png"],
      "table_body": null, "page": 2, "caption": "Sơ đồ tổ chức", "reference_id": "<chunk_id>",
      "metadata": {"source": "...", "page": 2, "document_id": "665f...", "chunk_index": null, "global_id": "<chunk_id>"},
      "doc": {"...": "..."} },
    { "text": "...", "score": 0.77, "modality": "table",
      "artifact_url": null, "artifact_urls": ["..."],
      "table_body": "| Ngành | Học phí |\n| --- | --- |\n| CNTT | 30tr |",
      "page": 4, "caption": "Bảng 1. Học phí", "reference_id": "<chunk_id>",
      "metadata": {"...": "..."}, "doc": {"...": "..."} }
  ],
  "references": [
    {"document_id": "665f...", "title": "Học phí 2026", "document_type": "pdf",
     "organization_unit_id": "...", "file_url": "https://minio/presigned/...", "source_url": null}
  ],
  "meta": {"latency_ms": 820, "requested_top_k": 10, "returned": 3, "candidates": 40, "fusion": "rrf"}
}
```

Field của mỗi document (`RetrievedDocV2`):
- `text`: nội dung chunk. `score`: điểm rerank (`null` khi rerank lỗi — fallback). `modality`: `text` | `image` | `table` | `equation` (từ `original_type` của chunk multimodal).
- `metadata`: `{source, page, document_id, chunk_index (luôn null), global_id (= chunk_id)}`.
- `doc`: document đã resolve qua Mongo (`resolver.resolve_doc_id`) — `{document_id, title, document_type, organization_unit_id, file_url, source_url}`; không resolve được → `null`.
- `reference_id`: chunk id.
- `artifact_url`: presigned URL (hết hạn sau `MINIO_PRESIGN_EXPIRES`) của `asset_key`; `null` khi chunk không có asset hoặc presign lỗi (tăng `presign_failures_total`).
- `artifact_urls`: presigned của **mọi ảnh của doc crawl** (`asset_store.presign_doc_images` — ảnh crawl section `Hình ảnh`, mục 2), trả cho cả chunk text của doc. Chỉ áp dụng doc crawl (`original_file_name` rỗng/`.txt`, **không có** `file_path`, **và** `metadata.source_url` khác rỗng); doc PDF/upload/seed text → `[]`, không list MinIO. Doc không resolve được qua Mongo hoặc store lỗi/missing method → `[]` (key giữ nguyên shape). `artifact_url` của chunk vẫn trả riêng cho mọi modality có `asset_key`.
- `table_body`: markdown của bảng (modality `table`); `page`, `caption`: trang và caption của item.

`references`: danh sách `doc` dedup theo `document_id`, chỉ có khi `include_references: true`. `meta`: `latency_ms`, `requested_top_k`, `returned`, `candidates` (số ứng viên tầng vector), `fusion` = `raw` | `rrf` | `empty` (cấu hình rerank fusion đã áp dụng).

Luồng: embed query dense+sparse (máy B; embedder legacy thì dense-only, mục 5.6) → hybrid search Qdrant, 2 prefetch fuse RRF ngay tại Qdrant, lấy `max(RETRIEVAL_CHUNK_TOP_K, top_k * 2)` ứng viên → rerank service (`RERANK_FUSION=rrf` thì xin full scores + fuse 2 nhóm modality, mục 5.2) → trả `top_k` document đầu. Lỗi rerank → trả chunk không điểm (fallback, đo `rerank_fallback_total`).

**Chẩn đoán 1 query** khi kết quả không như mong đợi (ảnh không lên top-k, chunk trả về vô nghĩa…): `scripts/diagnose_query.py` in ra hạng + điểm vector của từng ứng viên, điểm rerank raw, và kết luận rõ ảnh rơi ở tầng nào.

```bash
python scripts/diagnose_query.py --query "sân bóng đá cơ sở Hà Đông" --top-k 5
python scripts/diagnose_query.py --query "..." --output diag.json   # lưu lại để đối chiếu
```

- **Không có ảnh trong ứng viên vector** → vấn đề embed: kiểm tra caption (mục 5.3, ngôn ngữ), chunk (mục 5.5), hoặc index chưa có chunk ảnh.
- **Có ảnh trong ứng viên nhưng bị rerank cắt** → vấn đề fusion/weight (`RERANK_FUSION`, `RERANK_VISUAL_WEIGHT`, mục 5.2).
- Chỉ đọc Qdrant + embed/rerank server (máy B), không ghi gì, **không gọi LLM/Gemini**. Ngoài máy Docker cần trỏ `QDRANT_URL`/`MINIO_ENDPOINT` vào IP container tương ứng (tên service không resolve được từ host).

### POST /v2/rag/stream

Cùng request như `POST /v2/rag/`, trả NDJSON (`application/x-ndjson`), mỗi dòng một JSON:

1. `{"status": "retrieving"}`
2. Mỗi document một dòng `{"documents": [<1 object>]}`; object gồm `text`, `metadata` (bỏ 2 key `page` và `source`), và khi có giá trị: `modality` + `artifact_url` + `artifact_urls` (nhóm luôn xuất hiện cùng nhau) và `score`.
3. `{"status": "done"}`

```json
{"status": "retrieving"}
{"documents": [{"text": "Đoạn văn mô tả học phí ...", "metadata": {"document_id": "665f...", "chunk_index": null, "global_id": "<chunk_id>"}, "modality": "text", "artifact_url": null, "artifact_urls": [], "score": 0.93}]}
{"status": "done"}
```

### Admin (`/admin/*`)

| Endpoint | Mô tả |
|---|---|
| `GET /admin/pipeline_status` | `{workspace, doc_status_counts (DocStatusStore), queue_pending}` — lỗi Mongo/Redis tương ứng trả `{}` / `null` trong field, không 500 |
| `POST /admin/reprocess_failed` | publish lại event `updated` cho mọi doc `failed` trong registry → `{requeued}` |
| `POST /admin/reindex` | body `{"all": true}` (duyệt toàn bộ `organization_db.documents`) hoặc `{"document_ids": [...]}`; publish event `updated` → `{published}` |
| `GET /admin/documents/{id}` | trạng thái DocStatusStore: `document_id, status, stage, source, counts, page_count, parser, document_type, title, organization_unit_id, owner_id, error, updated_at` (ObjectId trả về dạng string, `null` nếu dòng chưa `indexed`); `404 {"detail": "Document not found"}` nếu chưa có |
| `GET /admin/documents/{id}/content` | từ `content_list.json` trên MinIO: `{document_id, markdown, blocks, tables}` (block có `type`, `page_idx`, `text`/`table_body`/`caption`/`latex`, `asset_url` presigned); `404 {"detail": "Content not found"}` nếu chưa có |

Khác: `GET /` (`{"service": "ami-rag", "version": "0.1.0"}`), `GET /healthz`, `GET /readyz`, `GET /metrics`.

## 7. CLI `ami-rag`

CLI phục vụ 4 việc: xem trạng thái doc, chạy lại doc lỗi/hỏng, reindex thủ công, dọn collection legacy. Nguồn sự thật: `ami_rag/cli.py` (entry point `ami-rag` = `ami_rag.cli:main`).

```bash
ami-rag status [--failed] [--stale] [--doc <id|path>] [--check-embed-server]
ami-rag retry (--doc <id|path> [...] | --all-failed) [--from-stage S] [--max-attempts N] [--yes] [--embed-server-url URL] [--embed-batch-size N]
ami-rag reindex [--stale | --all | --scan | --doc <id|path> [...]] [--from-stage S] [--dry-run] [--yes] [--limit N] [--sleep S] [--batch B] [--batch-sleep S] [--embed-server-url URL] [--embed-batch-size N]
ami-rag cleanup [--dry-run] [--yes] [--stale-models] [--purge-cache-model MODEL...] [--mongo-only | --qdrant-only]
```

### `status` — xem trạng thái (mặc định chỉ đọc local: 0 LLM, 0 embed, 0 model call)

In workspace, config đang dùng (`embed_model`/`embed_dim`/`chunker_version`), collection đang dùng theo quy ước `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`, số doc theo trạng thái, doc thiếu bản ghi (báo `pending`), doc `processing` quá lâu (`CLI_STUCK_PROCESSING_MINUTES`, mặc định 60 phút) được báo `TREO`.

| Cờ | Ý nghĩa |
|---|---|
| `--failed` | chỉ hiện doc `failed` (kèm stage lỗi, error, attempts, updated_at) |
| `--stale` | chỉ hiện doc `stale` (lệch embed_model/chunker_version, file đổi hash, pipeline cũ `processed`) |
| `--doc <id\|path>` | xem chi tiết một doc |
| `--check-embed-server` | gọi `/health` + handshake của embed server: sẵn sàng không, đúng model/dim theo config không |

### `retry` — chạy lại doc `failed` từ stage lỗi (bằng dữ liệu trung gian đã lưu)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--doc <id\|path> [...]` / `--all-failed` | - | chọn doc để retry (không phải `failed` thì bỏ qua) |
| `--from-stage` | stage lỗi đã lưu | `parse`/`describe`/`chunk`/`embed`; ép làm lại từ stage đó |
| `--max-attempts` | `3` | chỉ retry doc có `attempts < N` |
| `--yes` | hỏi xác nhận | bỏ qua xác nhận |
| `--embed-server-url` / `--embed-batch-size` | - | override tối thiểu |

- Trước khi chạy, kiểm tra embed server (`runner.preflight_embed()`: health + handshake model/dim); không với tới thì **dừng sớm** (mã thoát 2, không đánh fail từng doc nào).
- Một doc lỗi không dừng cả lô; cuối lệnh in tổng kết thành công/thất bại/bỏ qua. Mã thoát: `0` xong, `1` một số doc thất bại, `2` dừng sớm.
- Lockfile `./.ami_rag_cli.lock` chống chạy hai `retry`/`reindex` cùng lúc trên một kho dữ liệu.

### `reindex` — xử lý lại thủ công (chunk → embed từ dữ liệu parse + mô tả modal đã lưu)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--stale` | **mặc định** | chỉ doc `stale` |
| `--all` | - | mọi doc có bản ghi |
| `--scan` | - | quét `organization_db.documents` active: tạo `pending` cho doc chưa có bản ghi, rồi xử lý pending + stale (dùng cho migration từ pipeline cũ — chỉ đọc Mongo, 0 LLM/0 embed) |
| `--doc <id\|path> [...]` | - | doc cụ thể |
| `--from-stage` | theo từng doc | `chunk` khi `content_list.json` đã có trong MinIO, full parse khi thiếu; giá trị explicit override tất cả |
| `--dry-run` | - | in số doc, số chunk cần embed, số chunk trúng cache; không gọi embed, không ghi gì |
| `--yes` | hỏi xác nhận | bỏ qua xác nhận |
| `--embed-server-url` / `--embed-batch-size` | - | override tối thiểu |
| `--limit N` | `0` | chỉ xử lý N doc đầu (0 = không giới hạn); cắt ngay sau khi chọn — áp dụng cả `--dry-run` nên preview đúng scope |
| `--sleep S` | `0.2` | giây nghỉ giữa hai doc liên tiếp |
| `--batch B` | `20` | số doc mỗi lô (0 = một lô duy nhất) |
| `--batch-sleep S` | `2.0` | giây nghỉ khi vào lô mới (ngoài lô đầu) — tính theo vị trí thứ tự, doc fail không restart lô |

- Trước khi chạy, kiểm tra embed server (`runner.preflight_embed()`: health + handshake model/dim); không với tới thì **dừng sớm** (mã thoát 2, không đánh fail doc nào).
- Khoá bất đồng bộ: lệnh từ chối chạy nếu một `retry`/`reindex` khác đang giữ lockfile.
- **Migration sang BGE-M3 + hybrid Qdrant (nhiệm vụ hiện hành)**: `EMBED_MODEL` đổi sang `BAAI/bge-m3` (dim 1024) ⇒ collection đích tự thành `multimodal__bge-m3__v1` với named vectors `dense`+`sparse` (mục 5.6). `--from-stage embed` **dùng lại `chunks.json` đã lưu trong MinIO** — không parse/describe/chunk lại, không gọi LLM/Gemini:
  ```bash
  ami-rag status --check-embed-server                          # embed :8008 + rerank :8010 sẵn sàng
  ami-rag reindex --stale --from-stage embed --dry-run         # phạm vi + số chunk trúng cache
  ami-rag reindex --stale --from-stage embed --limit 20 --yes  # thử lô nhỏ trước, kiểm tra retrieve trước khi chạy thật
  ami-rag reindex --stale --from-stage embed --batch 20 --sleep 0.2 --batch-sleep 2.0 --yes
  ```
  - **Điều kiện**: doc phải đã có `chunks.json` (đã qua chunk). Doc chỉ có `content_list.json` → bỏ `--from-stage` cho per-doc tự chọn (nó chọn `chunk` — không gọi LLM — khi `content_list` có trong MinIO) và chạy `--yes` trực tiếp.
  - **Ngắt quãng / resume / lặp vô hạn**: doc xong embed được đánh dấu `indexed` kèm `embed_model` mới trong registry → effective không còn `stale`; doc chưa xong giữ `embed_model` cũ ⇒ vẫn `stale` và được lệnh `--stale` (mặc định) chọn lại. Đường dẫn resume = chạy lại cùng lệnh đến khi `ami-rag status` không còn doc `stale`; ngắt giữa chừng an toàn (stage embed luôn `delete_by_doc` trước khi upsert). Doc lỗi nằm ở `failed` kèm `error_stage`, xử lý `retry --from-stage embed`.
  - `--limit`/`--sleep`/`--batch`/`--batch-sleep` để chia lượt chậm, đè nhẹ embed/rerank server khi chạy cả nghìn doc; `--dry-run` vẫn tính đúng scope sau `--limit`.
  - Index mới OK (so sánh retrieve + live test `tests/test_retrieval_recall_live.py`) mới dọn: `ami-rag cleanup --stale-models --purge-cache-model "nvidia/llama-nemotron-embed-vl-1b-v2" --yes`.
- Các tình huống khác (lịch sử, cùng cơ chế): sinh lại caption ảnh `--from-stage describe` (1 call Gemini/ảnh, mục 5.3); doc crawl cũ chưa có ảnh `--from-stage parse` (mục 2); nâng `CHUNKER_VERSION` `--from-stage chunk` (mục 5.5).

Mã thoát: `0` xong, `1` một số doc thất bại, `2` dừng sớm (preflight embed server, lockfile, validation).

### `cleanup` — kiểm tra + xoá collection legacy của pipeline LightRAG cũ

Sau migration, các collection của LightRAG không còn được đọc/ghi nhưng vẫn chiếm chỗ. Lệnh dò và xoá chúng (`ami_rag/core/legacy_cleanup.py`):

- **Mongo** (`RAG_DB`): mọi collection `{WORKSPACE}_*` trừ registry `RAG_DOCUMENTS_COLLECTION` — tức `multimodal_full_docs`, `multimodal_text_chunks`, `multimodal_llm_response_cache`, `multimodal_doc_status`, `multimodal_chunk_entity_relation`(+`_edges`), `multimodal_parse_cache`, ... Registry được giữ nguyên vì DocStatusStore còn đọc dòng legacy `processed` (như `stale`) để `reindex --scan`.
- **Qdrant**: mọi collection `{WORKSPACE}_*` **một gạch** (`multimodal_chunks`, `multimodal_entities`, `multimodal_relationships`, ...). Các collection `{WORKSPACE}__*` (hai gạch — quy ước vector pipeline) mặc định luôn giữ; dùng `--stale-models` để dò + xoá các collection embed model/version **cũ** (không phải collection hiện tại) — ví dụ bản Qwen sau khi đổi sang Nemotron.
- Collection ngoài tiền tố workspace (dịch vụ khác dùng chung Mongo/Qdrant) không bao giờ bị đụng.

| Cờ | Ý nghĩa |
|---|---|
| `--dry-run` | chỉ liệt kê collection legacy kèm số document/points, không xoá |
| `--yes` | bỏ qua xác nhận (mặc định hỏi trước khi xoá — không hoàn tác được) |
| `--mongo-only` / `--qdrant-only` | giới hạn dò/xoá một loại DB |
| `--stale-models` | xoá thêm collection Qdrant của embed model cũ: mọi `{WORKSPACE}__*` không phải `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` hiện tại |
| `--purge-cache-model MODEL...` | xoá cache embedding SQLite của model (key bắt đầu `{MODEL}|`) — ví dụ `--purge-cache-model "Qwen/Qwen3-VL-Embedding-2B"` |

Mã thoát: `0` xong (hoặc không có gì xoá), `1` một số collection xoá lỗi, `2` không kết nối được Mongo/Qdrant.

Ví dụ:

```bash
ami-rag cleanup --dry-run       # xem cái gì sẽ bị xoá
ami-rag cleanup --yes           # xoá hết legacy (sau khi reindex --scan thành công)
ami-rag cleanup --qdrant-only   # chỉ xoá collection Qdrant legacy
# sau khi đổi EMBED_MODEL (ví dụ Qwen -> Nemotron) và retrieval trên index mới ổn:
ami-rag cleanup --stale-models --purge-cache-model "Qwen/Qwen3-VL-Embedding-2B" --yes
```

Lưu ý: MinIO `rag-assets/` không thuộc cleanup — assets/content_list.json vẫn dùng cho resume. Cache embedding key theo model nên tự vô hiệu khi đổi model, nhưng vẫn chiếm chỗ trên đĩa — dùng `--purge-cache-model` để xoá hẳn. Thư mục local `rag_storage/` của LightRAG cũ (nếu còn trên đĩa) xoá tay.

### 7.1 Chạy CLI khi dùng Docker

`docker-compose.ami.yml` chỉ có hai service: `ami-rag-api` và `ami-rag-worker`. Không có service `ami-rag` riêng; `ami-rag` là lệnh bên trong image (entry point `ami_rag.cli:main`). Chạy lệnh trong container đang chạy, dùng cùng `.env` (Mongo, Redis, Qdrant, MinIO, embed server) với service:

```bash
# từ thư mục gốc repo; -T tắt TTY (cần khi pipe/tee/cron, vô hại khi chạy tay)
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status --failed
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag status --check-embed-server
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --stale --dry-run
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag reindex --all --yes
docker compose -f docker-compose.ami.yml exec ami-rag-worker ami-rag retry --all-failed --yes
```

Rút gọn bằng Makefile (`make cli`, chạy `ami-rag <lệnh con>` trong `ami-rag-worker`):

```bash
make cli status
make cli status --failed
make cli -- reindex --stale --dry-run      # cờ dạng --xxx cần "--" để make không tự parse
make cli -- reindex --all --yes
make cli -- retry --all-failed --yes
make cli CLI_SERVICE=ami-rag-api status    # đổi container
```

Chạy nền cho lượng dữ liệu lớn (`reindex --all` trên cả nghìn doc): thêm `BG=1`. Lệnh chạy detached trong container (`exec -d`), sống tiếp khi đóng terminal/SSH; log ghi vào `/app/output/cli/cli-<thời gian>.log` (volume `rag-output`, symlink `latest.log`), dòng cuối `# exit=<mã>`.

```bash
make cli BG=1 -- reindex --all --yes    # in đường dẫn log, trả về ngay
make cli-logs                               # tail -f log của lệnh nền gần nhất (Ctrl-C chỉ dừng tail)
make cli-ps                                 # liệt kê mọi tiến trình `ami-rag <lệnh>` đang chạy (pid, thời gian)
make cli-stop                               # dừng lệnh nền gần nhất (chỉ pid đó)
make cli-stop PID=<pid>                     # dừng pid lấy từ cli-ps
```

`cli-stop` chỉ kill đúng pid đã ghi, không đụng worker/API hay phiên `ami-rag` mở tay khác. Dừng giữa chừng an toàn vì `retry`/`reindex` idempotent (stage embed luôn `delete_by_doc` trước khi upsert). `retry`/`reindex` bị khoá bởi lockfile `./.ami_rag_cli.lock`, nên không chạy hai lệnh này song song được.

Không có `--`, `make cli reindex --all` báo `unrecognized option '--all'` (lệnh con không cờ như `status` thì không cần). Target kiểm tra container đang chạy theo `docker compose exec`, nên cần `make start_docker` trước.

Sai thường gặp: `docker compose exec ami-rag reindex --all` lỗi vì `ami-rag` là tên lệnh chứ không phải tên service. Đúng: `exec <service> ami-rag <lệnh con>`, với `<service>` là `ami-rag-worker` hoặc `ami-rag-api`.

Chọn container:

| Lệnh | Nên chạy trong | Lý do |
|---|---|---|
| `status` | `ami-rag-worker` hoặc `ami-rag-api` | chỉ đọc Mongo; `--check-embed-server` cần `.env` trỏ đúng embed server |
| `retry`, `reindex` | `ami-rag-worker` | chạy trực tiếp trong tiến trình CLI; cần tới embed server + Mongo của worker |

Lưu ý:
- Lệnh `exec` dài nên chạy nền, có log: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag reindex --all --yes > reindex.log 2>&1 &`, rồi `tail -f reindex.log`. Hoặc dùng `tmux`/`screen`. Ngắt phiên SSH có thể làm dừng lệnh đang chạy ở foreground; chạy lại là an toàn (idempotent).
- Container phải đang chạy. Nếu chưa: `make start_docker` (hoặc `docker compose -f docker-compose.ami.yml up -d`). Nếu cần chạy mà không dựa vào container sẵn có (không đụng worker):
  `docker compose -f docker-compose.ami.yml run --rm --no-deps ami-rag-worker ami-rag status`
  (`run` tạo container tạm cùng image/`.env`/volume, không publish cổng 9109; `--rm` xoá khi xong).
- Image đang chạy phải có code CLI mới. Sau khi sửa `ami_rag/`, `docker compose -f docker-compose.ami.yml up -d --build` (lâu, xem mục 5) rồi mới chạy `retry`/`reindex`; image cũ báo `invalid choice` hoặc thiếu cờ mới.
- Mã thoát của `exec` là mã thoát của lệnh trong container, nên `retry`/`reindex` (trả 1 khi còn doc thất bại) dùng được trong script/CI: `docker compose -f docker-compose.ami.yml exec -T ami-rag-worker ami-rag retry --all-failed --yes || echo "còn doc failed"`.
- Ngoài Docker (máy dev, đã `uv sync`): `uv run ami-rag <lệnh con>` với cùng cờ; host trong `.env` phải truy cập được từ máy đó (tên container như `redis`, `mongo` chỉ phân giải trong `ami-network`; `EMBED_SERVER_URL` phải trỏ đúng địa chỉ embed server).

## 8. Runbook backfill dữ liệu cũ

Các lệnh `ami-rag …` dưới đây viết ở dạng ngắn; khi chạy bằng Docker thêm tiền tố `docker compose -f docker-compose.ami.yml exec ami-rag-worker` (mục 7.1).

Mục tiêu: nạp toàn bộ doc có sẵn trong `organization_db.documents` vào index mới (collection Qdrant `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}`, registry `organization_db.multimodal_rag_documents`). Doc cũ của pipeline LightRAG (status `processed` trong registry cũ) được `reindex --scan` quét lại thành `pending` rồi xử lý theo pipeline mới: parse → describe → chunk → embed. Doc cũ chỉ có text/bảng đơn giản trong Mongo vẫn được parse lại từ file MinIO (nếu `original_file_name` không rỗng/`.txt`); text Mongo của pdf/docx **không** được dùng.

1. Chuẩn bị: `.env` đúng, embed server máy B chạy (`ami-rag status --check-embed-server` OK), worker chạy (`make start_docker` hoặc `make start_worker`), `make health` OK, queue pending 0. Worker cần MinerU + model cache sẵn (xem mục 5).
2. Quét doc cũ chưa có bản ghi: `ami-rag reindex --scan --dry-run`. Lưu ý `--scan` **vẫn ghi** record `pending` cho doc thiếu bản ghi (chỉ đọc Mongo `documents`, 0 LLM/0 embed); `--dry-run` chỉ chặn bước chunk → embed và in số chunk cần embed.
3. Dry-run toàn bộ, kiểm tra cột `source` (`mongo_text`/`minio_parse`) hợp lý và số chunk trúng cache:
   `ami-rag reindex --all --dry-run`
4. Thử text/crawl nhỏ (nhanh, không cần MinerU):
   `ami-rag reindex --doc <id> --yes` với vài doc text trước.
5. Thử file thật (MinerU, chậm): `ami-rag reindex --doc <id> --from-stage parse --yes`. Kiểm tra `GET /admin/documents/{id}` (`counts` có `table`/`image`), `GET /admin/documents/{id}/content`, và một truy vấn `POST /v2/rag/`.
6. Chạy toàn bộ: `ami-rag reindex --all --yes` (hoặc `--scan --yes` để quét + xử lý). Trước đó xác nhận embed server sẵn sàng bằng `ami-rag status --check-embed-server`: server chết thì `preflight_embed` **dừng sớm** (mã thoát 2, không đánh fail doc nào); nếu server chết giữa chừng thì doc ở stage `embed` thành `failed` với lý do rõ ràng.
7. Theo dõi:
    - `ami-rag status` (đếm theo status, doc failed, doc treo); queue pending xem qua metric/admin API;
   - `curl -s localhost:8009/admin/pipeline_status` (thêm header `Authorization: Bearer ...` nếu có `RAG_API_KEY`);
   - metric worker `:9109/metrics`: `multimodal_rag_ingest_events_total{result}`, `..._stream_pending`, `..._stream_lag`, `..._documents{status}` (đếm theo `status` trong `organization_db.multimodal_rag_documents`), `..._parse_failures_total`; dashboard row "Ingest pipeline".
   - `make logs SERVICE=ami-rag-worker`.
8. Xử lý lỗi:
    - doc `failed`: `ami-rag retry --all-failed` (hoặc `ami-rag retry --doc <id>`);
    - nạp lại khi đổi model/chunker: `ami-rag reindex --stale` (doc được đánh `stale` tự động khi lệch `embed_model`/`chunker_version`);
    - ép làm lại toàn bộ từ parse: `ami-rag retry --doc <id> --from-stage parse`;
    - doc không cần nữa: xoá qua event `deleted` trong queue hoặc xoá dòng registry + asset thủ công (CLI không còn lệnh purge).
9. Kiểm tra index: `ami-rag status` (đếm theo status/stage; `--check-embed-server` khi nghi ngờ máy B). Doc lỗi nằm ở `failed` với lý do trong `error` + `error_stage`.
10. Hoàn tất khi không còn `processing`/`failed`/`pending` trong `ami-rag status`, số doc `indexed` xấp xỉ số doc active.
11. Dọn collection legacy của LightRAG: `ami-rag cleanup --dry-run` rồi `ami-rag cleanup --yes` (xem mục 7 `cleanup`). Chỉ chạy sau khi đã xác nhận retrieval trên index mới hoạt động tốt.

## 9. Monitoring

Chi tiết triển khai: [`monitoring/README.md`](../monitoring/README.md). Nguồn chốt danh sách metric là `ami_rag/observability.py`.

### 9.1 Kiến trúc

Prometheus (kube-prometheus-stack) scrape `/metrics` của 2 target rồi hiển thị qua Grafana: `ami-rag-api` port 8009 (named port `metrics`, prefix `multimodal_rag_retrieval_`) và `ami-rag-worker` port 9109 (`worker-metrics`, prefix `multimodal_rag_ingest_`). Khi API chạy worker nhúng (`WORKER_ENABLED=true`) metric ingest nằm chung `/metrics` của API; compose ép `WORKER_ENABLED=false` và chạy worker riêng. Grafana dùng 2 datasource: Prometheus uid `prometheus` (mọi panel metric) và Tempo uid `multimodal_rag_log` (panel "Recent traces"). Dashboard cố định timezone `Asia/Ho_Chi_Minh` để khớp label `day`/`hod` trong code — không đổi sang `browser`.

Tracing (OTLP/HTTP): chỉ truy vấn `POST /v2/rag` được trace — span `rag.retrieval` (input: attribute `input.query`/`input.mode`/`input.top_k`; output: danh sách document đầy đủ trong event `documents_json`) cùng span con theo stage (`embed_query`/`vector_search`/`rerank`). Worker không trace (ingest chỉ có metric Prometheus). FastAPI telemetry tự động bị tắt (`FastAPI(telemetry={"auto_configure": False, ...})` trong `ami_rag/api/main.py`), nếu không FastAPI ≥ 0.142 đọc `OTEL_EXPORTER_OTLP_ENDPOINT` rồi trace mọi route và đẩy metrics/logs tới `/v1/metrics`, `/v1/logs` (Tempo trả `404`: log `Failed to export metrics batch code: 404`). Tempo dùng riêng cho service này: release helm `tempo-multimodal-rag` (`monitoring/helm/tempo-multimodal-rag-values.yaml`, chart `grafana/tempo` 1.24.4, tách khỏi Tempo của conversational-agent; retention 744h, PVC local-path 10Gi). Datasource Grafana uid `multimodal_rag_log` nạp qua sidecar bằng ConfigMap `monitoring/helm/grafana-datasource-tempo-multimodal-rag.yaml`; URL in-cluster `http://tempo-multimodal-rag.monitoring.svc.cluster.local:3200`; `OTEL_EXPORTER_OTLP_ENDPOINT` trỏ tới NodePort OTLP/HTTP của service port `4318` (`kubectl get svc tempo-multimodal-rag -n monitoring`). App tự nối `/v1/traces` vào endpoint (`normalize_otlp_traces_url`). `TRACE_OUTPUT_MAX_LEN` (mặc định 2000000, 0 = không cắt) chặn event `documents_json` vượt ~4MB OTLP.

### 9.2 Metric

Metric API (`GET :8009/metrics`, prefix `multimodal_rag_retrieval_`; Counter hiển thị thêm hậu tố `_total`):

| Metric | Loại | Label | Ý nghĩa |
|---|---|---|---|
| `multimodal_rag_retrieval_requests_total` | Counter | `mode` (`vector`) | Tổng request retrieval |
| `multimodal_rag_retrieval_errors_total` | Counter | `mode` | Request retrieval lỗi |
| `multimodal_rag_retrieval_requests_by_day_total` / `_by_hour_of_day_total` / `_by_day_hour_total` | Counter | `day` / `hod` / `day,hod` | Đếm theo ngày/giờ (Asia/Ho_Chi_Minh) |
| `multimodal_rag_retrieval_duration_seconds` | Histogram | `mode` | Round-trip 1 request (query + rerank + resolve) |
| `multimodal_rag_retrieval_stage_duration_seconds` | Histogram | `stage` = `embed_query` / `vector_search` / `rerank` | Thời gian 1 stage pipeline |
| `multimodal_rag_retrieval_docs_returned` | Histogram | | Số document trả về mỗi request |
| `multimodal_rag_retrieval_chunks_retrieved` | Histogram | | Số chunk vector search trả trước rerank |
| `multimodal_rag_retrieval_requests_in_flight` | Gauge | | Request đang xử lý |
| `multimodal_rag_retrieval_rerank_fallback_total` | Counter | `reason` = `error` / `empty` | Rerank phải fallback về chunk không chấm điểm |
| `multimodal_rag_retrieval_docs_by_modality_total` | Counter | `modality` = `text` / `image` / `table` / `equation` | Document trả về theo modality |
| `multimodal_rag_retrieval_presign_failures_total` | Counter | | Tạo presigned URL MinIO lỗi |
| `multimodal_rag_retrieval_search_mode_total` | Counter | `mode` = `dense` / `hybrid` | Mode search thực tế (`hybrid` khi embedder trả sparse query, `dense` khi không) |
| `multimodal_rag_retrieval_artifact_urls_returned_total` | Counter | | Số entry `artifact_urls` trả về trong document |
| `multimodal_rag_retrieval_embed_failures_total` | Counter | `reason` (xem dưới) | Embed query lỗi trong đường retrieval |

Metric worker (`:9109/metrics` của `ami-rag-worker`; khi `WORKER_ENABLED=true` chúng nằm chung `/metrics` của API), prefix `multimodal_rag_ingest_`:

| Metric | Loại | Label | Ý nghĩa |
|---|---|---|---|
| `multimodal_rag_ingest_events_total` | Counter | `event` (`created`/`updated`/`deleted`), `result` (`processed`/`skipped`/`failed`/`retry`) | Ingest event đã xử lý theo kết quả |
| `multimodal_rag_ingest_in_flight` | Gauge | | Event ingest đang xử lý |
| `multimodal_rag_ingest_duration_seconds` | Histogram | `source` = `minio_parse` / `mongo_text` / `none` | Thời gian 1 event ingest |
| `multimodal_rag_ingest_stage_duration_seconds` | Histogram | `stage` | Stage `delete` của worker; timing parse/describe/chunk/embed nằm trong DocStatusStore, không lên Prometheus |
| `multimodal_rag_ingest_parse_failures_total` | Counter | `document_type` | Parse lỗi (tăng trong `VectorPipeline.run` khi stage parse lỗi) |
| `multimodal_rag_ingest_crawl_image_failures_total` | Counter | `reason` = `http` / `redirect` / `connect` / `ssl` / `minio` / `other` | Tải ảnh section "Hình ảnh" lỗi (tăng trong `CrawlImageWorker.ensure_in_minio`) |
| `multimodal_rag_ingest_items_total` | Counter | `modality` | Content-list item đã ingest theo modality |
| `multimodal_rag_ingest_pages_total` | Counter | | Số trang đã ingest |
| `multimodal_rag_ingest_stream_pending` / `_stream_lag` | Gauge | | Message chưa ack / chưa giao cho consumer group |
| `multimodal_rag_ingest_documents` | Gauge | `status` | Document trong `multimodal_rag_documents` theo status |
| `multimodal_rag_ingest_embed_failures_total` | Counter | `reason` (xem dưới) | Embed lỗi ở stage embed của ingest |
| `multimodal_rag_ingest_embed_items_total` | Counter | | Item đã đưa vào embed (mẫu số tỷ lệ cache hit) |
| `multimodal_rag_ingest_embed_cache_hits_total` | Counter | | Item đã có trong cache embedding (tử số tỷ lệ cache hit) |

`reason` của 2 metric embed (`multimodal_rag_retrieval_embed_failures_total` và `multimodal_rag_ingest_embed_failures_total`) là tập đóng trong `embed_failure_reason()`: `unreachable` / `circuit_open` / `auth` / `input_too_long` / `oom` / `model_mismatch` (theo lớp exception trong `ami_rag.core.embedder`) và `other`.

`requests_by_day` / `requests_by_hour_of_day` / `requests_by_day_hour` khai báo trong `observability.py` không kèm hậu tố `_total`; `prometheus_client` tự thêm khi phơi ra `/metrics`.

### 9.3 Alert

`monitoring/helm/prometheusrule.yaml` — PrometheusRule `multimodal-rag-retrieval`, label `release: kube-prometheus-stack` (phải khớp `ruleSelector` của Prometheus). Group `multimodal-rag-ingest`:

| Alert | Expr | Severity | Xử lý |
|---|---|---|---|
| `RagIngestBacklogHigh` | `sum(multimodal_rag_ingest_stream_pending) + sum(multimodal_rag_ingest_stream_lag) > 50` (for 15m) | warning | Hàng đợi `rag:ingest` tồn đọng; kiểm tra worker (GPU/MinerU) và `GET /admin/pipeline_status` |
| `RagIngestFailures` | `sum(increase(multimodal_rag_ingest_events_total{result="failed"}[30m])) > 5` | warning | Ingest lỗi liên tục; xem `/admin/pipeline_status` và log `ami-rag-worker` |
| `RagIngestParseFailures` | `sum(increase(multimodal_rag_ingest_parse_failures_total[30m])) > 3` | warning | MinerU parse lỗi (file hỏng, thiếu GPU/model, LibreOffice) |
| `RagCrawlImageFailures` | `sum(increase(multimodal_rag_ingest_crawl_image_failures_total[30m])) > 20` | warning | Ảnh crawl mất khỏi index; kiểm tra log worker và SSL allowlist PTIT |
| `RagIngestWorkerDown` | `absent(up{job=~".*multimodal-rag-retrieval.*", endpoint="worker-metrics"} == 1)` (for 5m) | critical | Target `worker-metrics` (:9109) không scrape được — worker chết hoặc Service/Endpoints sai |
| `RagIngestEmbedFailures` | `sum(rate(multimodal_rag_ingest_embed_failures_total[5m])) > 0` | warning | Embed lỗi khi ingest; kiểm tra embedding-server :8008 (`BGE_API_KEY`, GPU) và log worker |

Group `multimodal-rag-retrieval`:

| Alert | Expr | Severity | Xử lý |
|---|---|---|---|
| `RagRetrievalErrorRate` | `sum(rate(multimodal_rag_retrieval_errors_total[10m])) / clamp_min(sum(rate(multimodal_rag_retrieval_requests_total[10m])), 0.001) > 0.05` (for 10m) | warning | Tỷ lệ lỗi `/v2/rag` > 5%; xem log API và metric lỗi |
| `RagRerankFallback` | `sum(rate(multimodal_rag_retrieval_rerank_fallback_total[10m])) / clamp_min(sum(rate(multimodal_rag_retrieval_requests_total[10m])), 0.001) > 0.2` (for 10m) | warning | Rerank service thường xuyên fallback (lỗi/rỗng); kiểm tra rerank :8010 |
| `RagPresignFailures` | `sum(increase(multimodal_rag_retrieval_presign_failures_total[15m])) > 10` | warning | Presign MinIO lỗi; `artifact_url`/`file_url` sẽ null — kiểm tra MinIO và credentials |
| `RagMetricsDown` | `absent(up{job=~".*multimodal-rag-retrieval.*", endpoint="metrics"} == 1)` (for 5m) | critical | Target `metrics` (:8009) không scrape được — API chết hoặc Service/Endpoints sai |
| `RagEmbedFailures` | `sum(rate(multimodal_rag_retrieval_embed_failures_total[5m])) > 0` | warning | Embed query lỗi → `/v2/rag` trả lỗi; kiểm tra embedding-server :8008 (`BGE_API_KEY`, GPU, model/dim) |

### 9.4 Dashboard

`monitoring/dashboards/multimodal-rag-retrieval-dashboard.json` (Grafana v2 schema). Các nhóm panel (row): Tổng quan · Requests theo ngày / giờ · Latency & Errors · Pipeline stages · Docs & Chunks · Danh sách trace · Multimodal retrieval · Ingest pipeline. Panel chính:

- Tổng quan: Request Rate theo mode, Tổng request, Request hôm nay, Recent traces (datasource Tempo uid `multimodal_rag_log`).
- Requests theo ngày / giờ (plugin `volkovlabs-echarts-panel`).
- Latency & Errors: Retrieval Latency P50/P95/P99, Error Rate theo mode, Docs Returned per Request, requests in-flight.
- Pipeline stages: Stage latency P95, Stage time stacked, Rerank fallback rate, Chunks retrieved (trước rerank), Latency P95 theo mode.
- Docs & Chunks: Docs trả về theo modality, Presign failures.
- Ingest pipeline: Stream pending, Stream lag, Ingest in-flight, Ingest events theo kết quả, Ingest duration P50/P95, Ingest stage time stacked, Documents theo status, Parse failures, Items ingest theo modality, Crawl image failures.

Mỗi lần đổi metric trong `ami_rag/observability.py` phải cập nhật dashboard; test `tests/ami_service/test_observability.py::test_dashboard_metrics_exist_in_code` bắt tên metric không tồn tại trong code.

### 9.5 Deploy

- Scrape (service Docker trên host): điền `__NODE_IP__` trong `monitoring/k8s/multimodal-rag-retrieval-metrics-scrape.yaml` bằng INTERNAL-IP của node chạy Docker (`kubectl get nodes -o wide`) rồi `make metrics-scrape-apply`. File tạo Service + Endpoints (trỏ thẳng IP host) + ServiceMonitor, cả trong namespace `monitoring`. Khi service đã chạy trong k8s thì dùng `monitoring/helm/servicemonitor.yaml` (Service cần label `app: multimodal-rag-retrieval` + named port `metrics`→8009, `worker-metrics`→9109). Cả hai cấu hình scrape 2 endpoint `metrics` và `worker-metrics`, interval 15s.
- Dashboard + rule: `make dashboard-apply` sinh ConfigMap label `grafana_dashboard=1` (`monitoring/helm/dashboard-configmap.generated.yaml`, file sinh — không sửa tay) rồi apply ConfigMap + ServiceMonitor + PrometheusRule vào namespace `monitoring`. Label `release: kube-prometheus-stack` phải khớp `ruleSelector`/`serviceMonitorSelector` của Prometheus (`kubectl get prometheus -A -o jsonpath='{.items[*].spec.ruleSelector}'`).
- Tempo: `helm upgrade --install tempo-multimodal-rag grafana/tempo --version 1.24.4 -n monitoring -f monitoring/helm/tempo-multimodal-rag-values.yaml`; đặt `OTEL_EXPORTER_OTLP_ENDPOINT` (compose nạp `.env`) trỏ NodePort OTLP/HTTP của service port `4318` — tra NodePort thật bằng `kubectl get svc tempo-multimodal-rag -n monitoring`.

### 9.6 Giới hạn đã biết

- `multimodal_rag_ingest_embed_cache_hits_total` ~0 khi `EMBED_RETURN_SPARSE=true`: nhánh sparse của BGE-M3 khoá cache theo `model|dim` và gọi `use_cache=False`, nên `count_cache_hits` chỉ đọc cache dense và gần như không trúng. Tỷ lệ cache hit đo được trên dashboard vì vậy gần 0; thêm sparse cache thì bỏ hạn chế này. Chưa verify được trên cluster.
- `requests_by_day` / `requests_by_hour_of_day` / `requests_by_day_hour` sinh series mới mỗi ngày/giờ — cardinality tăng dần theo thời gian process chạy.
- `/metrics` không yêu cầu `RAG_API_KEY`; chỉ mở trong mạng nội bộ hoặc chặn ở reverse proxy.

## 10. Tests

Hai bộ chạy riêng. Trên Windows/PowerShell đặt `PYTHONIOENCODING=utf-8` trước khi chạy pytest để output tiếng Việt không lỗi mã hoá:

```powershell
# test của ami_rag (service; hoặc: make test)
$env:PYTHONIOENCODING="utf-8"; uv run --no-active pytest -q tests/ami_service/

# test gốc RAG-Anything (lib; hoặc: make test-lib)
uv run --no-active pytest -q tests --ignore=tests/ami_service

# lint (hoặc: make lint)
uv run --no-active ruff check ami_rag tests/ami_service
```

Toàn bộ test dùng fake (không cần dịch vụ ngoài, không cần lightrag — đã gỡ). Suite `tests/ami_service` gồm: API/worker/CLI/pipeline (`test_api.py`, `test_worker.py`, `test_cli.py`, `test_vector_pipeline.py`, `test_doc_status_store.py`) với fake embedder/vector store/parser; thêm `test_vector_store.py`, `test_settings_bge.py`, `test_crawl_images.py`, `test_asset_images.py`, observability và stream tests. Test gốc (`tests/*.py`) kiểm tra parser, modal processors, prompt, batch_parser của thư viện.

Lỗi đã biết, **pre-existing**, không thuộc scope thay đổi:
- 4 lỗi trong `tests/test_gateway_parity.py`: test cần repo gateway `nemotron-vl-vllm` (ngoài máy) — thiếu repo nên fail, không phải do code RAG.
- Lỗi ruff ở `tests/test_audio_processor.py` và `tests/test_video_processor.py`: nằm ngoài `tests/ami_service`, không thuộc lệnh lint service.

### 10.1 Test live retrieval (Hit@k + MRR)

`tests/test_retrieval_recall_live.py` gọi API thật `POST /v2/rag/` với bộ case `tests/retrieval_cases.json` và đo `Hit@5` / `Hit@20` / `Hit@40` + MRR theo modality (text/table/image).

Điều kiện:
- Stack ami-rag sống (API + Qdrant + embed + rerank), `/readyz` trả 200.
- Index đã reindex theo collection hiện tại (`EMBED_MODEL` + `CHUNKER_VERSION`); mỗi case trỏ `expected_doc_id`.
- Thiếu stack → test **skip**, không fail. CI chạy `-m "not live"` nên không gọi mạng.

Env (test đọc trực tiếp `os.environ`; pytest **không** tự nạp `.env` — `export` hoặc `uv run --env-file .env`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `RAG_BASE_URL` | `http://localhost:8009` | gốc API |
| `RAG_API_KEY` | rỗng | bearer nếu API bật auth |
| `RAG_TIMEOUT` | `60` | timeout mỗi request (giây) |

Lệnh:

```bash
export RAG_BASE_URL=http://localhost:8009 RAG_API_KEY=... RAG_TIMEOUT=60
uv run pytest -q -m live tests/test_retrieval_recall_live.py   # hoac make test-live
uv run pytest -q -m live tests/test_retrieval_recall_live.py -k summary  # chi bang tong hop
```

- "Hit" = top-k chứa đúng `expected_doc_id`; **không** so `expected_chunk_id` (hash MD5 đổi sau mỗi lần reindex / đổi caption → MISS giả).
- Case có `xfail: true` trong JSON là case kỳ vọng chưa trúng; `xfail_reason` giải thích. Bỏ `xfail` khi index đã cải thiện.
- `test_retrieval_summary_report` in bảng Hit@k/MRR theo modality và assert ngưỡng baseline (text/table/image). Ngưỡng gắn với dataset hiện tại — sửa `retrieval_cases.json` thì cập nhật assert.
- Xuất kết quả từng case: `python scripts/export_live_results.py --output result.json [--top-k N]` (dùng chung `run_case`, cùng cách tính Hit).
- Chẩn đoán 1 query rơi ở tầng nào: `python scripts/diagnose_query.py --query "..."`.

Khi muốn đo trước/sau một thay đổi chunk/caption: chạy bộ live này (index phải đã reindex) rồi đối chiếu Hit@k/MRR.

## 11. Phiên bản

Package `raganything` version **1.4.1** — `pyproject.toml` khai báo `dynamic = ["version"]`, giá trị lấy từ `raganything.__version__`. `pyproject.toml` không còn `lightrag-hku` — LightRAG đã gỡ hoàn toàn khỏi dependency. Extra `service` thêm: fastapi, uvicorn, pydantic-settings, redis, pymongo, qdrant-client, httpx, minio, prometheus-client, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http, openai, pdfplumber, pymupdf, python-docx. Embedding service nằm ở repo ngoài `embedding-server` (BGE-M3, máy B, port 8008), máy A không cài torch/transformers. Entry point: `ami-rag-api` (`ami_rag.api.main:run`), `ami-rag-worker` (`ami_rag.workers.ingest_worker:run`), `ami-rag` (`ami_rag.cli:main`).

`mineru[core]>=3.4.1,<4`: MinerU 4.x đổi CLI (`mineru parse <path>`, `-p` = pages) nên không tương thích với lệnh `mineru -p <file> -o <dir> -m ...` mà `raganything/parser.py` gọi; bắt buộc ghim `<4`. MinerU 3.4.x mặc định backend `hybrid-engine` (nặng VRAM) khi không truyền `-b`, và chọn thiết bị bằng biến môi trường `MINERU_DEVICE_MODE` (không có cờ `-d`); `MINERU_VIRTUAL_VRAM_SIZE` (GB) buộc MinerU chọn batch size như thể GPU có chừng đó VRAM.

Chunk multimodal: parse-only (`raganything/processor.py`) + mô tả modal (`generate_description_only`/`generate_chunk_sections`); riêng ảnh ở chế độ text-only (không VLM) do pipeline tự mô tả từ OCR + caption + ngữ cảnh (mục 5.3). Metadata có cấu trúc (`original_type`, `page_idx`, `asset_key`, `table_body`, `caption`) đi thẳng vào chunk payload để API trả về client.
