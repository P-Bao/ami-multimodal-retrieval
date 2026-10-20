# AMI Multimodal Retrieval

Service multimodal RAG cho hệ thống AMI: ingest worker parse tài liệu (PDF/DOCX crawl từ MongoDB `organization_db.documents`) bằng MinerU, caption ảnh tiếng Việt (Gemini 2.5 Flash Lite), chunk v2, embed BGE-M3 (dense + sparse) vào Qdrant hybrid. API truy xuất `POST /v2/rag` trả text + ảnh presigned từ MinIO.

Pipeline là vector thuần — không LightRAG (không entity/graph/neo4j).

## Kiến trúc

```
ami_data (MongoDB organization_db.documents + MinIO)
        │
        ▼
Redis Stream `rag:ingest`
        │
        ▼
ami-rag-worker ── stages: parse ─► describe ─► chunk ─► embed ─► indexed
  (hoặc worker nhúng trong API khi WORKER_ENABLED=true)
  parse:    MinerU (PARSE_TEXT_SOURCE=backend_data)
  describe: caption ảnh tiếng Việt, Gemini 2.5 Flash Lite
  chunk:    chunk v2 (CHUNK_SIZE/CHUNK_OVERLAP/CHUNK_MIN_CHARS)
  embed:    BGE-M3 dense 1024 + sparse lexical weights ─► upsert Qdrant
        │
        ▼
   Qdrant (hybrid: named vectors `dense` + `sparse`)
        ▲
AMI app ── POST /v2/rag ──► ami-rag-api :8009 ──► text + ảnh presigned MinIO
```

Các artifact trung gian (`content_list.json`, `descriptions.json`, `chunks.json`) được lưu lên MinIO, nên khi retry doc tiếp tục từ stage lỗi thay vì parse lại từ đầu.

## Dịch vụ ngoài (không nằm trong repo)

| Dịch vụ | Địa chỉ mặc định | Vai trò |
|---|---|---|
| MongoDB | `MONGO_URI` | Nguồn tài liệu crawl (`organization_db.documents`), registry trạng thái `multimodal_rag_documents` |
| Qdrant | `http://localhost:6333` | Vector store hybrid |
| Redis | `redis://localhost:6379/0` | Stream `rag:ingest` cho ingest worker |
| MinIO | `localhost:9000` | File gốc, ảnh/bảng đã tách, artifacts trung gian; presigned URL trả về client |
| Embed server BGE-M3 | `:8008` (repo `embedding-server`) | `BAAI/bge-m3`: dense 1024 + sparse, auth `Bearer BGE_API_KEY`, tối đa 64 text/request |
| Rerank BGE | `:8010` (repo `rerank`) | `BAAI/bge-reranker-v2-m3`: documents text-only, tối đa 100 docs/request |
| OCR fallback Vinter | `:8090` (tuỳ chọn) | `5CD-AI/Vintern-1B-v3_5` — phòng khi OCR service chính lỗi/treo/tra rỗng |

## Quickstart

Yêu cầu: Python 3.10+ và [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra service
cp .env.ami.example .env   # điền QDRANT_URL, MINIO_ENDPOINT/KEY, MONGO_URI, REDIS_URL,
                           # CAPTION_LLM_API_KEY, BGE_API_KEY (EMBED_SERVER_TOKEN), ...
make start_docker          # compose: ami-rag-api :8009 + ami-rag-worker :9109
```

Kiểm tra:

```bash
curl http://localhost:8009/healthz
make health   # gọi /healthz + đếm series metrics ở :8009 (API) và :9109 (worker)
```

GPU cho MinerU là tuỳ chọn: chạy được trên CPU (~5x chậm hơn). Compose mặc định chạy CPU; khi có GPU, bỏ comment block `deploy.resources` trong `docker-compose.ami.yml` (chạy CPU-only: giữ block comment và đặt `MINERU_DEVICE=cpu`).

## Biến môi trường

Đọc từ `.env` (copy từ `.env.ami.example`). Nhóm chính:

| Biến | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `API_PORT` | `8009` | Port API |
| `WORKER_ENABLED` | `true` | `true` = API nhúng worker; compose đặt `false` và chạy `ami-rag-worker` riêng |
| `WORKER_METRICS_PORT` | `9109` | Port `/metrics` của worker độc lập |
| `EMBED_BACKEND` | `bge` | Backend embed (`auto\|custom\|openai\|bge`) |
| `EMBED_SERVER_URL` | `http://localhost:8008` | Server BGE-M3 (trong Docker: `http://bge-m3-embed:8008`) |
| `EMBED_MODEL` | `BAAI/bge-m3` | Model embed |
| `EMBED_DIM` | `1024` | Số chiều dense |
| `EMBED_RETURN_SPARSE` | `true` | Lấy thêm sparse lexical weights cho hybrid |
| `EMBED_BATCH_SIZE` | `32` | ≤ 64 (giới hạn server BGE-M3) |
| `RERANK_BACKEND` | `legacy` | BGE reranker text-only (`auto\|legacy\|vllm`) |
| `RERANK_BASE_URL` | `http://localhost:8010` | Rerank service |
| `RERANK_MULTIMODAL` | `false` | Chỉ backend `vllm` mới rerank kèm ảnh |
| `RERANK_FUSION` | `raw` | `raw\|rrf` — fuse điểm 2 nhóm modality (image vs text) |
| `QDRANT_URL` | `http://localhost:6333` | (+ `QDRANT_API_KEY`) |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET` | `localhost:9000` / `minioadmin` / — / `ami-data-documents` | Storage asset |
| `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | (+ `RAG_DB`, `ORG_DB`, `DOC_COLLECTION`) |
| `REDIS_URL` / `RAG_STREAM` | `redis://localhost:6379/0` / `rag:ingest` | Queue ingest |
| `IMAGE_DESCRIBE_MODE` | `auto` | `auto\|gemini\|vision\|none` — cách mô tả ảnh |
| `CAPTION_LLM_API_KEY` | — | Bật caption Gemini (`CAPTION_LLM_MODEL=gemini-2.5-flash-lite`, `CAPTION_LLM_LANGUAGE=vi`) |
| `PARSE_TEXT_SOURCE` | `backend_data` | Nguồn text/bảng khi parse PDF/DOCX |
| `CHUNKER_VERSION` | `v1` | Một phần tên collection Qdrant |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (trống) | Tracing OTLP/HTTP tuỳ chọn (trống = tắt) |

Danh sách đầy đủ: `.env.ami.example` và [docs/ami_service.md](docs/ami_service.md) §4.

## API

- `POST /v2/rag/` — retrieval đồng bộ.
- `POST /v2/rag/stream` — cùng body, trả NDJSON: `{"status": "retrieving"}`, mỗi doc một dòng `{"documents": [...]}` (text, metadata, modality, score, `artifact_url`, `artifact_urls`), kết thúc `{"status": "done"}`.

Body:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "top_k": 8,
  "include_references": true
}
```

Response: `query`, `documents` (danh sách `RetrievedDocV2`), `references`, `meta` (latency_ms, requested_top_k, returned, candidates, fusion). Mỗi document gồm `text`, `score`, `metadata`, `modality`, `artifact_url` (1 ảnh của chunk) và `artifact_urls` (danh sách presigned mọi ảnh của doc: ảnh crawl + ảnh PDF).

Auth: `Authorization: Bearer $RAG_API_KEY` — để `RAG_API_KEY` trống trong `.env` = không auth. `/metrics` không cần key.

Luồng truy xuất: embed query (dense + sparse) → hybrid search trong Qdrant (fuse RRF của dense và sparse) → rerank BGE text-only trên caption → trả `top_k` document. `top_k` của request là số doc trả về cuối cùng; vector search lấy dư ứng viên (`max(RETRIEVAL_CHUNK_TOP_K, top_k*2)` — cap 2× để không dội quá nhiều POST lên rerank khi `top_k` lớn).

Ngoài ra còn có `/admin/*` — xem [docs/ami_service.md](docs/ami_service.md) §6.

## CLI `ami-rag`

Bảo trì pipeline, 4 lệnh:

```bash
ami-rag status [--failed] [--stale] [--doc ID] [--check-embed-server]
ami-rag retry (--doc ID ... | --all-failed) [--from-stage STAGE] [--max-attempts N] [--yes]
ami-rag reindex [--scan | --all | --doc ID ... | --stale] [--from-stage STAGE]
                [--dry-run] [--yes] [--limit N] [--sleep S] [--batch B] [--batch-sleep S]
ami-rag cleanup [--stale-models] [--purge-cache-model MODEL ...] [--dry-run] [--yes]
```

- `status` mặc định chỉ đọc local (0 LLM, 0 embed); `--check-embed-server` gọi `/health` + handshake server embed.
- `reindex` chạy ngắt quãng để tránh quá tải embed server: `--sleep` giữa 2 doc (mặc định 0.2s), `--batch 20` doc mỗi lô, `--batch-sleep 2.0`s giữa các lô, `--limit N` cắt số doc. `--from-stage` nhận `parse|describe|chunk|embed`; `--dry-run` in số doc/chunk cần embed và số trúng cache, không ghi gì.
- `cleanup --stale-models` xoá collection Qdrant của embed model/version cũ; `--purge-cache-model` xoá cache embedding SQLite theo model.

Khi chạy Docker dùng `make cli`: `make cli status`, `make cli -- reindex --stale --dry-run` (các flag `--xxx` cần dấu `--` để make không tự parse), `make cli BG=1 -- ...` để chạy nền.

## Collection Qdrant

Tên collection: `{WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}` — hiện tại `multimodal__bge-m3__v1`.

Đổi `EMBED_MODEL` hoặc `CHUNKER_VERSION` → service tự dùng collection mới; chạy `ami-rag reindex --scan` để index lại, rồi `ami-rag cleanup --stale-models` để dọn collection cũ.

## Cấu trúc repo

```
ami_rag/            # service: api/, core/, workers/, storage/, queue/
raganything/        # thư viện parser multimodal (dùng lại từ RAG-Anything)
tests/              # tests/ami_service/ = test service; phần còn lại = test thư viện
scripts/            # diagnose_query.py, eval_rerank_fusion.py, recaption_image.py, export_live_results.py
docs/ami_service.md # tài liệu đầy đủ của service
monitoring/         # Prometheus + Grafana dashboard + Tempo
Dockerfile.ami
docker-compose.ami.yml
Makefile
```

## Test và lint

```bash
uv run pytest -q tests/ami_service/                  # test service (không cần dịch vụ ngoài, dùng fake)
uv run pytest -q tests --ignore=tests/ami_service    # test thư viện RAG-Anything
uv run ruff check ami_rag tests/ami_service          # lint (tương đương `make lint`)
```

## Monitoring

Prometheus `/metrics`: API `:8009` với prefix `multimodal_rag_retrieval_`, worker `:9109` với prefix `multimodal_rag_ingest_` (khi `WORKER_ENABLED=true`, metric ingest nằm chung `/metrics` của API). Grafana dashboard + Tempo trace cho `/v2/rag` (release `tempo-multimodal-rag`, OTLP/HTTP qua `OTEL_EXPORTER_OTLP_ENDPOINT`). Apply lên cluster: `make dashboard-apply`, `make metrics-scrape-apply`.

Chi tiết: [monitoring/README.md](monitoring/README.md) và [docs/ami_service.md](docs/ami_service.md) §9.

## License

MIT — xem [LICENSE](LICENSE).
