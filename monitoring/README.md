# Monitoring — Multimodal RAG (API + ingest worker)

Layout (không có Prometheus/Grafana container — dùng kube-prometheus-stack + Grafana trung tâm trong cluster). Mọi lệnh `make` chạy ở gốc repo `ami-multimodal-retrieval` (thư mục chứa `Makefile`):

```
monitoring/
├── dashboards/multimodal-rag-retrieval-dashboard.json   # Grafana v2 schema
├── helm/servicemonitor.yaml                              # service chạy TRONG k8s (2 endpoint)
├── helm/prometheusrule.yaml                              # PrometheusRule: alert ingest + retrieval
└── k8s/multimodal-rag-retrieval-metrics-scrape.yaml      # service chạy Docker trên host (Service+Endpoints+ServiceMonitor)
```

| Target | Port | Named port | Metric prefix |
|---|---|---|---|
| `ami-rag-api` | 8009 | `metrics` | `multimodal_rag_retrieval_` |
| `ami-rag-worker` | 9109 (`WORKER_METRICS_PORT`) | `worker-metrics` | `multimodal_rag_ingest_` |

Khi API chạy worker nhúng (`WORKER_ENABLED=true`) thì metric ingest nằm chung `/metrics` của API; compose đặt `WORKER_ENABLED=false` và chạy worker riêng.

## Deploy

1. Chạy service: `make start_docker`, kiểm tra `make health` (đếm series `multimodal_rag_retrieval_*` ở `:8009` và `multimodal_rag_ingest_*` ở `:9109`).
2. Scrape (service Docker trên host): điền `__NODE_IP__` trong
   `k8s/multimodal-rag-retrieval-metrics-scrape.yaml` (`kubectl get nodes -o wide` → INTERNAL-IP), rồi
   `make metrics-scrape-apply`. Khi đã deploy trong k8s thì dùng `helm/servicemonitor.yaml` thay thế (Service cần label
   `app: multimodal-rag-retrieval` và named port `metrics` → 8009, `worker-metrics` → 9109).
3. Dashboard + rule: `make dashboard-apply` (sinh ConfigMap label `grafana_dashboard=1`, rồi apply ConfigMap + ServiceMonitor + PrometheusRule vào namespace `monitoring`).
   `helm/dashboard-configmap.generated.yaml` là file sinh ra, không sửa tay.
   Label `release: kube-prometheus-stack` phải khớp selector của Prometheus
   (`kubectl get prometheus -A -o jsonpath='{.items[*].spec.ruleSelector}'`).
4. Tracing (tuỳ chọn): đặt `OTEL_EXPORTER_OTLP_ENDPOINT` (vd `http://host.docker.internal:4318`, OTLP/HTTP của `tempo-multimodal-rag`) trong `.env`; compose nạp `.env` vào container. Chỉ API (`/v2/rag`, span `rag.retrieval` kèm input/output) gửi trace; worker không trace (compose để trống `OTEL_EXPORTER_OTLP_ENDPOINT`), ingest chỉ có metric Prometheus.

## Dashboard

Timezone cố định `Asia/Ho_Chi_Minh` — khớp label `day`/`hod` trong code; không đổi sang `browser`.
Rows: Tổng quan · Requests theo ngày/giờ · Latency & Errors · Pipeline stages · Docs & Chunks · Danh sách trace · **Multimodal retrieval** (docs theo modality, presign failures, search mode dense/hybrid — vintern luôn `dense`, xem Lưu ý, artifact URLs trả về, query embed failures theo reason) · **Ingest pipeline** (stream pending/lag, ingest in-flight, events theo kết quả, duration P50/P95, stage time, documents theo status, parse failures, items theo modality, crawl image failures, ingest embed failures theo reason, embed cache hit rate).

Yêu cầu:
- Prometheus datasource uid `prometheus`.
- Tempo datasource uid `multimodal_rag_log` (panel Recent traces). Tempo riêng cho service này: release `tempo-multimodal-rag` (`monitoring/helm/tempo-multimodal-rag-values.yaml`), datasource nạp qua sidecar bằng `monitoring/helm/grafana-datasource-tempo-multimodal-rag.yaml`. URL Grafana: `http://tempo-multimodal-rag.monitoring.svc.cluster.local:3200`; app ghi trace tới OTLP/HTTP port 4318 (`OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4318`; NodePort cụ thể xem bằng `kubectl get svc tempo-multimodal-rag -n monitoring`).
- Plugin `volkovlabs-echarts-panel` (panel Requests theo ngày/giờ).

Mỗi lần đổi metric trong `ami_rag/observability.py` cần cập nhật dashboard;
`tests/ami_service/test_observability.py::test_dashboard_metrics_exist_in_code` bắt tên metric không tồn tại.

## Metric ingest (worker, `:9109/metrics`)

| Metric | Loại | Label |
|---|---|---|
| `multimodal_rag_ingest_events_total` | Counter | `event`, `result` (processed/skipped/failed/retry) |
| `multimodal_rag_ingest_in_flight` | Gauge | |
| `multimodal_rag_ingest_duration_seconds` | Histogram | `source` (minio_parse/mongo_text/none) |
| `multimodal_rag_ingest_stage_duration_seconds` | Histogram | `stage` (delete; timing parse→describe→chunk→embed nằm trong DocStatusStore, không có metric Prometheus) |
| `multimodal_rag_ingest_parse_failures_total` | Counter | `document_type` (tăng trong `VectorPipeline.run` khi stage parse lỗi) |
| `multimodal_rag_ingest_crawl_image_failures_total` | Counter | `reason` (http/redirect/connect/ssl/minio/other; tăng trong `CrawlImageWorker.ensure_in_minio`) |
| `multimodal_rag_ingest_items_total` | Counter | `modality` (tăng theo `_count_by_type` ở `_stage_parse`) |
| `multimodal_rag_ingest_pages_total` | Counter | |
| `multimodal_rag_ingest_stream_pending` / `_stream_lag` | Gauge | |
| `multimodal_rag_ingest_documents` | Gauge | `status` |
| `multimodal_rag_ingest_embed_failures_total` | Counter | `reason` (unreachable/auth/input_too_long/oom/model_mismatch/circuit_open/other) |
| `multimodal_rag_ingest_embed_items_total` | Counter | (mẫu số của cache hit rate) |
| `multimodal_rag_ingest_embed_cache_hits_total` | Counter | (tử số của cache hit rate) |

Metric API mới so với bản cũ: `multimodal_rag_retrieval_docs_by_modality_total{modality}`, `_presign_failures_total`, `_chunks_retrieved`, `_rerank_fallback_total{reason}`, `_search_mode_total{mode}` (dense|hybrid — tỉ lệ hybrid vs dense), `_artifact_urls_returned_total`, `_embed_failures_total{reason}`. Danh sách đầy đủ: `docs/ami_service.md` mục 9.

## Alert (`helm/prometheusrule.yaml`)

| Alert | Điều kiện | Severity |
|---|---|---|
| `RagIngestBacklogHigh` | pending + lag > 50 trong 15 phút | warning |
| `RagIngestFailures` | > 5 event `failed` / 30 phút | warning |
| `RagIngestParseFailures` | > 3 parse lỗi / 30 phút | warning |
| `RagCrawlImageFailures` | > 20 URL ảnh crawl lỗi / 30 phút | warning |
| `RagIngestWorkerDown` | target `worker-metrics` không UP 5 phút | critical |
| `RagIngestEmbedFailures` | có lỗi embed ở ingest (rate > 0 / 5 phút) | warning |
| `RagRetrievalErrorRate` | error ratio > 5% trong 10 phút | warning |
| `RagRerankFallback` | fallback > 20% trong 10 phút | warning |
| `RagPresignFailures` | > 10 lỗi presign / 15 phút | warning |
| `RagMetricsDown` | target `metrics` không UP 5 phút | critical |
| `RagEmbedFailures` | có lỗi embed ở retrieval (rate > 0 / 5 phút) | warning |

## Lưu ý

- Backend embed `vintern` (mặc định nhánh này): multi-vector MaxSim **dense-only** — `search_mode_total{mode}` luôn `dense`; `hybrid` chỉ xuất hiện khi embedder legacy trả sparse query. `embed_failures_total{reason}` gồm `oom` (server trả 507 khi OOM cả batch=1 — hạ `EMB_MAX_INPUT_TOKENS` server / `EMBED_MAX_INPUT_TOKENS` client về 4096 để chặn forward lớn).
- Label `day` sinh series mới mỗi ngày (cardinality tăng dần theo thời gian chạy process).
- `/metrics` không yêu cầu `RAG_API_KEY`; chỉ mở trong mạng nội bộ hoặc chặn ở reverse proxy.
