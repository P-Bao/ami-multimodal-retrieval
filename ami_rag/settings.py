import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- LLM (qwen-selfhost, OpenAI-compatible): describe_func / answer_func ---
    QWEN_LLM_BASE_URL: str = "http://vllm:8000/v1"
    QWEN_LLM_MODEL: str = "Qwen/Qwen3-32B"
    QWEN_LLM_API_KEY: str = ""
    QWEN_VLM_MODEL: str = ""

    # --- Describe stage: caption ảnh (Gemini 2.5 Flash Lite, OpenAI-compatible) ---
    # auto | gemini | vision | none.
    # auto: có CAPTION_LLM_API_KEY -> gemini, elif có QWEN_VLM_MODEL -> vision,
    # else -> none (mô tả thô: caption + footnotes + ngữ cảnh, không gọi LLM).
    # Legacy "text" (OCR-ảnh text-only) đã bị xoá -> tự chuyển "none" kèm warning.
    IMAGE_DESCRIBE_MODE: str = "auto"
    CAPTION_LLM_BASE_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    CAPTION_LLM_MODEL: str = "gemini-2.5-flash-lite"
    # API key CHỈ từ env (.env / biến môi trường) - không commit vào repo
    CAPTION_LLM_API_KEY: str = ""
    CAPTION_LLM_TIMEOUT: int = 120
    # Số request caption song song (vừa fetch ảnh MinIO vừa gọi Gemini)
    CAPTION_LLM_MAX_CONCURRENCY: int = 4
    CAPTION_LLM_RETRIES: int = 2
    # Ngôn ngữ mô tả ảnh. Corpus và truy vấn là tiếng Việt nhưng caption mặc
    # định sinh tiếng Anh -> vector ảnh lệch xa truy vấn (đo thực tế: cos với
    # query "sân bóng đá cơ sở Hà Đông" là 0.34 với caption EN, 0.48 với caption
    # VI đầy đủ). Để rỗng = không thêm chỉ dẫn ngôn ngữ (prompt gốc).
    CAPTION_LLM_LANGUAGE: str = "vi"

    # --- Remote embedding server (máy B) ---
    # backend "bge": embedding-server BAAI/bge-m3 (:8008) — dense 1024 + sparse;
    # backend "custom": qwen-embedding-server (/info + /embed);
    # backend "openai": gateway nemotron-vl-vllm (máy B) — /health + /v1/embeddings
    # với {"input", "input_type"} cho nvidia/llama-nemotron-embed-vl-1b-v2.
    EMBED_SERVER_URL: str = "http://localhost:8008"
    # Chỉ từ env - không ghi token vào file config/commit
    EMBED_SERVER_TOKEN: str = ""
    EMBED_MODEL: str = "BAAI/bge-m3"
    # auto | custom | openai | bge (auto: model chứa "bge" -> bge, prefix "Qwen/" -> custom, còn lại -> openai)
    EMBED_BACKEND: str = "bge"
    # Dùng để xác minh với server lúc handshake (dim thực tế do server quyết định)
    EMBED_DIM: int = 1024
    EMBED_TIMEOUT: int = 60
    EMBED_BATCH_SIZE: int = 32  # server giới hạn tối đa 64 text/request
    EMBED_MAX_CONCURRENCY: int = 4
    EMBED_RETRIES: int = 3
    EMBED_CACHE_ENABLED: bool = True
    EMBED_CACHE_PATH: str = "./embed_cache.db"
    # Guard kích thước input embed (client-side): item vượt budget bị head-truncate
    # trước khi gửi (content đầy đủ vẫn lưu Qdrant). Default tính cho max seq 8192
    # của BGE-M3 - server phải chạy 8192, còn không thì giảm qua env.
    EMBED_MAX_INPUT_TOKENS: int = 8192
    # BGE-M3 text-only không xử lý ảnh -> không cần trừ budget token ảnh
    EMBED_IMAGE_TOKEN_RESERVE: int = 0
    # BGE-M3 trả thêm vector sparse (lexical weights) kèm dense
    EMBED_RETURN_SPARSE: bool = True

    MONGO_URI: str = "mongodb://localhost:27017/?directConnection=true"
    # Shares organization_db with the backend; every RAG collection is prefixed `multimodal_`
    RAG_DB: str = "organization_db"
    RAG_DOCUMENTS_COLLECTION: str = "multimodal_rag_documents"
    ORG_DB: str = "organization_db"
    DOC_COLLECTION: str = "documents"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    # Giới hạn byte mỗi request upsert (server Qdrant mặc định max_request_size_mb=32)
    QDRANT_UPSERT_MAX_MB: int = 16

    WORKSPACE: str = "multimodal"

    PARSER: str = "mineru"
    PARSE_METHOD: str = "auto"
    PARSER_OUTPUT_DIR: str = "./output"

    # Nguồn text/table cho PDF/DOCX khi parse: "backend_data" = pdfplumber/fitz +
    # OCR endpoint (hierarchy-parse) + bảng Qwen endpoint (phong cách
    # documents-management/backend_data), ảnh vẫn lấy từ MinerU; "mineru" =
    # giữ nguyên text/table của MinerU (hành vi cũ).
    PARSE_TEXT_SOURCE: str = "backend_data"
    # External OCR service (text hierarchy-parse + table qwen)
    OCR_SERVICE_URL: str = "http://171.226.10.153:12006"
    OCR_SERVICE_TIMEOUT: int = 300
    OCR_TABLE_ENDPOINT: str = "/api/v1/raw_miner_ocr/qwen"
    OCR_TEXT_ENDPOINT: str = "/api/v1/hierarchy-parse"

    # --- OCR fallback (Vinter, máy B) ---
    # Dự phòng khi OCR service chính lỗi/timeout/trả rỗng: gateway vintern-vllm
    # (repo máy B, FastAPI :8090 trước vLLM serving 5CD-AI/Vintern-1B-v3_5).
    # Client: ami_rag/core/ocr_client.py; inject qua BackendDataExtractor.
    OCR_FALLBACK_ENABLED: bool = True
    # Trên máy A phải trỏ sang IP máy B qua env (localhost:8090 chỉ là default dev)
    OCR_FALLBACK_URL: str = "http://localhost:8090"
    OCR_FALLBACK_MODEL: str = "5CD-AI/Vintern-1B-v3_5"  # chỉ để log/verify
    OCR_FALLBACK_TOKEN: str = ""  # chỉ từ env - không commit
    OCR_FALLBACK_TIMEOUT: int = 300
    OCR_FALLBACK_RETRIES: int = 1
    # Primary trả rỗng cũng chuyển sang fallback (đặt False để chỉ fallback khi lỗi)
    OCR_FALLBACK_ON_EMPTY: bool = True
    # Tắt fallback riêng từng loại: "text,table" | "text" | "table" | "" (tắt hết)
    OCR_FALLBACK_MODES: str = "text,table"

    # MinerU (only used when PARSER=mineru). Measured on Colab T4, 10-page PDF: `pipeline` peaks at
    # ~1.8 GiB with auto VRAM (1.1 GiB at MINERU_VIRTUAL_VRAM_SIZE=4, 3.3 GiB at 16) in ~50 s;
    # `hybrid-engine` (MinerU 3.4.x default when -b is omitted) peaks at ~14.5 GiB in ~286 s,
    # so always pass a backend.
    MINERU_BACKEND: str = "pipeline"
    MINERU_DEVICE: str = (
        ""  # "" = auto; cuda | cuda:0 | cpu (-> env MINERU_DEVICE_MODE)
    )
    MINERU_VIRTUAL_VRAM_SIZE: int = (
        0  # GB the parser may assume; 0 = auto (use GPU's real VRAM)
    )
    MINERU_LANG: str = ""  # OCR language hint (ch, en, latin, ...); "" = MinerU default
    MINERU_SOURCE: str = (
        ""  # model source: huggingface | modelscope | local; "" = default
    )
    MINERU_TIMEOUT: int = 1800  # seconds per document; 0 = no limit

    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 100
    # Chunk ngắn hơn ngưỡng này bị gộp vào chunk kề (doc crawl mỗi trường một
    # dòng, backend_data tách PDF theo dòng -> dễ sinh chunk mồ côi kiểu
    # "- Địa chỉ: Tầng 1 nhà A1" khớp query địa chỉ nhưng vô nghĩa). 0 = tắt.
    CHUNK_MIN_CHARS: int = 200

    REDIS_URL: str = "redis://localhost:6379/0"
    RAG_STREAM: str = "rag:ingest"
    RAG_CONSUMER_GROUP: str = "ami-rag"
    RAG_CONSUMER_NAME: str = ""
    WORKER_ENABLED: bool = True
    WORKER_BATCH: int = 10
    WORKER_POLL_BLOCK_MS: int = 5000
    WORKER_MAX_DELIVERY: int = 3
    # A failed message is re-delivered once it has been pending this long (ms)
    WORKER_RETRY_IDLE_MS: int = 600000
    WORKER_METRICS_PORT: int = 9109

    # --- Pipeline / CLI ---
    # Quy ước collection: {WORKSPACE}__{embed_model_slug}__{CHUNKER_VERSION}
    CHUNKER_VERSION: str = "v1"
    # doc `processing` quá lâu (tiến trình chết giữa chừng) được `status` báo là treo
    CLI_STUCK_PROCESSING_MINUTES: int = 60

    RERANK_BASE_URL: str = "http://localhost:8010"
    # legacy: BGE reranker text-only (:8010, BAAI/bge-reranker-v2-m3); vllm: gateway
    # nemotron-vl-vllm (máy B) serving nvidia/llama-nemotron-rerank-vl-1b-v2.
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # auto | legacy | vllm (auto: model rỗng hoặc chứa "bge" -> legacy BGE, còn lại -> vllm)
    RERANK_BACKEND: str = "legacy"
    # vLLM rerank kèm ảnh + text cho chunk image/table/equation (cần asset_key);
    # legacy BGE text-only -> luôn tắt
    RERANK_MULTIMODAL: bool = False
    RERANK_TOP_K: int = 5
    RERANK_TIMEOUT: int = 60
    # Guard kích thước text gửi lên rerank server (cùng vLLM 8192): document vượt
    # budget bị head-truncate (kèm ảnh -> trừ image token reserve).
    RERANK_MAX_INPUT_TOKENS: int = 6000
    RERANK_IMAGE_TOKEN_RESERVE: int = 1792
    # Fusion điểm rerank 2 nhóm (image vs phần còn lại): "raw" = so điểm tuyệt
    # đối (hành vi cũ), "rrf" = hạng nội bộ nhóm qua RRF + sàn điểm tuyệt đối
    # (item dưới floor không được promote). Chỉ áp dụng backend vllm (full scores).
    RERANK_FUSION: str = "raw"
    RERANK_RRF_K: int = 10
    RERANK_VISUAL_WEIGHT: float = 0.8
    RERANK_VISUAL_FLOOR: float = 0.01
    RERANK_TEXT_FLOOR: float = 0.05

    RETRIEVAL_CHUNK_TOP_K: int = 40
    # docs repaired in parallel by `reindex --repair` (low: avoids provider 429)

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "ami-data-documents"
    MINIO_SECURE: bool = False
    ASSET_PREFIX: str = "rag-assets"
    MINIO_PRESIGN_EXPIRES: int = 3600

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8009
    RAG_API_KEY: str = ""


def resolve_embed_backend(settings) -> str:
    """Resolve EMBED_BACKEND: explicit value, else auto-infer từ EMBED_MODEL.

    - EMBED_BACKEND đặt rõ -> dùng luôn (custom | openai | bge).
    - auto: model chứa "bge" -> "bge" (embedding-server BGE-M3 :8008),
      prefix "Qwen/" -> "custom" (qwen-embedding-server contract),
      còn lại -> "openai" (vLLM OpenAI-compatible, Nemotron VL embed).
    """
    backend = (getattr(settings, "EMBED_BACKEND", "auto") or "auto").lower()
    if backend != "auto":
        if backend not in ("custom", "openai", "bge"):
            raise ValueError(
                f"EMBED_BACKEND không hợp lệ: {backend} (custom | openai | bge | auto)"
            )
        return backend
    model = getattr(settings, "EMBED_MODEL", "")
    if "bge" in model.lower():
        return "bge"
    return "custom" if model.startswith("Qwen/") else "openai"


def resolve_rerank_backend(settings) -> str:
    """Resolve RERANK_BACKEND: explicit value, else auto-infer từ RERANK_MODEL.

    - RERANK_BACKEND đặt rõ -> dùng luôn (legacy | vllm).
    - auto: RERANK_MODEL rỗng hoặc chứa "bge" -> "legacy" (BGE reranker,
      documents text-only), còn lại -> "vllm" (Nemotron VL rerank, multimodal).
    """
    backend = (getattr(settings, "RERANK_BACKEND", "auto") or "auto").lower()
    if backend != "auto":
        if backend not in ("legacy", "vllm"):
            raise ValueError(
                f"RERANK_BACKEND không hợp lệ: {backend} (legacy | vllm | auto)"
            )
        return backend
    model = (getattr(settings, "RERANK_MODEL", "") or "").lower()
    return "vllm" if model and "bge" not in model else "legacy"


def resolve_image_describe_mode(settings) -> str:
    """Resolve IMAGE_DESCRIBE_MODE: "gemini" | "vision" | "none".

    - Giá trị rõ (gemini | vision | none) -> dùng luôn.
    - Legacy "text" (OCR-ảnh text-only, đã bị xoá ở phase caption Gemini) ->
      log warning + "none" (mô tả thô caption + footnotes + ngữ cảnh).
    - auto: có CAPTION_LLM_API_KEY -> "gemini"; elif có QWEN_VLM_MODEL ->
      "vision"; else -> "none".
    - Giá trị khác -> ValueError (cấu hình sai phải lộ diện lúc khởi động).
    """
    mode = (getattr(settings, "IMAGE_DESCRIBE_MODE", "auto") or "auto").lower()
    if mode == "text":
        logger.warning(
            'IMAGE_DESCRIBE_MODE="text" là giá trị legacy đã xoá (path OCR-ảnh text-only). '
            'Tự chuyển sang "none" (caption + footnotes + ngữ cảnh, không LLM). '
            'Đặt "gemini" để bật caption ảnh bằng Gemini.'
        )
        return "none"
    if mode in ("gemini", "vision", "none"):
        return mode
    if mode != "auto":
        raise ValueError(
            f"IMAGE_DESCRIBE_MODE không hợp lệ: {mode} (auto | gemini | vision | none)"
        )
    if getattr(settings, "CAPTION_LLM_API_KEY", ""):
        return "gemini"
    return "vision" if getattr(settings, "QWEN_VLM_MODEL", "") else "none"


def resolve_ocr_fallback_modes(settings) -> frozenset[str]:
    """Parse OCR_FALLBACK_MODES ("text,table") -> frozenset({"text", "table"}).

    - Chuỗi rỗng -> frozenset() (tắt fallback cho cả text lẫn table).
    - Giá trị lạ -> ValueError (để cấu hình sai được phát hiện lúc khởi động).
    """
    raw = (getattr(settings, "OCR_FALLBACK_MODES", "text,table") or "").strip()
    modes: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in ("text", "table"):
            raise ValueError(
                f"OCR_FALLBACK_MODES không hợp lệ: '{part}' (chỉ 'text', 'table', "
                "phân cách bằng dấu phẩy; chuỗi rỗng = tắt fallback)"
            )
        modes.add(part)
    return frozenset(modes)


def parser_kwargs(settings) -> dict:
    """Extra kwargs for `RAGAnything.parse_document` derived from MINERU_* settings.

    Empty for non-MinerU parsers. Device and VRAM cap go through the subprocess env
    (MINERU_DEVICE_MODE / MINERU_VIRTUAL_VRAM_SIZE), which is what the Colab VRAM sweep measured.
    """
    if getattr(settings, "PARSER", "mineru") != "mineru":
        return {}
    kwargs: dict = {}
    if backend := getattr(settings, "MINERU_BACKEND", ""):
        kwargs["backend"] = backend
    if lang := getattr(settings, "MINERU_LANG", ""):
        kwargs["lang"] = lang
    if source := getattr(settings, "MINERU_SOURCE", ""):
        kwargs["source"] = source
    if timeout := getattr(settings, "MINERU_TIMEOUT", 0):
        kwargs["timeout"] = int(timeout)
    env: dict[str, str] = {}
    if device := getattr(settings, "MINERU_DEVICE", ""):
        env["MINERU_DEVICE_MODE"] = device
    if vram := getattr(settings, "MINERU_VIRTUAL_VRAM_SIZE", 0):
        env["MINERU_VIRTUAL_VRAM_SIZE"] = str(int(vram))
    if env:
        kwargs["env"] = env
    return kwargs


@lru_cache
def get_settings() -> Settings:
    return Settings()
