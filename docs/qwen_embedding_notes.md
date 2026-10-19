# Ghi chú Qwen3-VL-Embedding-2B (cho embed server + RemoteEmbedder)

Nguồn:
- Model card: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B (đọc 2026-10)
- Code tham chiếu chính thức: https://github.com/QwenLM/Qwen3-VL-Embedding →
  `src/models/qwen3_vl_embedding.py` (vendor vào `qwen-embedding-server/app/qwen3_vl_embedding.py`)
- Technical report: arXiv 2601.04720

## Số chiều (dim) / MRL

- Embedding dim tối đa **2048** (2B model; bản 8B là 4096).
- **MRL support: Yes** — giảm chiều được, các mức tuỳ ý **64 → 2048**.
- Server expose `EMB_DIM` (mặc định 2048); dim < 2048 thực hiện bằng cắt N thành phần đầu
  rồi L2-normalize lại (chuẩn MRL). `dim` đọc từ config server, trả về qua `/info`.

## Pooling / Normalize (xác minh trong reference code)

- **Last-token pooling**: `_pooling_last` — lấy hidden state của token cuối cùng (flip
  attention mask, argmax vị trí 1 cuối). Processor khởi tạo với `padding_side='right'`.
- **L2 normalize mặc định**: `F.normalize(embeddings, p=2, dim=-1)` trong `process(..., normalize=True)`.
  Tương đồng = dot product (cùng chiều không gian).

## Instruction (instruction-aware: Yes)

- Đưa qua **system message**: `{"role": "system", "content": [{"type": "text", "text": instruction}]}`.
- Default: `"Represent the user's input."`.
- Reference tự thêm dấu `.` nếu instruction kết thúc không phải dấu câu
  (unicodedata category P*).
- Model card khuyến nghị: viết instruction **tiếng Anh**, tailor theo task (+1–5%).
- Server cấu hình 1 chỗ: `EMB_QUERY_INSTRUCTION`, `EMB_DOCUMENT_INSTRUCTION`;
  client KHÔNG gửi instruction, đọc từ `/info` để đưa vào metadata + cache key.

## Input

- Modalities: **text, ảnh, screenshot, video, và tổ hợp** (text+image, ...).
- Ảnh: qua `qwen_vl_utils.process_vision_info`; nhận PIL Image, đường dẫn (`file://`
  tự thêm), URL http(s). Resize theo `min_pixels`/`max_pixels`
  (default reference: MIN 4*32*32=4096 px, MAX 1800*32*32=1.8M px).
- Video: sample frame theo `fps` (1) / `max_frames` (64) / `total_pixels`.
- Text rỗng hoàn toàn → nội dung placeholder `"NULL"` (reference).

## Độ dài input / context

- Context length model: **32k**.
- Reference code đặt `MAX_LENGTH = 8192` (processor `truncation=True, max_length=...`,
  giữ special tokens khi cắt). Server mặc định `EMB_MAX_INPUT_TOKENS=8192` (theo
  reference, an toàn VRAM); text vượt → bị cắt (ghi `truncated` indices trong response).

## Dependency / phiên bản

- Model card requirements: `transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, `torch==2.8.0`.
- **flash_attention_2: KHÔNG bắt buộc** — chỉ khuyến nghị (tăng tốc + tiết kiệm memory);
  mặc định server dùng SDPA.
- Model class: `Qwen3VLForEmbedding` tự định nghĩa trong reference (kế thừa
  `Qwen3VLPreTrainedModel`, bọc `Qwen3VLModel` không LM head) → **phải vendor file**
  `qwen3_vl_embedding.py`, không có trên PyPI.

## VRAM / dtype

- Weights safetensors: **BF16** (2B params ≈ ~4 GB).
- Model card ví dụ: `torch_dtype=torch.float16` + flash_attention_2 (khuyến nghị);
  reference vLLM example dùng `dtype=bfloat16`.
- Server: `EMB_DTYPE` mặc định `bfloat16`, `EMB_DEVICE` auto cuda; OOM → chia nhỏ batch
  nội bộ (giảm dần tới 1) thay vì crash.

## Liên quan thiết kế

- Client không gửi instruction; server quyết định theo `type: query|document` của item.
- `/info` trả `model_name`, `dim`, `max_input_tokens`, `normalize`, instructions —
  client handshake so khớp trước khi ghi vector bất kỳ.
- Embed ảnh trực tiếp (payload base64 qua mạng) khả thi nhưng **TẮT mặc định**
  (`embed_images_directly=False`); mặc định chỉ embed text (gồm mô tả modal dạng text).
