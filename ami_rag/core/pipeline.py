"""PipelineRunner - interface giữa CLI và pipeline vector thuần.

Giai đoạn 4 triển khai `VectorPipeline` thật (chunk -> embed -> upsert Qdrant);
CLI (retry/reindex) chỉ orchestrate: chọn doc, preflight embed server, gọi runner
cho từng doc, đọc DocStatusStore để hiển thị.

Hợp đồng của runner:
- Ghi DocStatusStore ở MỖI stage (nếu không CLI không có gì để đọc).
- Idempotent: stage embed luôn delete_by_doc trước khi upsert - chạy lại không
  sinh vector trùng.
- reindex/retry KHÔNG gọi lại parse hay LLM mô tả (kết quả parse + mô tả modal
  đã lưu ở MinIO), trừ khi from_stage ép làm lại từ đầu.
"""

from dataclasses import dataclass, field
from typing import Protocol

from ami_rag.storage.doc_status import STAGES


@dataclass
class StageOutcome:
    """Kết quả chạy pipeline cho một doc."""

    doc_id: str
    stage: str = ""  # stage đạt được (dù ok hay lỗi)
    ok: bool = True
    error: str = ""  # message + traceback rút gọn
    chunk_count: int = 0
    cache_hits: int = 0
    details: dict = field(default_factory=dict)


class PipelineRunner(Protocol):
    async def preflight_embed(self) -> None:
        """Kiểm tra embed server (health + handshake model/dim).

        Raise EmbedderError (EmbedServerUnreachable/EmbedModelMismatch) khi không
        sẵn sàng - CLI dừng sớm, không đánh fail từng doc.
        """
        ...

    async def run(
        self,
        doc_id: str,
        *,
        from_stage: str | None = None,
        dry_run: bool = False,
    ) -> StageOutcome:
        """Chạy pipeline cho một doc.

        from_stage: bắt đầu từ stage này (parse/describe/chunk/embed);
            None = tiếp tục từ stage đã lưu trong DocStatusStore.
        dry_run: chỉ đếm (số chunk, cache hits) - không gọi embed server,
            không ghi vector, không ghi DocStatusStore.
        """
        ...


def validate_stage(stage: str | None) -> str | None:
    """Trả về stage hợp lệ hoặc raise ValueError; None cho phép (resume).

    from_stage chỉ chấp nhận các stage trước `indexed` (resume từ `indexed`
    vô nghĩa).
    """
    if stage is None:
        return None
    resumable = STAGES[:-1]
    if stage not in resumable:
        raise ValueError(
            f"stage không hợp lệ: {stage} (hợp lệ: {', '.join(resumable)})"
        )
    return stage


def create_runner(settings=None) -> PipelineRunner:
    """Cắm pipeline runner thật (trả về VectorPipeline)."""
    from ami_rag.core.factory import build_pipeline

    return build_pipeline(settings)
