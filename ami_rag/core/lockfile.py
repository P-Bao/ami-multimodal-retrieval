"""Lockfile chống chạy hai tiến trình reindex/retry cùng lúc trên một kho dữ liệu.

Ghi pid + thời điểm vào file khoá. Tiến trình còn sống -> LockHeld; pid chết
hoặc file quá cũ -> lấy lại khoá (stale lock).
"""

import os
import time
from pathlib import Path

# Lock không xác định được chủ -> coi là stale sau tuổi này (giây)
STALE_LOCK_AGE = 6 * 3600


class LockHeld(RuntimeError):
    """Đã có tiến trình reindex/retry khác đang giữ khoá."""


def _pid_alive(pid: int) -> bool | None:
    """True/False nếu xác định được; None nếu không (-> dùng tuổi file)."""
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        return None


class Lockfile:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> tuple[int | None, float | None]:
        try:
            text = self.path.read_text(encoding="utf-8")
            pid = None
            held_at = None
            for line in text.splitlines():
                if line.startswith("pid:"):
                    pid = int(line.split(":", 1)[1].strip() or 0)
                elif line.startswith("held_at:"):
                    held_at = float(line.split(":", 1)[1].strip() or 0)
            return pid, held_at
        except (OSError, ValueError):
            return None, None

    def acquire(self) -> None:
        """Lấy khoá. Tiến trình khác đang sống giữ khoá -> LockHeld."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            pid, held_at = self._read()
            alive = _pid_alive(pid) if pid else None
            if alive:
                raise LockHeld(
                    f"Tiến trình khác đang giữ khoá {self.path} (pid={pid}). "
                    "Kiểm tra bằng `ami-rag status` rồi thử lại sau."
                )
            age = time.time() - (held_at or self.path.stat().st_mtime)
            if alive is None and age < STALE_LOCK_AGE:
                raise LockHeld(
                    f"Khoá {self.path} tồn tại (pid không xác định, {age/60:.0f} phút). "
                    "Xoá file nếu chắc chắn không còn tiến trình nào chạy."
                )
            # pid chết hoặc khoá quá cũ -> lấy lại
            self.path.unlink()
        self.path.write_text(
            f"pid:{os.getpid()}\nheld_at:{time.time():.0f}\n", encoding="utf-8"
        )

    def release(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "Lockfile":  # noqa: PYI034 (Self chỉ có từ Python 3.11)
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
