"""Cache embedding phía máy A (SQLite) - tiết kiệm lượt gọi mạng sang máy B.

Khoá cache do ``RemoteEmbedder`` tạo: ``(model, dim, instruction_ns, sha256(text))``.
Đổi embed_model, dim hoặc instruction phía server -> khoá đổi -> cache cũ tự hết hiệu lực.
"""

import sqlite3
import struct
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for key in set(keys):
            row = self._conn.execute(
                "SELECT dim, vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                dim, blob = row
                out[key] = list(struct.unpack(f"<{dim}f", blob))
        return out

    def put_many(self, mapping: dict[str, list[float]]) -> None:
        rows = [
            (key, len(vec), struct.pack(f"<{len(vec)}f", *vec))
            for key, vec in mapping.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def delete_by_prefix(self, prefix: str) -> int:
        """Xoá mọi entry có key bắt đầu bằng `prefix` (ví dụ `f"{model}|"`);
        trả về số entry đã xoá. Wildcard `%`/`_` trong prefix được escape."""
        escaped = (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        cur = self._conn.execute(
            "DELETE FROM embeddings WHERE key LIKE ? ESCAPE '\\'", (escaped + "%",)
        )
        self._conn.commit()
        return int(cur.rowcount)

    def close(self) -> None:
        self._conn.close()
