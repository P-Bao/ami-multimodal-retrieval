import logging
import socket
import uuid

import redis.asyncio as aioredis

from ami_rag.queue.events import RagEvent, event_fields, event_from_fields

logger = logging.getLogger(__name__)


class RagStreamQueue:
    """Redis Streams queue for RAG ingest events.

    The publisher (backend docs) XADDs thin events (ids only); this consumer
    group reads them and acks after success. A message whose processing failed
    is left pending and re-claimed (XAUTOCLAIM) once it has been idle for
    `retry_idle_ms`, until the worker's delivery budget is exhausted and it acks.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        stream: str,
        group: str,
        consumer: str | None = None,
        retry_idle_ms: int = 600_000,
    ):
        self._redis = redis_client
        self._stream = stream
        self._group = group
        self._consumer = consumer or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._retry_idle_ms = retry_idle_ms

    async def ensure_group(self) -> None:
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _claim_stale(self, count: int) -> list[tuple[str, RagEvent]]:
        """Pending messages idle for `retry_idle_ms` (failed earlier, or from a dead consumer)."""
        try:
            resp = await self._redis.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._retry_idle_ms,
                start_id="0-0",
                count=count,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to new messages only
            logger.warning("xautoclaim failed for %s: %s", self._stream, exc)
            return []
        claimed = resp[1] if resp and len(resp) > 1 else []
        batch: list[tuple[str, RagEvent]] = []
        for message_id, fields in claimed or []:
            event = event_from_fields(fields) if fields else None
            if event is not None:
                batch.append((message_id, event))
        return batch

    async def read_batch(self, count: int = 10, block_ms: int = 5000) -> list[tuple[str, RagEvent]]:
        stale = await self._claim_stale(count)
        if stale:
            return stale
        resp = await self._redis.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=count,
            block=block_ms,
        )
        batch: list[tuple[str, RagEvent]] = []
        for _stream, messages in resp or []:
            for message_id, fields in messages:
                event = event_from_fields(fields)
                if event is not None:
                    batch.append((message_id, event))
        return batch

    async def ack(self, message_id: str) -> None:
        await self._redis.xack(self._stream, self._group, message_id)

    async def publish(self, event: RagEvent) -> str:
        return await self._redis.xadd(self._stream, event_fields(event))

    async def pending_count(self) -> int:
        summary = await self._redis.xpending(self._stream, self._group)
        if not summary:
            return 0
        return int(summary.get("pending", 0))

    async def stats(self) -> tuple[int, int]:
        """(pending, lag) of the consumer group via XINFO GROUPS; (0, 0) on any error."""
        try:
            for info in await self._redis.xinfo_groups(self._stream) or []:
                name = info.get("name")
                if isinstance(name, bytes):
                    name = name.decode()
                if name == self._group:
                    return int(info.get("pending") or 0), int(info.get("lag") or 0)
        except Exception as exc:  # noqa: BLE001 - metrics must never break the worker
            logger.debug("xinfo_groups failed for %s: %s", self._stream, exc)
        return 0, 0
