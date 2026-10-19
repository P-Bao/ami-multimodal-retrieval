from ami_rag.queue.events import RagEvent
from ami_rag.queue.streams import RagStreamQueue


class FakeAsyncRedis:
    def __init__(self):
        self.streams = {}
        self.groups = {}
        self.pending = {}

    async def xgroup_create(self, stream, group, id="0", mkstream=False):
        groups = self.groups.setdefault(stream, {})
        if group in groups:
            raise Exception("BUSYGROUP Consumer Group name already exists")
        groups[group] = id
        self.pending.setdefault((stream, group), {})

    async def xadd(self, stream, fields):
        entries = self.streams.setdefault(stream, [])
        msg_id = f"{len(entries) + 1}-1"
        entries.append((msg_id, dict(fields)))
        return msg_id

    async def xreadgroup(self, group, consumer, streams, count=10, block=None):
        result = []
        for stream, _ in streams.items():
            entries = self.streams.get(stream, [])
            delivered = self._delivered_ids(stream, group)
            fresh = [(mid, f) for mid, f in entries if mid not in delivered][:count]
            for mid, f in fresh:
                delivered[mid] = True
                self.pending[(stream, group)][mid] = {"consumer": consumer}
            if fresh:
                result.append((stream, fresh))
        return result

    def _delivered_ids(self, stream, group):
        return self._pending_entry(stream, group).setdefault("_delivered", {})

    def _pending_entry(self, stream, group):
        return self.pending.setdefault((stream, group), {})

    async def xautoclaim(self, stream, group, consumer, min_idle_time=0, start_id="0-0", count=10):
        # fake: every delivered-but-unacked message counts as idle long enough
        pending = self._pending_entry(stream, group)
        ids = [k for k in pending if k != "_delivered"]
        entries = dict(self.streams.get(stream, []))
        claimed = [(mid, entries[mid]) for mid in ids if mid in entries][:count]
        for mid, _ in claimed:
            pending[mid] = {"consumer": consumer}
        return ["0-0", claimed, []]

    async def xack(self, stream, group, message_id):
        pending = self._pending_entry(stream, group)
        pending.pop(message_id, None)

    async def xpending(self, stream, group):
        pending = self._pending_entry(stream, group)
        count = sum(1 for k in pending if k != "_delivered")
        return {"pending": count}

    async def xinfo_groups(self, stream):
        if stream not in self.groups:
            raise Exception("ERR no such key")
        return [
            {
                "name": group,
                "pending": len(
                    [k for k in self._pending_entry(stream, group) if k != "_delivered"]
                ),
                "lag": len(self.streams.get(stream, []))
                - len(self._pending_entry(stream, group).get("_delivered", {})),
            }
            for group in self.groups[stream]
        ]


def test_ensure_group_idempotent():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1")
        await queue.ensure_group()
        await queue.ensure_group()

    asyncio.run(main())


def test_publish_and_read_and_ack():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1")
        await queue.ensure_group()
        await queue.publish(RagEvent(event="created", document_id="doc-1"))
        batch = await queue.read_batch(count=10, block_ms=10)
        assert len(batch) == 1
        message_id, event = batch[0]
        assert event.document_id == "doc-1"
        assert await queue.pending_count() == 1
        await queue.ack(message_id)
        assert await queue.pending_count() == 0

    asyncio.run(main())


def test_stats_pending_and_lag():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1")
        await queue.ensure_group()
        for i in range(3):
            await queue.publish(RagEvent(event="created", document_id=f"doc-{i}"))
        assert await queue.stats() == (0, 3)
        await queue.read_batch(count=2, block_ms=10)
        assert await queue.stats() == (2, 1)

    asyncio.run(main())


def test_stats_returns_zero_on_error():
    import asyncio

    async def main():
        # Group never created -> fake raises; stats must not propagate.
        queue = RagStreamQueue(FakeAsyncRedis(), "rag:ingest", "ami-rag", "c1")
        assert await queue.stats() == (0, 0)

        class Broken:
            async def xinfo_groups(self, stream):
                return [{"name": "ami-rag", "pending": 1, "lag": None}]

        # Redis may report lag=None; treated as 0.
        assert await RagStreamQueue(Broken(), "s", "ami-rag", "c1").stats() == (1, 0)

    asyncio.run(main())


def test_malformed_entry_skipped():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()
        await redis.xadd("rag:ingest", {"event": "bogus", "document_id": "x"})
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1")
        await queue.ensure_group()
        batch = await queue.read_batch(count=10, block_ms=10)
        assert batch == []

    asyncio.run(main())


def test_unacked_message_is_redelivered_until_acked():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1", retry_idle_ms=0)
        await queue.ensure_group()
        await queue.publish(RagEvent(event="created", document_id="doc-1"))

        first = await queue.read_batch(count=10, block_ms=10)
        assert [e.document_id for _, e in first] == ["doc-1"]
        # worker failed and did not ack: the same message comes back
        again = await queue.read_batch(count=10, block_ms=10)
        assert [(m, e.document_id) for m, e in again] == [(first[0][0], "doc-1")]

        await queue.ack(first[0][0])
        assert await queue.read_batch(count=10, block_ms=10) == []

    asyncio.run(main())


def test_claim_failure_falls_back_to_new_messages():
    import asyncio

    async def main():
        redis = FakeAsyncRedis()

        async def boom(*args, **kwargs):
            raise RuntimeError("redis < 6.2")

        redis.xautoclaim = boom
        queue = RagStreamQueue(redis, "rag:ingest", "ami-rag", "c1")
        await queue.ensure_group()
        await queue.publish(RagEvent(event="created", document_id="doc-1"))
        batch = await queue.read_batch(count=10, block_ms=10)
        assert [e.document_id for _, e in batch] == ["doc-1"]

    asyncio.run(main())
