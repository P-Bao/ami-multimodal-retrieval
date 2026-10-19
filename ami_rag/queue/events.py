from collections.abc import Mapping
from dataclasses import dataclass

EVENT_CREATED = "created"
EVENT_UPDATED = "updated"
EVENT_DELETED = "deleted"
VALID_EVENTS = frozenset({EVENT_CREATED, EVENT_UPDATED, EVENT_DELETED})


@dataclass(frozen=True)
class RagEvent:
    event: str
    document_id: str
    content_hash: str | None = None
    document_type: str | None = None
    org_id: str | None = None
    ts: str | None = None


def event_from_fields(fields: Mapping[str, str]) -> RagEvent | None:
    """Parse Redis stream fields into a RagEvent; None for malformed entries."""
    event = (fields.get("event") or "").strip()
    document_id = (fields.get("document_id") or "").strip()
    if event not in VALID_EVENTS or not document_id:
        return None
    return RagEvent(
        event=event,
        document_id=document_id,
        content_hash=fields.get("content_hash") or None,
        document_type=fields.get("document_type") or None,
        org_id=fields.get("org_id") or None,
        ts=fields.get("ts") or None,
    )


def event_fields(event: RagEvent) -> dict:
    """Serialize a RagEvent into str->str Redis stream fields."""
    return {
        "event": event.event,
        "document_id": event.document_id,
        "content_hash": event.content_hash or "",
        "document_type": event.document_type or "",
        "org_id": event.org_id or "",
        "ts": event.ts or "",
    }
