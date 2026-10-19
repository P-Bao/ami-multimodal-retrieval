from ami_rag.queue.events import RagEvent, event_fields, event_from_fields


def test_roundtrip_all_fields():
    event = RagEvent(
        event="created",
        document_id="doc-1",
        content_hash="abc",
        document_type="pdf",
        org_id="org-1",
        ts="123",
    )
    parsed = event_from_fields(event_fields(event))
    assert parsed == event


def test_roundtrip_minimal_fields():
    event = RagEvent(event="deleted", document_id="doc-2")
    parsed = event_from_fields(event_fields(event))
    assert parsed.event == "deleted"
    assert parsed.document_id == "doc-2"
    assert parsed.content_hash is None


def test_invalid_event_rejected():
    assert event_from_fields({"event": "bogus", "document_id": "doc-1"}) is None
    assert event_from_fields({"event": "created", "document_id": ""}) is None
    assert event_from_fields({}) is None


def test_whitespace_normalized():
    parsed = event_from_fields({"event": " created ", "document_id": " doc-1 "})
    assert parsed.event == "created"
    assert parsed.document_id == "doc-1"
