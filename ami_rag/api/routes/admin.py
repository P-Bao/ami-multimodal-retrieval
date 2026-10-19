import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ami_rag.api.routes.rag import get_asset_store, verify_api_key
from ami_rag.queue.events import EVENT_UPDATED, RagEvent
from ami_rag.settings import get_settings

router = APIRouter(tags=["admin"], dependencies=[Depends(verify_api_key)])

_queue_instance = None


def get_queue():
    global _queue_instance
    if _queue_instance is None:
        from ami_rag.queue.streams import RagStreamQueue
        from ami_rag.workers.ingest_worker import aioredis_from_url

        settings = get_settings()
        _queue_instance = RagStreamQueue(
            redis_client=aioredis_from_url(settings.REDIS_URL),
            stream=settings.RAG_STREAM,
            group=settings.RAG_CONSUMER_GROUP,
        )
    return _queue_instance


def get_state_repo():
    from ami_rag.storage.doc_status import DocStatusStore

    settings = get_settings()
    return DocStatusStore(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.RAG_DB,
        collection_name=settings.RAG_DOCUMENTS_COLLECTION,
    )


def get_docs_repo():
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    settings = get_settings()
    return MongoDocumentRepo(
        mongo_uri=settings.MONGO_URI,
        db_name=settings.ORG_DB,
        collection_name=settings.DOC_COLLECTION,
    )


@router.get("/pipeline_status")
async def pipeline_status(queue=Depends(get_queue), state_repo=Depends(get_state_repo)):
    settings = get_settings()
    try:
        status_counts = await asyncio.to_thread(state_repo.counts)
    except Exception:
        status_counts = {}
    try:
        queue_pending = await queue.pending_count()
    except Exception:
        queue_pending = None
    return {
        "workspace": settings.WORKSPACE,
        "doc_status_counts": status_counts,
        "queue_pending": queue_pending,
    }


@router.post("/reprocess_failed")
async def reprocess_failed(
    queue=Depends(get_queue),
    state_repo=Depends(get_state_repo),
):
    failed_rows = await asyncio.to_thread(state_repo.failed_rows)
    requeued = 0
    for row in failed_rows:
        event = RagEvent(
            event=EVENT_UPDATED,
            document_id=str(row.get("_id")),
            content_hash=row.get("content_hash") or None,
        )
        await queue.publish(event)
        requeued += 1
    return {"requeued": requeued}


def _join_caption(value) -> str | None:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v)
    return value or None


def build_content_blocks(content_list: list[dict], presign=lambda key: None) -> list[dict]:
    """Normalize a parsed content_list into API blocks (asset_url = presigned asset_key)."""
    blocks = []
    for item in content_list or []:
        block_type = item.get("type") or "text"
        block = {"type": block_type, "page_idx": item.get("page_idx")}
        if block_type == "table":
            block["table_body"] = item.get("table_body")
            block["caption"] = _join_caption(item.get("table_caption"))
            block["asset_url"] = presign(item.get("asset_key"))
        elif block_type == "image":
            block["caption"] = _join_caption(item.get("image_caption"))
            block["asset_url"] = presign(item.get("asset_key"))
        elif block_type == "equation":
            block["latex"] = item.get("latex") or item.get("text")
            block["text"] = item.get("text")
        else:
            block["text"] = item.get("text") or ""
        blocks.append(block)
    return blocks


def blocks_to_markdown(blocks: list[dict]) -> str:
    """Reading-order markdown: paragraphs, tables (caption in italics above), images, equations."""
    parts = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "table":
            if not block.get("table_body"):
                continue
            if block.get("caption"):
                parts.append(f"*{block['caption']}*")
            parts.append(block["table_body"])
        elif block_type == "image":
            if block.get("asset_url"):
                parts.append(f"![{block.get('caption') or ''}]({block['asset_url']})")
        elif block_type == "equation":
            latex = block.get("latex")
            if latex:
                parts.append(f"$${latex}$$")
        else:
            text = (block.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _str_or_none(value):
    return None if value is None else str(value)


@router.get("/documents/{document_id}")
async def document_status(document_id: str, state_repo=Depends(get_state_repo)):
    state = await asyncio.to_thread(state_repo.get, document_id)
    if not state:
        raise HTTPException(status_code=404, detail="Document not found")
    updated_at = state.get("updated_at")
    return {
        "document_id": document_id,
        "status": state.get("status"),
        "stage": state.get("stage"),
        "source": state.get("source"),
        "counts": state.get("counts") or {},
        "page_count": state.get("page_count"),
        "parser": state.get("parser"),
        "document_type": state.get("document_type"),
        "title": state.get("title"),
        "organization_unit_id": _str_or_none(state.get("organization_unit_id")),
        "owner_id": _str_or_none(state.get("owner_id")),
        "error": state.get("error") or None,
        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
    }


@router.get("/documents/{document_id}/content")
async def document_content(document_id: str, asset_store=Depends(get_asset_store)):
    content_list = await asyncio.to_thread(asset_store.load_content_list, document_id)
    if content_list is None:
        raise HTTPException(status_code=404, detail="Content not found")
    blocks = await asyncio.to_thread(build_content_blocks, content_list, asset_store.presign)
    return {
        "document_id": document_id,
        "markdown": blocks_to_markdown(blocks),
        "blocks": blocks,
        "tables": [block for block in blocks if block["type"] == "table"],
    }


@router.post("/reindex")
async def reindex(
    payload: dict | None = None,
    queue=Depends(get_queue),
    docs_repo=Depends(get_docs_repo),
):
    payload = payload or {}
    document_ids = payload.get("document_ids") or []
    if payload.get("all"):
        documents = list(await asyncio.to_thread(lambda: list(docs_repo.iter_all())))
    else:
        documents = []
        for doc_id in document_ids:
            doc = await asyncio.to_thread(docs_repo.find_by_id, doc_id)
            if doc:
                documents.append(doc)
    published = 0
    for doc in documents:
        event = RagEvent(
            event=EVENT_UPDATED,
            document_id=str(doc["_id"]),
            content_hash=doc.get("content_hash"),
            document_type=doc.get("document_type"),
            org_id=str(doc["organization_unit_id"]) if doc.get("organization_unit_id") else None,
        )
        await queue.publish(event)
        published += 1
    return {"published": published}
