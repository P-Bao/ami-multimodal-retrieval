"""Sinh thử caption cho MỘT vài ảnh mà KHÔNG đụng index (dùng để kiểm chứng
trước khi reindex `--from-stage describe` cho cả doc).

Đi đúng đường caption của pipeline (cùng `ImageModalProcessor`, cùng prompt,
cùng ngữ cảnh `content_list` và ngôn ngữ `CAPTION_LLM_LANGUAGE`), nhưng dừng ở
bước sinh mô tả: không ghi `descriptions.json`, không chunk, không embed.

Usage:
    # xem ảnh nào trong doc (không gọi LLM)
    python scripts/recaption_image.py --doc <doc_id> --list

    # caption thử 1 trang (mỗi ảnh 1 call Gemini/VLM)
    python scripts/recaption_image.py --doc <doc_id> --page 13

    # xem prompt sẽ gửi, KHÔNG gọi API
    python scripts/recaption_image.py --doc <doc_id> --page 13 --prompt-only

Sau khi ưng, chạy thật cho cả doc:
    ami-rag reindex --doc <doc_id> --from-stage describe --yes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ami_rag.core.vector_pipeline import VectorPipeline, _format_modal_chunk
from ami_rag.settings import get_settings, resolve_image_describe_mode
from ami_rag.textnorm import with_title

PREVIEW = 600


async def _materialize(asset_store, item: str, temp_dir: str) -> str:
    data = await asyncio.to_thread(asset_store.get_bytes, item)
    if not data:
        raise RuntimeError(f"không đọc được ảnh từ MinIO: {item}")
    name = hashlib.sha1(item.encode()).hexdigest()[:12] + VectorPipeline._image_ext(data, item)
    path = Path(temp_dir) / name
    await asyncio.to_thread(path.write_bytes, data)
    return str(path)


async def run(args: argparse.Namespace) -> None:
    from ami_rag.core.factory import (
        _build_caption_func,
        _build_modal_processors,
        _build_vision_func,
        get_asset_store,
    )
    from ami_rag.storage.mongo_docs import MongoDocumentRepo

    settings = get_settings()
    doc_id = args.doc
    asset_store = get_asset_store()

    content_list = await asyncio.to_thread(asset_store.load_content_list, doc_id)
    if not content_list:
        print(f"doc {doc_id}: không có content_list.json trong MinIO (cần parse lại)")
        return

    images = [(i, it) for i, it in enumerate(content_list) if it.get("type") == "image"]
    if args.list or not images:
        print(f"doc {doc_id}: {len(content_list)} items, {len(images)} ảnh")
        for i, it in images:
            page = it.get("page_idx")
            print(f"  idx={i:<5} page={page:<5} asset={it.get('asset_key')}")
        return

    if args.page is not None:
        images = [(i, it) for i, it in images if it.get("page_idx") == args.page]
        if not images:
            print(f"doc {doc_id}: không có ảnh nào ở trang {args.page}")
            return
    if args.index is not None:
        images = [(i, it) for i, it in images if i == args.index]
        if not images:
            print(f"doc {doc_id}: không có ảnh nào ở index {args.index}")
            return

    mode = resolve_image_describe_mode(settings)
    print(f"doc {doc_id} | chế độ caption: {mode} | "
          f"ngôn ngữ: {settings.CAPTION_LLM_LANGUAGE or '(prompt gốc)'} | "
          f"model: {settings.CAPTION_LLM_MODEL} | ảnh: {len(images)}")
    if mode == "none":
        print("IMAGE_DESCRIBE_MODE=none -> không có caption func; hãy đặt CAPTION_LLM_API_KEY "
              "hoặc QWEN_VLM_MODEL")
        return

    saved: list[str] = []
    caption_func = _build_caption_func(settings) or _build_vision_func(settings)
    processors = _build_modal_processors(settings, caption_func=caption_func)
    proc = processors.get("image") or processors["generic"]
    set_source = getattr(proc, "set_content_source", None)
    if callable(set_source):
        set_source(content_list, "minerU")

    doc = await asyncio.to_thread(
        MongoDocumentRepo(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.ORG_DB,
            collection_name=settings.DOC_COLLECTION,
        ).find_by_id,
        doc_id,
    )
    title = (doc or {}).get("title") or ""

    language = (settings.CAPTION_LLM_LANGUAGE or "").strip()
    with tempfile.TemporaryDirectory(prefix="ami-recaption-") as tmp:
        for idx, item in images:
            if not item.get("asset_key"):
                print(f"  idx={idx} page={item.get('page_idx')}: bỏ qua (không có asset_key)")
                continue
            item = dict(item)
            item["img_path"] = await _materialize(asset_store, item["asset_key"], tmp)
            info = {"page_idx": item.get("page_idx", 0), "index": idx, "type": "image"}
            if language:
                info["caption_language"] = language
            if title:
                info["doc_title"] = title

            print("\n" + "=" * 78)
            print(f"idx={idx} page={item.get('page_idx')} asset={item['asset_key']}")
            print(f"doc title: {title!r}")

            if args.prompt_only:
                from raganything.modalprocessors import PROMPTS

                ctx = proc._get_context_for_item(info) if hasattr(proc, "_get_context_for_item") else ""
                base = (
                    PROMPTS["vision_prompt_with_context"].format(
                        context=ctx or "(none)",
                        section_path="None",
                        entity_name="unique descriptive name for this image",
                        image_path=item["img_path"],
                        captions=item.get("image_caption") or "None",
                        footnotes=item.get("image_footnote") or "None",
                    )
                    if ctx
                    else PROMPTS["vision_prompt"].format(
                        section_path="None",
                        entity_name="unique descriptive name for this image",
                        image_path=item["img_path"],
                        captions=item.get("image_caption") or "None",
                        footnotes=item.get("image_footnote") or "None",
                    )
                )
                extra = ""
                if language:
                    extra += PROMPTS["vision_prompt_language_note"].format(language=language)
                if title:
                    extra += PROMPTS["vision_prompt_doc_title_note"].format(doc_title=title)
                print("--- prompt (chưa gọi API) ---")
                print(base + extra)
                continue

            sections = await proc.generate_chunk_sections(
                modal_content=item, content_type="image", item_info=info, entity_name=None
            )
            desc = str(sections[0]["description"]) if sections else ""
            chunk_text = with_title(
                _format_modal_chunk("image", item, desc), title
            )
            print(f"--- description ({len(desc)} ký tự) ---")
            print(desc[:PREVIEW] + ("…" if len(desc) > PREVIEW else ""))
            print(f"--- chunk content sẽ ghi vào Qdrant ({len(chunk_text)} ký tự) ---")
            print(chunk_text[: PREVIEW + 100] + ("…" if len(chunk_text) > PREVIEW + 100 else ""))
            saved.append(chunk_text)

    args._saved = "\n\n---\n\n".join(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh thử caption ảnh, không đụng index")
    parser.add_argument("--doc", required=True, help="doc id (Mongo ObjectId)")
    parser.add_argument("--page", type=int, help="chỉ ảnh ở trang này")
    parser.add_argument("--index", type=int, help="chỉ ảnh ở index content_list này")
    parser.add_argument("--list", action="store_true", help="liệt kê ảnh của doc, không gọi LLM")
    parser.add_argument("--prompt-only", action="store_true", help="in prompt, không gọi API")
    parser.add_argument("--out", help="ghi description sinh được ra file (để so sánh/đo lại)")
    args = parser.parse_args()
    if args.page is None and args.index is None and not args.list and not args.prompt_only:
        parser.error("cần --list, --page, --index hoặc --prompt-only")
    asyncio.run(run(args))
    if args.out and getattr(args, "_saved", None):
        Path(args.out).write_text(args._saved, encoding="utf-8")
        print(f"\nđã ghi description vào {args.out}")


if __name__ == "__main__":
    main()