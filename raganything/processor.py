"""Document parsing for RAGAnything (vector pipeline thuần - chỉ parse).

Parse-only mixin: không ghi LightRAG storage, không entity/graph. Parse result
+ content_list được VectorPipeline lưu ở MinIO; parse cache của LightRAG bỏ.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from raganything.parser import MineruParser, MineruExecutionError, get_parser

logger = logging.getLogger(__name__)

AUDIO_FILE_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".opus"}
)
VIDEO_FILE_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv", ".wmv", ".m4v"}
)


def compute_mdhash_id(content, prefix: str = "id-") -> str:
    """Stable id từ nội dung (thay compute_mdhash_id của LightRAG)."""
    import hashlib
    import json as _json

    if isinstance(content, (dict, list)):
        content = _json.dumps(content, ensure_ascii=False, sort_keys=True)
    return f"{prefix}{hashlib.md5(str(content).encode()).hexdigest()}"


class ProcessorMixin:
    """Parsing-only mixin (không LightRAG)."""

    @staticmethod
    def _file_content_fingerprint(file_path: Path) -> str:
        """Return a streaming SHA-256 fingerprint for file identity."""
        import hashlib

        digest = hashlib.sha256()
        with file_path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """Generate doc_id based on document content."""
        content_hash_data = []
        for item in content_list:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    content_hash_data.append(str(item))
        content_signature = "\n".join(content_hash_data)
        return compute_mdhash_id(content_signature, prefix="doc-")

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse document (không cache, không ghi storage).

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to log content statistics
            **kwargs: Additional parameters for parser (e.g., lang, device,
                backend, source, timeout, env)

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        callback_file = str(file_path)
        callback_manager = getattr(self, "callback_manager", None)
        parse_start_time = time.time()
        if callback_manager is not None:
            callback_manager.dispatch(
                "on_parse_start",
                file_path=callback_file,
                parser=self.config.parser,
            )

        ext = file_path.suffix.lower()
        try:
            doc_parser = getattr(self, "doc_parser", None)
            if doc_parser is None:
                doc_parser = get_parser(self.config.parser)
                self.doc_parser = doc_parser

            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in AUDIO_FILE_EXTENSIONS or ext in VIDEO_FILE_EXTENSIONS:
                # Media files need no document parser: they become a single
                # audio/video content item handled by the modal processors.
                kind = "audio" if ext in AUDIO_FILE_EXTENSIONS else "video"
                self.logger.info(
                    f"Detected {kind} file; building {kind} content item "
                    "(no document parser involved)..."
                )
                content_list = [
                    {
                        "type": kind,
                        f"{kind}_path": str(file_path.absolute()),
                        f"{kind}_caption": [],
                        "page_idx": 0,
                        "media_sha256": self._file_content_fingerprint(file_path),
                    }
                ]
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                try:
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                except NotImplementedError:
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = await asyncio.to_thread(
                        MineruParser().parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            else:
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )
        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise

        self.logger.info(
            f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        )
        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        doc_id = self._generate_content_based_doc_id(content_list)

        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1
            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        if callback_manager is not None:
            duration = time.time() - parse_start_time
            callback_manager.dispatch(
                "on_parse_complete",
                file_path=callback_file,
                content_blocks=len(content_list),
                doc_id=doc_id,
                duration_seconds=duration,
            )

        return content_list, doc_id
