"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

from __future__ import annotations

import base64
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_caption_list(value: Any) -> List[str]:
    """Return captions and footnotes as a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_table_body(item: Dict[str, Any]) -> Any:
    """Read table content across common content-list alias fields."""
    if item.get("table_body") not in (None, ""):
        return item.get("table_body")
    if item.get("table_data") not in (None, ""):
        return item.get("table_data")
    return item.get("text", "")


def format_table_body(table_body: Any) -> str:
    """Serialize table content for prompts and chunks without dropping aliases.

    Strings are passed through unchanged. List-of-lists (the common
    ``table_data`` shape from non-MinerU parsers) are rendered as a simple
    Markdown table so the LLM sees structured rows instead of a Python repr.
    Other shapes fall back to a newline-joined string of ``str(...)`` items.
    """
    if isinstance(table_body, str):
        return table_body
    if isinstance(table_body, list):
        if not table_body:
            return ""
        if all(isinstance(row, (list, tuple)) for row in table_body):
            rendered_rows = [
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in table_body
            ]
            if len(rendered_rows) >= 1:
                column_count = max(len(row) for row in table_body)
                separator = "| " + " | ".join(["---"] * column_count) + " |"
                rendered_rows.insert(1, separator)
            return "\n".join(rendered_rows)
        return "\n".join(str(row) for row in table_body)
    return str(table_body)


def format_asset_line(item: Dict[str, Any], path_prefix: str, path: Any) -> str:
    """Render the asset reference line of an image/table chunk.

    When the item carries an ``asset_key`` (object-store key set by the
    caller) it is printed instead of the temporary local path.
    """
    asset_key = item.get("asset_key") if isinstance(item, dict) else None
    if asset_key:
        return f"Asset: {asset_key}"
    return f"{path_prefix}{path}"


def build_modal_chunk_metadata(
    content_type: str, item: Any, page_idx: Any = None
) -> Dict[str, Any]:
    """Structured metadata stored next to a multimodal chunk in ``text_chunks``.

    Missing values are ``None`` (never invented). ``table_body`` is the raw
    table content before prompt formatting; ``caption`` joins the image/table
    captions of the item.
    """
    if not isinstance(item, dict):
        item = {}
    if page_idx is None:
        page_idx = item.get("page_idx", 0)

    captions: List[str] = []
    for key in ("image_caption", "img_caption", "table_caption"):
        for cap in normalize_caption_list(item.get(key)):
            if cap not in captions:
                captions.append(cap)

    table_body = None
    if content_type == "table":
        raw_body = get_table_body(item)
        if raw_body not in (None, ""):
            table_body = (
                raw_body if isinstance(raw_body, str) else format_table_body(raw_body)
            )

    return {
        "is_multimodal": True,
        "original_type": content_type,
        "page_idx": page_idx,
        "asset_key": item.get("asset_key") or None,
        "table_body": table_body,
        "caption": ", ".join(captions) if captions else None,
    }


def get_equation_text_and_format(item: Dict[str, Any]) -> Tuple[str, str]:
    """Read equation content while preserving LaTeX aliases from content lists.

    Field priority follows MinerU first (``text`` + ``text_format``), then
    falls back to ``latex`` and ``equation`` aliases used by other parsers.
    The textual description is intentionally NOT concatenated into the
    equation body: the ``equation_chunk`` template has a separate
    ``enhanced_caption`` slot for that.
    """
    text = str(item.get("text", "") or "").strip()
    latex = str(item.get("latex", "") or "").strip()
    equation = str(item.get("equation", "") or "").strip()
    equation_format = str(item.get("text_format", "") or "").strip()

    if text:
        equation_text = text
    elif latex:
        equation_text = latex
        if not equation_format:
            equation_format = "latex"
    elif equation:
        equation_text = equation
    else:
        equation_text = ""

    return equation_text, equation_format


def extract_section_path_from_content_list(
    content_list: List[Dict[str, Any]], current_index: int
) -> str:
    """Build a hierarchical section path from preceding heading blocks.

    MinerU content lists keep document order, and heading blocks are exposed as
    text items with a positive ``text_level``.  For a given item index, we walk
    the preceding items and keep the latest heading at each level to reconstruct
    a stable chapter/section path such as ``Introduction > Method > Ablation``.
    """
    if not content_list or current_index is None:
        return ""

    try:
        limit = max(0, int(current_index))
    except (TypeError, ValueError):
        return ""

    heading_chain: List[Tuple[int, str]] = []

    for item in content_list[:limit]:
        if not isinstance(item, dict):
            continue

        if item.get("type", "text") != "text":
            continue

        text = str(item.get("text", "") or "").strip()
        if not text:
            continue

        try:
            level = int(item.get("text_level", 0) or 0)
        except (TypeError, ValueError):
            continue

        if level <= 0:
            continue

        while heading_chain and heading_chain[-1][0] >= level:
            heading_chain.pop()
        heading_chain.append((level, text))

    return " > ".join(text for _, text in heading_chain)


def extract_neighbor_text_from_content_list(
    content_list: List[Dict[str, Any]], current_index: int, window_size: int = 3
) -> str:
    """Collect nearby text blocks around an item index from MinerU content list."""
    if not content_list or current_index is None:
        return ""

    try:
        idx = int(current_index)
    except (TypeError, ValueError):
        return ""

    if idx < 0 or idx >= len(content_list):
        return ""

    start_idx = max(0, idx - window_size)
    end_idx = min(len(content_list), idx + window_size + 1)

    parts: List[str] = []
    for pos in range(start_idx, end_idx):
        if pos == idx:
            continue
        item = content_list[pos]
        if not isinstance(item, dict):
            continue
        if item.get("type", "text") != "text":
            continue

        text = str(item.get("text", "") or "").strip()
        if text:
            parts.append(text)

    return " ".join(parts)


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for index, item in enumerate(content_list):
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = str(item.get("text", "") or "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_item = dict(item)
            multimodal_item.setdefault("_content_list_index", index)
            if content_type == "image":
                multimodal_item.setdefault(
                    "_section_path",
                    extract_section_path_from_content_list(content_list, index),
                )
                multimodal_item.setdefault(
                    "_neighbor_text",
                    extract_neighbor_text_from_content_list(content_list, index),
                )
            multimodal_items.append(multimodal_item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(f"  - Text content length: {len(text_content)} characters")
    logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists and is not a symlink (for security)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        if path.is_symlink():
            logger.warning(f"Blocking symlink for security: {image_path}")
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    elif content_type == "audio":
        # Fall back to generic when audio processing is disabled / deps missing
        return modal_processors.get("audio") or modal_processors.get("generic")
    elif content_type == "video":
        # Fall back to generic when video processing is disabled / deps missing
        return modal_processors.get("video") or modal_processors.get("generic")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "audio": [
            "Speech-to-text transcription",
            "Timestamped segment extraction",
            "Audio entity extraction",
        ],
        "video": [
            "Scene detection and keyframe analysis",
            "Visual + audio dual-channel understanding",
            "Video entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
