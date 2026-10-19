"""RAGAnything - parser + multimodal processors (vector pipeline thuần).

Không LightRAG: chỉ giữ parser (get_parser/parse_document), modal processors
(describe stage) và query helpers (aquery/aquery_data trên embedder +
vector_store). Ingest/index/query thật của service chạy qua VectorPipeline
(ami_rag.core.vector_pipeline).
"""

import atexit
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from raganything.callbacks import CallbackManager
from raganything.config import RAGAnythingConfig
from raganything.parser import SUPPORTED_PARSERS, MineruParser, get_parser
from raganything.processor import ProcessorMixin
from raganything.query import QueryMixin
from raganything.utils import get_processor_supports

logger = logging.getLogger(__name__)

# Specialized processors
from raganything.modalprocessors import (
    ContextConfig,
    ContextExtractor,
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)
from raganything.modalprocessors_audio import (
    AudioModalProcessor,
    audio_deps_available,
)
from raganything.modalprocessors_video import (
    VideoModalProcessor,
    video_deps_available,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin):
    """Parser + multimodal processors container (không LightRAG)."""

    # Optional deps cho query (embed + search)
    embedder: Any | None = field(default=None)
    """Embedder (RemoteEmbedder) cho aquery_data."""
    vector_store: Any | None = field(default=None)
    """VectorStore cho aquery_data."""
    collection: str = field(default="")
    """Qdrant collection cho aquery_data."""

    llm_model_func: Callable | None = field(default=None)
    """LLM model function for text analysis."""
    vision_model_func: Callable | None = field(default=None)
    """Vision model function for image analysis."""

    config: RAGAnythingConfig | None = field(default=None)
    """Configuration object, if None will create with environment variables."""

    # Internal State
    modal_processors: dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: ContextExtractor | None = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    callback_manager: CallbackManager = field(
        default_factory=CallbackManager, init=False, repr=False
    )
    """Processing callbacks manager (optional hooks for observability and metrics)."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """Flag to track if parser installation has been checked."""

    def __post_init__(self):
        """Post-initialization setup."""
        if self.config is None:
            self.config = RAGAnythingConfig()

        self.working_dir = self.config.working_dir
        self.logger = logger
        self.doc_parser = get_parser(self.config.parser)
        atexit.register(self.close)

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

    def close(self):
        """No resources to clean up (không LightRAG storages)."""

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from RAGAnything config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor (không cần LightRAG tokenizer)."""
        return ContextExtractor(
            config=self._create_context_config(), tokenizer=None
        )

    def _initialize_processors(self):
        """Initialize multimodal processors (lightrag=None tolerant)."""
        self.context_extractor = self._create_context_extractor()
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=None,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=None,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=None,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_audio_processing:
            if not audio_deps_available():
                self.logger.warning(
                    "enable_audio_processing=True but audio dependencies are missing. "
                    "Audio content will fall back to the generic processor."
                )
            else:
                self.modal_processors["audio"] = AudioModalProcessor(
                    lightrag=None,
                    modal_caption_func=self.llm_model_func,
                    context_extractor=self.context_extractor,
                )

        if self.config.enable_video_processing:
            if not video_deps_available():
                self.logger.warning(
                    "enable_video_processing=True but video dependencies are missing. "
                    "Video content will fall back to the generic processor."
                )
            else:
                self.modal_processors["video"] = VideoModalProcessor(
                    lightrag=None,
                    modal_caption_func=self.vision_model_func or self.llm_model_func,
                    context_extractor=self.context_extractor,
                    # Reuse the audio processor's whisper model when audio is enabled
                    audio_processor=self.modal_processors.get("audio"),
                )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=None,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    def check_parser_installation(self) -> bool:
        """Check if the configured parser is properly installed"""
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> dict[str, Any]:
        """Get current configuration information"""
        return {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
        }

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors"""
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration"""
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        if self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                for processor in self.modal_processors.values():
                    processor.context_extractor = self.context_extractor
                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "parser_installation": {
                parser_name: get_parser(parser_name).check_installation()
                for parser_name in SUPPORTED_PARSERS
            },
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedder
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}
            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
