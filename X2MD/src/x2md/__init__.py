"""X2MD - Convert various file formats to Markdown for RAG chunking and ingestion."""

from __future__ import annotations

from x2md.config import Config, load_config
from x2md.converters import convert, get_converter, get_converter_by_name, supported_extensions
from x2md.converters.base import BaseConverter
from x2md.converters.docx_fast import DocxFastConverter
from x2md.vit import VitExtractor, process_images
from x2md.text_cleaner import TextCleaner, clean_text, process_text
from x2md.chunk import (
    Chunk, ChunkMetadata, ChunkType, ChunkSplitter,
    SmallerChunksStrategy, ChunkMaxLimitError,
    IntervalSearch, aggregate_chunk_text, split_chunk_text,
)

__all__ = [
    "convert", "get_converter", "get_converter_by_name", "supported_extensions",
    "BaseConverter", "DocxFastConverter", "Config", "load_config",
    "VitExtractor", "process_images",
    "TextCleaner", "clean_text", "process_text",
    "Chunk", "ChunkMetadata", "ChunkType", "ChunkSplitter",
    "SmallerChunksStrategy", "ChunkMaxLimitError",
    "IntervalSearch", "aggregate_chunk_text", "split_chunk_text",
]
__version__ = "0.2.0"
