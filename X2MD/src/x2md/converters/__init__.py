from __future__ import annotations

from pathlib import Path

from x2md.converters.base import BaseConverter
from x2md.converters.docx import DocxConverter
from x2md.converters.docx_fast import DocxFastConverter
from x2md.converters.html import HtmlConverter
from x2md.converters.image import ImageConverter
from x2md.converters.legacy_office import DocConverter, XlsConverter, PptConverter
from x2md.converters.pdf import PdfConverter
from x2md.converters.pptx import PptxConverter
from x2md.converters.text import TextConverter
from x2md.converters.xlsx import XlsxConverter

_CONVERTERS: list[BaseConverter] = [
    PdfConverter(),
    DocxConverter(),       # .docx → 完整版 (含图片提取和 LLM), 默认选择
    DocxFastConverter(),   # .docx → 快速版 (跳过图片和 LLM, 仅通过 converter= 参数选用)
    XlsxConverter(),
    PptxConverter(),
    ImageConverter(),
    HtmlConverter(),
    TextConverter(),
    DocConverter(),  # 老版本Word .doc
    XlsConverter(),  # 老版本Excel .xls
    PptConverter(),  # 老版本PowerPoint .ppt
]

_EXTENSION_MAP: dict[str, BaseConverter] = {}
_CONVERTER_BY_NAME: dict[str, BaseConverter] = {}

# ★ 扩展名映射: 同扩展名多个 converter 时, 第一个注册的作为默认
# (DocxConverter 先注册 → .docx 默认走完整版)
_seen_extensions: set[str] = set()
for _conv in _CONVERTERS:
    _CONVERTER_BY_NAME[_conv.__class__.__name__] = _conv
    for _ext in _conv.extensions:
        if _ext not in _seen_extensions:
            _EXTENSION_MAP[_ext] = _conv
            _seen_extensions.add(_ext)


def get_converter(file_path: str | Path, converter: str | None = None) -> BaseConverter | None:
    """获取文件对应的转换器.

    Args:
        file_path: 文件路径, 用于推断扩展名
        converter: 可选转换器类名 (如 "DocxFastConverter"), 用于选择同扩展名的不同实现
    """
    if converter:
        conv = _CONVERTER_BY_NAME.get(converter)
        if conv is not None:
            return conv
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def get_converter_by_name(name: str) -> BaseConverter | None:
    """按类名获取指定转换器 (用于选择同扩展名的不同实现)."""
    return _CONVERTER_BY_NAME.get(name)


def convert(file_path: str | Path, **kwargs) -> str:
    converter_name = kwargs.pop("converter", None)
    converter_obj = get_converter(file_path, converter=converter_name)
    if converter_obj is None:
        ext = Path(file_path).suffix
        raise ValueError(f"Unsupported file format: {ext}")
    return converter_obj.convert(Path(file_path), **kwargs)


def supported_extensions() -> list[str]:
    return sorted(_EXTENSION_MAP.keys())
