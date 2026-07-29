from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar


class BaseConverter(ABC):
    extensions: ClassVar[list[str]] = []

    @abstractmethod
    def convert(self, file_path: Path, **kwargs) -> str:
        ...

    def convert_with_elements(self, file_path: Path, **kwargs) -> dict[str, Any]:
        """默认实现：调一次 convert，返回 text + 空 element 数据。

        子类（特别是 PdfConverter）应重写以在单次解析中同时返回 element 数据。
        """
        text = self.convert(file_path, **kwargs)
        return {
            "text": text,
            "source": file_path.name,
            "element_indexes": [],
            "element_pages": [],
            "element_bboxes": [],
            "element_types": [],
        }

    @classmethod
    def supports(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in cls.extensions
