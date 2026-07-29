from __future__ import annotations

from pathlib import Path

from x2md.converters.base import BaseConverter
from x2md.utils import clean_markdown, split_contact_info


class TextConverter(BaseConverter):
    extensions = [
        ".txt", ".md", ".csv", ".json", ".xml",
        ".yaml", ".yml", ".log", ".ini", ".cfg",
        ".conf", ".toml",
    ]

    def convert(self, file_path: Path, **kwargs) -> str:
        encoding: str = kwargs.get("encoding", "utf-8")

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()

        content = split_contact_info(content)
        return clean_markdown(content)
