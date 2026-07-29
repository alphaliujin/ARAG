from __future__ import annotations

from pathlib import Path

from x2md.converters.base import BaseConverter
from x2md.utils import clean_markdown

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class ImageConverter(BaseConverter):
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    def convert(self, file_path: Path, **kwargs) -> str:
        lang: str = kwargs.get("ocr_lang", "eng")
        # enable_ocr=False (来自 CLI --no-ocr): 直接返回占位符, 不运行 tesseract
        if not kwargs.get("enable_ocr", True):
            return self._placeholder(file_path)

        if not _OCR_AVAILABLE:
            return self._placeholder(file_path)

        try:
            # 用 context manager 关闭 PIL 句柄, 避免循环中 fd 泄漏
            with Image.open(file_path) as img:
                text = pytesseract.image_to_string(img, lang=lang)
            if not text.strip():
                return self._placeholder(file_path)
            return clean_markdown(text)
        except (IOError, OSError) as e:
            # 图片文件读取错误
            return self._placeholder(file_path)
        except ImportError:
            # pytesseract 未安装
            return self._placeholder(file_path)
        except Exception:
            # 其他错误（如 OCR 失败）
            return self._placeholder(file_path)

    @staticmethod
    def _placeholder(file_path: Path) -> str:
        return f"[Image: {file_path.name}]\n"
