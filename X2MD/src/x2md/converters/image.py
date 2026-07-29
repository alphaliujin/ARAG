from __future__ import annotations

from pathlib import Path

from x2md.converters.base import BaseConverter
from x2md.utils import clean_markdown

# tesseract 回退引擎 (服务器未装 tesseract/pytesseract, 仅本地或历史兼容用)
try:
    import pytesseract  # type: ignore
    _TESS_AVAILABLE = True
except ImportError:
    _TESS_AVAILABLE = False


# ---------------------------------------------------------------------------
# PaddleOCR (PP-OCRv6) 主引擎
# ---------------------------------------------------------------------------
# 实例化 + 模型加载昂贵 (CPU 上数秒), 进程内用单例复用.
# X2MD 由 preprocess.py 逐文件子进程调用, 每个图片文件 = 1 进程 = 1 次实例化.
_PADDLE_OCR = None
_PADDLE_INIT_ERR: Exception | None = None


def _tess_lang_to_paddle(lang: str) -> str:
    """tesseract 语言码 -> PaddleOCR lang."""
    if not lang:
        return "en"
    l = lang.lower()
    if "chi" in l or l.startswith("ch"):
        return "ch"
    if "jpn" in l:
        return "japan"
    if "kor" in l:
        return "korean"
    return "en"


def _get_paddle_ocr(model_base: str, lang: str):
    """惰性创建并缓存 PaddleOCR (PP-OCRv6) 实例. 失败记错误, 返回 None."""
    global _PADDLE_OCR, _PADDLE_INIT_ERR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    if _PADDLE_INIT_ERR is not None:
        return None
    try:
        import paddle.inference as _pi
        # aarch64 (DGX Spark) 上 paddlepaddle 的 PIR (new IR) 推理路径在
        # AnalysisPredictor::PreparePirProgram -> SaveOrLoadPirParameters 处 segfault.
        # PaddleX runner.py 默认 config.enable_new_ir(True); 这里强制 False,
        # 走老的 PrepareProgram 路径规避崩溃 (PP-OCRv6 medium 的 det+rec 均验证可用).
        _orig_enr = _pi.Config.enable_new_ir
        _pi.Config.enable_new_ir = lambda self, flag=True: _orig_enr(self, False)

        from paddleocr import PaddleOCR
        _PADDLE_OCR = PaddleOCR(
            text_detection_model_name=f"{model_base}_det",
            text_recognition_model_name=f"{model_base}_rec",
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as e:
        _PADDLE_INIT_ERR = e
    return _PADDLE_OCR


def _paddle_rec_texts(ocr, file_path: Path) -> str:
    """跑一次 PP-OCR, 拼接所有识别文本行."""
    result = ocr.predict(str(file_path))
    out: list[str] = []
    for r in (result if isinstance(result, list) else [result]):
        texts = None
        try:
            texts = r.get("rec_texts") if hasattr(r, "get") else r["rec_texts"]
        except Exception:
            try:
                texts = r.json["res"]["rec_texts"]
            except Exception:
                texts = None
        if texts:
            out.extend(t for t in texts if t)
    return "\n".join(out)


class ImageConverter(BaseConverter):
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    def convert(self, file_path: Path, **kwargs) -> str:
        # --no-ocr: 直接占位符, 不跑任何 OCR
        if not kwargs.get("enable_ocr", True):
            return self._placeholder(file_path)

        engine = str(kwargs.get("ocr_engine") or "paddleocr").lower()
        tess_lang = kwargs.get("ocr_lang", "eng") or "eng"
        model_base = kwargs.get("ocr_paddle_model") or "PP-OCRv6_medium"

        text = ""
        if engine == "tesseract":
            text = self._tesseract(file_path, tess_lang)
        else:
            # 默认 paddleocr (PP-OCRv6); 不可用时回退 tesseract (若装了)
            ocr = _get_paddle_ocr(model_base, _tess_lang_to_paddle(tess_lang))
            if ocr is not None:
                try:
                    text = _paddle_rec_texts(ocr, file_path)
                except Exception:
                    text = ""
            if not text and _TESS_AVAILABLE:
                text = self._tesseract(file_path, tess_lang)

        if not text.strip():
            return self._placeholder(file_path)
        return clean_markdown(text)

    @staticmethod
    def _tesseract(file_path: Path, lang: str) -> str:
        if not _TESS_AVAILABLE:
            return ""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return pytesseract.image_to_string(img, lang=lang)
        except Exception:
            return ""

    @staticmethod
    def _placeholder(file_path: Path) -> str:
        return f"[Image: {file_path.name}]\n"
