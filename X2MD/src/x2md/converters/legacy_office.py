"""老版本Office文件转换器 (.doc, .xls)"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from x2md.converters.base import BaseConverter
from x2md.utils import clean_markdown

logger = logging.getLogger(__name__)


def _run_soffice(
    src: Path, target_format: str, tmpdir: str
) -> tuple[Path | None, str]:
    """Run `soffice --headless --convert-to <fmt>` and return (output_path, stderr_text).

    并发 soffice 调用会争抢 ~/.config/libreoffice profile,导致死锁或随机失败。
    通过 -env:UserInstallation=file://<tmpdir>/uno_profile 给每次调用独立 profile。

    返回 (None, stderr) 表示失败;调用方据 stderr 给用户清晰诊断。
    """
    profile = Path(tmpdir) / "uno_profile"
    cmd = [
        "soffice",
        "-env:UserInstallation=file://" + str(profile),
        "--headless",
        "--convert-to", target_format,
        "--outdir", tmpdir,
        str(src),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120, env=None)
    except subprocess.TimeoutExpired:
        return None, "soffice timed out after 120s"
    except FileNotFoundError:
        return None, "soffice (LibreOffice) not found in PATH"

    stderr_text = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    if result.returncode != 0:
        return None, f"soffice exited {result.returncode}: {stderr_text[:500]}"

    expected = Path(tmpdir) / src.with_suffix("." + target_format).name
    if not expected.exists():
        # 即使 returncode==0,LibreOffice 偶尔会因 corrupt / encrypted 输入静默不产物
        return None, f"soffice returned 0 but no output file produced; stderr: {stderr_text[:300]}"
    return expected, stderr_text


class DocConverter(BaseConverter):
    """老版本Word (.doc) 转换器

    使用antiword或LibreOffice转换为文本
    """
    extensions = [".doc"]

    def convert(self, file_path: Path, **kwargs) -> str:
        # 尝试使用antiword
        # text=True 会按 locale 解码 antiword 输出, GBK→UTF-8 locale 下会 UnicodeDecodeError;
        # 改为字节流读取后用 errors="replace" 解码, 保证不会因编码崩溃。
        antiword_error: str | None = None
        try:
            result = subprocess.run(
                ["antiword", str(file_path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                stdout_text = result.stdout.decode("utf-8", errors="replace")
                if stdout_text.strip():
                    return clean_markdown(stdout_text)
                antiword_error = "antiword returned empty output"
            else:
                stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
                antiword_error = f"antiword exit {result.returncode}: {stderr[:200]}"
                # 加密文件 antiword 常报特定关键字
                if "encrypted" in stderr.lower() or "password" in stderr.lower():
                    raise RuntimeError(
                        f"文件 {file_path.name} 似乎已加密或受密码保护,无法转换"
                    )
        except subprocess.TimeoutExpired:
            antiword_error = "antiword timed out after 30s"
        except FileNotFoundError:
            antiword_error = "antiword not installed"

        if antiword_error:
            logger.info(f"antiword failed for {file_path.name}: {antiword_error}; falling back to soffice")

        # 备用1: macOS 自带 textutil (无需额外安装)
        textutil_err: str | None = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = Path(tmpdir) / "out.txt"
                result = subprocess.run(
                    ["textutil", "-convert", "txt", "-output", str(out_file), str(file_path)],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode == 0 and out_file.exists():
                    text = out_file.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        return clean_markdown(text)
                    textutil_err = "textutil returned empty output"
                else:
                    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
                    textutil_err = f"textutil exit {result.returncode}: {stderr[:200]}"
        except subprocess.TimeoutExpired:
            textutil_err = "textutil timed out after 60s"
        except FileNotFoundError:
            textutil_err = "textutil not available (non-macOS)"

        if textutil_err:
            logger.info(f"textutil failed for {file_path.name}: {textutil_err}; falling back to soffice")

        # 备用2：使用LibreOffice转换为docx再处理
        with tempfile.TemporaryDirectory() as tmpdir:
            docx_path, soffice_err = _run_soffice(file_path, "docx", tmpdir)
            if docx_path:
                from x2md.converters.docx import DocxConverter
                return DocxConverter().convert(docx_path, **kwargs)
            logger.warning(f"soffice failed for {file_path.name}: {soffice_err}")
            # 加密检测: soffice stderr 中常见关键字
            if "password" in soffice_err.lower() or "encrypted" in soffice_err.lower():
                raise RuntimeError(
                    f"文件 {file_path.name} 似乎已加密或受密码保护,无法转换"
                )

        raise RuntimeError(
            f"无法转换 .doc 文件 {file_path.name}。\n"
            f"  antiword: {antiword_error or 'unavailable'}\n"
            f"  textutil: {textutil_err or 'unavailable'}\n"
            f"  soffice:  {soffice_err}\n"
            "请安装其中之一: brew install antiword 或 brew install --cask libreoffice"
        )


class XlsConverter(BaseConverter):
    """老版本Excel (.xls) 转换器

    使用xlrd库读取xls文件
    """
    extensions = [".xls"]

    def convert(self, file_path: Path, **kwargs) -> str:
        try:
            import xlrd
        except ImportError:
            raise RuntimeError(
                "无法转换 .xls 文件。请安装 xlrd: "
                "pip install xlrd"
            )

        try:
            workbook = xlrd.open_workbook(str(file_path))
        except xlrd.XLRDError as e:
            # XLRDError 涵盖加密 / corrupt / 错误格式
            msg = str(e).lower()
            if "password" in msg or "encrypted" in msg:
                raise RuntimeError(f"文件 {file_path.name} 加密或受密码保护,无法转换") from e
            raise RuntimeError(f"无法解析 .xls 文件 {file_path.name}: {e}") from e

        parts: list[str] = []

        for sheet_name in workbook.sheet_names():
            sheet = workbook.sheet_by_name(sheet_name)
            if sheet.nrows == 0 or sheet.ncols == 0:
                continue

            # 每个 sheet 独立组装 markdown 表, 最后整体加到 parts。
            # 旧实现用 parts.insert(3, separator) 这种"绝对位置 3"的写法,
            # 第二个 sheet 会插到第一个 sheet 数据中间,把表错乱。
            sheet_lines: list[str] = [f"## 工作表: {sheet_name}", ""]
            for row_idx in range(sheet.nrows):
                row_values = []
                for col_idx in range(sheet.ncols):
                    cell_value = sheet.cell_value(row_idx, col_idx)
                    row_values.append(str(cell_value))
                if any(v.strip() for v in row_values):
                    sheet_lines.append("| " + " | ".join(row_values) + " |")

            # 在 sheet 第一行数据后插入 Markdown 表头分隔符: heading, 空行, 第一行, ---, 余下行
            if len(sheet_lines) >= 3:
                separator = "|" + "---|" * sheet.ncols
                sheet_lines.insert(3, separator)

            parts.extend(sheet_lines)
            parts.append("")

        return clean_markdown("\n".join(parts))


class PptConverter(BaseConverter):
    """老版本PowerPoint (.ppt) 转换器

    使用LibreOffice转换为pptx再处理
    """
    extensions = [".ppt"]

    def convert(self, file_path: Path, **kwargs) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            pptx_path, soffice_err = _run_soffice(file_path, "pptx", tmpdir)
            if pptx_path:
                from x2md.converters.pptx import PptxConverter
                return PptxConverter().convert(pptx_path, **kwargs)

        logger.warning(f"soffice failed for {file_path.name}: {soffice_err}")
        if "password" in soffice_err.lower() or "encrypted" in soffice_err.lower():
            raise RuntimeError(
                f"文件 {file_path.name} 似乎已加密或受密码保护,无法转换"
            )
        raise RuntimeError(
            f"无法转换 .ppt 文件 {file_path.name}: {soffice_err}\n"
            "请安装 LibreOffice: brew install --cask libreoffice 或 apt-get install libreoffice"
        )
