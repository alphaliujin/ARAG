from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

from x2md.config import Config, load_config
from x2md.converters import get_converter, supported_extensions
from x2md.converters.pdf import PdfConverter
from x2md.utils import insert_image_descriptions
from x2md.chunk import Chunk, ChunkSplitter, SmallerChunksStrategy, aggregate_chunk_text


@click.command()
@click.argument("input_path", type=click.Path(), required=False)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output path. Defaults to <input>.md or config output_dir.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to x2md.conf. Auto-detected from current directory.",
)
@click.option(
    "--ocr-lang",
    default=None,
    help="OCR language for image conversion. Overrides config.",
)
@click.option(
    "--no-ocr",
    is_flag=True,
    default=False,
    help="Disable OCR for image files (return placeholder instead of running tesseract).",
)
@click.option(
    "--encoding",
    default=None,
    help="Text encoding for text/html files. Overrides config.",
)
@click.option(
    "--no-tables",
    is_flag=True,
    default=False,
    help="Skip table extraction in PDF files.",
)
@click.option(
    "--sheet",
    default=None,
    help="Specific sheet name to convert in Excel files.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Disable LLM-based content analysis.",
)
@click.option(
    "--no-vit",
    is_flag=True,
    default=False,
    help="Disable ViT image feature extraction.",
)
@click.option(
    "--vit-only",
    is_flag=True,
    default=False,
    help="Only run ViT image feature extraction (skip document conversion).",
)
@click.option(
    "--list-formats",
    is_flag=True,
    default=False,
    help="List all supported file formats and exit.",
)
@click.option(
    "--chunk",
    is_flag=True,
    default=False,
    help="Output structured chunks with metadata instead of flat Markdown.",
)
@click.option(
    "--parent-child",
    is_flag=True,
    default=False,
    help="Use parent-child chunk strategy (small child for retrieval, large parent for context).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Override chunk size from config.",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Override chunk overlap from config.",
)
def main(
    input_path, output, config_path,
    ocr_lang, no_ocr, encoding, no_tables, sheet,
    no_llm, no_vit, vit_only, list_formats,
    chunk, parent_child, chunk_size, chunk_overlap,
):
    """Convert various file formats to Markdown for RAG chunking and ingestion.

    INPUT_PATH is the file or directory to convert. Defaults to the input_dir
    specified in x2md.conf (or 'from/' if not configured).

    Output directory structure mirrors the input directory structure,
    rooted at output_dir from x2md.conf (or 'to/' by default).
    """
    cfg = load_config(config_path)

    if list_formats:
        exts = supported_extensions()
        click.echo("Supported formats: " + ", ".join(exts))
        return

    input_dir = cfg.input_dir
    # -o 既可指定输出目录,也可指定 .md 完整文件路径(单文件转换时常用)。
    # 当输入是 .md 文件路径时,绕过 _resolve_output_path 的"相对路径派生 + _external 兜底"逻辑,
    # 直接把 -o 当作目标文件,避免 backend 调用时多出一层 _external_<hash>/ 子目录。
    explicit_output_file: Path | None = None
    if output:
        op = Path(output)
        if op.suffix.lower() == ".md":
            explicit_output_file = op
            output_dir = op.parent
        else:
            output_dir = op
    else:
        output_dir = cfg.output_dir

    if input_path is None:
        input_path = str(input_dir)
        if not input_dir.exists():
            click.echo("Error: Missing argument 'INPUT_PATH'.", err=True)
            click.echo("Use --list-formats to see supported formats.", err=True)
            sys.exit(1)

    path = Path(input_path)
    if not path.exists():
        click.echo(f"Error: Path not found: {path}", err=True)
        sys.exit(1)

    md_files: list[Path] = []
    chunk_files: list[Path] = []
    # 缓存每个 md_path 对应的 element 数据（PDF 一次解析拿全）
    element_cache: dict[Path, dict] = {}

    # CLI 显式覆盖优先于 x2md.conf 配置
    eff_chunk_size = chunk_size if chunk_size is not None else cfg.chunk_size
    eff_chunk_overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap

    # 初始化切片器（使用配置参数）
    splitter = ChunkSplitter(
        chunk_size=eff_chunk_size,
        chunk_overlap=eff_chunk_overlap,
        separators=cfg.chunk_separators,
        separator_rule=cfg.chunk_separator_rule,
        max_chunk_limit=cfg.chunk_max_limit,
    )

    inject_abstract = cfg.chunk_inject_abstract

    if not vit_only:
        llm_client = None
        if not no_llm:
            llm_client = cfg.create_llm_client()
            if llm_client is None and cfg.llm_enabled and not no_llm:
                click.echo(
                    f"Warning: LLM enabled (model={cfg.llm_model}) "
                    f"but Ollama not available at {cfg.llm_base_url}. "
                    "Continuing without LLM.",
                    err=True,
                )

        kwargs = _build_kwargs(
            cfg, llm_client, ocr_lang, no_ocr, encoding, no_tables, sheet
        )

        if path.is_dir():
            md_files, element_cache = _convert_directory(path, input_dir, output_dir, kwargs)
        else:
            md_path, ed = _convert_file(
                path, input_dir, output_dir, kwargs,
                explicit_output=explicit_output_file,
            )
            if md_path:
                md_files.append(md_path)
                if ed:
                    element_cache[md_path] = ed

        # chunk 模式：对每个 md 文件进行切片输出
        if chunk or parent_child:
            for md_path in md_files:
                content = md_path.read_text(encoding="utf-8")
                source = md_path.stem

                # 优先用缓存的 element 数据，避免再次打开 PDF
                element_data = element_cache.get(md_path)

                # 提取摘要（如果 LLM 可用）
                abstract = ""
                if inject_abstract and llm_client and llm_client.is_available():
                    try:
                        from x2md.llm import OllamaClient
                        abstract_prompt = (
                            "请用一句话概括以下文档的核心内容：\n\n"
                            + content[:3000]
                        )
                        abstract = llm_client.generate(abstract_prompt, use_cache=True)
                    except Exception:
                        abstract = ""

                if parent_child:
                    # A4: 显式传 parent_overlap=eff_chunk_overlap*2 (~200), 同时把 child_overlap
                    # 最低值从 20 抬到 60, 避免句子被硬切到两个 child 都低召回
                    strategy = SmallerChunksStrategy(
                        parent_chunk_size=eff_chunk_size * 4,
                        child_chunk_size=eff_chunk_size,
                        parent_overlap=max(eff_chunk_overlap * 2, 100),
                        child_overlap=max(eff_chunk_overlap // 2, 60),
                        separators=cfg.chunk_separators,
                        separator_rule=cfg.chunk_separator_rule,
                    )
                    parents, children = strategy.split(
                        text=content,
                        source=source,
                        abstract=abstract,
                        document_name=source,
                        **(
                            {
                                "element_indexes": element_data["element_indexes"],
                                "element_pages": element_data["element_pages"],
                                "element_bboxes": element_data["element_bboxes"],
                                "element_types": element_data["element_types"],
                            }
                            if element_data
                            else {}
                        ),
                    )
                    # 输出父子块
                    parent_path = md_path.with_suffix(".parents.json")
                    child_path = md_path.with_suffix(".children.json")
                    _write_chunks_json(parents, parent_path)
                    _write_chunks_json(children, child_path)
                    chunk_files.extend([parent_path, child_path])
                    click.echo(f"Parent chunks: {parent_path} ({len(parents)} chunks)")
                    click.echo(f"Child chunks: {child_path} ({len(children)} chunks)")
                else:
                    # 普通 chunk 模式
                    chunks = splitter.split_documents(
                        text=content,
                        source=source,
                        abstract=abstract,
                        document_name=source,
                        **(
                            {
                                "element_indexes": element_data["element_indexes"],
                                "element_pages": element_data["element_pages"],
                                "element_bboxes": element_data["element_bboxes"],
                                "element_types": element_data["element_types"],
                            }
                            if element_data
                            else {}
                        ),
                    )
                    chunk_path = md_path.with_suffix(".chunks.json")
                    _write_chunks_json(chunks, chunk_path)
                    chunk_files.append(chunk_path)
                    click.echo(f"Chunks: {chunk_path} ({len(chunks)} chunks)")

    if not no_vit and cfg.vit_enabled:
        _run_vit(cfg, output_dir)
        _insert_image_descriptions(md_files)


def _build_kwargs(
    cfg: Config, llm_client, ocr_lang, no_ocr, encoding, no_tables, sheet,
) -> dict:
    kwargs: dict = {}

    kwargs["llm_client"] = llm_client

    pdf_cfg = cfg.section("pdf")
    # --no-tables 显式 True → 强制关;未传 → 用 config 默认。
    # 旧实现 `not no_tables if no_tables else cfg` 三目逻辑很绕,意义同此。
    if no_tables:
        kwargs["extract_tables"] = False
    else:
        kwargs["extract_tables"] = pdf_cfg.get("extract_tables", True)
    kwargs["page_separator"] = pdf_cfg.get("page_separator", "\n\n---\n\n")

    xlsx_cfg = cfg.section("xlsx")
    kwargs["include_all_sheets"] = xlsx_cfg.get("include_all_sheets", True)
    kwargs["sheet_name"] = sheet or xlsx_cfg.get("default_sheet", None) or None

    img_cfg = cfg.section("image")
    kwargs["ocr_lang"] = ocr_lang or img_cfg.get("ocr_lang", "eng")
    # OCR 总开关: --no-ocr 时 ImageConverter 跳过 tesseract 直接返回占位符
    kwargs["enable_ocr"] = not no_ocr

    kwargs["encoding"] = encoding or cfg.encoding

    return kwargs


def _run_vit(cfg: Config, output_dir: Path):
    try:
        from x2md.vit import VitExtractor
    except ImportError:
        click.echo(
            "ViT dependencies not installed. "
            "Run: pip install x2md[vit]",
            err=True,
        )
        return

    images_dirs = set()
    for images_dir in output_dir.rglob("images"):
        if images_dir.is_dir():
            images_dirs.add(images_dir)

    if not images_dirs:
        return

    extractor = VitExtractor(
        model_name=cfg.vit_model,
        device=cfg.vit_device,
        local_files_only=cfg.vit_local_files_only,
        mirror=cfg.vit_mirror,
        labels=cfg.vit_labels,
    )

    for images_dir in sorted(images_dirs):
        image_files = [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"
            }
        ]
        if not image_files:
            continue

        click.echo(f"ViT: Processing {len(image_files)} images in {images_dir}")

        try:
            output_files = extractor.process_directory(images_dir)
        except Exception as e:
            click.echo(
                f"ViT: Failed to process {images_dir}: {e}",
                err=True,
            )
            click.echo(
                "Hint: Set mirror in x2md.conf [vit] section "
                "(e.g. mirror = https://hf-mirror.com) or download "
                "the model locally and set model to the local path.",
                err=True,
            )
            return

        for md_path in output_files:
            click.echo(f"ViT: {md_path}")


def _insert_image_descriptions(md_files: list[Path]):
    for md_path in md_files:
        images_dir = md_path.parent / "images"
        if not images_dir.exists():
            continue

        content = md_path.read_text(encoding="utf-8")
        if not re.search(r'##[^#]+##', content):
            continue

        new_content = insert_image_descriptions(content, images_dir)
        if new_content != content:
            md_path.write_text(new_content, encoding="utf-8")
            click.echo(f"Updated: {md_path} (image descriptions inserted)")


def _resolve_output_path(file_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map input file path to output path, preserving relative directory structure.

    若 file_path 不在 input_dir 下,旧实现 fallback 为 `output_dir / file_path.name`,
    多个同名输入(`/a/foo.docx`, `/b/foo.docx`)会互相覆盖。改为附加内容哈希前缀。
    """
    try:
        relative = file_path.relative_to(input_dir)
    except ValueError:
        # 用文件绝对路径的 SHA-1 前 8 位作目录前缀,保留 .name 便于阅读
        import hashlib
        digest = hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()[:8]
        relative = Path(f"_external_{digest}") / file_path.name
    return output_dir / relative.with_suffix(".md")


def _convert_file(
    file_path: Path, input_dir: Path, output_dir: Path, kwargs: dict,
    explicit_output: Path | None = None,
) -> tuple[Path | None, dict | None]:
    """转换单个文件，返回 (md_path, element_data)。

    关键：使用 convert_with_elements 一次解析拿到 element 数据，
    避免 parent-child 模式下被重复打开 PDF。

    explicit_output 不为 None 时(由 CLI 的 -o file.md 指定)直接用该路径,
    跳过 _resolve_output_path 的"相对路径派生"逻辑。
    """
    converter = get_converter(file_path)
    if converter is None:
        ext = file_path.suffix
        click.echo(f"Error: Unsupported file format '{ext}'", err=True)
        click.echo(
            f"Supported formats: {', '.join(supported_extensions())}", err=True
        )
        sys.exit(1)

    out_path = explicit_output if explicit_output else _resolve_output_path(file_path, input_dir, output_dir)

    kwargs["md_output_path"] = out_path

    try:
        # 优先尝试 convert_with_elements（PdfConverter 一次解析拿全数据）
        if isinstance(converter, PdfConverter):
            result = converter.convert_with_elements(file_path, **kwargs)
            md_content = result["text"]
            element_data = {
                "element_indexes": result["element_indexes"],
                "element_pages": result["element_pages"],
                "element_bboxes": result["element_bboxes"],
                "element_types": result["element_types"],
            }
        else:
            md_content = converter.convert(file_path, **kwargs)
            element_data = None
    except Exception as e:
        click.echo(f"Error converting {file_path}: {e}", err=True)
        return None, None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    click.echo(f"Converted: {file_path} -> {out_path}")
    return out_path, element_data


def _convert_directory(
    dir_path: Path, input_dir: Path, output_dir: Path, kwargs: dict
) -> tuple[list[Path], dict[Path, dict]]:
    """转换目录下所有文件，返回 (md_files, element_data_by_md_path)。

    element_data_by_md_path 缓存每个 md 文件对应的 element 元数据，
    后续 chunk 阶段直接用，避免重复打开 PDF。
    """
    exts = set(supported_extensions())
    files = [
        f for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]

    if not files:
        click.echo(f"No supported files found in {dir_path}", err=True)
        return [], {}

    md_files: list[Path] = []
    element_cache: dict[Path, dict] = {}
    for file_path in sorted(files):
        out_path = _resolve_output_path(file_path, input_dir, output_dir)

        converter = get_converter(file_path)
        if converter is None:
            continue

        kwargs["md_output_path"] = out_path

        try:
            if isinstance(converter, PdfConverter):
                result = converter.convert_with_elements(file_path, **kwargs)
                md_content = result["text"]
                element_cache[out_path] = {
                    "element_indexes": result["element_indexes"],
                    "element_pages": result["element_pages"],
                    "element_bboxes": result["element_bboxes"],
                    "element_types": result["element_types"],
                }
            else:
                md_content = converter.convert(file_path, **kwargs)
        except Exception as e:
            click.echo(f"Error converting {file_path}: {e}", err=True)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_content, encoding="utf-8")
        click.echo(f"Converted: {file_path} -> {out_path}")
        md_files.append(out_path)

    return md_files, element_cache


def _write_chunks_json(chunks: list[Chunk], output_path: Path):
    """将 Chunk 列表写入 JSON 文件"""
    data = [chunk.to_dict() for chunk in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_source_file(md_path: Path, input_dir: Path, output_dir: Path) -> Path | None:
    """根据输出 md 文件路径反查输入源文件"""
    try:
        relative = md_path.relative_to(output_dir)
    except ValueError:
        relative = Path(md_path.name)

    # 尝试匹配各种扩展名
    for ext in supported_extensions():
        candidate = input_dir / relative.with_suffix(ext)
        if candidate.exists():
            return candidate

    # 在输入目录下搜索同名文件
    stem = md_path.stem
    for ext in supported_extensions():
        for candidate in input_dir.rglob(f"{stem}{ext}"):
            return candidate

    return None


if __name__ == "__main__":
    main()
