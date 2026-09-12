from __future__ import annotations

import re
import zipfile
from pathlib import Path


def clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip() + "\n"
    return text


def escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def table_to_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    normalized = []
    for row in rows:
        padded = [escape_pipe(str(cell).strip()) for cell in row]
        padded += [""] * (col_count - len(padded))
        normalized.append(padded)

    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(col_count)) + " |"
    body_lines = []
    for row in normalized[1:]:
        body_lines.append("| " + " | ".join(row) + " |")

    parts = [header, separator] + body_lines
    return "\n".join(parts)


def table_to_lines(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    normalized = []
    for row in rows:
        padded = [str(cell).strip() for cell in row]
        padded += [""] * (col_count - len(padded))
        normalized.append(padded)

    headers = normalized[0]
    parts: list[str] = []
    for row in normalized[1:]:
        row_parts: list[str] = []
        for i, cell in enumerate(row):
            header_name = headers[i] if i < len(headers) else f"列{i + 1}"
            if cell:
                row_parts.append(f"{header_name}: {cell}")
        if row_parts:
            parts.append("\n".join(row_parts))
            parts.append("---")

    if parts and parts[-1] == "---":
        parts.pop()

    return "\n".join(parts)


def image_marker(filename: str) -> str:
    return f"##{filename}##"


def extract_images_from_zip(
    zip_path: Path,
    media_prefix: str,
    output_dir: Path,
    source_stem: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not zipfile.is_zipfile(zip_path):
        return mapping

    # 确保 output_dir 使用绝对路径, 用于路径穿越校验
    resolved_output_dir = output_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        media_files = [
            n for n in zf.namelist() if n.startswith(media_prefix) and not n.endswith("/")
        ]
        for idx, media_name in enumerate(sorted(media_files), start=1):
            # Zip Slip 防护:
            # 1) 先校验 archive 中的原始路径无 `..` / 绝对路径片段, 防止恶意 zip。
            #    旧实现仅在构造后的 image_path 上做 is_relative_to,
            #    但 image_path 是用 source_stem + idx 重命名的,本来就在 output_dir 内,
            #    校验形同虚设。
            # 2) 再用 os.path.normpath 双保险。
            normalized = media_name.replace("\\", "/")
            if normalized.startswith("/") or ".." in Path(normalized).parts:
                continue
            # 用 source_stem 重新命名,丢弃 archive 内目录结构(保留扩展名)
            image_path = resolved_output_dir / f"{source_stem}_{idx:03d}{Path(media_name).suffix or '.png'}"
            try:
                # Python 3.9+: is_relative_to;Path.relative_to 抛 ValueError 兼容更老版本
                image_path.resolve().relative_to(resolved_output_dir)
            except ValueError:
                continue  # 跳过路径穿越条目

            resolved_output_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(media_name) as src, open(image_path, "wb") as dst:
                dst.write(src.read())
            mapping[media_name] = image_path.name

    return mapping


# 座机分支必须区号以 0 开头 (中国区号 010/021/0755...): 否则 "2024-12345678"、
# "1990-11223344" 等文档/合同/登记编号 (YYYY-NNNNNNNN) 会被误当电话, 从原文扣字
# 改写为 "电话:" 行, 破坏文档内容 (合同编号/订单号/登记号全中招)。
_RE_PHONE = re.compile(
    r"(?<!\d)(1[3-9]\d{9}|0\d{2,3}-\d{7,8})(?!\d)"
)
_RE_ID_CARD = re.compile(
    r"(?<!\d)([1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
)
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)


def split_contact_info(text: str) -> str:
    if not text.strip():
        return text

    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        phones = _RE_PHONE.findall(line)
        ids = _RE_ID_CARD.findall(line)
        emails = _RE_EMAIL.findall(line)

        has_contacts = bool(phones or ids or emails)
        if not has_contacts:
            result.append(line)
            continue

        remaining = line
        extracted: list[str] = []

        for phone in phones:
            extracted.append(f"电话: {phone}")
            remaining = remaining.replace(phone, "", 1)
        for id_num in ids:
            extracted.append(f"身份证: {id_num}")
            remaining = remaining.replace(id_num, "", 1)
        for email in emails:
            extracted.append(f"邮箱: {email}")
            remaining = remaining.replace(email, "", 1)

        remaining = re.sub(r"\s{2,}", " ", remaining).strip()
        if remaining:
            result.append(remaining)
        result.extend(extracted)

    return "\n".join(result)


def strip_toc_markers(text: str) -> str:
    text = re.sub(r"^\.{2,}\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+\.{3,}\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text


def get_image_output_dir(md_output_path: Path | None, source_path: Path) -> Path:
    if md_output_path:
        base = md_output_path.parent
    else:
        base = source_path.parent
    return base / "images"


def insert_image_descriptions(md_content: str, images_dir: Path) -> str:
    if not images_dir.exists():
        return md_content

    image_md_files: dict[str, str] = {}
    for md_file in images_dir.glob("*.md"):
        image_md_files[md_file.stem] = md_file.read_text(encoding="utf-8").strip()

    if not image_md_files:
        return md_content

    def _replace_marker(match: re.Match) -> str:
        filename = match.group(1)
        stem = Path(filename).stem
        if stem in image_md_files:
            return image_md_files[stem]
        return match.group(0)

    pattern = re.compile(r"##([^#]+)##")
    return pattern.sub(_replace_marker, md_content)
