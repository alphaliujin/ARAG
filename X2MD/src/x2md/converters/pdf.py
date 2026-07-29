from __future__ import annotations

from pathlib import Path

import pdfplumber

from x2md.converters.base import BaseConverter
from x2md.llm import OllamaClient, llm_batch_strip_noise
from x2md.utils import (
    clean_markdown,
    get_image_output_dir,
    split_contact_info,
    strip_toc_markers,
    table_to_lines,
)


class PdfConverter(BaseConverter):
    extensions = [".pdf"]

    def convert(self, file_path: Path, **kwargs) -> str:
        return self._convert_internal(file_path, **kwargs)[0]

    def convert_with_elements(self, file_path: Path, **kwargs) -> dict:
        """
        转换 PDF 并返回 element 级元数据（indexes, pages, bboxes, types）

        Returns:
            {
                "text": str,             # 转换后的 Markdown
                "source": str,           # 文件名
                "element_indexes": list, # [[start, end], ...]
                "element_pages": list,   # [page_num, ...]
                "element_bboxes": list,  # [[x1,y1,x2,y2], ...]
                "element_types": list,   # ["text", "Table", ...]
            }
        """
        md_text, element_data = self._convert_internal(file_path, **kwargs)
        return {
            "text": md_text,
            "source": file_path.name,
            "element_indexes": element_data["indexes"],
            "element_pages": element_data["pages"],
            "element_bboxes": element_data["bboxes"],
            "element_types": element_data["types"],
        }

    def _convert_internal(self, file_path: Path, **kwargs) -> tuple[str, dict]:
        page_separator: str = kwargs.get("page_separator", "\n\n---\n\n")
        extract_tables: bool = kwargs.get("extract_tables", True)
        llm_client: OllamaClient | None = kwargs.get("llm_client")
        md_output_path: Path | None = kwargs.get("md_output_path")

        image_dir = get_image_output_dir(md_output_path, file_path)
        image_dir.mkdir(parents=True, exist_ok=True)

        # 单次打开 PDF, 提取每页的文本行(+bbox) 与 表格(+bbox)。
        # element = 文本行 或 表格; 其字符区间随后统一指向"最终返回的 markdown"。
        page_text_lines: list[list[str]] = []                    # 每页文本行
        page_text_bboxes: list[list[list[float]]] = []           # 每页每行 bbox (与文本行对齐)
        page_tables: list[list[tuple[str, list[float]]]] = []    # 每页 [(表格md, bbox), ...]
        page_text_cache: list[str] = []                          # 原始 extract_text, 供页眉页脚检测

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                page_text_cache.append(text)

                lines = text.split("\n") if text else []
                page_text_lines.append(lines)
                page_text_bboxes.append([self._get_line_bbox(page, ln) for ln in lines])

                tables_md: list[tuple[str, list[float]]] = []
                if extract_tables:
                    for table in (page.extract_tables() or []):
                        if not table:
                            continue
                        md_table = table_to_lines(table)
                        if md_table:
                            tables_md.append((md_table, self._get_table_bbox(page, table)))
                page_tables.append(tables_md)

        # 页眉/页脚检测, 并从每页文本行里同步移除 (bbox 一并丢弃), 使最终 markdown 与
        # element 列表都只含正文本。此前 _filter_header_footer_elements 用"页内文本行数
        # 当元素数"算偏移, 漏算表格元素 -> 第 2 页起 local_line_idx 错位, 页眉页脚过滤失效。
        header_footer_texts = self._detect_header_footer_from_texts(page_text_cache)
        if header_footer_texts:
            for i, lines in enumerate(page_text_lines):
                kept = [
                    (ln, bb) for ln, bb in zip(lines, page_text_bboxes[i])
                    if ln.strip() not in header_footer_texts
                ]
                page_text_lines[i] = [ln for ln, _ in kept]
                page_text_bboxes[i] = [bb for _, bb in kept]

        # 过滤后每页正文 (文本行用 \n 连接), 供噪声页检测 + 作为最终 markdown 的页正文
        page_texts: list[str] = ["\n".join(lines) for lines in page_text_lines]

        # 噪声页检测 (LLM): 命中的页整页跳过, 不进最终 markdown 也不进 element 列表
        noise_indices: set[int] = set()
        if llm_client and llm_client.is_available():
            candidate_texts: list[str] = []
            candidate_indices: list[int] = []
            for i, pt in enumerate(page_texts):
                if pt.strip():
                    candidate_texts.append(pt)
                    candidate_indices.append(i)
            if candidate_texts:
                batch_noise = llm_batch_strip_noise(llm_client, candidate_texts)
                for local_idx in batch_noise:
                    noise_indices.add(candidate_indices[local_idx])

        # 构造最终 markdown: 每个非空非噪声页一块, 前缀 <!-- Page N -->,
        # 页正文+表格用 \n\n 连接, 页间用 page_separator 连接 (与原实现输出一致)。
        parts: list[str] = []
        for i, pt in enumerate(page_texts):
            if i in noise_indices:
                continue
            table_strs = [md for md, _ in page_tables[i]]
            page_content_parts: list[str] = []
            if pt.strip():
                page_content_parts.append(pt)
            page_content_parts.extend(table_strs)
            if not page_content_parts:
                continue
            parts.append(f"<!-- Page {i + 1} -->\n")
            parts.append("\n\n".join(page_content_parts))

        result = page_separator.join(parts)
        result = strip_toc_markers(result)
        result = split_contact_info(result)
        result = clean_markdown(result)

        # ★ element indexes 必须指向"最终返回的 markdown"(经 strip_toc /
        # split_contact_info / clean_markdown 变换后), 因为 chunker 用同一份 markdown 切片、
        # 并用 text.find() 定位 (chunk.py:475), 再用 IntervalSearch(element_indexes) 把 chunk
        # 区间映射到 element。此前 indexes 按一个"假想的 \n 拼接串"重建, 与真正返回的串
        # (含 <!-- Page N -->、page_separator、表格、噪声页剔除、三道文本变换)完全错位 ->
        # 每个 PDF chunk 的 bbox/页码元数据全错。
        # 改为对最终串做游标式 find, 逐个 element 定位其真实字符区间: 游标只前进, 故即使
        # 同名行重复出现, 也按文档顺序匹配到各自位置。被 split_contact_info / strip_toc
        # 改动后不再出现的行/表跳过 (其 bbox 无法可靠关联, 不写入而非写错位置)。
        element_indexes: list[list[int]] = []
        element_pages: list[int] = []
        element_bboxes: list[list[float]] = []
        element_types: list[str] = []
        cursor = 0
        for i, (lines, bboxes, tables) in enumerate(
            zip(page_text_lines, page_text_bboxes, page_tables)
        ):
            if i in noise_indices:
                continue
            # 文本行元素 (页内顺序)
            for ln, bb in zip(lines, bboxes):
                if not ln.strip():
                    continue  # 空白行无法在最终串里唯一定位, 且 bbox 无意义
                pos = result.find(ln, cursor)
                if pos == -1:
                    continue  # 行文本被 split_contact_info / strip_toc 改动后不再出现, 跳过
                element_indexes.append([pos, pos + len(ln)])
                element_pages.append(i + 1)
                element_bboxes.append(bb)
                element_types.append("text")
                cursor = pos + len(ln)
            # 表格元素
            for md, bb in tables:
                pos = result.find(md, cursor)
                if pos == -1:
                    continue
                element_indexes.append([pos, pos + len(md)])
                element_pages.append(i + 1)
                element_bboxes.append(bb)
                element_types.append("Table")
                cursor = pos + len(md)

        element_data = {
            "indexes": element_indexes,
            "pages": element_pages,
            "bboxes": element_bboxes,
            "types": element_types,
        }
        return result, element_data

    @staticmethod
    def _get_line_bbox(page, line_text: str) -> list[float]:
        """尝试获取某行文本的 bbox，失败则返回空列表。

        优化：单遍 Counter 统计 top 频率，避免 3 遍 chars 扫描。
        """
        try:
            chars = page.chars
            if not chars:
                return []

            # 单遍：累加 top 出现次数
            top_counts: dict[float, int] = {}
            for c in chars:
                if c.get("text", "") and c["text"] in line_text:
                    rounded = round(c["top"], 1)
                    top_counts[rounded] = top_counts.get(rounded, 0) + 1

            if not top_counts:
                return []

            dominant_top = max(top_counts, key=top_counts.get)

            # 单遍：收集 dominant_top ± 2pt 范围内的字符
            line_chars = [
                c for c in chars
                if c.get("text", "") and abs(c["top"] - dominant_top) < 2
            ]

            if not line_chars:
                return []

            x0 = min(c["x0"] for c in line_chars)
            y0 = min(c["top"] for c in line_chars)
            x1 = max(c["x1"] for c in line_chars)
            y1 = max(c["bottom"] for c in line_chars)
            return [x0, y0, x1, y1]
        except Exception:
            return []

    @staticmethod
    def _get_table_bbox(page, table: list) -> list[float]:
        """尝试获取表格的 bbox"""
        try:
            found_tables = page.find_tables()
            for ft in found_tables:
                ft_data = ft.extract()
                if ft_data and len(ft_data) == len(table):
                    bbox = ft.bbox
                    return list(bbox)
            return []
        except Exception:
            return []

    @staticmethod
    def _detect_header_footer(pdf, threshold: float = 0.6) -> set[str]:
        # 兼容旧接口：仍然接受 pdf 对象，内部委托给基于缓存文本的实现
        page_count = len(pdf.pages)
        if page_count < 3:
            return set()
        page_text_cache = [page.extract_text() or "" for page in pdf.pages]
        return PdfConverter._detect_header_footer_from_texts(page_text_cache, threshold)

    @staticmethod
    def _detect_header_footer_from_texts(
        page_texts: list[str], threshold: float = 0.6
    ) -> set[str]:
        """基于已缓存的每页文本检测 header/footer（不再调 extract_text）。"""
        page_count = len(page_texts)
        if page_count < 3:
            return set()

        line_pages: dict[str, int] = {}
        for text in page_texts:
            if not text:
                continue
            seen_on_page: set[str] = set()
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in seen_on_page:
                    line_pages[stripped] = line_pages.get(stripped, 0) + 1
                    seen_on_page.add(stripped)

        min_occurrences = max(3, int(page_count * threshold))
        return {
            line for line, count in line_pages.items()
            if count >= min_occurrences
        }
