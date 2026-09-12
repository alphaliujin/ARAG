"""
文本清洗与质量增强模块
提供文档预处理、清洗、去重、元数据提取、高级切片等功能
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from x2md.chunk import Chunk, ChunkMetadata, ChunkType, ChunkSplitter, ChunkMaxLimitError


# 全角 -> 半角 转换表（模块级一次性构建，translate() 走 C 路径）
# - U+3000 全角空格 -> 半角空格
# - U+FF01..U+FF5E 全角 ASCII -> 对应半角 ASCII（差 0xFEE0）
def _build_fullwidth_to_halfwidth() -> dict[int, int]:
    table: dict[int, int] = {0x3000: 0x20}
    table.update({cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)})
    return table


_FULLWIDTH_TO_HALFWIDTH = _build_fullwidth_to_halfwidth()

_RE_MULTI_WS = re.compile(r'[ \t]+')

# 模块级预编译正则（类内多处复用）
_SKIP_SECTION_RE = [
    re.compile(p, re.IGNORECASE) for p in (
        r'^\s*目\s*录\s*$',
        r'^\s*contents?\s*$',
        r'^\s*参考文献\s*$',
        r'^\s*references?\s*$',
        r'^\s*附录\s*[A-Z]?\s*$',
        r'^\s*appendix\s*[A-Z]?\s*$',
        r'^\s*致谢\s*$',
        r'^\s*acknowledgements?\s*$',
        r'^\s*公式\s*$',
        r'^\s*equations?\s*$',
        r'^\s*索引\s*$',
        r'^\s*index\s*$',
        r'^\s*图\s*表?\s*目\s*录\s*$',
        r'^\s*list\s+of\s+(figures|tables)\s*$',
        # 注: 摘要/abstract/关键词/keywords 是 RAG 高价值检索内容, 不在此过滤。
        # 早期版本曾过滤这些章节, filter_content 命中标题后会连续跳过后续正文行,
        # 导致摘要正文与关键词列表被整段丢弃, 严重降低检索召回质量。
    )
]

_FORMULA_RE = [re.compile(p) for p in (
    r'^\s*\$+.+\$+\s*$',
    r'^\s*\\\[.+\\\]\s*$',
    r'^\s*\\\(.+\\\)\s*$',
    # 早期第 4 条 r'^\s*[\w\s]*=.+[+\-*/=^].*$' 过宽, 会误删 "E = mc^2"、
    # "Revenue = price * quantity" 等含等号+算符的正常说明文字。已移除:
    # LaTeX 风格 ($...$/\[...\]/\(...\)) 与纯数字算式已由其余规则覆盖,
    # 含变量的行内公式即使残留为文本也无害 (仍可被检索), 优于静默删除。
    r'^\s*\d+\s*[+\-*/=^]\s*\d+',
)]

_GARBAGE_RE = [re.compile(p) for p in (
    r'[\x00-\x08\x0b-\x0c\x0e-\x1f]',
    r'[�﻿]',
    r'[　]{3,}',
    r'[^\w\s一-鿿　-〿＀-￯]{10,}',
)]

_LOW_QUALITY_RE = [re.compile(p) for p in (
    r'^\s*\d+\s*$',
    r'^\s*[^\w一-鿿]+\s*$',
    r'^\s*[_\-]{3,}\s*$',
    r'^\s*[=]{3,}\s*$',
    r'^\s*[-]{3,}>\s*$',
)]

# 共用的小模式
_RE_KV_LINE = re.compile(r'^[\w一-鿿\s]+:\s+\S')
_RE_MD_TABLE_SEP = re.compile(r'^\|[\s\-:]+\|$')
_RE_FULLWIDTH_SPACE3 = re.compile(r'[　]{3,}')
_RE_CTRL_BOM = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f﻿]')
_RE_SECTION_BREAK = re.compile(
    r'^(第[一二三四五六七八九十\d]+[章节条]|\d+[\.\s]+|[一二三四五六七八九十][、\.\s])'
)
# 目录/参考文献章节内的"条目行"特征: 有序号标记 (1. / 一、 / [1] / (1)) 或点线
# (...12)。用于判断 skip-section 标题后紧跟的是不是真章节内容: 若不是 (标题后
# 直接是正文), 立即退出跳过, 防止吞掉正文第一段。注意不单用"行尾数字"判页码,
# 否则 "附录\n\n附表A数据 12\n\n正文" 这类正文行会被误当目录条目吞掉。
_RE_SECTION_ENTRY = re.compile(
    r'^\s*(?:'
    r'\d+[、．.]?\s*[一-鿿A-Za-z]'      # "1 概述" / "1. 概述" / "1、概述"
    r'|第[一二三四五六七八九十\d]+[章节条例]'   # "第一章"
    r'|[一二三四五六七八九十]+[、\.]\s*[一-鿿]'  # "一、概述"
    r'|\[\d+\]'                        # "[1] 参考文献"
    r'|\(?\d+\)\s*[一-鿿A-Za-z]'       # "(1) 概述"
    r')'
    r'|\.{2,}'                          # 点线 "……12"
)
_RE_REPEAT_PUNCT = re.compile(r'([。！？，、；：])\1+')
_RE_DEDUP_PUNCT = re.compile(r'[\s　，。！？；：、（）【】「」""''…—]+')
_RE_CN_EN_NO_SPACE = re.compile(r'([一-鿿])\s+([a-zA-Z0-9])')
_RE_EN_CN_NO_SPACE = re.compile(r'([a-zA-Z0-9])\s+([一-鿿])')
_RE_HEADING = re.compile(r'^(#{1,6})\s+(.+)$')
_RE_SENT_END = re.compile(r'[。！？；\.\!\?]')
_RE_SENT_END_GROUP = re.compile(r'([。！？；\.\!\?])')
_RE_CLAUSE_SPLIT = re.compile(r'([，、,])')
_RE_CN_TOKENS = re.compile(r'[一-鿿]{2,4}|[a-zA-Z]{3,}')
_RE_CN_LONG = re.compile(r'[一-鿿]{2,4}')
_RE_EN_WORD = re.compile(r'[a-zA-Z]{3,}')
_RE_CN_CHAR = re.compile(r'[一-鿿]')
_RE_URL_DOMAIN = re.compile(r'https?://([^/]+)')
_RE_SENT_END_STRIP = re.compile(r'^([。！？；\.\!\?])+$')
_RE_HEADING_PREFIX = re.compile(r'^#{1,6}\s')
_RE_PURE_DIGIT = re.compile(r'^\s*\d+\s*$')

# 停用词（模块级 frozenset，省去每次方法调用重新构造）
_STOP_WORDS = frozenset({
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这',
    'the', 'and', 'for', 'are', 'but', 'not', 'you',
    'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out',
    'day', 'get', 'has', 'him', 'his',
})


class TextCleaner:
    """文本清洗器"""

    # 需要过滤的章节标题模式（目录、参考文献、公式等）
    SKIP_SECTION_PATTERNS = [
        r'^\s*目\s*录\s*$',
        r'^\s*contents?\s*$',
        r'^\s*参考文献\s*$',
        r'^\s*references?\s*$',
        r'^\s*附录\s*[A-Z]?\s*$',
        r'^\s*appendix\s*[A-Z]?\s*$',
        r'^\s*致谢\s*$',
        r'^\s*acknowledgements?\s*$',
        r'^\s*公式\s*$',
        r'^\s*equations?\s*$',
        r'^\s*索引\s*$',
        r'^\s*index\s*$',
        r'^\s*图\s*表?\s*目\s*录\s*$',
        r'^\s*list\s+of\s+(figures|tables)\s*$',
        r'^\s*摘\s*要\s*$',
        r'^\s*abstract\s*$',
        r'^\s*关\s*键\s*词\s*$',
        r'^\s*keywords?\s*$',
    ]

    # 公式模式（LaTeX 风格、等号开头的行等）
    FORMULA_PATTERNS = [
        r'^\s*\$+.+\$+\s*$',           # $...$ 或 $$...$$
        r'^\s*\\\[.+\\\]\s*$',        # \[...\]
        r'^\s*\\\(.+\\\)\s*$',        # \(...\)
        r'^\s*[\w\s]*=.+[+\-*/=^].*$',  # x = y + z 形式
        r'^\s*\d+\s*[+\-*/=^]\s*\d+',   # 数字运算符数字
    ]

    # 乱码检测模式
    GARBAGE_PATTERNS = [
        r'[\x00-\x08\x0b-\x0c\x0e-\x1f]',  # 控制字符
        r'[�﻿]',                   # 替换字符、BOM
        r'[　]{3,}',                      # 连续3个以上全角空格
        r'[^\w\s一-鿿　-〿＀-￯]{10,}',  # 连续10个以上非文字符号
    ]

    # 低质量文本模式
    LOW_QUALITY_PATTERNS = [
        r'^\s*\d+\s*$',                      # 纯数字行
        r'^\s*[^\w一-鿿]+\s*$',      # 纯符号行
        r'^\s*[_\-]{3,}\s*$',                # 分隔线
        r'^\s*[=]{3,}\s*$',                  # 等号分隔线
        r'^\s*[-]{3,}>\s*$',                 # 箭头分隔
    ]

    # URL 模式
    # 负向后顾 (?<!\]) 跳过已是 markdown 链接目标的 URL ([text](http://...)),
    # 否则 normalize_links 会把它再包一层 [链接: ...](...) 破坏既有链接语法;
    # 二次 normalize 时同样靠它跳过已包裹的 URL。
    URL_PATTERN = re.compile(
        r'(?<!\]\()https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        re.IGNORECASE
    )

    # 邮箱模式
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )

    # 日期模式
    DATE_PATTERN = re.compile(
        r'(?:\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 500)
        self.chunk_overlap = self.config.get('chunk_overlap', 0)
        self.separators = self.config.get('separators', ["\n\n", "\n", "。", ".", " ", ""])
        self.separator_rule = self.config.get('separator_rule', None)
        self.max_chunk_limit = self.config.get('max_chunk_limit', 10000)
        self.chunk_splitter = ChunkSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            separator_rule=self.separator_rule,
            max_chunk_limit=self.max_chunk_limit,
        )

    def clean_text(self, text: str) -> str:
        """
        清洗文本：空格、换行、全角半角归一
        """
        if not text:
            return ""

        # 1. 全角字符转半角（保留中文）：全角空格 -> 半角空格；FF01-FF5E -> 对应半角
        text = text.translate(_FULLWIDTH_TO_HALFWIDTH)

        # 2. 处理连续空格和制表符
        text = _RE_MULTI_WS.sub(' ', text)

        # 3. 处理换行：保留段落结构，但删除多余空行
        lines = text.split('\n')
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                # 保留一个空行作为段落分隔
                cleaned_lines.append('')
                prev_empty = True

        # 4. 去除首尾空行
        while cleaned_lines and cleaned_lines[0] == '':
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1] == '':
            cleaned_lines.pop()

        text = '\n'.join(cleaned_lines)

        # 5. 处理中文与英文/数字之间的空格
        text = _RE_CN_EN_NO_SPACE.sub(r'\1\2', text)
        text = _RE_EN_CN_NO_SPACE.sub(r'\1\2', text)

        # 6. 处理重复标点
        text = _RE_REPEAT_PUNCT.sub(r'\1', text)

        return text

    def detect_garbage(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        检测文本中的乱码/垃圾字符

        Returns:
            (是否包含乱码, 乱码比例, 乱码片段列表)
        """
        if not text:
            return False, 0.0, []

        garbage_count = 0
        garbage_fragments = []

        for pattern in _GARBAGE_RE:
            for match in pattern.finditer(text):
                m = match.group()
                garbage_count += len(m)
                if m:
                    garbage_fragments.append(m[:50])

        # 去重
        garbage_fragments = list(set(garbage_fragments))

        ratio = garbage_count / len(text) if text else 0.0
        has_garbage = ratio > 0.01 or len(garbage_fragments) > 5

        return has_garbage, ratio, garbage_fragments

    def remove_garbage(self, text: str) -> str:
        """移除乱码字符"""
        if not text:
            return text

        # 移除控制字符和BOM
        text = _RE_CTRL_BOM.sub('', text)
        # 移除替换字符
        text = text.replace('�', '')
        # 将连续3个以上全角空格替换为单个
        text = _RE_FULLWIDTH_SPACE3.sub('　', text)

        return text

    def is_skip_section(self, line: str) -> bool:
        """判断是否为需要跳过的章节"""
        line = line.strip()
        if not line:
            return False

        for pattern in _SKIP_SECTION_RE:
            if pattern.match(line):
                return True

        return False

    def is_formula_line(self, line: str) -> bool:
        """判断是否为公式行"""
        line = line.strip()
        if not line:
            return False

        for pattern in _FORMULA_RE:
            if pattern.match(line):
                return True

        return False

    def filter_content(self, text: str) -> str:
        """过滤目录、参考文献、公式等内容"""
        if not text:
            return ""

        lines = text.split('\n')
        filtered_lines = []
        skip_section = False
        skip_depth = 0
        # skip-section 标题后是否已见过首个非空行: 首行必须是章节条目 (目录编号/
        # 点线/页码) 才继续跳; 若是正文 (标题后只有单空行), 立即恢复并保留该行,
        # 否则 "目录\n\n这是正文第一段…" 会吞掉正文第一段 (常见文档的内容丢失)。
        first_content_seen = False

        for line in lines:
            original_line = line
            stripped = line.strip()

            # 检测章节标题
            if self.is_skip_section(stripped):
                skip_section = True
                skip_depth = 0
                first_content_seen = False
                continue

            # 检测章节结束
            if skip_section:
                if not stripped:
                    skip_depth += 1
                    if skip_depth >= 2:
                        skip_section = False
                        skip_depth = 0
                    continue
                else:
                    if not first_content_seen:
                        # 首行判定: 是章节条目则继续跳 (跳过该行), 否则退出跳过保留正文
                        first_content_seen = True
                        if _RE_SECTION_ENTRY.match(stripped):
                            continue
                        skip_section = False
                        skip_depth = 0
                    elif _RE_SECTION_ENTRY.match(stripped):
                        # 仍是目录条目 (如 "2. 方法……5"), 继续跳;
                        # 必须放在 _RE_SECTION_BREAK 之前, 否则编号条目会被当成
                        # "下一章标题" 提前结束跳过而泄漏进正文
                        continue
                    elif _RE_SECTION_BREAK.match(stripped):
                        skip_section = False
                        skip_depth = 0
                    elif skip_section:
                        continue

            # 跳过公式行
            if self.is_formula_line(stripped):
                continue

            filtered_lines.append(original_line)

        return '\n'.join(filtered_lines)

    def is_low_quality_line(self, line: str) -> bool:
        """判断是否为低质量行"""
        stripped = line.strip()
        if not stripped:
            return True

        for pattern in _LOW_QUALITY_RE:
            if pattern.match(stripped):
                return True

        return False

    def filter_low_quality_lines(self, text: str) -> str:
        """过滤低质量行"""
        if not text:
            return text

        lines = text.split('\n')
        filtered = [line for line in lines if not self.is_low_quality_line(line)]
        return '\n'.join(filtered)

    def paragraph_similarity(self, p1: str, p2: str) -> float:
        """计算两个段落的相似度（简化版Jaccard）"""
        s1 = self._tokens_of(p1)
        s2 = self._tokens_of(p2)
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _tokens_of(text: str) -> set[str]:
        if not text:
            return set()
        return set(_RE_CN_TOKENS.findall(text))

    def deduplicate_paragraphs(self, text: str, similarity_threshold: float = 0.9) -> str:
        """
        去除文档内重复的段落

        Args:
            text: 输入文本
            similarity_threshold: 相似度阈值，超过视为重复
        """
        if not text:
            return text

        paragraphs = text.split('\n\n')
        unique_paragraphs: list[str] = []
        unique_token_cache: list[set[str]] = []  # 缓存每个 unique 段落的 token
        seen_normalized: set[str] = set()

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 精确指纹：去空格 + 去标点后的前 100 字符
            normalized = _RE_DEDUP_PUNCT.sub('', para)[:100]

            if normalized in seen_normalized:
                continue

            # 对较短段落（<50字符）做 Jaccard 近似匹配，防止微调型重复
            is_similar = False
            if len(normalized) < 50 and unique_paragraphs:
                para_tokens = self._tokens_of(para)
                for prev_tokens in unique_token_cache[-5:]:
                    if not prev_tokens or not para_tokens:
                        continue
                    inter = len(prev_tokens & para_tokens)
                    union = len(prev_tokens | para_tokens)
                    # ★ 短段 token 集极小 (常只有标签词, 如 "合同编号: A100" 只有
                    # {合同编号} 一个 token), Jaccard=1.0 会把 A100/A101 判为重复
                    # 而删掉合法编号行。要求至少 2 个共同 token 才近似判重。
                    if inter < 2:
                        continue
                    sim = inter / union if union else 0.0
                    if sim > similarity_threshold:
                        is_similar = True
                        break

            if not is_similar:
                unique_paragraphs.append(para)
                unique_token_cache.append(self._tokens_of(para))
                seen_normalized.add(normalized)

        return '\n\n'.join(unique_paragraphs)

    def normalize_headings(self, text: str) -> str:
        """规范化标题层级"""
        if not text:
            return text

        lines = text.split('\n')
        result = []
        prev_level = 0

        for line in lines:
            stripped = line.strip()

            heading_match = _RE_HEADING.match(stripped)
            if heading_match:
                current_level = len(heading_match.group(1))

                # 修复跳级问题
                if current_level > prev_level + 1:
                    current_level = prev_level + 1

                result.append('#' * current_level + ' ' + heading_match.group(2))
                prev_level = current_level
            else:
                result.append(line)
                if stripped and not stripped.startswith('#'):
                    prev_level = 0

        return '\n'.join(result)

    def normalize_links(self, text: str) -> str:
        """规范化链接"""
        if not text:
            return text

        def replace_url(match):
            url = match.group(0)
            domain = _RE_URL_DOMAIN.search(url)
            if domain:
                return f"[链接: {domain.group(1)}]({url})"
            return url

        return self.URL_PATTERN.sub(replace_url, text)

    def extract_keywords_simple(self, text: str, top_n: int = 10) -> List[str]:
        """简单关键词提取"""
        if not text:
            return []

        # 停用词（方法内构造一次）
        stop_words = _STOP_WORDS

        # 合并 update 省去中间 list
        word_counts: Counter = Counter()
        word_counts.update(
            w for w in _RE_CN_LONG.findall(text) if w not in stop_words
        )
        word_counts.update(
            w for w in _RE_EN_WORD.findall(text.lower()) if w not in stop_words
        )

        return [word for word, _ in word_counts.most_common(top_n)]

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取元数据
        """
        metadata = {
            'url_list': [],
            'email_list': [],
            'date_list': [],
            'keywords': [],
            'title': '',
            'word_count': len(text),
        }

        if not text:
            return metadata

        # 共享一次 text.split('\n') 给标题检测用
        lines = text.split('\n')

        # 提取 URL
        urls = self.URL_PATTERN.findall(text)
        metadata['url_list'] = list(set(urls))

        # 提取邮箱
        emails = self.EMAIL_PATTERN.findall(text)
        metadata['email_list'] = list(set(emails))

        # 提取日期 — DATE_PATTERN 无捕获组，findall 返回完整匹配字符串列表
        # 不能用 d[0]/d[1] 索引（那是取字符串的单字符），应直接使用完整匹配
        dates = self.DATE_PATTERN.findall(text)
        metadata['date_list'] = list(set(dates))

        # 提取标题（复用 lines）
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                metadata['title'] = line[2:].strip()
                break
            elif line.startswith('## '):
                metadata['title'] = line[3:].strip()
                break
        if not metadata['title'] and lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and not first_line.startswith('---'):
                metadata['title'] = first_line

        # 提取关键词
        metadata['keywords'] = self.extract_keywords_simple(text)

        return metadata

    def calculate_quality_score(self, text: str) -> Dict[str, Any]:
        """
        计算文本质量评分
        """
        if not text:
            return {
                'overall_score': 0,
                'readability': 0,
                'structure': 0,
                'content_richness': 0,
                'issues': ['文本为空']
            }

        issues = []

        # 共享 lines（避免重复 split）
        lines = text.split('\n')
        total_lines = len(lines) if lines else 0

        # 1. 可读性评分
        avg_line_length = sum(len(line) for line in lines) / total_lines if total_lines else 0

        readability = 100
        if avg_line_length > 200:
            readability -= 20
            issues.append('存在过长行')
        if avg_line_length < 10:
            readability -= 20
            issues.append('存在过短行')

        has_garbage, garbage_ratio, _ = self.detect_garbage(text)
        if has_garbage:
            readability -= int(garbage_ratio * 100)
            issues.append(f'包含乱码字符({garbage_ratio:.1%})')

        readability = max(0, readability)

        # 2. 结构评分（用预编译的标题正则）
        structure = 100
        prev_level = 0
        heading_count = sum(1 for _ in _RE_HEADING.finditer(text))
        if heading_count == 0 and len(text) > 1000:
            structure -= 30
            issues.append('长文档缺少标题结构')

        for match in _RE_HEADING.finditer(text):
            level = len(match.group(1))
            if level > prev_level + 1:
                structure -= 10
            prev_level = level

        structure = max(0, structure)

        # 3. 内容丰富度
        content_richness = 100

        total_chars = len(text)
        chinese_chars = len(_RE_CN_CHAR.findall(text))
        english_words = len(_RE_EN_WORD.findall(text))

        effective_ratio = (chinese_chars + english_words * 2) / total_chars if total_chars else 0
        if effective_ratio < 0.3:
            content_richness -= 30
            issues.append('有效内容比例低')

        paragraphs = text.split('\n\n')
        unique_paragraphs = set(p[:100] for p in paragraphs if len(p) > 50)
        if len(paragraphs) > 5 and len(unique_paragraphs) / len(paragraphs) < 0.8:
            content_richness -= 20
            issues.append('存在重复段落')

        content_richness = max(0, content_richness)

        # 综合评分
        overall = int((readability * 0.4 + structure * 0.3 + content_richness * 0.3))

        return {
            'overall_score': overall,
            'readability': readability,
            'structure': structure,
            'content_richness': content_richness,
            'issues': issues[:5]
        }

    def semantic_chunk(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        按句子/语义块切分长文本
        """
        if not text:
            return []

        paragraphs = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                continue

            sentences = _RE_SENT_END_GROUP.split(para)

            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences) and _RE_SENT_END_STRIP.match(sentences[i + 1]):
                    sentence = sentences[i] + sentences[i + 1]
                    i += 2
                else:
                    sentence = sentences[i]
                    i += 1

                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_size = len(sentence)

                if sentence_size > max_chunk_size:
                    if current_chunk:
                        chunks.append(''.join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    sub_sentences = _RE_CLAUSE_SPLIT.split(sentence)
                    j = 0
                    temp_chunk = []
                    temp_size = 0
                    while j < len(sub_sentences):
                        part = sub_sentences[j]
                        if j + 1 < len(sub_sentences):
                            part += sub_sentences[j + 1]
                            j += 2
                        else:
                            j += 1

                        part = part.strip()
                        if not part:
                            continue

                        if temp_size + len(part) > max_chunk_size and temp_chunk:
                            chunks.append(''.join(temp_chunk))
                            temp_chunk = [part]
                            temp_size = len(part)
                        else:
                            temp_chunk.append(part)
                            temp_size += len(part)

                    if temp_chunk:
                        chunks.append(''.join(temp_chunk))

                elif current_size + sentence_size > max_chunk_size and current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return self._merge_small_chunks(chunks, min_size=100)

    def _merge_small_chunks(self, chunks: List[str], min_size: int = 100) -> List[str]:
        """合并过小的块"""
        if not chunks:
            return []

        merged = []
        buffer = []
        buffer_size = 0

        for chunk in chunks:
            chunk_size = len(chunk)

            if chunk_size >= min_size:
                if buffer:
                    merged.append(''.join(buffer))
                    buffer = []
                    buffer_size = 0
                merged.append(chunk)
            else:
                buffer.append(chunk)
                buffer_size += chunk_size

                if buffer_size >= min_size:
                    merged.append(''.join(buffer))
                    buffer = []
                    buffer_size = 0

        if buffer:
            if merged:
                merged[-1] = merged[-1] + ''.join(buffer)
            else:
                merged.append(''.join(buffer))

        return merged

    def detect_table_blocks(self, text: str) -> List[Tuple[int, int]]:
        """检测文本中的表格块（Markdown 表格和 key-value 行格式）"""
        lines = text.split("\n")
        blocks: List[Tuple[int, int]] = []
        in_table = False
        start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Markdown 表格行
            is_md_table = stripped.startswith("|") and stripped.endswith("|")
            # 分隔行
            is_separator = bool(_RE_MD_TABLE_SEP.match(stripped))
            # key-value 行格式 (Header: Cell)
            is_kv = bool(_RE_KV_LINE.match(stripped))
            # 表格分隔线 ---
            is_divider = stripped == "---"

            if is_md_table or is_separator or (is_kv and not in_table):
                if not in_table:
                    in_table = True
                    start = i
            elif in_table and is_divider and is_kv:
                continue  # key-value 表格内的分隔行
            elif in_table:
                # 表格结束
                if i - start >= 2:  # 至少 2 行才算表格
                    blocks.append((start, i))
                in_table = False
            else:
                continue

        if in_table and len(lines) - start >= 2:
            blocks.append((start, len(lines)))

        return blocks

    def chunk_with_metadata(
        self,
        text: str,
        source: str = "",
        page: Optional[int] = None,
        abstract: str = "",
        document_name: str = "",
        element_indexes: Optional[List[List[int]]] = None,
        element_pages: Optional[List[int]] = None,
        element_bboxes: Optional[List[List[float]]] = None,
        element_types: Optional[List[str]] = None,
    ) -> Tuple[str, List[Chunk], Dict[str, Any]]:
        """
        完整处理流程 + 结构化 chunk 元数据

        Returns:
            (清洗后的文本, Chunk列表, 质量报告)
        """
        quality_report: Dict[str, Any] = {
            'original_length': len(text),
            'steps': [],
            'metadata': {},
            'quality_score': {}
        }

        # 1-7 同 process()
        text = self.remove_garbage(text)
        text = self.clean_text(text)
        quality_report['steps'].append('文本清洗完成')
        text = self.filter_content(text)
        quality_report['steps'].append('过滤特殊章节')
        text = self.filter_low_quality_lines(text)
        quality_report['steps'].append('过滤低质量行')
        text = self.deduplicate_paragraphs(text)
        quality_report['steps'].append('去重段落')
        text = self.normalize_headings(text)
        quality_report['steps'].append('规范化标题')
        text = self.normalize_links(text)
        quality_report['steps'].append('规范化链接')

        # 8. 提取元数据
        metadata = self.extract_metadata(text)
        quality_report['metadata'] = metadata

        # 9. 质量评分
        quality_report['quality_score'] = self.calculate_quality_score(text)

        # 10. 使用 ChunkSplitter 切片（替代原 semantic_chunk）
        # 检测表格块，标记为 TABLE 类型
        table_blocks = self.detect_table_blocks(text)
        lines = text.split("\n")
        # 构建逐行的 element 级数据（如果没有外部数据）
        if element_indexes is None:
            # 逐行构造 indexes/pages/types
            char_pos = 0
            element_indexes = []
            element_types_local = []
            for i, line in enumerate(lines):
                line_len = len(line) + 1  # +1 for the \n
                element_indexes.append([char_pos, char_pos + line_len])
                # 检查此行是否在表格块中
                in_table_block = any(
                    start <= i < end for start, end in table_blocks
                )
                if in_table_block:
                    element_types_local.append("Table")
                elif line.strip().startswith("#"):
                    element_types_local.append("Title")
                else:
                    element_types_local.append("text")
                char_pos += line_len

            if element_types is None:
                element_types = element_types_local

        try:
            chunks = self.chunk_splitter.split_documents(
                text=text,
                source=source,
                page=page,
                abstract=abstract or metadata.get('title', ''),
                document_name=document_name or metadata.get('title', ''),
                element_indexes=element_indexes,
                element_pages=element_pages,
                element_bboxes=element_bboxes,
                element_types=element_types,
            )
        except ChunkMaxLimitError as e:
            quality_report['error'] = str(e)
            # 回退到原 semantic_chunk
            plain_chunks = self.semantic_chunk(text, max_chunk_size=self.chunk_size)
            chunks = [
                Chunk(
                    text=c,
                    metadata=ChunkMetadata(
                        chunk_index=i,
                        source=source,
                        page=page,
                        chunk_type=ChunkType.TEXT,
                        abstract=abstract or metadata.get('title', ''),
                        document_name=document_name or metadata.get('title', ''),
                    ),
                )
                for i, c in enumerate(plain_chunks)
            ]

        quality_report['chunks_count'] = len(chunks)
        quality_report['final_length'] = len(text)

        return text, chunks, quality_report

    def process(self, text: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        完整处理流程

        Returns:
            (清洗后的文本, 语义块列表, 质量报告)
        """
        quality_report = {
            'original_length': len(text),
            'steps': [],
            'metadata': {},
            'quality_score': {}
        }

        # 1. 移除乱码
        text = self.remove_garbage(text)
        has_garbage, garbage_ratio, _ = self.detect_garbage(text)
        if has_garbage:
            quality_report['steps'].append(f'移除乱码: {garbage_ratio:.1%}')

        # 2. 清洗文本
        text = self.clean_text(text)
        quality_report['steps'].append('文本清洗完成')

        # 3. 过滤特殊章节
        text = self.filter_content(text)
        quality_report['steps'].append('过滤特殊章节')

        # 4. 过滤低质量行
        text = self.filter_low_quality_lines(text)
        quality_report['steps'].append('过滤低质量行')

        # 5. 去重
        text = self.deduplicate_paragraphs(text)
        quality_report['steps'].append('去重段落')

        # 6. 规范化标题
        text = self.normalize_headings(text)
        quality_report['steps'].append('规范化标题')

        # 7. 规范化链接
        text = self.normalize_links(text)
        quality_report['steps'].append('规范化链接')

        # 8. 提取元数据
        metadata = self.extract_metadata(text)
        quality_report['metadata'] = metadata

        # 9. 质量评分
        quality_report['quality_score'] = self.calculate_quality_score(text)

        # 10. 语义切分
        chunks = self.semantic_chunk(text, max_chunk_size=self.chunk_size)
        quality_report['chunks_count'] = len(chunks)
        quality_report['final_length'] = len(text)

        return text, chunks, quality_report


def clean_text(text: str, config: Optional[Dict[str, Any]] = None) -> str:
    """便捷函数：清洗文本"""
    cleaner = TextCleaner(config)
    return cleaner.clean_text(text)


def process_text(text: str, config: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str], Dict[str, Any]]:
    """便捷函数：完整处理"""
    cleaner = TextCleaner(config)
    return cleaner.process(text)
