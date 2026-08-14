"""DocScan 文档扫描入库服务.

新设计 (2026-06-14 重构):
  上传文件 → 保存到 DocScan 目录 (不自动处理)
  预处理 → X2MD 转换 (MD + .parents.json + .children.json)
  生成向量 → 读取切片 → 批量嵌入 → 向量值写入 JSON 带 /ARAG-begin//ARAG-end/ 标记
  比对 → 从 JSON 读取向量 → 层级比对(摘要→父块→子块) → 相似度写入 /ARAG-end/ 后

层级比对策略:
  1. 先比对摘要向量 → 如果超过阈值 → 整篇文章标记为高度相似, 不再逐块比对
  2. 再比对父块向量 → 如果父块超过阈值 → 该父块的所有子块标记为高度相似, 不再比对
  3. 最后比对剩余子块向量 → 逐一与三级库比对

向量存储格式 (在 .parents.json / .children.json 中):
  嵌入后: {"text":"...", "metadata":{...}, "vector": "/ARAG-begin/[0.123,...]/ARAG-end/"}
  比对后: {"text":"...", "metadata":{...}, "vector": "/ARAG-begin/[0.123,...]/ARAG-end/public:0.85,restricted:0.72,confidential:0.91"}
  跳过块: {"text":"...", "metadata":{...}, "vector": "/ARAG-begin/[0.123,...]/ARAG-end/SKIPPED:高度相似(public:0.92)"}
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.vector_db import vector_db_service
from app.services.reranker import reranker
from app.services.chinese_norm import normalize_for_ngram


# 向量标记常量
VECTOR_BEGIN = "/ARAG-begin/"
VECTOR_END = "/ARAG-end/"

# 可上传/预处理的源文档扩展名 (与 endpoints 白名单一致, 用于识别"源文件" vs 派生产物)
SOURCE_FILE_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".html", ".md")

# bge-m3 不相关文本基线 (全项目统一)
BGE_M3_BASELINE = 0.37

# 文本重叠检测参数
NGRAM_SIZE = 8  # 8字符滑动窗口 (中文约 4-6 个汉字, 足够识别短句重复)


# ---------------------------------------------------------------------------
# 层级比对配置 (P2-7: 收敛魔法数, 改阈值不再要全文搜索)
# ---------------------------------------------------------------------------

class DocScanConfig:
    """DocScan 层级比对/相似度合成的所有阈值与权重.

    全部以模块级常量形式暴露, 调参时改这一处即可。如未来要做成
    可热更新, 把这些字段挪进 settings.* 并加 SettingsService 分支即可。
    """

    # 跳过下级的相似度门槛: 摘要/父块达到此值, 整篇/该父块的所有子块直接 skip
    # ★ 0.92 对应 raw≈0.95 (近乎同一篇), 不再因为"主题相同"就整篇判高 —
    #   旧值 0.8 (raw≈0.77) 在 bge-m3 上是同领域常态, 是大量假阳性的根因 (A1)
    SKIP_THRESHOLD: float = 0.92

    # 摘要级跳过的 n-gram 核验门槛: 摘要 sim 高但库内 longest_run 占比 < 此值时
    # 拒绝整篇跳过, 让父块/子块逐一比对决定 (避免 LLM 摘要泛化导致的假阳性)
    ABSTRACT_SKIP_NGRAM_GUARD: float = 0.4

    # UI 命中片段展示门槛: 低于此值的匹配不返回给前端 (避免噪声)
    MATCH_DISPLAY_THRESHOLD: float = 0.6

    # combined_similarity 段落抄袭判定 (单文档内连续命中字符数, 用 per-doc 索引)
    LONG_RUN_HIGH: int = 80           # ≥80 字 + 占 query 70%+ → 段落级抄袭通道, 0.85+
    LONG_RUN_LOW: int = 50            # ≥50 字 → 计入 run_strength 信号


def _atomic_write_json(path: Path, data: Any) -> None:
    """原子写入 JSON: 先写 .tmp 再 os.replace, 避免 docscan_compare 长流程
    被并发写入或中断时撕裂 JSON 文件 (P2-6).

    os.replace 在 POSIX 上是原子操作; tmp 文件落在同目录确保跨设备 rename 不会失败。
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_text(text: str) -> str:
    """归一化文本: 去除所有空白字符 + 繁→简 (B3).

    用于 n-gram 比对前的文本归一化:
    - 消除 X2MD 转换不同版本之间的空白差异 (PDF 解析换行位置不同)
    - 消除繁简差异: "形勢變化的反應" 与 "形势变化的反应" n-gram 通道能匹配
    简繁转换由 chinese_norm.normalize_for_ngram 提供, OpenCC 优先, fallback 到
    内置高频映射 (~200 字), 没装 OpenCC 也不至于 100% 失效。
    """
    return normalize_for_ngram(text)


def text_ngram_overlap(query: str, target: str, n: int = NGRAM_SIZE) -> float:
    """计算 query 文本与 target 文本的字符级 n-gram 重叠率.

    返回: query 中出现在 target 里的 n-gram 占比 (0~1)
    用于检测小段落抄袭: 即使整段不相似, 只要有连续 n 字相同就能被识别。
    ★ 比对前先归一化空白字符, 消除 X2MD 转换版本间的空白差异。
    """
    query = _normalize_text(query)
    target = _normalize_text(target)
    if not query or not target or len(query) < n:
        return 1.0 if query and target and query in target else 0.0

    query_grams = set()
    for i in range(len(query) - n + 1):
        query_grams.add(query[i:i + n])

    if not query_grams:
        return 0.0

    matched = sum(1 for g in query_grams if g in target)
    return matched / len(query_grams)


def build_ngram_index(texts: List[str], n: int = NGRAM_SIZE) -> set:
    """为多个 target 文本构建 n-gram 全局集合 (合并所有库的 n-gram).

    返回: 一个集合, 包含所有 target 文本中出现过的所有 n-gram。
    用于全库扫描 n-gram 重叠 — query 的 n-gram 在此集合中查找命中即可。
    ★ 索引构建时先归一化空白字符, 与 query 端保持一致。
    """
    all_grams = set()
    for text in texts:
        normalized = _normalize_text(text)
        if not normalized or len(normalized) < n:
            continue
        for i in range(len(normalized) - n + 1):
            all_grams.add(normalized[i:i + n])
    return all_grams


def build_per_doc_ngram_index(
    texts: List[str], doc_ids: List[str], n: int = NGRAM_SIZE,
) -> Dict[str, set]:
    """按文档分桶的 n-gram 索引 — 用于 longest_matching_run 严格限定在单文档内.

    全库聚合的 n-gram 索引会让"连续 50 字匹配"假阳性: 这 50 个 8-gram
    可能散落在不同文档里。按 doc 分桶后, longest_matching_run 必须在
    某一篇 target 文档内找到连续命中段, 才算真实的"段落抄袭"。

    Args:
        texts: 库中各 chunk 的原文 (与 doc_ids 一一对应)
        doc_ids: 每条 text 所属的源文档名 (来自 metadata.document_name / source)
                 同一文档内的多条 chunk 会合并到同一个 set
    """
    per_doc: Dict[str, set] = {}
    for text, doc_id in zip(texts, doc_ids):
        if not doc_id:
            continue
        normalized = _normalize_text(text)
        if not normalized or len(normalized) < n:
            continue
        bucket = per_doc.setdefault(doc_id, set())
        for i in range(len(normalized) - n + 1):
            bucket.add(normalized[i:i + n])
    return per_doc


def query_ngram_overlap_against_index(query: str, ngram_index: set, n: int = NGRAM_SIZE) -> float:
    """计算 query 与全库 n-gram 集合的重叠率 — 全库扫描专用."""
    query = _normalize_text(query)
    if not query or len(query) < n:
        return 0.0
    query_grams = set()
    for i in range(len(query) - n + 1):
        query_grams.add(query[i:i + n])
    if not query_grams:
        return 0.0
    matched = sum(1 for g in query_grams if g in ngram_index)
    return matched / len(query_grams)


def longest_matching_run(query: str, ngram_index: set, n: int = NGRAM_SIZE) -> int:
    """找出 query 中"最长连续匹配段"的字符长度.

    算法: 标记 query 每个位置 i 的 8-gram 是否在库中,
    找最长连续命中段 (run), 返回其字符长度。
    ★ 比对前先归一化空白字符。
    ⚠️ 注意: 此函数对全库聚合索引会产生"漂浮匹配"假阳性 —
    当索引来自多个文档合并时, 连续命中可能跨文档拼接出来。
    若要严格限定在单文档内, 用 longest_matching_run_per_doc()。
    """
    query = _normalize_text(query)
    if not query or len(query) < n or not ngram_index:
        return 0

    hits = [(query[i:i + n] in ngram_index) for i in range(len(query) - n + 1)]
    if not hits:
        return 0

    max_run = 0
    current = 0
    for h in hits:
        if h:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0

    return max_run + n - 1 if max_run > 0 else 0


def longest_matching_run_per_doc(
    query: str, per_doc_index: Dict[str, set], n: int = NGRAM_SIZE,
) -> Tuple[int, str]:
    """在按文档分桶的 n-gram 索引上查"单文档内最长连续匹配段".

    返回 (max_run_chars, source_doc_name)。在每篇 target 文档的 n-gram set
    上独立计算 longest run, 取所有文档中最大的那个。这样"连续 N 字匹配"
    保证来自同一篇 target — 排除"漂浮匹配"假阳性。
    """
    if not per_doc_index:
        return 0, ""
    best_run = 0
    best_doc = ""
    for doc_id, idx in per_doc_index.items():
        run = longest_matching_run(query, idx, n)
        if run > best_run:
            best_run = run
            best_doc = doc_id
    return best_run, best_doc


def _pair_text_signals(query: str, candidate_text: str, n: int = NGRAM_SIZE) -> Tuple[float, int]:
    """逐对计算 (text_overlap, longest_run) — query 在单个 candidate_text 上的字面信号.

    A5: 取代库级聚合的 text_overlap/longest_run, 让每个 (query, candidate) 对
    有自己独立的字面相似度信号; 防止某个候选向量近但文字完全不重合的"语义噪声"
    被库级 text_overlap 替它撑分。
    """
    if not candidate_text:
        return 0.0, 0
    norm_cand = _normalize_text(candidate_text)
    if not norm_cand or len(norm_cand) < n:
        return 0.0, 0
    cand_grams = set()
    for i in range(len(norm_cand) - n + 1):
        cand_grams.add(norm_cand[i:i + n])
    overlap = text_ngram_overlap(query, candidate_text, n)
    run = longest_matching_run(query, cand_grams, n)
    return overlap, run


def _fuse_with_reranker(pair_combined: float, rerank_score: Optional[float]) -> float:
    """B2: 把 cross-encoder reranker 分数与启发式 combined_similarity 融合.

    Cross-encoder 分数已经是一个高质量相关性信号 (sigmoid 后 0~1, 训练目标
    就是"相关 vs 不相关"), 可以直接信任:
    - rerank ≥ 0.6: 强相关, rerank 主导 (0.7 权重) + combined 字面补强
    - rerank < 0.2: 强不相关, 即使 combined 高也大概率是 boilerplate, 压到 ≤ 0.4
    - 0.2~0.6: 不确定, 加权 50/50

    若 rerank_score is None (reranker 未启用 / 加载失败), 直接返回 pair_combined,
    保证启发式路径完全可用。
    """
    if rerank_score is None:
        return pair_combined
    if rerank_score >= 0.6:
        return min(1.0, 0.7 * rerank_score + 0.3 * pair_combined)
    if rerank_score < 0.2:
        return min(pair_combined, 0.4)
    return 0.5 * rerank_score + 0.5 * pair_combined


def combined_similarity(vec_sim: float, text_overlap: float, longest_run: int = 0, query_len: int = 0) -> float:
    """综合向量相似度、文本重叠率、最长连续匹配段 — 多通道分级.

    设计原则:
    1. **段落级抄袭通道**: 单文档内连续 ≥ LONG_RUN_HIGH(80) 字, 且占 query ≥ 70%
       → 真实整段抄袭, 0.85+ 起步, 配合 run_strength 上调
    2. **强语义通道**: vec_sim ≥ 0.7 直接信任向量 (高维语义已经接近一致)
       → 以 vec_sim 为基础, 文本信号微调
    3. **中等向量通道**: vec_sim 0.4~0.7, 需要其他信号佐证, 加权融合 0.55/0.3/0.15
    4. **低向量通道**: vec_sim < 0.4, 即使有文本重叠也限制在 0.55 以下 (避免
       boilerplate / 漂浮匹配把不相关文档判成高相似)

    longest_run 应来自 longest_matching_run_per_doc (单文档内), 否则会引入漂浮假阳性。
    """
    # run 强度: 仅当 ≥ LONG_RUN_LOW 才记分, 用占 query 的比例衡量
    run_strength = 0.0
    if query_len > 0 and longest_run >= DocScanConfig.LONG_RUN_LOW:
        run_strength = min(1.0, longest_run / query_len)

    # ★ 通道 1: 段落级抄袭 (单文档连续 ≥80 字, 占 query ≥70%)
    if (query_len > 0
            and longest_run >= DocScanConfig.LONG_RUN_HIGH
            and longest_run / query_len >= 0.7):
        return min(1.0, max(0.85, 0.7 + run_strength * 0.3, vec_sim))

    # ★ 通道 2: 强语义 — vec_sim ≥ 0.7 直接信任
    if vec_sim >= 0.7:
        # 文本信号最多上调 0.1 (vec 已经够强, 不需要太多支持)
        text_bonus = min(0.1, (text_overlap + run_strength) * 0.1)
        return min(1.0, vec_sim + text_bonus)

    # ★ 通道 3+4 合并: 加权融合 + 按 vec_sim 强度动态调整上限
    # A2 修正: 之前通道 3/4 用不同公式导致 vec=0.49→0.44, vec=0.50→0.34 的非单调跳变。
    # 现在统一加权 (vec 主导), 仅在 vec_sim < 0.5 (低向量, 怀疑 boilerplate) 时
    # 把上限压到 0.5 (低于 MATCH_DISPLAY_THRESHOLD 0.6, UI 不显示)。
    weighted = 0.55 * vec_sim + 0.3 * text_overlap + 0.15 * run_strength
    if vec_sim < 0.5:
        return min(weighted, 0.5)
    return weighted


def adjust_similarity(raw_similarity: float) -> float:
    """将原始 cosine similarity 线性重标定 - 供 scanner / dedup 使用.

    (raw - baseline)/(1 - baseline): raw=0.37(基线)->0, raw=1.0->1。
    DocScan 比对不使用本函数, 改用 adjust_similarity_docscan (√映射, 放大低位信号)。
    二者单调递增, 对"是否超阈值"判定等价, 但数值尺度不同, 故 scanner 阈值与
    DocScan 显示阈值 (display_threshold_for_level) 不可直接互换。
    """
    return max(0.0, (raw_similarity - BGE_M3_BASELINE) / (1.0 - BGE_M3_BASELINE))


def adjust_similarity_docscan(raw_similarity: float) -> float:
    """DocScan 专用相似度重标定 — √映射, 比线性更敏感.

    DocScan 比对场景与 scanner/dedup 不同:
    - scanner/dedup: 同主题文档在知识库中是正常的, 0.50 raw 只需标为 0.21 "弱相关"
    - DocScan: 待检文件与库中文档有任何语义相似都应引起警觉,
      0.50 raw (含片段引用) 应标为 ≥0.45 "可疑"

    √映射公式:  sqrt( max(0, (raw - baseline) / (1 - baseline)) )
    效果:
      raw 0.45 (含片段) → √0.13 = 0.36 (线性 0.13, 提升 2.8x)
      raw 0.50 (含片段) → √0.21 = 0.46 (线性 0.21, 提升 2.2x)
      raw 0.65 (大段相同) → √0.44 = 0.67 (线性 0.44, 提升 1.5x)
      raw 1.00 (完全相同) → √1.00 = 1.00 (不变)
      raw 0.37 (无关) → 0.00 (不变)

    低位信号被放大, 高位信号保持不变, 0.37 基线仍然归零.
    """
    t = max(0.0, (raw_similarity - BGE_M3_BASELINE) / (1.0 - BGE_M3_BASELINE))
    return math.sqrt(t)


def display_threshold_for_level(level: str) -> float:
    """返回某密级命中的显示阈值 (在 adjust_similarity_docscan 的 √ 尺度上).

    DocScan 原先对所有密级用统一的 DocScanConfig.MATCH_DISPLAY_THRESHOLD (0.6),
    完全忽略用户在 runtime_settings 配的 per-level 阈值 (SIMILARITY_THRESHOLD_*),
    导致"设置"页调灵敏度对文件扫描无效。现按密级取 runtime 阈值并换算到 √ 尺度:
    因 adjust_similarity_docscan 单调递增, m["sim"] >= 该值 等价于 raw_sim >= 原始阈值,
    即严格按用户配置的相似度门槛过滤, 与 scanner (线性, 同样保序) 的判定一致。

    public 非敏感密级, 无 runtime 阈值, 回退到 MATCH_DISPLAY_THRESHOLD。
    """
    if level == "confidential":
        t = getattr(settings, "SIMILARITY_THRESHOLD_CONFIDENTIAL", None)
    elif level == "restricted":
        t = getattr(settings, "SIMILARITY_THRESHOLD_RESTRICTED", None)
    else:
        t = None
    if t is None:
        return DocScanConfig.MATCH_DISPLAY_THRESHOLD
    return adjust_similarity_docscan(float(t))


def format_vector(values: List[float]) -> str:
    """将向量值格式化为 /ARAG-begin/[...]/ARAG-end/ 标记字符串."""
    # 保留足够精度 (float32 约 7 位有效数字)
    formatted = "[" + ",".join(f"{v:.7f}" for v in values) + "]"
    return VECTOR_BEGIN + formatted + VECTOR_END


def parse_vector(vector_str: str) -> Optional[List[float]]:
    """从标记字符串中解析向量值.

    格式: /ARAG-begin/[0.123,...]/ARAG-end/ 或
          /ARAG-begin/[0.123,...]/ARAG-end/public:0.85,...

    返回: 向量值列表, 或 None (无标记或解析失败)
    """
    if not vector_str or VECTOR_BEGIN not in vector_str:
        return None
    if VECTOR_END not in vector_str:
        return None
    begin_idx = vector_str.index(VECTOR_BEGIN) + len(VECTOR_BEGIN)
    end_idx = vector_str.index(VECTOR_END)
    array_str = vector_str[begin_idx:end_idx]
    try:
        values = json.loads(array_str)
        return [float(v) for v in values]
    except (json.JSONDecodeError, ValueError):
        return None


def parse_similarity_after_end(vector_str: str) -> Optional[Dict[str, float]]:
    """从 /ARAG-end/ 后面解析相似度.

    格式: /ARAG-end/public:0.85,restricted:0.72,confidential:0.91
    或:   /ARAG-end/SKIPPED:高度相似(public:0.92)

    返回: {"public": 0.85, "restricted": 0.72, "confidential": 0.91} 或 None
    """
    if not vector_str or VECTOR_END not in vector_str:
        return None
    after_end = vector_str[vector_str.index(VECTOR_END) + len(VECTOR_END):]
    if not after_end:
        return None

    # 处理 SKIPPED 标记
    if after_end.startswith("SKIPPED:"):
        # SKIPPED:高度相似(public:0.92)
        detail = after_end[len("SKIPPED:"):]
        return _parse_level_similarities(detail)

    return _parse_level_similarities(after_end)


def _parse_level_similarities(s: str) -> Dict[str, float]:
    """解析 public:0.85,restricted:0.72,confidential:0.91 格式."""
    result = {}
    # 可能包裹在 中文标签(...) 中
    # 先去掉外层标签如 "高度相似(...)"
    if '(' in s and s.endswith(')'):
        s = s[s.index('(') + 1:-1]
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            key, val = part.split(":", 1)
            try:
                result[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return result


def write_similarity_to_vector(vector_str: str, level_sims: Dict[str, float]) -> str:
    """在 /ARAG-end/ 后追加相似度值.

    输入: /ARAG-begin/[...]/ARAG-end/
    输出: /ARAG-begin/[...]/ARAG-end/public:0.85,restricted:0.72,confidential:0.91
    """
    if VECTOR_END not in vector_str:
        return vector_str  # 无标记, 不修改

    end_idx = vector_str.index(VECTOR_END) + len(VECTOR_END)
    before_end = vector_str[:end_idx]
    after_end = vector_str[end_idx:]

    # 如果已有相似度数据, 先清除
    if after_end and not after_end.startswith("SKIPPED:"):
        # 已有 level:sim 数据, 替换
        after_end = ""

    sim_parts = ",".join(f"{k}:{v:.4f}" for k, v in level_sims.items())
    return before_end + sim_parts


def write_skipped_to_vector(vector_str: str, reason: str, level_sims: Dict[str, float]) -> str:
    """在 /ARAG-end/ 后追加 SKIPPED 标记.

    输入: /ARAG-begin/[...]/ARAG-end/
    输出: /ARAG-begin/[...]/ARAG-end/SKIPPED:高度相似(public:0.92)
    """
    if VECTOR_END not in vector_str:
        return vector_str

    end_idx = vector_str.index(VECTOR_END) + len(VECTOR_END)
    before_end = vector_str[:end_idx]

    sim_parts = ",".join(f"{k}:{v:.4f}" for k, v in level_sims.items())
    return before_end + f"SKIPPED:{reason}({sim_parts})"


class DocScanService:
    """文档扫描入库服务 - 保存 → 预处理 → 生成向量 → 层级比对."""

    def __init__(self):
        self._embedder = None

    def _validate_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        safe = Path(filename).name
        docscan_dir = Path(settings.DOCSCAN_DIR)
        target = (docscan_dir / safe).resolve()
        # 用 relative_to 而非 startswith,避免前缀混淆(/tmp/DocScan vs /tmp/DocScan_evil)
        try:
            target.relative_to(docscan_dir.resolve())
        except ValueError:
            raise ValueError(f"Invalid filename: path traversal detected")
        return safe

    # ------------------------------------------------------------------
    # Embedder 管理
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """获取或创建 embedder (与 DataIngestionService 相同配置)."""
        if self._embedder is not None:
            return self._embedder

        from md2rag.embedder import create_embedder

        if settings.EMBEDDING_MODEL == "ollama-bge-m3":
            self._embedder = create_embedder(
                model_type="ollama",
                ollama_url=settings.OLLAMA_BASE_URL,
                ollama_model="bge-m3:latest",
                ollama_cache_path=os.path.join(settings.VECTOR_DB_DIR, "md2rag_embedding_cache.sqlite"),
            )
        elif settings.EMBEDDING_MODEL == "mps-bge-m3":
            self._embedder = create_embedder(
                model_type="mps",
                device="mps",
            )
        else:
            self._embedder = create_embedder(model_type="chromadb-default")

        return self._embedder

    # ------------------------------------------------------------------
    # 步骤 1: 预处理 (X2MD 转换)
    # ------------------------------------------------------------------

    def preprocess_file(self, filename: str, **x2md_kwargs) -> Dict:
        """对 DocScan 目录下的单个文件执行 X2MD 转换.

        Args:
            filename: DocScan 目录下的文件名 (不含路径)
            **x2md_kwargs: chunk_strategy, enable_llm, encoding, extract_tables,
                           chunk_size, chunk_overlap

        Returns:
            Dict: {"success": bool, "md_path": str, "message": str}
        """
        filename = self._validate_filename(filename)
        docscan_dir = Path(settings.DOCSCAN_DIR)
        source_path = docscan_dir / filename

        if not source_path.exists():
            return {"success": False, "message": f"文件不存在: {filename}"}

        return self.convert_file_to_md(str(source_path), **x2md_kwargs)

    def convert_file_to_md(
        self,
        source_path: str,
        chunk_strategy: str = 'parent-child',
        enable_llm: Optional[bool] = None,
        encoding: Optional[str] = None,
        extract_tables: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        ocr_lang: Optional[str] = None,
        enable_ocr: Optional[bool] = None,
    ) -> Dict:
        """将单个文件通过 X2MD 转换为 Markdown + 切片 JSON.

        Args:
            source_path: 源文件路径
            chunk_strategy: 切片策略
            其他参数同 PreprocessService (None → settings.* 默认值, 由 SettingsService 维护)

        Returns:
            Dict: {"success": bool, "md_path": str, "message": str}
        """
        # None → 用户在 Settings 页保存的默认值 (P1: 打通 Settings 联动)
        if enable_llm is None:
            enable_llm = settings.ENABLE_LLM_DEFAULT
        if extract_tables is None:
            extract_tables = settings.EXTRACT_TABLES_DEFAULT
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE_DEFAULT
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP_DEFAULT
        if ocr_lang is None:
            ocr_lang = settings.OCR_LANG_DEFAULT
        if enable_ocr is None:
            enable_ocr = settings.ENABLE_OCR_DEFAULT

        source_file = Path(source_path)
        if not source_file.exists():
            return {"success": False, "message": f"文件不存在: {source_path}"}

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        docscan_dir = Path(settings.DOCSCAN_DIR)
        output_path = docscan_dir / f"{source_file.stem}.md"

        x2md_src = base_dir / 'X2MD' / 'src'
        cmd = [
            sys.executable,
            str(x2md_src / 'x2md' / '_invoke.py'),
            str(source_file),
            '-o', str(output_path),
        ]

        if chunk_strategy == 'parent-child':
            cmd.append('--parent-child')
        elif chunk_strategy == 'chunk':
            cmd.append('--chunk')
        if not enable_llm:
            cmd.append('--no-llm')
        if encoding:
            cmd.extend(['--encoding', encoding])
        if not extract_tables:
            cmd.append('--no-tables')
        if chunk_size is not None:
            cmd.extend(['--chunk-size', str(chunk_size)])
        if chunk_overlap is not None:
            cmd.extend(['--chunk-overlap', str(chunk_overlap)])
        if ocr_lang:
            cmd.extend(['--ocr-lang', ocr_lang])
        if not enable_ocr:
            cmd.append('--no-ocr')

        env = os.environ.copy()
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(base_dir), timeout=300, env=env,
            )
            if result.returncode == 0:
                return {"success": True, "md_path": str(output_path), "message": "X2MD 转换成功"}
            else:
                return {"success": False, "md_path": None, "message": result.stderr or "X2MD 转换失败"}
        except subprocess.TimeoutExpired:
            return {"success": False, "md_path": None, "message": "X2MD 转换超时 (5分钟)"}
        except Exception as e:
            return {"success": False, "md_path": None, "message": str(e)}

    # ------------------------------------------------------------------
    # 步骤 2: 生成向量 (读取切片 → 嵌入 → 写入 JSON 带标记)
    # ------------------------------------------------------------------

    def embed_file_vectors(self, filename: str, progress_callback=None, cancel_event=None) -> Dict:
        """对预处理后的文件生成向量, 写入切片 JSON 文件带 /ARAG-begin//ARAG-end/ 标记.

        分批嵌入(每批 100), 逐批上报进度并检查取消; 父块/子块各段写盘后即落盘,
        取消或异常时已嵌入部分保留。供后台任务调用, 避免长文档 HTTP 超时
        (《鄧小平時代》港版 ~3719 chunk 串行嵌入约 15 分钟, 单次 HTTP 必超时)。
        """
        filename = self._validate_filename(filename)
        docscan_dir = Path(settings.DOCSCAN_DIR)
        stem = Path(filename).stem
        chunk_dir = docscan_dir

        parents_file = chunk_dir / f"{stem}.parents.json"
        children_file = chunk_dir / f"{stem}.children.json"

        if not parents_file.exists() and not children_file.exists():
            return {"status": "error", "message": "切片文件不存在, 请先执行预处理"}

        embedder = self._get_embedder()
        BATCH = 100

        # 预加载切片 + 统计总量(用于进度)
        parents_data = json.loads(parents_file.read_text(encoding="utf-8")) if parents_file.exists() else None
        children_data = json.loads(children_file.read_text(encoding="utf-8")) if children_file.exists() else None
        n_parents = len(parents_data) if parents_data else 0
        n_children = len(children_data) if children_data else 0
        total = n_parents + n_children
        if total == 0:
            return {"status": "success", "message": "无切片需嵌入",
                    "parents_embedded": 0, "children_embedded": 0, "abstract_embedded": False}

        p_done = 0
        c_done = 0
        abstract_embedded = False

        def _cancelled():
            return cancel_event is not None and cancel_event.is_set()

        def _report(msg):
            if progress_callback:
                done = p_done + c_done
                progress_callback({
                    "phase": "progress",
                    "progress": done / total,
                    "message": msg,
                    "embedded": done,
                    "total": total,
                })

        _report(f"开始嵌入, 共 {total} 个切片 (父 {n_parents} + 子 {n_children})")

        # --- 嵌入父块 (分批 + 进度 + 取消) ---
        if parents_data:
            texts = [item.get("text", "") for item in parents_data]
            all_embs = []
            for i in range(0, len(texts), BATCH):
                if _cancelled():
                    _atomic_write_json(parents_file, parents_data)
                    return {"status": "cancelled", "message": f"已取消 (父块 {p_done}/{n_parents})",
                            "parents_embedded": p_done, "children_embedded": 0, "abstract_embedded": False}
                batch = texts[i:i + BATCH]
                embs = embedder.embed(batch)
                all_embs.extend(embs)
                for j, emb in enumerate(embs):
                    parents_data[i + j]["vector"] = format_vector(emb)
                p_done += len(batch)
                _report(f"嵌入父块 {p_done}/{n_parents}")

            # 摘要向量: 前 N 个父块向量平均后 L2 归一化 (多锚点, 比单 LLM 摘要稳定;
            # 旧版仅嵌 LLM 摘要"这是关于X的报告"与所有 X 主题文档余弦都高, 触发整篇假阳性)
            if all_embs:
                n_anchors = min(3, len(all_embs))
                anchor_vecs = all_embs[:n_anchors]
                dim = len(anchor_vecs[0])
                summed = [0.0] * dim
                for v in anchor_vecs:
                    for k in range(dim):
                        summed[k] += v[k]
                avg = [x / n_anchors for x in summed]
                norm = math.sqrt(sum(x * x for x in avg))
                summary_emb = [x / norm for x in avg] if norm > 0 else avg
                parents_data[0]["summary_vector"] = format_vector(summary_emb)
                abstract_embedded = True

            _atomic_write_json(parents_file, parents_data)

        # --- 嵌入子块 (分批 + 进度 + 取消) ---
        if children_data and not _cancelled():
            texts = [item.get("text", "") for item in children_data]
            for i in range(0, len(texts), BATCH):
                if _cancelled():
                    _atomic_write_json(children_file, children_data)
                    return {"status": "cancelled", "message": f"已取消 (子块 {c_done}/{n_children})",
                            "parents_embedded": n_parents, "children_embedded": c_done, "abstract_embedded": abstract_embedded}
                batch = texts[i:i + BATCH]
                embs = embedder.embed(batch)
                for j, emb in enumerate(embs):
                    children_data[i + j]["vector"] = format_vector(emb)
                c_done += len(batch)
                _report(f"嵌入子块 {c_done}/{n_children}")
            _atomic_write_json(children_file, children_data)

        return {
            "status": "success",
            "message": f"向量生成完成: {n_parents} 父块 + {n_children} 子块",
            "parents_embedded": n_parents,
            "children_embedded": n_children,
            "abstract_embedded": abstract_embedded,
        }

    # ------------------------------------------------------------------
    # 步骤 3: 层级比对 (摘要->父块->子块, 逐级跳过)
    # ------------------------------------------------------------------

    def compare_file_hierarchical(self, filename: str, n_results: int = 10) -> Dict:
        """层级比对: 摘要→父块→子块, 超阈值则跳过下级.

        比对策略:
        1. 先比对摘要向量 → 如果 max(三密级相似度) > 阈值 → 整篇文章标记为高度相似
        2. 再比对父块向量 → 如果父块 > 阈值 → 该父块所有子块标记为高度相似
        3. 最后比对剩余子块向量 → 逐一与三级库比对

        比对结果写入 /ARAG-end/ 后:
        - 正常: /ARAG-end/public:0.85,restricted:0.72,confidential:0.91
        - 跳过: /ARAG-end/SKIPPED:高度相似(public:0.92)

        Args:
            filename: DocScan 目录下的源文件名
            n_results: 每个密级检索的 top-n 数

        Returns:
            Dict: 比对结果统计 + 逐块详情
        """
        try:
            return self._compare_file_hierarchical_impl(filename, n_results)
        finally:
            # ★ 比对结束必然释放: reranker 模型 (~1GB), embedder 引用, gc
            # compare 路径会触发 reranker._try_load 把 ~568M params 模型常驻;
            # 一旦比对结束就该卸载, 否则与 ingestion.embedder 共存时进程 RSS 翻倍。
            try:
                reranker.release()
            except Exception:
                pass
            self.cleanup_memory()

    def _compare_file_hierarchical_impl(self, filename: str, n_results: int = 10) -> Dict:
        filename = self._validate_filename(filename)
        docscan_dir = Path(settings.DOCSCAN_DIR)
        stem = Path(filename).stem
        chunk_dir = docscan_dir

        parents_file = chunk_dir / f"{stem}.parents.json"
        children_file = chunk_dir / f"{stem}.children.json"

        if not parents_file.exists() and not children_file.exists():
            return {"status": "error", "message": "切片文件不存在"}

        # 层级比对阈值 (adjusted_similarity 尺度)
        skip_threshold = DocScanConfig.SKIP_THRESHOLD  # ≥80% 视为高度相似, 跳过下级

        # ★ 复用 vector_db_service 的共享 ChromaDB client,避免多客户端锁竞争
        client = vector_db_service.get_client()

        level_names = ["public", "restricted", "confidential"]
        level_collections = {}
        for level in level_names:
            coll_name = f"md2rag_{level}_child"
            try:
                level_collections[level] = client.get_collection(name=coll_name)
            except Exception:
                level_collections[level] = None

        # ★ 为每个密级构建 n-gram 索引 — 同时维护两份:
        #   - level_ngram_indices[level]: 全库聚合 set (用于 text_overlap 整体覆盖率)
        #   - level_per_doc_indices[level]: 按文档分桶 (用于 longest_matching_run 单文档内连续段)
        # 后者是治本: longest_run 必须在某一篇 target 内连续, 才不算"漂浮拼接"
        level_ngram_indices = {}
        level_per_doc_indices: Dict[str, Dict[str, set]] = {}
        for level in level_names:
            coll = level_collections.get(level)
            if coll is None or coll.count() == 0:
                level_ngram_indices[level] = set()
                level_per_doc_indices[level] = {}
                continue
            try:
                all_data = coll.get(include=["documents", "metadatas"])
                all_docs = all_data.get("documents") or []
                all_metas = all_data.get("metadatas") or []
                if hasattr(all_docs, 'tolist'):
                    all_docs = all_docs.tolist()
                if hasattr(all_metas, 'tolist'):
                    all_metas = all_metas.tolist()
                # 提取每条 chunk 的源文档名
                doc_ids = []
                for meta in all_metas:
                    meta = meta or {}
                    doc_ids.append(
                        meta.get("document_name")
                        or meta.get("source")
                        or "(unknown)"
                    )
                level_ngram_indices[level] = build_ngram_index(all_docs)
                level_per_doc_indices[level] = build_per_doc_ngram_index(all_docs, doc_ids)
                print(
                    f"[DEDUP] {level} n-gram index: {len(level_ngram_indices[level])} grams "
                    f"from {len(all_docs)} chunks across {len(level_per_doc_indices[level])} docs"
                )
            except Exception as e:
                print(f"[DEDUP] Failed to build n-gram index for {level}: {e}")
                level_ngram_indices[level] = set()
                level_per_doc_indices[level] = {}

        # 比对统计
        stats = {
            "abstract_skipped_all": False,
            "abstract_max_sim": {},
            "parents_total": 0,
            "parents_skipped": 0,
            "parents_compared": 0,
            "children_total": 0,
            "children_skipped_by_abstract": 0,
            "children_skipped_by_parent": 0,
            "children_compared": 0,
            "level_max_sim": {level: 0.0 for level in level_names},
        }

        # ★ 第 1 级: 比对摘要向量
        abstract_skip = False
        abstract_sims = {}
        abstract_best_match = {}  # level -> {"text", "doc", "sim"}
        abstract_text = ""

        if parents_file.exists():
            p_data = json.loads(parents_file.read_text(encoding="utf-8"))
            # 查找摘要向量
            summary_vec = None
            for item in p_data:
                sv = item.get("summary_vector", "")
                if sv and VECTOR_BEGIN in sv:
                    summary_vec = parse_vector(sv)
                    abstract_text = (item.get("metadata") or {}).get("abstract", "")
                    break

            if summary_vec is not None:
                for level in level_names:
                    coll = level_collections.get(level)
                    if coll is None or coll.count() == 0:
                        abstract_sims[level] = 0.0
                        continue
                    n = min(n_results, coll.count())
                    results = coll.query(
                        query_embeddings=[summary_vec],
                        n_results=n,
                        include=["documents", "metadatas", "distances"],
                    )
                    if results.get("distances") and results["distances"][0]:
                        distances = results["distances"][0]
                        documents = (results.get("documents") or [[]])[0]
                        metadatas = (results.get("metadatas") or [[]])[0]
                        min_idx = min(range(len(distances)), key=lambda k: distances[k])
                        min_dist = distances[min_idx]
                        raw_sim = max(0.0, min(1.0, 1.0 - min_dist / 2.0))
                        adj_sim = adjust_similarity_docscan(raw_sim)
                        abstract_sims[level] = adj_sim
                        # 记录该密级最佳匹配的文本与来源
                        if min_idx < len(documents):
                            best_doc_text = documents[min_idx] or ""
                            best_doc_meta = metadatas[min_idx] if min_idx < len(metadatas) else {}
                            best_doc_meta = best_doc_meta or {}
                            abstract_best_match[level] = {
                                "sim": adj_sim,
                                "matched_text": best_doc_text,
                                "matched_doc": (
                                    best_doc_meta.get("document_name")
                                    or best_doc_meta.get("source")
                                    or ""
                                ),
                            }
                    else:
                        abstract_sims[level] = 0.0

                max_abstract_sim = max(abstract_sims.values()) if abstract_sims else 0.0
                stats["abstract_max_sim"] = abstract_sims
                stats["level_max_sim"] = {
                    level: max(stats["level_max_sim"].get(level, 0.0), abstract_sims.get(level, 0.0))
                    for level in level_names
                }

                if max_abstract_sim >= skip_threshold:
                    # ★ A1: 跳过前做一次 n-gram 核验 — 摘要+前 2 个父块拼接, 在
                    # "摘要最高级"的 per-doc 索引内查 longest_run 占比, 太低则拒绝跳过。
                    # 这避免 LLM 泛化摘要导致的"主题相同但内容不同"误整篇跳过。
                    guard_query_parts = [abstract_text]
                    if parents_file.exists():
                        for item in p_data[:2]:
                            txt = item.get("text", "")
                            if txt:
                                guard_query_parts.append(txt)
                    guard_query = "\n".join(p for p in guard_query_parts if p)

                    # 取摘要相似度最高那一级的 per-doc 索引
                    top_level = max(abstract_sims, key=abstract_sims.get) if abstract_sims else None
                    guard_run = 0
                    guard_query_norm_len = len(_normalize_text(guard_query))
                    if top_level and guard_query_norm_len > 0:
                        guard_run, _ = longest_matching_run_per_doc(
                            guard_query, level_per_doc_indices.get(top_level, {})
                        )
                    guard_ratio = guard_run / max(guard_query_norm_len, 1)

                    if guard_ratio >= DocScanConfig.ABSTRACT_SKIP_NGRAM_GUARD:
                        abstract_skip = True
                        stats["abstract_skipped_all"] = True
                        stats["abstract_skip_guard_ratio"] = round(guard_ratio, 3)
                    else:
                        # 摘要可疑但 n-gram 不支持 → 拒绝整篇跳过, 让逐块比对决定
                        stats["abstract_skip_rejected"] = {
                            "max_abstract_sim": round(max_abstract_sim, 3),
                            "guard_ratio": round(guard_ratio, 3),
                            "reason": "摘要高相似但库内长连续匹配占比不足, 进入逐块比对",
                        }

        # ★ 如果摘要超过阈值 → 整篇文章标记为高度相似
        if abstract_skip:
            # 所有父块和子块标记为 SKIPPED:高度相似
            p_data: List[Dict[str, Any]] = []
            c_data: List[Dict[str, Any]] = []
            if parents_file.exists():
                p_data = json.loads(parents_file.read_text(encoding="utf-8"))
                for item in p_data:
                    if "vector" in item and VECTOR_BEGIN in item["vector"]:
                        item["vector"] = write_skipped_to_vector(
                            item["vector"], "高度相似(摘要超阈值)", abstract_sims
                        )
                _atomic_write_json(parents_file, p_data)

            if children_file.exists():
                c_data = json.loads(children_file.read_text(encoding="utf-8"))
                for item in c_data:
                    if "vector" in item and VECTOR_BEGIN in item["vector"]:
                        item["vector"] = write_skipped_to_vector(
                            item["vector"], "高度相似(摘要超阈值)", abstract_sims
                        )
                _atomic_write_json(children_file, c_data)

            stats["parents_total"] = len(p_data)
            stats["children_total"] = len(c_data)
            stats["children_skipped_by_abstract"] = stats["children_total"]

            # 摘要超阈值时, 把摘要级 best match 作为唯一一条命中记录返回
            abstract_matches: List[Dict[str, Any]] = []
            for level, m in abstract_best_match.items():
                if m["sim"] >= display_threshold_for_level(level):
                    abstract_matches.append({
                        "scanned_chunk_type": "abstract",
                        "scanned_chunk_index": -1,
                        "scanned_text": abstract_text,
                        "matched_text": m["matched_text"],
                        "matched_doc": m["matched_doc"],
                        "matched_level": level,
                        "similarity": round(m["sim"], 4),
                    })
            abstract_matches.sort(key=lambda x: x["similarity"], reverse=True)

            return {
                "status": "success",
                "message": "摘要相似度超过阈值, 整篇文章标记为高度相似",
                "skip_reason": "abstract",
                "stats": stats,
                "matches": abstract_matches,
            }

        # ★ 第 2 级: 比对父块向量 + 文本重叠检测
        parent_skip_doc_ids = set()  # 超阈值的父块 doc_id 集合
        parent_sims = {}  # parent_index -> {level: combined_sim}
        parent_best_match = {}  # parent_index -> {"sim", "level", "matched_text", "matched_doc"}

        if parents_file.exists():
            p_data = json.loads(parents_file.read_text(encoding="utf-8"))
            stats["parents_total"] = len(p_data)

            # 批量提取父块向量 + 文本
            parent_vectors = []
            parent_texts = []
            parent_indices = []
            for i, item in enumerate(p_data):
                v = item.get("vector", "")
                vec = parse_vector(v)
                if vec is not None:
                    parent_vectors.append(vec)
                    parent_texts.append(item.get("text", ""))
                    parent_indices.append(i)

            # 批量 ChromaDB 查询 — 同时取回 documents 用于文本重叠检测
            if parent_vectors:
                for level in level_names:
                    coll = level_collections.get(level)
                    if coll is None or coll.count() == 0:
                        for idx in parent_indices:
                            parent_sims.setdefault(idx, {})[level] = 0.0
                        continue
                    n = min(n_results, coll.count())
                    batch_results = coll.query(
                        query_embeddings=parent_vectors,
                        n_results=n,
                        include=["documents", "metadatas", "distances"],
                    )
                    # 全库 n-gram 索引 (确保即使向量检索漏掉, 文本相同也能识别)
                    ngram_idx = level_ngram_indices.get(level, set())
                    per_doc_idx = level_per_doc_indices.get(level, {})

                    if batch_results.get("distances"):
                        per_q_docs = batch_results.get("documents") or []
                        per_q_metas = batch_results.get("metadatas") or []
                        for qi, idx in enumerate(parent_indices):
                            distances = batch_results["distances"][qi] if qi < len(batch_results["distances"]) else []
                            docs_qi = per_q_docs[qi] if qi < len(per_q_docs) else []
                            metas_qi = per_q_metas[qi] if qi < len(per_q_metas) else []
                            query_text = parent_texts[qi]
                            ql = len(_normalize_text(query_text))

                            # B2: 一次性 reranker 评分本 query 的所有候选 (返回 None 表示未启用)
                            cand_texts_for_rr = [
                                (docs_qi[k] if k < len(docs_qi) else "") for k in range(len(distances))
                            ]
                            rerank_scores = reranker.rerank(query_text, cand_texts_for_rr) if cand_texts_for_rr else None

                            # ★ A5: 逐对 combined — 对 top-N 每个候选独立算字面信号 + 向量相似度 + reranker
                            best_pair_combined = 0.0
                            best_pair_text = ""
                            best_pair_doc = ""
                            for k in range(len(distances)):
                                cand_text = cand_texts_for_rr[k]
                                cand_meta = (metas_qi[k] if k < len(metas_qi) else {}) or {}
                                cand_doc_name = (
                                    cand_meta.get("document_name")
                                    or cand_meta.get("source")
                                    or ""
                                )
                                raw = max(0.0, min(1.0, 1.0 - distances[k] / 2.0))
                                pair_vec = adjust_similarity_docscan(raw)
                                pair_overlap, pair_run = _pair_text_signals(query_text, cand_text)
                                pair_combined = combined_similarity(pair_vec, pair_overlap, pair_run, ql)
                                rscore = rerank_scores[k] if rerank_scores and k < len(rerank_scores) else None
                                fused = _fuse_with_reranker(pair_combined, rscore)
                                if fused > best_pair_combined:
                                    best_pair_combined = fused
                                    best_pair_text = cand_text
                                    best_pair_doc = cand_doc_name

                            # 库级兜底信号 — 防 top-N 漏掉真实抄袭目标
                            global_overlap = query_ngram_overlap_against_index(query_text, ngram_idx)
                            global_run, global_run_doc = longest_matching_run_per_doc(query_text, per_doc_idx)
                            # 兜底通道用 vec_sim=0 — 字面信号若强能独立站得住; 否则 combined 给低分
                            global_combined = combined_similarity(0.0, global_overlap, global_run, ql)

                            combined = max(best_pair_combined, global_combined)
                            parent_sims.setdefault(idx, {})[level] = combined
                            stats["level_max_sim"][level] = max(
                                stats["level_max_sim"].get(level, 0.0), combined
                            )

                            # 跨密级 best match: 优先用对级胜出的候选, 否则用库级 run_doc
                            prev = parent_best_match.get(idx)
                            if combined > 0 and (prev is None or combined > prev["sim"]):
                                if best_pair_combined >= global_combined:
                                    parent_best_match[idx] = {
                                        "sim": combined, "level": level,
                                        "matched_text": best_pair_text,
                                        "matched_doc": best_pair_doc,
                                    }
                                else:
                                    parent_best_match[idx] = {
                                        "sim": combined, "level": level,
                                        "matched_text": "",  # 库级兜底无具体配对文本
                                        "matched_doc": global_run_doc,
                                    }

            # 判断每个父块是否超阈值
            for idx, sims in parent_sims.items():
                max_sim = max(sims.values()) if sims else 0.0
                if max_sim >= skip_threshold:
                    # 找到该父块的 doc_id
                    meta = p_data[idx].get("metadata", {})
                    doc_id = meta.get("doc_id", "")
                    if doc_id:
                        parent_skip_doc_ids.add(doc_id)
                    stats["parents_skipped"] += 1
                    # 写入 SKIPPED 标记
                    p_data[idx]["vector"] = write_skipped_to_vector(
                        p_data[idx]["vector"], "高度相似(父块超阈值)", sims
                    )
                else:
                    stats["parents_compared"] += 1
                    # 写入正常相似度
                    p_data[idx]["vector"] = write_similarity_to_vector(
                        p_data[idx]["vector"], sims
                    )

            _atomic_write_json(parents_file, p_data)

        # ★ 第 3 级: 比对子块向量 + 文本重叠检测 (跳过摘要超阈值和父块超阈值的子块)
        child_sims = {}  # child_index -> {level: combined_sim}
        child_best_match: Dict[int, Dict[str, Any]] = {}
        c_data: List[Dict[str, Any]] = []

        if children_file.exists():
            c_data = json.loads(children_file.read_text(encoding="utf-8"))
            stats["children_total"] = len(c_data)

            # 分类: 跳过的子块 vs 需比对的子块
            compare_indices = []
            compare_vectors = []
            compare_texts = []
            skip_indices = []

            for i, item in enumerate(c_data):
                v = item.get("vector", "")
                vec = parse_vector(v)
                if vec is None:
                    continue  # 无向量标记, 跳过

                meta = item.get("metadata", {})
                parent_doc_id = meta.get("parent_doc_id", "") or meta.get("doc_id", "")

                if parent_doc_id in parent_skip_doc_ids:
                    # 父块已超阈值 → 子块标记为高度相似
                    skip_indices.append(i)
                    stats["children_skipped_by_parent"] += 1
                else:
                    compare_indices.append(i)
                    compare_vectors.append(vec)
                    compare_texts.append(item.get("text", ""))

            # 对跳过的子块写入 SKIPPED 标记 (使用其父块的相似度)
            for i in skip_indices:
                meta = c_data[i].get("metadata", {})
                parent_doc_id = meta.get("parent_doc_id", "") or meta.get("doc_id", "")
                # 找到父块的相似度
                parent_sim = {}
                for idx, sims in parent_sims.items():
                    p_meta = p_data[idx].get("metadata", {}) if p_data else {}
                    if p_meta.get("doc_id") == parent_doc_id:
                        parent_sim = sims
                        break
                c_data[i]["vector"] = write_skipped_to_vector(
                    c_data[i]["vector"], "高度相似(父块超阈值)", parent_sim
                )

            # 批量比对剩余子块 — 向量相似度 + 全库 n-gram 文本重叠
            if compare_vectors:
                for level in level_names:
                    coll = level_collections.get(level)
                    if coll is None or coll.count() == 0:
                        for idx in compare_indices:
                            child_sims.setdefault(idx, {})[level] = 0.0
                        continue
                    n = min(n_results, coll.count())
                    batch_results = coll.query(
                        query_embeddings=compare_vectors,
                        n_results=n,
                        include=["documents", "metadatas", "distances"],
                    )
                    # 全库 n-gram 索引
                    ngram_idx = level_ngram_indices.get(level, set())
                    per_doc_idx = level_per_doc_indices.get(level, {})

                    if batch_results.get("distances"):
                        per_q_docs = batch_results.get("documents") or []
                        per_q_metas = batch_results.get("metadatas") or []
                        for qi, idx in enumerate(compare_indices):
                            distances = batch_results["distances"][qi] if qi < len(batch_results["distances"]) else []
                            docs_qi = per_q_docs[qi] if qi < len(per_q_docs) else []
                            metas_qi = per_q_metas[qi] if qi < len(per_q_metas) else []
                            query_text = compare_texts[qi]
                            ql = len(_normalize_text(query_text))

                            # B2: 一次性 reranker 评分
                            cand_texts_for_rr = [
                                (docs_qi[k] if k < len(docs_qi) else "") for k in range(len(distances))
                            ]
                            rerank_scores = reranker.rerank(query_text, cand_texts_for_rr) if cand_texts_for_rr else None

                            # ★ A5: 逐对 combined
                            best_pair_combined = 0.0
                            best_pair_text = ""
                            best_pair_doc = ""
                            for k in range(len(distances)):
                                cand_text = cand_texts_for_rr[k]
                                cand_meta = (metas_qi[k] if k < len(metas_qi) else {}) or {}
                                cand_doc_name = (
                                    cand_meta.get("document_name")
                                    or cand_meta.get("source")
                                    or ""
                                )
                                raw = max(0.0, min(1.0, 1.0 - distances[k] / 2.0))
                                pair_vec = adjust_similarity_docscan(raw)
                                pair_overlap, pair_run = _pair_text_signals(query_text, cand_text)
                                pair_combined = combined_similarity(pair_vec, pair_overlap, pair_run, ql)
                                rscore = rerank_scores[k] if rerank_scores and k < len(rerank_scores) else None
                                fused = _fuse_with_reranker(pair_combined, rscore)
                                if fused > best_pair_combined:
                                    best_pair_combined = fused
                                    best_pair_text = cand_text
                                    best_pair_doc = cand_doc_name

                            # 库级兜底
                            global_overlap = query_ngram_overlap_against_index(query_text, ngram_idx)
                            global_run, global_run_doc = longest_matching_run_per_doc(query_text, per_doc_idx)
                            global_combined = combined_similarity(0.0, global_overlap, global_run, ql)

                            combined = max(best_pair_combined, global_combined)
                            child_sims.setdefault(idx, {})[level] = combined
                            stats["level_max_sim"][level] = max(
                                stats["level_max_sim"].get(level, 0.0), combined
                            )

                            prev = child_best_match.get(idx)
                            if combined > 0 and (prev is None or combined > prev["sim"]):
                                if best_pair_combined >= global_combined:
                                    child_best_match[idx] = {
                                        "sim": combined, "level": level,
                                        "matched_text": best_pair_text,
                                        "matched_doc": best_pair_doc,
                                    }
                                else:
                                    child_best_match[idx] = {
                                        "sim": combined, "level": level,
                                        "matched_text": "",
                                        "matched_doc": global_run_doc,
                                    }

                stats["children_compared"] = len(compare_indices)

            # 写入子块相似度
            for idx, sims in child_sims.items():
                c_data[idx]["vector"] = write_similarity_to_vector(
                    c_data[idx]["vector"], sims
                )

            _atomic_write_json(children_file, c_data)

        # ─── 构建命中片段列表 (按相似度从高到低, 按密级用 runtime 阈值过滤) ───
        matches: List[Dict[str, Any]] = []

        # 摘要级命中 (即使未触发 skip, 0.6~0.8 区间仍有信息量)
        for level, m in abstract_best_match.items():
            if m["sim"] >= display_threshold_for_level(level):
                matches.append({
                    "scanned_chunk_type": "abstract",
                    "scanned_chunk_index": -1,
                    "scanned_text": abstract_text,
                    "matched_text": m["matched_text"],
                    "matched_doc": m["matched_doc"],
                    "matched_level": level,
                    "similarity": round(m["sim"], 4),
                })

        # 父块命中 (parent_best_match 仅在 parents_file 存在时才填充)
        if parents_file.exists() and p_data:
            for idx, m in parent_best_match.items():
                if m["sim"] >= display_threshold_for_level(m["level"]):
                    matches.append({
                        "scanned_chunk_type": "parent",
                        "scanned_chunk_index": idx,
                        "scanned_text": p_data[idx].get("text", ""),
                        "matched_text": m["matched_text"],
                        "matched_doc": m["matched_doc"],
                        "matched_level": m["level"],
                        "similarity": round(m["sim"], 4),
                    })

        # 子块命中 (child_best_match 仅在 children_file 存在时才填充)
        if children_file.exists() and c_data:
            for idx, m in child_best_match.items():
                if m["sim"] >= display_threshold_for_level(m["level"]):
                    matches.append({
                        "scanned_chunk_type": "child",
                        "scanned_chunk_index": idx,
                        "scanned_text": c_data[idx].get("text", ""),
                        "matched_text": m["matched_text"],
                        "matched_doc": m["matched_doc"],
                        "matched_level": m["level"],
                        "similarity": round(m["sim"], 4),
                    })

        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "message": f"层级比对完成: 摘要{'跳过' if abstract_skip else '已比对'}, "
                       f"{stats['parents_skipped']} 父块跳过, "
                       f"{stats['children_compared']} 子块逐一比对",
            "stats": stats,
            "matches": matches,
        }

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_file_status(self, filename: str) -> Dict:
        """获取文件在 DocScan 流程中的状态 (saved/preprocessed/embedded/compared)."""
        filename = self._validate_filename(filename)
        docscan_dir = Path(settings.DOCSCAN_DIR)
        stem = Path(filename).stem
        source_file = None
        for ext in SOURCE_FILE_EXTENSIONS:
            candidate = docscan_dir / f"{stem}{ext}"
            if candidate.exists():
                source_file = candidate
                break

        # 也检查带序号的文件名 (上传冲突时被改名为 stem_1.ext / stem_2.ext...)
        # P2-10: 严格用正则限定 "_<digit>+", 避免 'report' 误匹配 'report_v2' / 'reportX'
        if source_file is None:
            stem_re = re.compile(r"^" + re.escape(stem) + r"(_\d+)?$")
            for f in docscan_dir.iterdir():
                if f.is_file() and not f.name.startswith('.') and stem_re.match(f.stem):
                    source_file = f
                    break

        result = {
            "filename": filename,
            "saved": source_file is not None,
            "preprocessed": False,
            "embedded": False,
            "compared": False,
        }

        if source_file is None:
            return result

        actual_stem = source_file.stem
        # 与 embed_file_vectors / compare_file_hierarchical 一致: 切片 JSON 与 .md
        # 并列在 DocScan 目录下, 不能把 ".md 文件" 当目录拼接 (此处为此前漏修的第三处,
        # 导致 get_file_status 恒返回 preprocessed/embedded/compared=False)。
        chunk_dir = docscan_dir
        parents_file = chunk_dir / f"{actual_stem}.parents.json"

        if parents_file.exists():
            result["preprocessed"] = True
            # 检查是否有向量标记
            try:
                p_data = json.loads(parents_file.read_text(encoding="utf-8"))
                if p_data and VECTOR_BEGIN in (p_data[0].get("vector", "") or ""):
                    result["embedded"] = True
                    # 检查是否有相似度 (ARAG-end 后有数据)
                    vector_str = p_data[0].get("vector", "")
                    if VECTOR_END in vector_str:
                        after_end = vector_str[vector_str.index(VECTOR_END) + len(VECTOR_END):]
                        if after_end:
                            result["compared"] = True
            except Exception:
                pass

        return result

    def cleanup_memory(self) -> Dict:
        """释放 Embedder + Reranker 引用, 触发 GC + torch 缓存回收.

        DocScan 内部持有的两大块常驻内存:
        - self._embedder: bge-m3 (~2GB on MPS) 或 Ollama (轻量, 只持 LRU cache)
        - reranker 模块单例 (~568M params, ~1GB+ on MPS)
        三按钮流程结束 / 比对结束时必须一并卸载, 否则与 ingestion.embedder 共存
        时进程 RSS 显著翻倍。
        """
        # 嵌入器内部如果有 LRU cache 也顺手清, 防止 Ollama 路径下漏掉
        try:
            if self._embedder is not None and hasattr(self._embedder, "release"):
                self._embedder.release()
        except Exception:
            pass
        self._embedder = None
        try:
            reranker.release()
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass
        return {"status": "success", "message": "内存已释放"}

    def get_docscan_stats(self) -> Dict:
        """获取 DocScan 目录文件统计."""
        docscan_dir = Path(settings.DOCSCAN_DIR)
        if not docscan_dir.exists():
            return {"total_files": 0, "preprocessed": 0, "embedded": 0, "compared": 0}

        files = [f for f in docscan_dir.iterdir()
                 if f.is_file() and not f.name.startswith('.')]

        # 辅助切片 JSON 不计入 total_files (非文档), 也不作为文档计数
        AUX_SUFFIXES = (".parents.json", ".children.json")
        doc_files = [f for f in files if not f.name.endswith(AUX_SUFFIXES)]

        # 一次 iterdir 收集所有 .parents.json 的 stem (每个 = 一个已预处理文档)。
        # 避免逐文件调 get_file_status 各自 iterdir 搜源文件 (O(N²) 目录扫描)。
        # ★ 按 stem 去重遍历: report.pdf 与 report.md 同 stem, 只算一个文档
        # (旧实现遍历 files, 两者都命中 parents_stems, preprocessed/embedded/compared 翻倍)。
        parents_stems = {
            f.name[:-len(".parents.json")]
            for f in files
            if f.name.endswith(".parents.json")
        }
        # 源文件 stem 集合 (排除派生产物: .md/.parents.json/.children.json 均非源文件)。
        # ★ 孤儿切片 (源文件已被删除, 如 delete_docscan_file 只删源文件的历史遗留)
        # 不计入任何统计, 与 get_file_status 的"要求源文件存在"口径保持一致。
        source_stems = {
            f.name[:-len(f.suffix)]
            for f in files
            if f.suffix.lower() in SOURCE_FILE_EXTENSIONS
        }

        preprocessed = 0
        embedded = 0
        compared = 0
        for stem in parents_stems:
            if stem not in source_stems:
                continue  # 孤儿切片: 源文件已删, 跳过
            preprocessed += 1
            # embedded/compared 需读 parents.json[0]["vector"] 标记
            parents_file = docscan_dir / f"{stem}.parents.json"
            try:
                p_data = json.loads(parents_file.read_text(encoding="utf-8"))
                if p_data:
                    vector_str = p_data[0].get("vector", "") or ""
                    if VECTOR_BEGIN in vector_str:
                        embedded += 1
                        if VECTOR_END in vector_str:
                            after_end = vector_str[vector_str.index(VECTOR_END) + len(VECTOR_END):]
                            if after_end:
                                compared += 1
            except Exception:
                pass

        return {
            "total_files": len(doc_files),
            "preprocessed": preprocessed,
            "embedded": embedded,
            "compared": compared,
        }


docscan_service = DocScanService()
