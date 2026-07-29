"""繁简归一化 — 把繁体中文转简体, 统一 n-gram 比对的字面通道.

设计:
- OpenCC 可选: 装了 (pip install opencc-python-reimplemented 或 OpenCC) 自动启用,
  没装 fallback 到内置小规模映射 (~200 个高频繁体字), 不阻塞主流程。
- 仅用于 _normalize_text 的字面通道; **不改 chunk.text 原文**, 不影响嵌入。
- L2 归一化向量是字符无关的 (bge-m3 模型本身处理简繁), 所以这层只服务 n-gram。

为什么需要: bge-m3 嵌入端能容忍简繁差异, 但 docscan.py 的字面 ngram_overlap /
longest_run 是字符级精确匹配, "形勢"≠"形势" → n-gram 通道对简繁文档 100% 失效。

如何启用 OpenCC (推荐, ~200 行 C++ 加速 + 完整繁体覆盖):
    pip install opencc-python-reimplemented
或更高性能版本:
    pip install OpenCC
"""
from __future__ import annotations

from typing import Callable, Optional


_converter: Optional[Callable[[str], str]] = None
_load_attempted = False
_load_status = ""


def _try_load_opencc() -> Optional[Callable[[str], str]]:
    """尝试加载 OpenCC; 失败返回 None, fallback 到内置映射."""
    global _load_status

    # 1) opencc (C++) — 最快最全
    try:
        import opencc
        cc = opencc.OpenCC("t2s")
        _load_status = "opencc (C++)"
        return cc.convert
    except Exception:
        pass

    # 2) opencc-python-reimplemented (pure-Python, 兼容性好)
    try:
        from opencc_python_reimplemented import OpenCC as _OpenCCPy
        cc = _OpenCCPy("t2s")
        _load_status = "opencc-python-reimplemented"
        return cc.convert
    except Exception:
        pass

    # 3) zhconv (轻量备选)
    try:
        import zhconv
        _load_status = "zhconv"
        return lambda s: zhconv.convert(s, "zh-cn")
    except Exception:
        pass

    return None


# 内置 fallback: 仅覆盖最高频的繁简差异. 不全, 但比"100% 失效"好。
# 真正需要严格简繁支持的部署应安装 OpenCC。
_FALLBACK_T2S = str.maketrans({
    "勢": "势", "變": "变", "應": "应", "響": "响", "與": "与", "經": "经",
    "濟": "济", "進": "进", "個": "个", "時": "时", "間": "间", "問": "问",
    "題": "题", "後": "后", "對": "对", "點": "点", "從": "从", "現": "现",
    "實": "实", "認": "认", "識": "识", "說": "说", "話": "话", "讓": "让",
    "麼": "么", "這": "这", "裡": "里", "為": "为", "會": "会", "發": "发",
    "達": "达", "處": "处", "場": "场", "員": "员", "團": "团", "務": "务",
    "報": "报", "據": "据", "標": "标", "準": "准", "係": "系", "條": "条",
    "結": "结", "構": "构", "頭": "头", "腳": "脚", "馬": "马", "車": "车",
    "車": "车", "業": "业", "學": "学", "習": "习", "體": "体", "驗": "验",
    "戰": "战", "爭": "争", "選": "选", "擇": "择", "決": "决", "議": "议",
    "論": "论", "證": "证", "據": "据", "確": "确", "認": "认", "證": "证",
    "鐵": "铁", "錢": "钱", "銀": "银", "貨": "货", "資": "资", "貿": "贸",
    "邊": "边", "頁": "页", "顯": "显", "視": "视", "聽": "听", "覺": "觉",
    "歷": "历", "華": "华", "國": "国", "黨": "党", "軍": "军", "戶": "户",
    "産": "产", "業": "业", "醫": "医", "藥": "药", "藝": "艺", "術": "术",
    "計": "计", "劃": "划", "測": "测", "試": "试", "驗": "验", "備": "备",
    "勝": "胜", "負": "负", "贏": "赢", "輸": "输", "競": "竞",
    "極": "极", "難": "难", "易": "易", "簡": "简", "單": "单", "複": "复",
    "級": "级", "層": "层", "類": "类", "種": "种", "樣": "样",
    "屬": "属", "於": "于", "並": "并", "幾": "几", "麼": "么",
    "賽": "赛", "獎": "奖", "獲": "获", "勵": "励", "稱": "称",
    "養": "养", "傷": "伤", "勞": "劳", "動": "动", "靜": "静",
    "歡": "欢", "樂": "乐", "苦": "苦", "風": "风", "雨": "雨",
    "陽": "阳", "陰": "阴", "電": "电", "腦": "脑", "機": "机",
    "網": "网", "絡": "络", "頻": "频", "畫": "画", "圖": "图",
    "聲": "声", "響": "响", "鬱": "郁", "鬆": "松", "緊": "紧",
    "張": "张", "開": "开", "關": "关", "閉": "闭", "護": "护",
    "辦": "办", "幫": "帮", "別": "别", "屆": "届", "週": "周",
    "繞": "绕", "圍": "围", "圓": "圆", "圈": "圈", "團": "团",
    "釋": "释", "釋": "释", "兒": "儿", "親": "亲", "戀": "恋",
    "終": "终", "始": "始", "處": "处", "辭": "辞", "雙": "双",
    "獨": "独", "傳": "传", "統": "统", "黨": "党", "聯": "联",
    "盤": "盘", "縣": "县", "區": "区", "鄉": "乡", "鎮": "镇",
    "倆": "俩", "倆": "俩", "倆": "俩", "係": "系", "東": "东",
    "兩": "两", "個": "个",
})


def _fallback_t2s(text: str) -> str:
    """字符级 translate, 覆盖高频繁简差异."""
    return text.translate(_FALLBACK_T2S)


def t2s(text: str) -> str:
    """繁体 → 简体. 优先用 OpenCC, fallback 到内置映射."""
    global _converter, _load_attempted
    if not text:
        return text
    if not _load_attempted:
        _load_attempted = True
        _converter = _try_load_opencc()
        if _converter is not None:
            print(f"[CHINESE_NORM] using {_load_status}")
        else:
            print("[CHINESE_NORM] OpenCC not installed, using fallback dict (incomplete)")
    if _converter is not None:
        try:
            return _converter(text)
        except Exception:
            pass
    return _fallback_t2s(text)


def normalize_for_ngram(text: str) -> str:
    """字面通道归一化: 去空白 + 繁→简. 仅供 n-gram / longest_run 使用."""
    if not text:
        return ""
    # 去所有空白字符 (与原 _normalize_text 行为一致)
    no_space = ''.join(c for c in text if not c.isspace())
    return t2s(no_space)
