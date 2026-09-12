from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        call_interval: float = 0.5,
        max_concurrent: int = 4,
        max_cache_size: int = 1000,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.call_interval = call_interval
        self.max_concurrent = max_concurrent
        self._cache: dict[str, Any] = {}
        self._max_cache_size = max_cache_size
        # 串行速率限制 + 并发支持：每个并发调用在锁内抢占下一个允许的时间点
        # 这样 max_concurrent 线程可以真正并行排队，而不是被全局 sleep 串行化
        self._next_call_time: float = 0.0
        self._rate_lock = threading.Lock()
        # _cache 被 ThreadPoolExecutor 并发读写, 需独立锁保护 (与 _rate_lock 职责不同)
        self._cache_lock = threading.Lock()

    def _throttle(self):
        """抢占下一个允许的调用时刻；只在已过期时才立即返回。

        旧实现用全局 sleep 让所有并发线程都阻塞；新实现让每个线程拿到自己的时间槽。
        time.time() 可能因 NTP / DST 跳变,导致 scheduled 爆炸性大 → 无限等待。
        time.monotonic() 不受系统时间调整影响。
        """
        with self._rate_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_call_time)
            # 推进下一个时间点（基于本调用开始时刻）
            self._next_call_time = scheduled + self.call_interval
        # 只在槽位还没到时睡眠
        if scheduled > now:
            time.sleep(scheduled - now)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict:
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def _add_to_cache(self, key: str, value: str) -> None:
        # _cache 被 ThreadPoolExecutor 并发读写 (llm_batch_strip_noise), 必须加锁:
        # check-then-act (len 判断 -> 删一半 -> 写入) 否则并发会过度删除。
        with self._cache_lock:
            if len(self._cache) >= self._max_cache_size:
                # FIFO: 丢弃最早插入的一半
                keys_to_discard = list(self._cache.keys())[:self._max_cache_size // 2]
                for k in keys_to_discard:
                    del self._cache[k]
            self._cache[key] = value

    def _cache_get(self, key: str) -> str | None:
        """线程安全的缓存读取 (与 _add_to_cache 共用 _cache_lock)。"""
        with self._cache_lock:
            return self._cache.get(key)

    def generate(self, prompt: str, system: str = "", use_cache: bool = True) -> str:
        cache_key = self._cache_key(prompt)
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        result = self._post("/api/generate", payload)
        response = result.get("response", "").strip()

        if use_cache:
            self._add_to_cache(cache_key, response)
        return response

    def chat(
        self,
        messages: list[dict[str, str]],
        use_cache: bool = True,
    ) -> str:
        cache_key = self._cache_key(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        result = self._post("/api/chat", payload)
        response = result.get("message", {}).get("content", "").strip()

        if use_cache:
            self._add_to_cache(cache_key, response)
        return response

    def generate_with_images(
        self,
        prompt: str,
        images: list[str],
        system: str = "",
        use_cache: bool = True,
    ) -> str:
        cache_key = self._cache_key(prompt + "".join(images))
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
        }
        if system:
            payload["system"] = system
        result = self._post("/api/generate", payload)
        response = result.get("response", "").strip()

        if use_cache:
            self._add_to_cache(cache_key, response)
        return response

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


_PROMPT_BATCH_NOISE = (
    "你是一个文档内容分析助手。请判断以下每个段落是否属于"
    "页眉、页脚、目录、修订信息、版本记录、文档属性、版权声明等非正文内容。\n\n"
    "对每个段落，回复该段落编号+YES（噪声）或NO（正文）。\n"
    "格式：编号:YES 或 编号:NO，每行一个，不要输出其他内容。\n\n"
    "{items}"
)

_PROMPT_EXTRACT_CONTACTS = (
    "你是一个信息提取助手。请从以下文本中识别并提取出：\n"
    "- 联系人姓名\n"
    "- 电话号码\n"
    "- 身份证号码\n"
    "- 地址信息\n\n"
    "请将识别到的每项信息单独一行输出，格式为：\n"
    "类型: 内容\n\n"
    "例如：\n"
    "联系人: 张三\n"
    "电话: 13800138000\n"
    "身份证: 110101199001011234\n"
    "地址: 北京市朝阳区某某路\n\n"
    "如果文本中没有以上信息，回复 NONE。\n"
    "只输出识别结果，不要输出其他内容。\n\n"
    "文本：\n"
    "{content}"
)

_NOISE_KEYWORDS = frozenset({
    # ★ 仅保留"无歧义的文档戳记"型关键词: 命中即噪声, 直接删, 不送 LLM。
    #   泛词 (http/https、目录、日期、状态、附件、草稿、制定/审核/批准 等) 一律
    #   移出: 它们常出现在合法正文 (官网链接、目录结构说明、修订状态行), 启发式
    #   命中会静默吞掉真内容。这些文本改由 LLM 判定 (保守路径), LLM 不可用时
    #   整个噪声剥离本就不执行。
    # 页眉页脚
    "页眉", "页脚", "header", "footer",
    # 目录/修订记录 (明确的无内容标记)
    "table of contents", "toc", "修订记录", "版本记录", "revision",
    "version history", "变更记录", "changelog",
    # 版权/免责
    "版权所有", "copyright", "all rights reserved", "版权声明",
    "免责声明", "disclaimer",
    # 密级/保密戳记
    "机密", "confidential", "保密", "绝密", "内部文件", "内部资料",
    "internal use", "内部使用",
    # 审批/责任人戳记
    "审批人", "审核人", "批准人", "拟定人", "复核人",
    # 文档/文件编号
    "文档编号", "文件编号",
    # 密级字段
    "密级", "密等",
    # 正本/副本/草稿 (文档状态戳记)
    "正本", "副本", "草稿", "draft",
    # 页码
    "第 1 页", "第1页", "第 2 页", "第2页",
    "page 1", "page 2", "共 1 页", "共1页", "共页",
    # 作者/部门戳记 (字段: 值 形态, 正文中罕见)
    "作者：", "作者:", "作者 ",
    "部门：", "部门:", "部门 ",
    # 文档状态戳记
    "文档状态",
})

_MIN_NOISE_LEN = 2
_MAX_NOISE_LEN = 200


def _is_likely_noise_by_heuristic(text: str) -> bool | None:
    stripped = text.strip()
    length = len(stripped)
    if length < _MIN_NOISE_LEN:
        return None
    if length > _MAX_NOISE_LEN:
        return False
    lower = stripped.lower()
    for kw in _NOISE_KEYWORDS:
        if kw in lower:
            return True
    return None


def llm_batch_strip_noise(
    client: OllamaClient,
    texts: list[str],
) -> set[int]:
    if not texts:
        return set()

    candidates: dict[int, str] = {}
    noise_indices: set[int] = set()

    for i, text in enumerate(texts):
        if not text.strip():
            continue
        heuristic = _is_likely_noise_by_heuristic(text)
        if heuristic is True:
            noise_indices.add(i)
            continue
        if heuristic is False:
            continue
        candidates[i] = text

    if not candidates:
        return noise_indices

    batch_size = 50
    items_list = list(candidates.items())

    def _process_batch(batch: list[tuple[int, str]]) -> list[tuple[int, int]]:
        items_text = ""
        local_index_map: dict[int, int] = {}

        for local_idx, (orig_idx, text) in enumerate(batch, start=1):
            items_text += f"[{local_idx}] {text[:200]}\n\n"
            local_index_map[local_idx] = orig_idx

        prompt = _PROMPT_BATCH_NOISE.format(items=items_text)
        results: list[tuple[int, int]] = []

        try:
            result = client.generate(prompt)
            for line in result.strip().split("\n"):
                line = line.strip()
                if ":" not in line:
                    continue
                parts = line.split(":", 1)
                try:
                    local_idx = int(parts[0].strip())
                except ValueError:
                    continue
                verdict = parts[1].strip().upper()
                # 模型幻觉的越界编号 (如 10 条 batch 里回 "99: NO"): 跳过该行,
                # 不要因单个坏行 KeyError 被外层 except 吞掉, 否则整个 batch 的
                # 判定结果全部丢弃 (含已解析的合法 YES), 噪声行会静默保留。
                if local_idx not in local_index_map:
                    continue
                results.append((local_index_map[local_idx], 1 if verdict.startswith("YES") else 0))
        except Exception:
            pass

        return results

    batches = [
        items_list[i:i + batch_size]
        for i in range(0, len(items_list), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=client.max_concurrent) as executor:
        futures = [executor.submit(_process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                for orig_idx, is_noise in future.result():
                    if is_noise:
                        noise_indices.add(orig_idx)
            except Exception:
                pass

    return noise_indices


def llm_strip_noise(client: OllamaClient, text: str) -> bool:
    if not text.strip():
        return False
    heuristic = _is_likely_noise_by_heuristic(text)
    if heuristic is not None:
        return heuristic
    result = llm_batch_strip_noise(client, [text])
    return 0 in result


def llm_extract_contacts(client: OllamaClient, text: str) -> list[str] | None:
    if not text.strip():
        return None
    prompt = _PROMPT_EXTRACT_CONTACTS.format(content=text[:1000])
    try:
        result = client.generate(prompt)
        if result.upper().strip() == "NONE":
            return None
        lines = [line.strip() for line in result.split("\n") if line.strip()]
        return lines if lines else None
    except Exception:
        return None
