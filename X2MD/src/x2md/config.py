from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

from x2md.llm import OllamaClient

_DEFAULTS: dict[str, dict[str, Any]] = {
    "x2md": {
        "input_dir": "../from",
        "output_dir": "../to",
        "encoding": "utf-8",
    },
    "llm": {
        "enabled": "true",
        "model": "qwen2.5:7b-instruct",
        "base_url": "http://localhost:11434",
        "timeout": "120",
        "call_interval": "0.5",
        "max_concurrent": "4",
    },
    "pdf": {
        "extract_tables": "true",
        "page_separator": "\n\n---\n\n",
    },
    "docx": {
        "heading_style": "true",
    },
    "xlsx": {
        "include_all_sheets": "true",
        "default_sheet": "",
    },
    "pptx": {
        "slide_separator": "\n\n---\n\n",
    },
    "image": {
        "ocr_enabled": "true",
        "ocr_lang": "eng",
    },
    "vit": {
        "enabled": "true",
        "model": "openai/clip-vit-base-patch32",
        "device": "cpu",
        "local_files_only": "false",
        "mirror": "",
        "labels": "",
    },
    "html": {
        "encoding": "utf-8",
    },
    "chunk": {
        "chunk_size": "500",
        "chunk_overlap": "0",
        "separators": "\\n\\n,\\n,。,., ,",
        "separator_rule": "after,after,after,after,after,after",
        "max_chunk_limit": "10000",
        "inject_abstract": "true",
    },
}


def _coerce_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


# 仅这些字面量被无条件解释为 bool;数字字符串如 "0"/"1" 改由调用方按字段类型解释,
# 否则 default_sheet = 0 这种数字配置会被错误解读为 False。
_BOOL_LITERALS = frozenset({"true", "false", "yes", "no", "on", "off"})


def _coerce_value(value: str) -> bool | str:
    if value.lower() in _BOOL_LITERALS:
        return _coerce_bool(value)
    return value.replace("\\n", "\n")


def _to_str(value: Any) -> str:
    """配置值标准化为字符串,供 int()/float() 等类型化解释使用.

    避免链路 raw → _coerce_value → bool/str → int(True) == 1 这种悄无声息的失真。
    """
    if isinstance(value, bool):
        # bool 在 Python 中是 int 的子类,直接 int(value) 会得到 0/1。
        # 但配置里出现 True/False 时,我们希望调用方至少能拿到原字符串再决定怎么解释。
        return "true" if value else "false"
    return str(value)


class Config:
    def __init__(self, config_path: str | Path | None = None):
        self._parser = configparser.ConfigParser()
        self._load_defaults()
        if config_path is not None:
            self._load_file(Path(config_path))

    def _load_defaults(self):
        for section, options in _DEFAULTS.items():
            self._parser.add_section(section)
            for key, value in options.items():
                self._parser.set(section, key, str(value))

    def _load_file(self, path: Path):
        if not path.exists():
            return
        try:
            self._parser.read(str(path), encoding="utf-8")
        except configparser.Error as e:
            # 配置文件格式错误，使用默认配置
            print(f"Warning: Failed to parse config file {path}: {e}")
        except Exception as e:
            # 其他读取错误
            print(f"Warning: Error reading config file {path}: {e}")

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        raw = self._parser.get(section, key, fallback=None)
        if raw is None:
            return fallback
        return _coerce_value(raw)

    def get_raw(self, section: str, key: str, fallback: Any = None) -> Any:
        """获取原始字符串(未经 bool 推断),供数值类字段使用,避免 'on'→True→1 误判."""
        raw = self._parser.get(section, key, fallback=None)
        if raw is None:
            return fallback
        return raw.replace("\\n", "\n")

    def section(self, name: str) -> dict[str, Any]:
        if not self._parser.has_section(name):
            return {}
        return {k: _coerce_value(v) for k, v in self._parser.items(name)}

    def as_kwargs(self, section: str) -> dict[str, Any]:
        return self.section(section)

    @property
    def input_dir(self) -> Path:
        raw = self.get("x2md", "input_dir", "../from")
        p = Path(raw)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / raw
        return p

    @property
    def output_dir(self) -> Path:
        raw = self.get("x2md", "output_dir", "../to")
        p = Path(raw)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / raw
        return p

    @property
    def encoding(self) -> str:
        return self.get("x2md", "encoding", "utf-8")

    @property
    def llm_enabled(self) -> bool:
        return self.get("llm", "enabled", True)

    @property
    def llm_model(self) -> str:
        return self.get("llm", "model", "qwen2.5:7b-instruct")

    @property
    def llm_base_url(self) -> str:
        return self.get("llm", "base_url", "http://localhost:11434")

    @property
    def llm_timeout(self) -> int:
        return int(self.get_raw("llm", "timeout", "120"))

    @property
    def llm_call_interval(self) -> float:
        return float(self.get_raw("llm", "call_interval", "0.5"))

    @property
    def llm_max_concurrent(self) -> int:
        return int(self.get_raw("llm", "max_concurrent", "4"))

    def create_llm_client(self) -> OllamaClient | None:
        if not self.llm_enabled:
            return None
        client = OllamaClient(
            model=self.llm_model,
            base_url=self.llm_base_url,
            timeout=self.llm_timeout,
            call_interval=self.llm_call_interval,
            max_concurrent=self.llm_max_concurrent,
        )
        if not client.is_available():
            return None
        return client

    @property
    def vit_enabled(self) -> bool:
        return self.get("vit", "enabled", True)

    @property
    def vit_model(self) -> str:
        return self.get("vit", "model", "openai/clip-vit-base-patch32")

    @property
    def vit_device(self) -> str:
        return self.get("vit", "device", "cpu")

    @property
    def vit_local_files_only(self) -> bool:
        return self.get("vit", "local_files_only", False)

    @property
    def vit_mirror(self) -> str:
        return self.get("vit", "mirror", "")

    @property
    def vit_labels(self) -> list[str] | None:
        raw = self.get("vit", "labels", "")
        if not raw:
            return None
        return [label.strip() for label in raw.split(",") if label.strip()]

    @property
    def chunk_size(self) -> int:
        return int(self.get_raw("chunk", "chunk_size", "500"))

    @property
    def chunk_overlap(self) -> int:
        return int(self.get_raw("chunk", "chunk_overlap", "0"))

    @property
    def chunk_separators(self) -> list[str]:
        raw = self._parser.get("chunk", "separators", fallback=None)
        if raw is None:
            return ["\n\n", "\n", "。", ".", " ", ""]
        parts = raw.split(",")
        result: list[str] = []
        for p in parts:
            p = p.strip(" ").replace("\\n", "\n")
            result.append(p)
        # 保证兜底 "" 存在且只出现一次,放在末尾(字符级切分)
        if "" in result:
            # 去掉所有 "",再补一个到末尾,避免末尾逗号产生重复
            result = [s for s in result if s != ""]
        result.append("")
        return result

    @property
    def chunk_separator_rule(self) -> list[str]:
        raw = self.get("chunk", "separator_rule", "after,after,after,after,after,after")
        parts = [p.strip() for p in raw.split(",")]
        return parts if parts else ["after", "after", "after", "after", "after", "after"]

    @property
    def chunk_max_limit(self) -> int:
        return int(self.get_raw("chunk", "max_chunk_limit", "10000"))

    @property
    def chunk_inject_abstract(self) -> bool:
        return self.get("chunk", "inject_abstract", True)


def find_config(start: Path | None = None) -> Path | None:
    """搜索 x2md.conf,从 start 向上但不越过若干安全边界.

    安全边界:
      - 文件系统根 /
      - 用户主目录 ~ (不读 ~/x2md.conf,避免误用同事配置)
      - 任一含 .git 的目录(到达 git 仓库根即停)
      - 任一含 pyproject.toml 的目录(项目根)
    """
    current = (start or Path.cwd()).resolve()
    home = Path.home().resolve()
    while True:
        candidate = current / "x2md.conf"
        if candidate.exists():
            return candidate

        # 到达 git/项目根 → 仅在该层继续匹配 x2md.conf 后停止
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            break
        # 到达 home 目录 → 不再向上(避免触发 /Users/xxx/x2md.conf 这种共享配置)
        if current == home:
            break

        parent = current.parent
        if parent == current:  # 文件系统根
            break
        current = parent
    return None


def load_config(config_path: str | Path | None = None) -> Config:
    if config_path is not None:
        return Config(config_path)
    found = find_config()
    if found is not None:
        return Config(found)
    return Config()
