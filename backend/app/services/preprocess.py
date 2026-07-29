"""X2MD 预处理服务 - 将 DOC 文档转换为 Markdown"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional
from app.core.config import settings
from md2rag.loader import CLASSIFICATION_DIR_MAP


def _resolve_preprocess_defaults(
    enable_llm: Optional[bool],
    extract_tables: Optional[bool],
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    ocr_lang: Optional[str] = None,
    enable_ocr: Optional[bool] = None,
):
    """请求体里 None 表示"用默认", 这里回退到 settings.* (由 SettingsService 实时维护).

    六个字段的 settings 源被 PUT /settings/preprocess 更新, 因此调用方传 None 时
    实际生效的是用户在 Settings 页保存的值, 而不是 X2MD CLI 的硬编码 500/50。
    """
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
    return enable_llm, extract_tables, chunk_size, chunk_overlap, ocr_lang, enable_ocr


class PreprocessService:
    """文档预处理服务，使用 X2MD 转换文档"""

    # 派生自 md2rag.loader.CLASSIFICATION_DIR_MAP,保持唯一权威定义,避免再分叉。
    # key 是密级目录名(如 "0Public"),映射到 DOC/MD 下的源/输出子路径。
    LEVEL_MAP = {
        dir_name: {'source': f'DOC/{dir_name}', 'output': f'MD/{dir_name}'}
        for dir_name in CLASSIFICATION_DIR_MAP.values()
    }

    def process_level(self, level: str, chunk_strategy: str = 'parent-child',
                      enable_llm: Optional[bool] = None,
                      encoding: str = None,
                      extract_tables: Optional[bool] = None,
                      chunk_size: Optional[int] = None,
                      chunk_overlap: Optional[int] = None,
                      ocr_lang: Optional[str] = None,
                      enable_ocr: Optional[bool] = None,
                      progress_callback: Optional[Callable] = None,
                      cancel_event=None) -> Dict:
        """处理单个密级的所有文档

        Args:
            level: 密级 (0Public, 1Restricted, 2Confidential)
            chunk_strategy: 切片策略 (chunk 或 parent-child)
            enable_llm: 是否启用 LLM 增强 (None → settings.ENABLE_LLM_DEFAULT)
            encoding: text/html 文件的字符编码 (None 用 X2MD config 默认; P2-5)
            extract_tables: PDF 是否提取表格 (None → settings.EXTRACT_TABLES_DEFAULT)
            chunk_size/chunk_overlap: None → settings.CHUNK_*_DEFAULT (运行时可改)
            ocr_lang: OCR 语言, 透传到 X2MD --ocr-lang (None → settings.OCR_LANG_DEFAULT)
            enable_ocr: 图片 OCR 总开关 (None → settings.ENABLE_OCR_DEFAULT, False → CLI --no-ocr)
            progress_callback: 进度回调,接收 dict {"phase", "file", "index", "total", ...}

        Returns:
            Dict: 处理结果统计
        """
        enable_llm, extract_tables, chunk_size, chunk_overlap, ocr_lang, enable_ocr = _resolve_preprocess_defaults(
            enable_llm, extract_tables, chunk_size, chunk_overlap, ocr_lang, enable_ocr
        )
        if level not in self.LEVEL_MAP:
            return {"error": f"Invalid level: {level}"}

        # 获取目录路径
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        source_dir = base_dir / self.LEVEL_MAP[level]['source']
        output_dir = base_dir / self.LEVEL_MAP[level]['output']

        if not source_dir.exists():
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "logs": [],
                "message": f"Source directory does not exist: {source_dir}"
            }

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 扫描源文件
        source_files = []
        for ext in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.txt', '.html', '.md']:
            source_files.extend(source_dir.rglob(f'*{ext}'))

        # 过滤掉隐藏文件
        source_files = [f for f in source_files if not f.name.startswith('.')]
        total_files = len(source_files)

        if progress_callback:
            progress_callback({
                "phase": "start",
                "level": level,
                "message": f"开始预处理 {level}, 共 {total_files} 个文件",
                "total": total_files,
            })

        logs = []
        success_count = 0
        failed_count = 0

        for i, file_path in enumerate(source_files):
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                result = {
                    "processed": i,
                    "success": success_count,
                    "failed": failed_count,
                    "logs": logs,
                    "source_dir": str(source_dir),
                    "output_dir": str(output_dir),
                    "cancelled": True,
                    "message": f"预处理在处理第 {i+1} 个文件时被用户取消",
                }
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "level": level,
                        "message": result["message"],
                        "progress": i / total_files if total_files > 0 else 0,
                    })
                return result

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "level": level,
                    "message": f"正在处理 {file_path.name} ({i+1}/{total_files})",
                    "progress": (i + 0.5) / total_files if total_files > 0 else 0.5,
                    "file": file_path.name,
                    "index": i,
                    "total": total_files,
                    "success_so_far": success_count,
                    "failed_so_far": failed_count,
                })

            try:
                # 构建输出路径 - 保留相对子目录结构,避免不同子目录下同名(同 stem)
                # 文件的输出互相覆盖。file_path 来自 source_dir.rglob,必在 source_dir 下;
                # X2MD _convert_file 会对 out_path.parent 做 mkdir(parents=True),嵌套目录自动创建。
                try:
                    rel = file_path.relative_to(source_dir)
                    output_path = output_dir / rel.with_suffix(".md")
                except ValueError:
                    output_path = output_dir / f"{file_path.stem}.md"

                # 使用 wrapper 脚本方式调用 X2MD,避免 python3 -c f-string 命令注入
                # 将 sys.path 和参数通过环境变量/命令行参数传递,不再拼接 Python 代码字符串
                x2md_src = base_dir / 'X2MD' / 'src'
                cmd = [
                    sys.executable,
                    str(x2md_src / 'x2md' / '_invoke.py'),
                    str(file_path),
                    '-o', str(output_path),
                ]

                # 添加切片策略参数
                if chunk_strategy == 'parent-child':
                    cmd.append('--parent-child')
                elif chunk_strategy == 'chunk':
                    cmd.append('--chunk')

                # 添加 LLM 开关
                if not enable_llm:
                    cmd.append('--no-llm')

                # P2-5: 之前漏传的两个开关
                if encoding:
                    cmd.extend(['--encoding', encoding])
                if not extract_tables:
                    cmd.append('--no-tables')
                # 切片参数 (None 时用 X2MD config 默认值)
                if chunk_size is not None:
                    cmd.extend(['--chunk-size', str(chunk_size)])
                if chunk_overlap is not None:
                    cmd.extend(['--chunk-overlap', str(chunk_overlap)])
                # OCR 语言 (透传到 X2MD --ocr-lang)
                if ocr_lang:
                    cmd.extend(['--ocr-lang', ocr_lang])
                # OCR 总开关
                if not enable_ocr:
                    cmd.append('--no-ocr')

                # 执行转换
                # 内网环境: 禁止 HuggingFace 联网检查更新, 强制使用本地缓存
                env = os.environ.copy()
                env['HF_HUB_OFFLINE'] = '1'
                env['TRANSFORMERS_OFFLINE'] = '1'
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(base_dir),
                    timeout=300,  # 5分钟超时
                    env=env,
                )

                if result.returncode == 0:
                    success_count += 1
                    logs.append({
                        "file": file_path.name,
                        "status": "success",
                        "message": "Converted successfully"
                    })
                else:
                    failed_count += 1
                    logs.append({
                        "file": file_path.name,
                        "status": "error",
                        "message": result.stderr or "Unknown error"
                    })

            except subprocess.TimeoutExpired:
                failed_count += 1
                logs.append({
                    "file": file_path.name,
                    "status": "error",
                    "message": "Processing timeout (5 minutes)"
                })
            except Exception as e:
                failed_count += 1
                logs.append({
                    "file": file_path.name,
                    "status": "error",
                    "message": str(e)
                })

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "level": level,
                    "message": f"完成 {file_path.name} ({i+1}/{total_files})",
                    "progress": (i + 1) / total_files if total_files > 0 else 1.0,
                    "file": file_path.name,
                    "index": i,
                    "total": total_files,
                    "success_so_far": success_count,
                    "failed_so_far": failed_count,
                })

        result = {
            "processed": total_files,
            "success": success_count,
            "failed": failed_count,
            "logs": logs,
            "source_dir": str(source_dir),
            "output_dir": str(output_dir)
        }

        if progress_callback:
            progress_callback({
                "phase": "done",
                "level": level,
                "message": f"预处理完成 {level}: 成功 {success_count}, 失败 {failed_count}",
                "progress": 1.0,
                **result,
            })

        return result

    def process_all_levels(self, chunk_strategy: str = 'parent-child',
                           enable_llm: bool = False,
                           encoding: str = None,
                           extract_tables: bool = True,
                           chunk_size: Optional[int] = None,
                           chunk_overlap: Optional[int] = None,
                           ocr_lang: Optional[str] = None,
                           enable_ocr: Optional[bool] = None,
                           progress_callback: Optional[Callable] = None,
                           cancel_event=None) -> Dict:
        """依次处理所有密级(0Public → 1Restricted → 2Confidential).

        Args:
            progress_callback: 每个密级开始/完成时回调,进度按密级位置估算。
        """
        levels = list(self.LEVEL_MAP.keys())  # ["0Public", "1Restricted", "2Confidential"]
        total_levels = len(levels)
        results = {}

        for i, level in enumerate(levels):
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "level": level,
                        "message": f"预处理在 {level} 开始前被用户取消",
                        "progress": i / total_levels,
                    })
                break

            if progress_callback:
                progress_callback({
                    "phase": "start",
                    "level": level,
                    "message": f"开始预处理 {level} ({i+1}/{total_levels})",
                    "progress": i / total_levels,
                })

            # 子回调: 把单密级进度映射到全局进度
            def _sub_callback(detail, _level_index=i, _total=total_levels):
                if progress_callback:
                    sub_progress = detail.get("progress", 0.5)
                    global_progress = (_level_index + sub_progress) / _total
                    detail["global_progress"] = global_progress
                    progress_callback(detail)

            result = self.process_level(
                level, chunk_strategy, enable_llm, encoding, extract_tables,
                chunk_size, chunk_overlap, ocr_lang, enable_ocr,
                progress_callback=_sub_callback,
                cancel_event=cancel_event,
            )
            results[level] = result

            if progress_callback:
                progress_callback({
                    "phase": "done",
                    "level": level,
                    "message": f"完成 {level}: 成功 {result.get('success',0)}, 失败 {result.get('failed',0)}",
                    "progress": (i + 1) / total_levels,
                })

        # 合计
        total_processed = sum(r.get("processed", 0) for r in results.values())
        total_success = sum(r.get("success", 0) for r in results.values())
        total_failed = sum(r.get("failed", 0) for r in results.values())
        was_cancelled = cancel_event and cancel_event.is_set()

        return {
            "status": "cancelled" if was_cancelled else "success",
            "results": results,
            "total_processed": total_processed,
            "total_success": total_success,
            "total_failed": total_failed,
        }

    def retry_failed_files(self, files: list, chunk_strategy: str = 'parent-child',
                           enable_llm: Optional[bool] = None, encoding: str = None,
                           extract_tables: Optional[bool] = None, chunk_size: Optional[int] = None,
                           chunk_overlap: Optional[int] = None,
                           ocr_lang: Optional[str] = None,
                           enable_ocr: Optional[bool] = None,
                           progress_callback: Optional[Callable] = None,
                           cancel_event=None) -> Dict:
        """重试转换失败的文件.

        Args:
            files: 失败文件列表, 每项含 level(密级目录名) 和 file(文件名)
            其他参数同 process_level (None → settings.* 默认值)
            progress_callback: 进度回调
            cancel_event: 取消事件

        Returns:
            Dict: 重试结果, 包含成功和仍失败的文件清单
        """
        enable_llm, extract_tables, chunk_size, chunk_overlap, ocr_lang, enable_ocr = _resolve_preprocess_defaults(
            enable_llm, extract_tables, chunk_size, chunk_overlap, ocr_lang, enable_ocr
        )
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        success_list = []
        still_failed_list = []
        total = len(files)

        if progress_callback:
            progress_callback({
                "phase": "start",
                "message": f"开始重试 {total} 个失败文件",
                "progress": 0.0,
            })

        for i, item in enumerate(files):
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "message": "重试任务被用户取消",
                        "progress": i / total if total > 0 else 0,
                    })
                break

            level = item.get("level", "")
            file_name = item.get("file", "")

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"正在重试 {file_name} ({i+1}/{total})",
                    "progress": (i + 0.5) / total if total > 0 else 0.5,
                    "file": file_name,
                    "index": i,
                    "total": total,
                })

            if level not in self.LEVEL_MAP or not file_name:
                still_failed_list.append({
                    "level": level,
                    "file": file_name,
                    "reason": "无效的密级或文件名",
                })
                continue

            source_dir = base_dir / self.LEVEL_MAP[level]['source']
            output_dir = base_dir / self.LEVEL_MAP[level]['output']
            output_dir.mkdir(parents=True, exist_ok=True)

            # Path traversal validation: sanitize file_name before path operations
            safe_name = Path(file_name).name
            target = (source_dir / safe_name).resolve()
            # 用 relative_to 而非 startswith,避免前缀混淆
            try:
                target.relative_to(source_dir.resolve())
            except ValueError:
                still_failed_list.append({
                    "level": level,
                    "file": file_name,
                    "reason": "Invalid file name: path traversal detected",
                })
                continue
            file_name = safe_name

            # 找到源文件
            source_file = None
            # 文件名可能带后缀或不带后缀
            candidate = source_dir / file_name
            if candidate.exists():
                source_file = candidate
            if not source_file:
                for ext in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.txt', '.html', '.md']:
                    candidate = source_dir / f"{file_name}{ext}"
                    if candidate.exists():
                        source_file = candidate
                        break
            if not source_file:
                stem = Path(file_name).stem
                for f in source_dir.rglob(f'{stem}.*'):
                    if not f.name.startswith('.') and f.suffix.lower() in ['.pdf', '.docx', '.doc', '.xlsx', '.pptx', '.txt', '.html', '.md']:
                        source_file = f
                        break

            if not source_file:
                still_failed_list.append({
                    "level": level,
                    "file": file_name,
                    "reason": "源文件不存在",
                })
                continue

            # 保留相对子目录结构,避免同名文件互相覆盖 (与 process_level 一致)。
            # source_file 来自 source_dir.rglob,必在 source_dir 下。
            try:
                rel = source_file.relative_to(source_dir)
                output_path = output_dir / rel.with_suffix(".md")
            except ValueError:
                output_path = output_dir / f"{source_file.stem}.md"
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

            try:
                # 内网环境: 禁止 HuggingFace 联网检查更新, 强制使用本地缓存
                env = os.environ.copy()
                env['HF_HUB_OFFLINE'] = '1'
                env['TRANSFORMERS_OFFLINE'] = '1'
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(base_dir),
                    timeout=300,
                    env=env,
                )
                if result.returncode == 0:
                    success_list.append({
                        "level": level,
                        "file": file_name,
                    })
                else:
                    still_failed_list.append({
                        "level": level,
                        "file": file_name,
                        "reason": result.stderr or "转换失败",
                    })
            except subprocess.TimeoutExpired:
                still_failed_list.append({
                    "level": level,
                    "file": file_name,
                    "reason": "超时 (5分钟)",
                })
            except Exception as e:
                still_failed_list.append({
                    "level": level,
                    "file": file_name,
                    "reason": str(e),
                })

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"完成重试 {file_name} ({i+1}/{total})",
                    "progress": (i + 1) / total if total > 0 else 1.0,
                    "file": file_name,
                    "index": i,
                    "total": total,
                })

        was_cancelled = cancel_event and cancel_event.is_set()

        result = {
            "status": "cancelled" if was_cancelled else "success",
            "retried_count": len(files),
            "success_count": len(success_list),
            "still_failed_count": len(still_failed_list),
            "success_list": success_list,
            "still_failed_list": still_failed_list,
        }

        if progress_callback:
            progress_callback({
                "phase": "done",
                "message": f"重试完成: 成功 {len(success_list)}, 仍失败 {len(still_failed_list)}",
                "progress": 1.0,
                **result,
            })

        return result


preprocess_service = PreprocessService()