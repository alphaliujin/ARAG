from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import logging
import os
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)
from app.services.scanner import scanner_service
from app.services.ingestion import data_ingestion_service
from app.services.preprocess import preprocess_service
from app.services.docscan import docscan_service
from app.services.task_manager import task_manager, TASK_PENDING, TASK_RUNNING
from app.core.config import settings
from md2rag.loader import CLASSIFICATION_DIR_MAP, VALID_CLASSIFICATIONS

# 延迟导入 dedup 服务（避免启动时 MD2RAG 路径问题）
deduplication_service = None

def _get_dedup_service():
    global deduplication_service
    if deduplication_service is None:
        # MD2RAG path is configured in app/main.py at startup; no need to manipulate sys.path here.
        from app.services.dedup import DataDeduplicationService
        deduplication_service = DataDeduplicationService()
    return deduplication_service

router = APIRouter()


# DOC 目录映射(派生自 CLASSIFICATION_DIR_MAP,避免重复维护)
DOC_LEVEL_MAP = {
    dir_name: f'../DOC/{dir_name}'
    for dir_name in CLASSIFICATION_DIR_MAP.values()
}


class TextScanRequest(BaseModel):
    # 限制单次扫描文本长度, 防止超大 body 直接进内存 + 线程造成 DoS
    text: str = Field(..., max_length=200000)


class TextScanResponse(BaseModel):
    total_chunks: int
    has_sensitive: bool
    segments: List[dict]
    chunk_results: List[dict]
    summary: dict


class IngestRequest(BaseModel):
    classification_level: Optional[str] = None
    estimate_only: bool = False
    force: bool = False  # P2-3: True 时跳过"已入库则 skip"的去重预检


class PreprocessRequest(BaseModel):
    level: str
    chunk_strategy: str = 'parent-child'
    enable_llm: bool = False
    # P2-5: 之前 UI 的 encoding/extract_tables 开关不传后端, 控件无效
    encoding: Optional[str] = None       # text/html 文件编码; None = X2MD config 默认
    extract_tables: bool = True          # PDF 是否提取表格 (False → CLI --no-tables)
    chunk_size: Optional[int] = None     # 切片大小; None = settings.CHUNK_SIZE_DEFAULT
    chunk_overlap: Optional[int] = None  # 切片重叠; None = settings.CHUNK_OVERLAP_DEFAULT
    ocr_lang: Optional[str] = None       # OCR 语言; None = settings.OCR_LANG_DEFAULT
    enable_ocr: Optional[bool] = None    # OCR 总开关; None = settings.ENABLE_OCR_DEFAULT


class DBStatsResponse(BaseModel):
    collections: dict
    total_vectors: int



class DocScanPreprocessRequest(BaseModel):
    """DocScan 预处理请求参数."""
    filename: str
    chunk_strategy: str = 'parent-child'
    enable_llm: bool = False
    encoding: Optional[str] = None
    extract_tables: bool = True
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    ocr_lang: Optional[str] = None
    enable_ocr: Optional[bool] = None

@router.post("/scan/file")
async def scan_document(file: UploadFile = File(...)):
    """文件上传 — 只保存到 DocScan 目录, 不自动处理.

    后续操作通过 /docscan/preprocess, /docscan/embed, /docscan/compare 三个按钮触发。
    """
    # 先校验扩展名再读 body: 避免对 .exe 等非法文件白读最多 50MB 进内存
    allowed_extensions = {'.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.html', '.md'}
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    # Enforce upload size limit even when file.size is None (common in FastAPI)
    MAX_UPLOAD_SIZE = settings.MAX_UPLOAD_SIZE
    docscan_dir = Path(settings.DOCSCAN_DIR)
    docscan_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 确定目标路径 (不依赖文件内容, 先于读取完成)
        raw_name = file.filename or f"upload{file_ext}"
        safe_filename = Path(raw_name).name  # strip any directory components
        # Verify no path traversal — 用 relative_to 而非 startswith,
        # 避免 /tmp/DocScan 与 /tmp/DocScan_evil 这种前缀混淆。
        target_path = (docscan_dir / safe_filename).resolve()
        try:
            target_path.relative_to(docscan_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")
        docscan_path = docscan_dir / safe_filename
        if docscan_path.exists():
            stem = docscan_path.stem
            counter = 1
            while (docscan_dir / f"{stem}_{counter}{file_ext}").exists():
                counter += 1
            docscan_path = docscan_dir / f"{stem}_{counter}{file_ext}"

        # 流式写盘: 读一块写一块, 不再把整个文件攒进内存 (原 chunks=[] + b"".join
        # 对 50MB 文件峰值占 ~100MB+ 内存)。超限时删除已写的部分文件, 不留残骸。
        total_size = 0
        too_large = False
        async with aiofiles.open(str(docscan_path), 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    too_large = True
                    break
                await f.write(chunk)
        if too_large:
            try:
                docscan_path.unlink()
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=f"File too large (>{MAX_UPLOAD_SIZE // (1024*1024)}MB)")

        return {
            "status": "success",
            "saved_path": str(docscan_path),
            "saved_filename": docscan_path.name,
            "message": f"文件已保存: {docscan_path.name}",
        }

    except HTTPException:
        # 让校验类 4xx (如 Invalid filename) 原样返回, 不被下面的 500 吞掉
        raise
    except Exception as e:
        logger.error(f"Error in scan_document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/scan/text", response_model=TextScanResponse)
async def scan_text(request: TextScanRequest):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    try:
        scan_result = await asyncio.to_thread(scanner_service.scan_text, request.text)
        return scan_result
    except Exception as e:
        logger.error(f"Error in scan_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ==================== DocScan 操作 API ====================

@router.post("/docscan/preprocess")
async def docscan_preprocess(request: DocScanPreprocessRequest):
    """DocScan 预处理 — 对已保存的文件执行 X2MD 转换."""
    try:
        result = await asyncio.to_thread(
            docscan_service.preprocess_file,
            request.filename,
            chunk_strategy=request.chunk_strategy,
            enable_llm=request.enable_llm,
            encoding=request.encoding,
            extract_tables=request.extract_tables,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            ocr_lang=request.ocr_lang,
            enable_ocr=request.enable_ocr,
        )
        return result
    except Exception as e:
        logger.error(f"Error in docscan_preprocess: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/docscan/embed")
async def docscan_embed(filename: str = Query(..., description="文件名")):
    """DocScan 生成向量 — 读取切片 → 批量嵌入 → 向量写入 JSON 带标记."""
    existing = task_manager.get_task_by_type("docscan_embed")
    if existing:
        return {"task_id": existing.task_id, "status": existing.status,
                "message": "已有生成向量任务在运行, 请等待完成后再启动"}
    task_id = task_manager.create_task("docscan_embed", {"filename": filename})

    def _run():
        info = task_manager.get_task(task_id)
        cancel_evt = info.cancel_event if info else None
        callback = task_manager.make_progress_callback(task_id)
        return docscan_service.embed_file_vectors(
            filename, progress_callback=callback, cancel_event=cancel_evt
        )

    task_manager.run_in_thread(task_id, _run)
    return {"task_id": task_id, "status": "pending"}


@router.post("/docscan/compare")
async def docscan_compare(filename: str = Query(..., description="文件名"), n_results: int = Query(10, description="每密级top-n数")):
    """DocScan 层级比对 — 摘要→父块→子块, 超阈值跳过下级."""
    try:
        result = await asyncio.to_thread(
            docscan_service.compare_file_hierarchical,
            filename,
            n_results,
        )
        return result
    except Exception as e:
        logger.error(f"Error in docscan_compare: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/docscan/status")
async def docscan_file_status(filename: str = Query(..., description="文件名")):
    """获取文件在 DocScan 流程中的状态."""
    try:
        return await asyncio.to_thread(docscan_service.get_file_status, filename)
    except Exception as e:
        logger.error(f"Error in docscan_file_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/docscan/files")
async def list_docscan_files():
    """列出 DocScan 目录下所有已扫描上传的文件."""
    def _list():
        docscan_dir = Path(settings.DOCSCAN_DIR)
        if not docscan_dir.exists():
            return {"files": [], "count": 0}

        files = []
        for f in docscan_dir.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "size_display": f"{stat.st_size / 1024:.1f} KB" if stat.st_size < 1024 * 1024
                                  else f"{stat.st_size / (1024 * 1024):.1f} MB",
                    "modified": stat.st_mtime,
                })

        # 按修改时间降序排列
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": files, "count": len(files)}

    try:
        # 阻塞 I/O (iterdir/stat) 放到线程池, 避免卡住事件循环 (与 get_status 一致)
        return await asyncio.to_thread(_list)
    except Exception as e:
        logger.error(f"Error in list_docscan_files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/docscan/files")
async def delete_docscan_file(name: str = Query(..., description="文件名 (仅基名, 不允许路径)")):
    """删除 DocScan 目录下指定的文件.

    安全: 与 dedup 相同的路径穿越校验。
    """
    if not name or any(c in name for c in ("/", "\\", "\x00")) or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    def _delete():
        # 阻塞 I/O (resolve/exists/unlink/rmtree) 在线程池执行
        docscan_dir = Path(settings.DOCSCAN_DIR).resolve()
        target = (docscan_dir / name).resolve()

        # 路径穿越防御
        try:
            target.relative_to(docscan_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="Path traversal denied")

        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {name}")

        logger.info(f"[AUDIT] Delete operation: docscan file '{name}' deleted")
        os.unlink(target)
        # 同步清理 X2MD 派生的切片产物, 否则重传同名文件时会复用过期嵌入产生 stale
        # 比对结果, 且 get_docscan_stats 会误报已删文档仍已处理。
        # 三者均在 docscan_dir 根: <stem>.md (文件, docscan.py 的 output_path)、
        # <stem>.parents.json、<stem>.children.json; 兼容旧布局的 <stem>.md/ 目录。
        import shutil
        stem = Path(name).stem
        for derived_name in (f"{stem}.md", f"{stem}.parents.json", f"{stem}.children.json"):
            p = (docscan_dir / derived_name).resolve()
            try:
                p.relative_to(docscan_dir)  # 防穿越: 必须在 docscan_dir 内
            except ValueError:
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    logger.info(f"[AUDIT] Also removed orphan chunk dir: {p.name}")
                elif p.is_file():
                    p.unlink()
                    logger.info(f"[AUDIT] Also removed derived file: {p.name}")
            except OSError as e:
                logger.warning(f"[AUDIT] Failed to clean derived file {p.name}: {e}")
        return {"status": "success", "message": f"已删除: {name}"}

    try:
        return await asyncio.to_thread(_delete)
    except HTTPException:
        raise  # 400/404 原样上抛, 不被下面的 500 吞掉
    except Exception as e:
        logger.error(f"Error in delete_docscan_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/docscan/files/all")
async def delete_all_docscan_files():
    """清空 DocScan 目录下全部内容(源文件 + .md + .parents/.children JSON + images/),
    保留目录本身。仅删 docscan_dir 内的条目、不跟随符号链接(防穿越)。
    """
    def _delete_all():
        docscan_dir = Path(settings.DOCSCAN_DIR).resolve()
        if not docscan_dir.exists():
            return {"status": "success", "message": "DocScan 目录不存在, 无需清理", "deleted": 0}

        deleted = 0
        for entry in docscan_dir.iterdir():
            if entry.is_symlink():
                continue  # 防符号链接穿越
            try:
                entry.resolve().relative_to(docscan_dir)  # 必须落在 docscan_dir 内
            except ValueError:
                continue
            try:
                if entry.is_file():
                    entry.unlink()
                elif entry.is_dir():
                    import shutil
                    shutil.rmtree(entry, ignore_errors=True)
                deleted += 1
            except Exception as e:
                logger.warning(f"[AUDIT] Failed to delete {entry.name}: {e}")

        logger.info(f"[AUDIT] Delete-all: cleared {deleted} entries from DocScan dir")
        return {"status": "success", "message": f"已清空 DocScan ({deleted} 项)", "deleted": deleted}

    try:
        # 阻塞 I/O (iterdir/unlink/rmtree) 放到线程池, 避免卡住事件循环
        return await asyncio.to_thread(_delete_all)
    except Exception as e:
        logger.error(f"Error in delete_all_docscan_files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/docscan/stats")
async def get_docscan_stats():
    """获取 DocScan 向量库统计."""
    try:
        return await asyncio.to_thread(docscan_service.get_docscan_stats)
    except Exception as e:
        logger.error(f"Error in get_docscan_stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ingest")
async def ingest_documents(request: Optional[IngestRequest] = None):
    try:
        if request and request.classification_level:
            if request.classification_level not in VALID_CLASSIFICATIONS:
                raise HTTPException(status_code=400, detail="Invalid classification level")
            result = await asyncio.to_thread(
                data_ingestion_service.ingest_directory,
                request.classification_level,
                None,
                "auto",
                True,
                request.force,
            )
        else:
            force = bool(request and request.force)
            result = await asyncio.to_thread(
                data_ingestion_service.ingest_all_levels, None, force
            )

        return result
    except HTTPException:
        # 让校验类 4xx (如 Invalid classification level) 原样返回, 不被下面的 500 吞掉
        raise
    except Exception as e:
        logger.error(f"Error in ingest_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=DBStatsResponse)
async def get_database_stats():
    try:
        stats = await asyncio.to_thread(data_ingestion_service.get_database_stats)
        return stats
    except Exception as e:
        logger.error(f"Error in get_database_stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_status():
    """获取数据入库状态.

    密级语义（统一规则,与 md2rag.loader.CLASSIFICATION_DIR_MAP 对齐）:
      public       → 0Public 目录
      restricted   → 1Restricted 目录（受限）
      confidential → 2Confidential 目录（机密）
    secret 别名已于 2026-06-09 彻底废除,前端请勿再传。
    """
    try:
        # 阻塞 I/O (文件系统 estimate_records + ChromaDB get_database_stats) 放到线程池,
        # 避免逐密级查询时阻塞事件循环 (含 /health 在内的其他请求被卡住)。
        def _compute():
            # 各密级的待入库文件数
            pending = {cls: 0 for cls in CLASSIFICATION_DIR_MAP}
            for level in CLASSIFICATION_DIR_MAP:
                result = data_ingestion_service.estimate_records(level)
                pending[level] = result.get("file_count", 0)

            # 已入库数量 - 使用 get_database_stats() 获取准确的统计
            try:
                db_stats = data_ingestion_service.get_database_stats()
                indexed = {
                    "public": db_stats["collections"]["public_documents"],
                    "restricted": db_stats["collections"]["restricted_documents"],
                    "confidential": db_stats["collections"]["confidential_documents"],
                }
            except Exception:
                indexed = {cls: 0 for cls in CLASSIFICATION_DIR_MAP}
            return pending, indexed

        pending, indexed = await asyncio.to_thread(_compute)

        return {
            "pending": pending,
            "pending_total": sum(pending.values()),
            "indexed": indexed,
            "indexed_total": sum(indexed.values()),
            "embedding_model": settings.EMBEDDING_MODEL,
        }
    except Exception as e:
        logger.error(f"Error in get_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/reset")
async def reset_database():
    try:
        logger.info("[AUDIT] Delete operation: database reset (all collections cleared)")
        # reset_all() 删除 ChromaDB 集合, cleanup_memory() 释放嵌入器缓存, 均为阻塞操作,
        # 放到线程池避免卡住事件循环 (与 /ingest, /status 一致)。
        result = await asyncio.to_thread(data_ingestion_service.reset_all)
        # busy = 已有 reset 在跑, 本请求什么都没做; 跳过 cleanup, 避免与在跑的 reset 竞争 indexer 释放
        if result.get("status") != "busy":
            await asyncio.to_thread(data_ingestion_service.cleanup_memory)
        # 透传真实的 before/after 计数, 便于前端验证清空确实生效
        if result.get("status") == "error":
            logger.error(f"Reset operation returned error: {result.get('message')}")
            raise HTTPException(status_code=500, detail="Internal server error")
        return {
            "status": result.get("status", "success"),
            "message": result.get("message", "Database reset successfully"),
            "vectors_before": result.get("vectors_before"),
            "vectors_after": result.get("vectors_after"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reset_database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/preprocess")
async def preprocess_documents(request: PreprocessRequest):
    """预处理指定密级的文档（使用 X2MD 转换为 Markdown）"""
    try:
        if request.level not in DOC_LEVEL_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid level: {request.level}")

        result = await asyncio.to_thread(
            preprocess_service.process_level,
            request.level,
            request.chunk_strategy,
            request.enable_llm,
            request.encoding,
            request.extract_tables,
            request.chunk_size,
            request.chunk_overlap,
            request.ocr_lang,
            request.enable_ocr,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in preprocess_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/scan-source-dir")
async def scan_source_dir(level: str = Query(..., description="密级: 0Public, 1Restricted, 2Confidential")):
    """扫描源目录获取文件数量"""
    try:
        if level not in DOC_LEVEL_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid level: {level}")

        # 获取backend目录作为基准
        base_dir = Path(__file__).resolve().parent.parent.parent
        source_dir = base_dir / DOC_LEVEL_MAP[level]

        if not source_dir.exists():
            return {
                "level": level,
                "source_dir": str(source_dir),
                "file_count": 0,
                "files": []
            }

        # rglob + stat 是阻塞 I/O, 放到线程池避免卡住事件循环 (含 /health)。
        # 与 /status 的 _compute 同理。
        def _scan():
            files = []
            for f in source_dir.rglob("*"):
                if f.is_file() and not f.name.startswith('.'):
                    files.append({
                        "name": f.name,
                        "path": str(f.relative_to(source_dir)),
                        "size": f.stat().st_size,
                    })
            return files

        files = await asyncio.to_thread(_scan)

        return {
            "level": level,
            "source_dir": str(source_dir),
            "file_count": len(files),
            "files": files
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scan_source_dir: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dedup")
async def run_deduplication():
    """执行数据去重比对.

    比对三组数据，结果按相似度自动分为三档：
    1. 受限 vs 公开
    2. 机密 vs 公开
    3. 机密 vs 受限

    三档划分：≥0.8 高度相似 / 0.65-0.8 中度相似 / 0.5-0.65 弱相关
    """
    try:
        dedup_service = _get_dedup_service()
        result = await asyncio.to_thread(
            dedup_service.run_deduplication,
        )
        return result
    except Exception as e:
        logger.error(f"Error in run_deduplication: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dedup/results")
async def get_dedup_results():
    """获取去重结果文件列表."""
    def _list():
        dedup_dir = Path(settings.DATA_DIR).parent / "dedup_results"
        if not dedup_dir.exists():
            return {"files": [], "count": 0}

        files = []
        for f in dedup_dir.glob("*.md"):
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })

        return {"files": files, "count": len(files)}

    try:
        # 阻塞 I/O (glob/stat) 放到线程池, 避免卡住事件循环
        return await asyncio.to_thread(_list)
    except Exception as e:
        logger.error(f"Error in get_dedup_results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dedup/results/file")
async def get_dedup_result_file(name: str = Query(..., description="文件名 (仅基名, 不允许路径)")):
    """返回 dedup_results 目录下某个 .md 文件的内容.

    P2-4: 前端原来用 window.open('file://...') 被浏览器禁止, 改为通过这个 API 拉取.
    安全: 只接受 basename, 强制后缀 .md, 拒绝任何含 / \\ .. 的输入, 校验最终路径仍在
    dedup_dir 内 (防符号链接 + 路径穿越).
    """
    # 入参基本校验
    if not name or any(c in name for c in ("/", "\\", "\x00")) or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not name.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files allowed")

    dedup_dir = (Path(settings.DATA_DIR).parent / "dedup_results").resolve()
    target = (dedup_dir / name).resolve()

    # 路径穿越终极防御: resolve 后 target 必须仍在 dedup_dir 下
    try:
        target.relative_to(dedup_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal denied")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {name}")

    return FileResponse(
        path=str(target),
        media_type="text/markdown; charset=utf-8",
        filename=name,
    )


@router.delete("/dedup/results/file")
async def delete_dedup_result_file(name: str = Query(..., description="文件名 (仅基名, 不允许路径)")):
    """删除 dedup_results 目录下指定的 .md 文件.

    安全: 与 GET /dedup/results/file 相同的路径穿越校验。
    """
    if not name or any(c in name for c in ("/", "\\", "\x00")) or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not name.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files allowed")

    dedup_dir = (Path(settings.DATA_DIR).parent / "dedup_results").resolve()
    target = (dedup_dir / name).resolve()

    try:
        target.relative_to(dedup_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal denied")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {name}")

    try:
        logger.info(f"[AUDIT] Delete operation: dedup result file '{name}' deleted")
        os.unlink(target)
        return {"status": "success", "message": f"已删除: {name}"}
    except Exception as e:
        logger.error(f"Error in delete_dedup_result_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


class DedupApplyRequest(BaseModel):
    dedup_threshold: float = 0.75


class PreprocessRetryRequest(BaseModel):
    files: List[dict]  # [{"level": "1Restricted", "file": "附件3.doc"}, ...]
    chunk_strategy: str = 'parent-child'
    enable_llm: bool = False
    encoding: Optional[str] = None
    extract_tables: bool = True
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    ocr_lang: Optional[str] = None
    enable_ocr: Optional[bool] = None


@router.post("/preprocess/retry")
async def preprocess_retry(request: PreprocessRetryRequest):
    """启动后台失败重试任务."""
    existing = task_manager.get_task_by_type("preprocess_retry")
    if existing and existing.status in (TASK_PENDING, TASK_RUNNING):
        return {"task_id": existing.task_id, "status": existing.status,
                "message": "重试任务已在运行,请等待完成后再启动新任务"}

    task_id = task_manager.create_task("preprocess_retry", request.dict())

    def _run():
        callback = task_manager.make_progress_callback(task_id)
        info = task_manager.get_task(task_id)
        cancel_evt = info.cancel_event if info else None
        return preprocess_service.retry_failed_files(
            request.files, request.chunk_strategy,
            request.enable_llm, request.encoding,
            request.extract_tables, request.chunk_size,
            request.chunk_overlap, request.ocr_lang,
            request.enable_ocr,
            progress_callback=callback,
            cancel_event=cancel_evt,
        )

    task_manager.run_in_thread(task_id, _run)
    return {"task_id": task_id, "status": "pending"}


@router.post("/dedup/apply")
async def apply_dedup(request: DedupApplyRequest):
    """按照比对结果文件去重: 从高密级库删除相似度超过阈值的 chunk.

    读取 JSON 结果文件, 提取超过阈值的 source_chunk_id, 从对应的高密级 child collection 中删除.
    操作只在向量数据库中进行, 不涉及 md 和原始文件.
    """
    try:
        dedup_service = _get_dedup_service()
        result = await asyncio.to_thread(
            dedup_service.apply_dedup_from_results,
            request.dedup_threshold,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Unknown error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in apply_dedup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ==================== 系统设置 API ====================

VALID_SETTINGS_CATEGORIES = {"general", "model", "preprocess", "database"}


class SettingsUpdateRequest(BaseModel):
    values: dict


@router.get("/settings")
async def get_all_settings():
    """获取所有系统设置."""
    try:
        from app.services.settings_service import settings_service
        return settings_service.get_all()
    except Exception as e:
        logger.error(f"Error in get_all_settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/settings/{category}")
async def get_settings_category(category: str):
    """获取指定类别的设置."""
    if category not in VALID_SETTINGS_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid: {sorted(VALID_SETTINGS_CATEGORIES)}",
        )
    try:
        from app.services.settings_service import settings_service
        return settings_service.get_category(category)
    except Exception as e:
        logger.error(f"Error in get_settings_category({category!r}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/settings/{category}")
async def update_settings_category(category: str, request: SettingsUpdateRequest):
    """更新指定类别的设置."""
    if category not in VALID_SETTINGS_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid: {sorted(VALID_SETTINGS_CATEGORIES)}",
        )
    try:
        from app.services.settings_service import settings_service
        updated = settings_service.update_category(category, request.values)
        response = {"status": "success", "category": category, "values": updated}
        # vectorDbDir / collectionPrefix 改动需重启后端才生效 (运行中的 vector_db_service
        # 仍持有旧路径/client), 提示用户重启
        if category == "database" and any(
            k in updated for k in ("vectorDbDir", "collectionPrefix")
        ):
            response["warning"] = (
                "vectorDbDir/collectionPrefix 已更新, 需重启后端服务后生效 "
                "(运行中的向量库连接仍使用旧路径)"
            )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in update_settings_category: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/settings/{category}/reset")
async def reset_settings_category(category: str):
    """重置指定类别的设置为默认值."""
    if category not in VALID_SETTINGS_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid: {sorted(VALID_SETTINGS_CATEGORIES)}",
        )
    try:
        from app.services.settings_service import settings_service
        reset = settings_service.reset_category(category)
        return {"status": "success", "category": category, "values": reset}
    except Exception as e:
        logger.error(f"Error in reset_settings_category: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ==========================================================================
# 后台任务 API — 异步执行长时间操作,支持进度查询
# ==========================================================================

class TaskPreprocessRequest(BaseModel):
    level: Optional[str] = None   # 单密级(如 "0Public"),空=全部密级
    chunk_strategy: str = "parent-child"
    enable_llm: bool = False
    encoding: Optional[str] = None
    extract_tables: bool = True
    chunk_size: Optional[int] = None     # 切片大小
    chunk_overlap: Optional[int] = None  # 切片重叠
    ocr_lang: Optional[str] = None       # OCR 语言
    enable_ocr: Optional[bool] = None    # OCR 总开关


class TaskIngestRequest(BaseModel):
    classification: Optional[str] = None  # 单密级或空=全部
    strategy: str = "auto"
    include_images: bool = True
    force: bool = False
    embedding_model: Optional[str] = None


class DedupRequest(BaseModel):
    """去重任务参数 — Pydantic 校验,防止任意 key 注入.

    auto_dedup=True 会触发从高密级库删除相似度超阈值的 chunk,
    属于不可逆操作, 默认关闭, 需前端显式勾选。
    """
    auto_dedup: bool = False
    dedup_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


@router.post("/tasks/preprocess")
async def start_preprocess_task(request: TaskPreprocessRequest):
    """启动后台预处理任务."""
    # 校验 level: 必须是 CLASSIFICATION_DIR_MAP 的合法值,否则 process_level
    # 会返回 {"error": ...} 让 task_manager 把任务标记为 COMPLETED,客户端无从感知错误。
    if request.level is not None:
        valid_levels = set(CLASSIFICATION_DIR_MAP.values())
        if request.level not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level: {request.level!r}. Must be one of: {sorted(valid_levels)}"
            )

    # 防止重复启动
    existing = task_manager.get_task_by_type("preprocess")
    if existing:
        return {"task_id": existing.task_id, "status": existing.status,
                "message": "预处理任务已在运行,请等待完成后再启动新任务"}

    task_id = task_manager.create_task("preprocess", request.dict())

    def _run():
        callback = task_manager.make_progress_callback(task_id)
        info = task_manager.get_task(task_id)
        cancel_evt = info.cancel_event if info else None
        if request.level:
            return preprocess_service.process_level(
                request.level, request.chunk_strategy, request.enable_llm,
                request.encoding, request.extract_tables,
                request.chunk_size, request.chunk_overlap, request.ocr_lang,
                request.enable_ocr,
                progress_callback=callback,
                cancel_event=cancel_evt,
            )
        else:
            return preprocess_service.process_all_levels(
                request.chunk_strategy, request.enable_llm,
                request.encoding, request.extract_tables,
                request.chunk_size, request.chunk_overlap, request.ocr_lang,
                request.enable_ocr,
                progress_callback=callback,
                cancel_event=cancel_evt,
            )

    task_manager.run_in_thread(task_id, _run)
    return {"task_id": task_id, "status": "pending"}


@router.post("/tasks/ingest")
async def start_ingest_task(request: TaskIngestRequest):
    """启动后台入库任务."""
    existing = task_manager.get_task_by_type("ingest")
    if existing:
        return {"task_id": existing.task_id, "status": existing.status,
                "message": "入库任务已在运行,请等待完成后再启动新任务"}

    # 更新嵌入模型设置 (支持 ollama-bge-m3 / mps-bge-m3)
    # 嵌入模型切换 (支持 ollama-bge-m3 / mps-bge-m3)
    # ★ 仅当传入模型与当前不同时才切换 (幂等): 相同模型直接入库, 不触发清库.
    #   切换需 persist + 清库 (维度不一致) + 释放 indexer + 重建 vector_db embedding_fn,
    #   否则入库/检索仍用旧模型. 此前只做内存赋值, 不持久化/不清库/不重建.
    _switch_model = None
    if request.embedding_model:
        from app.services.settings_service import settings_service
        if request.embedding_model not in settings_service._ALLOWED_EMBEDDING_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedding_model: {request.embedding_model}. "
                       f"Allowed: {list(settings_service._ALLOWED_EMBEDDING_MODELS)}",
            )
        if request.embedding_model != settings.EMBEDDING_MODEL:
            _switch_model = request.embedding_model

    # Validate classification BEFORE creating the task (client gets 400, not hidden error)
    if request.classification and request.classification not in VALID_CLASSIFICATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid classification: {request.classification}. Valid: {VALID_CLASSIFICATIONS}")

    task_id = task_manager.create_task("ingest", request.dict())

    def _run():
        callback = task_manager.make_progress_callback(task_id)
        # 切模型 (若请求): persist + 清库 + 释放 indexer + 重建兜底 embedding_fn.
        # reset 是阻塞 IO, 放后台线程不卡 HTTP; 失败则任务 failed, 不混入旧维度向量.
        if _switch_model:
            data_ingestion_service.switch_embedding_model(_switch_model)
        info = task_manager.get_task(task_id)
        cancel_evt = info.cancel_event if info else None
        if request.classification:
            # 单密级模式：注入 level 到回调，使进度映射生效
            def _level_callback(data):
                data["level"] = request.classification
                callback(data)
            return data_ingestion_service.ingest_directory(
                request.classification, _level_callback,
                request.strategy, request.include_images, request.force,
                cancel_event=cancel_evt,
            )
        else:
            return data_ingestion_service.ingest_all_levels(
                callback, force=request.force, strategy=request.strategy,
                include_images=request.include_images, cancel_event=cancel_evt,
            )

    task_manager.run_in_thread(task_id, _run)
    return {"task_id": task_id, "status": "pending"}


@router.post("/tasks/dedup")
async def start_dedup_task(request: Optional[DedupRequest] = None):
    """启动后台去重任务. 结果按相似度自动分为三档.

    request 参数通过 Pydantic DedupRequest 校验:
    - auto_dedup: bool, 是否自动执行去重删除(不可逆操作,默认关闭)
    - dedup_threshold: float, 去重相似度阈值 [0,1], 默认 0.75
    """
    if request is None:
        request = DedupRequest()

    existing = task_manager.get_task_by_type("dedup")
    if existing:
        return {"task_id": existing.task_id, "status": existing.status,
                "message": "去重任务已在运行,请等待完成后再启动新任务"}

    task_id = task_manager.create_task("dedup", request.dict())

    def _run():
        dedup_service = _get_dedup_service()
        callback = task_manager.make_progress_callback(task_id)
        info = task_manager.get_task(task_id)
        cancel_evt = info.cancel_event if info else None
        return dedup_service.run_deduplication(
            progress_callback=callback,
            cancel_event=cancel_evt,
            dedup_threshold=request.dedup_threshold,
            auto_dedup=request.auto_dedup,
        )

    task_manager.run_in_thread(task_id, _run)
    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/active")
async def get_active_tasks():
    """列出所有运行中的任务."""
    tasks = task_manager.get_active_tasks()
    return {"tasks": [t.to_dict() for t in tasks]}


@router.get("/tasks/latest/{task_type}")
async def get_latest_task(task_type: str):
    """查询指定类型最近的任务(运行中或已完成)."""
    task = task_manager.get_task_by_type(task_type)
    if not task:
        # 没有运行中的任务,查看最近完成的(使用公共方法,不访问 _tasks 私有属性)
        task = task_manager.get_latest_task_by_type(task_type)
    if not task:
        return {"task_id": None, "status": "none", "task_type": task_type}
    return task.to_dict()


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """请求取消后台任务(协作式).

    设置取消标志后,工作线程会在下一个检查点自行退出。
    如果任务不存在或已结束,返回 400。
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    ok = task_manager.cancel_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Task cannot be cancelled (current status: {task.status})",
        )
    return {"task_id": task_id, "status": "cancelled", "message": "任务已请求取消"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询后台任务状态和进度."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()