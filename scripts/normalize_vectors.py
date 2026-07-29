#!/usr/bin/env python3
"""向量归一化迁移脚本 — 修复 ChromaDB 中可能存在的未归一化向量。

背景:
  2026-06-14 之前, _OllamaEmbeddingFn 返回 Ollama 原始向量 (norm≈26, 未归一化),
  如果有向量通过 backend add_documents() 写入, 这些向量与 MD2RAG Indexer
  写入的归一化向量 (norm=1) 混在同一 collection 中, 导致:
  1. ChromaDB HNSW 索引结构损坏 (norm≈26 与 norm=1 混合)
  2. similarity = 1 - d²/2 公式对未归一化向量不成立

修复策略:
  1. 扫描每个 collection 的全部向量, 检测 norm 偏离 1.0 的向量
  2. 如果发现未归一化向量 → 归一化后用 collection.upsert() 写回
     (upsert 更新向量但保留 id/documents/metadata)
  3. 如果无偏离 → 跳过 (正常状态)

使用方法:
  python scripts/normalize_vectors.py           # 扫描 + 报告
  python scripts/normalize_vectors.py --fix     # 扫描 + 修复
  python scripts/normalize_vectors.py --rebuild # 删除 + 重建 collection (最彻底)
"""

import argparse
import math
import sys
from pathlib import Path

# 默认向量库路径 (相对于脚本位置)
DEFAULT_DB_DIR = str(Path(__file__).resolve().parent.parent / "vector_db")

# 所有 collection 名称模式
PATTERNS = [
    # 三级文本
    "md2rag_public_parent",
    "md2rag_public_child",
    "md2rag_restricted_parent",
    "md2rag_restricted_child",
    "md2rag_confidential_parent",
    "md2rag_confidential_child",
    # 三级图片
    "md2rag_public_images",
    "md2rag_restricted_images",
    "md2rag_confidential_images",
    # DocScan
    "md2rag_docscan_docscan_parent",
    "md2rag_docscan_docscan_child",
]

# 允许的 norm 偏离范围 (浮点误差容忍)
NORM_TOLERANCE = 0.01  # norm 在 [0.99, 1.01] 视为正常


def check_and_fix_collection(client, collection_name, fix=False, rebuild=False):
    """检查单个 collection 的向量归一化状态."""
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        print(f"  [SKIP] Collection '{collection_name}' 不存在")
        return None

    count = collection.count()
    if count == 0:
        print(f"  [OK]   Collection '{collection_name}' 为空 (0 vectors)")
        return {"name": collection_name, "count": 0, "bad_vectors": 0, "fixed": 0}

    print(f"  [SCAN] {collection_name}: {count} vectors...")

    # 获取全部向量 + 文本 + 元数据
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = data.get("embeddings")
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if embeddings is None or len(embeddings) == 0:
        print(f"  [SKIP] {collection_name}: 无嵌入数据")
        return {"name": collection_name, "count": count, "bad_vectors": 0, "fixed": 0}

    # 检测每个向量的 norm
    bad_indices = []
    norms = []
    for i, emb in enumerate(embeddings):
        norm = math.sqrt(sum(x * x for x in emb))
        norms.append(norm)
        if abs(norm - 1.0) > NORM_TOLERANCE:
            bad_indices.append(i)

    # 统计
    min_norm = min(norms) if norms else 0
    max_norm = max(norms) if norms else 0
    avg_norm = sum(norms) / len(norms) if norms else 0

    print(f"         Norm 统计: min={min_norm:.4f}, max={max_norm:.4f}, avg={avg_norm:.4f}")

    if not bad_indices:
        print(f"  [OK]   {collection_name}: 全部 {count} 向量 norm 在 [{1.0-NORM_TOLERANCE:.2f}, {1.0+NORM_TOLERANCE:.2f}] 范围内")
        return {"name": collection_name, "count": count, "bad_vectors": 0, "fixed": 0}

    print(f"  [WARN] {collection_name}: 发现 {len(bad_indices)}/{count} 向量 norm 偏离 1.0")
    # 显示前5个坏向量的详情
    for idx in bad_indices[:5]:
        print(f"         id={ids[idx]}, norm={norms[idx]:.4f}")

    if not fix and not rebuild:
        print(f"         → 需要 --fix 或 --rebuild 参数来修复")
        return {"name": collection_name, "count": count, "bad_vectors": len(bad_indices), "fixed": 0}

    if rebuild:
        # 最彻底修复: 删除 collection 并重建 (需要重新入库全部数据)
        print(f"  [REBUILD] 删除 {collection_name}...")
        client.delete_collection(name=collection_name)
        # 重建空 collection (保留原始 metadata 配置)
        new_coll = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"MD2RAG {collection_name} collection (rebuilt)"},
        )
        print(f"  [REBUILD] {collection_name} 已重建 (空), 需要重新入库数据")
        return {"name": collection_name, "count": count, "bad_vectors": len(bad_indices), "fixed": count, "rebuilt": True}

    # fix: 归一化坏向量并用 upsert 写回
    fixed_embeddings = list(embeddings)  # 复制
    for idx in bad_indices:
        emb = fixed_embeddings[idx]
        norm = norms[idx]
        if norm > 0:
            fixed_embeddings[idx] = [x / norm for x in emb]

    # 同时修复所有"正常"向量的微小浮点误差
    for i in range(len(fixed_embeddings)):
        norm = norms[i]
        if abs(norm - 1.0) > 1e-12 and norm > 0:
            fixed_embeddings[i] = [x / norm for x in fixed_embeddings[i]]

    # upsert: 更新向量但保留 id/documents/metadata
    print(f"  [FIX]  正在 upsert {count} 归一化向量到 {collection_name}...")
    collection.upsert(
        ids=ids,
        embeddings=fixed_embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"  [FIX]  {collection_name}: {len(bad_indices)} 坏向量已归一化, 全部 {count} 向量已 upsert")

    # 验证修复后结果
    verify_data = collection.get(include=["embeddings"])
    verify_embs = verify_data.get("embeddings", [])
    verify_bad = 0
    for emb in verify_embs:
        norm = math.sqrt(sum(x * x for x in emb))
        if abs(norm - 1.0) > NORM_TOLERANCE:
            verify_bad += 1

    if verify_bad == 0:
        print(f"  [VERIFY] {collection_name}: 验证通过, 全部向量 norm≈1.0")
    else:
        print(f"  [VERIFY] {collection_name}: 仍有 {verify_bad} 向量 norm 偏离! 可能需要 --rebuild")

    return {"name": collection_name, "count": count, "bad_vectors": len(bad_indices), "fixed": len(bad_indices)}


def main():
    parser = argparse.ArgumentParser(description="向量归一化迁移脚本")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="ChromaDB 目录路径")
    parser.add_argument("--fix", action="store_true", help="修复发现的未归一化向量 (upsert)")
    parser.add_argument("--rebuild", action="store_true", help="删除并重建 collection (最彻底, 需重新入库)")
    parser.add_argument("--extra", nargs="*", help="额外检查的 collection 名称")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    db_path = str(Path(args.db_dir).resolve())
    if not Path(db_path).exists():
        print(f"[ERROR] ChromaDB 目录不存在: {db_path}")
        sys.exit(1)

    print(f"[INIT] ChromaDB path: {db_path}")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # 获取实际存在的 collection 列表
    existing = client.list_collections()
    existing_names = {c.name if hasattr(c, "name") else str(c) for c in existing}

    # 合合标准列表 + 用户指定额外 + 实际存在的
    check_names = set(PATTERNS)
    if args.extra:
        check_names.update(args.extra)

    # 也检查实际存在但不在标准列表中的 md2rag_* collection
    for name in existing_names:
        if name.startswith("md2rag_") and name not in check_names:
            check_names.add(name)
            print(f"[INFO] 发现非标准 collection: {name}, 加入检查")

    # 只检查实际存在的
    to_check = sorted(n for n in check_names if n in existing_names)

    print(f"\n[SCAN] 将检查 {len(to_check)} 个 collection...")
    print("=" * 60)

    results = []
    total_bad = 0
    total_fixed = 0

    for name in to_check:
        result = check_and_fix_collection(client, name, fix=args.fix, rebuild=args.rebuild)
        if result:
            results.append(result)
            total_bad += result.get("bad_vectors", 0)
            total_fixed += result.get("fixed", 0)

    # 报告不存在于实际列表中的标准 collection
    missing = sorted(n for n in check_names if n not in existing_names and n in PATTERNS)
    if missing:
        print(f"\n[MISSING] 标准 collection 未找到: {missing}")
        print("          这些 collection 尚未创建 (未入库对应密级数据)")

    print("\n" + "=" * 60)
    print(f"[SUMMARY] 检查 {len(results)} 个 collection")
    print(f"          发现 {total_bad} 个未归一化向量")
    print(f"          修复 {total_fixed} 个向量")

    if total_bad > 0 and not args.fix and not args.rebuild:
        print(f"\n[ACTION] 发现未归一化向量! 请执行以下操作之一:")
        print(f"  python scripts/normalize_vectors.py --fix       # upsert 归一化向量 (推荐)")
        print(f"  python scripts/normalize_vectors.py --rebuild   # 删除重建 (需要重新入库全部数据)")
    elif total_bad == 0:
        print(f"\n[OK] 所有向量已归一化, 无需修复")

    if args.rebuild and total_bad > 0:
        print(f"\n[IMPORTANT] 使用了 --rebuild, 以下 collection 已被清空, 需要重新入库数据:")
        for r in results:
            if r.get("rebuilt"):
                print(f"  - {r['name']} (原有 {r['count']} 向量)")
        print(f"\n重新入库方法:")
        print(f"  cd MD2RAG && python -m md2rag.indexer index <数据目录>")
        print(f"  或通过 backend API: POST /api/v1/index")


if __name__ == "__main__":
    main()
