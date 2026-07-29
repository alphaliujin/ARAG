"""MD2RAG CLI - 命令行工具."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from md2rag.config import MD2RAGConfig, load_config
from md2rag.indexer import Indexer
from md2rag.loader import VALID_CLASSIFICATIONS

# CLI 的密级 choice 统一从 loader 派生,sorted 让 --help 顺序稳定
_CLS_CHOICES = sorted(VALID_CLASSIFICATIONS)


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to md2rag.conf configuration file.",
)
@click.pass_context
def cli(ctx, config_path):
    """MD2RAG - 将 X2MD 生成的 Markdown 切片向量化并存入向量数据库."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@cli.command()
@click.argument("md_file", type=click.Path(exists=True))
@click.option(
    "--classification",
    "-c",
    type=click.Choice(_CLS_CHOICES),
    default=None,
    help="Document classification level. Auto-detected if not specified.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["auto", "chunk", "parent-child", "parents-only", "children-only"]),
    default="auto",
    help="Chunk loading strategy.",
)
@click.pass_context
def index_file(ctx, md_file, classification, strategy):
    """索引单个 MD 文件."""
    config = ctx.obj["config"]
    indexer = Indexer(config)

    click.echo(f"Indexing: {md_file}")
    result = indexer.index_md_file(md_file, classification, strategy)

    click.echo(f"Status: {result.status}")
    click.echo(f"Message: {result.message}")
    if result.chunks_added > 0:
        click.echo(f"Chunks added: {result.chunks_added}")
    if result.errors:
        click.echo(f"Errors: {len(result.errors)}")


@cli.command()
@click.option(
    "--classification",
    "-c",
    type=click.Choice(_CLS_CHOICES),
    default=None,
    help="Filter by classification level.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["auto", "chunk", "parent-child", "parents-only", "children-only"]),
    default="auto",
    help="Chunk loading strategy.",
)
@click.pass_context
def index_all(ctx, classification, strategy):
    """索引所有 MD 文件."""
    config = ctx.obj["config"]
    indexer = Indexer(config)

    click.echo("Scanning MD directory...")
    result = indexer.index_directory(classification, strategy)

    click.echo(f"Status: {result.status}")
    click.echo(f"Message: {result.message}")
    if result.errors:
        click.echo(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            click.echo(f"  - {error}")


@cli.command()
@click.pass_context
def stats(ctx):
    """查看向量数据库统计信息."""
    config = ctx.obj["config"]
    indexer = Indexer(config)

    stats = indexer.get_stats()
    click.echo("Vector Database Statistics:")
    click.echo(f"  Public:       {stats.get('public', 0)} documents")
    click.echo(f"  Restricted:   {stats.get('restricted', 0)} documents")
    click.echo(f"  Confidential: {stats.get('confidential', 0)} documents")
    click.echo(f"  Total:        {sum(stats.values())} documents")


@cli.command()
@click.option(
    "--classification",
    "-c",
    type=click.Choice(_CLS_CHOICES),
    default=None,
    help="Clear specific classification. Clears all if not specified.",
)
@click.confirmation_option(
    prompt="Are you sure you want to clear the vector database?"
)
@click.pass_context
def clear(ctx, classification):
    """清空向量数据库."""
    config = ctx.obj["config"]
    indexer = Indexer(config)

    indexer.clear(classification)
    if classification:
        click.echo(f"Cleared {classification} collection.")
    else:
        click.echo("Cleared all collections.")


@cli.command()
@click.option(
    "--classification",
    "-c",
    type=click.Choice(_CLS_CHOICES),
    default=None,
    help="Filter by classification level.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["auto", "chunk", "parent-child", "parents-only", "children-only"]),
    default="auto",
    help="Chunk loading strategy.",
)
@click.pass_context
def dry_run(ctx, classification, strategy):
    """试运行：扫描并显示将要索引的文件，但不执行实际索引."""
    from md2rag.loader import ChunkLoader

    config = ctx.obj["config"]
    loader = ChunkLoader(config.md_dir)

    click.echo("=== Dry Run ===")
    click.echo(f"Strategy: {strategy}")
    click.echo(f"Classification: {classification or 'all'}")
    click.echo()

    files = list(loader.discover_files(classification))
    click.echo(f"Found {len(files)} chunk files:")

    for file_path in files:
        records = loader.load_file(file_path)
        click.echo(f"  {file_path.name} ({len(records)} chunks)")


if __name__ == "__main__":
    cli()
