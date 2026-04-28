"""
知识库构建入口模块 (knowledge_base/build_kb.py)

本模块是整个 RAG 知识库的构建入口，负责将原始文档转化为可检索的向量知识库。
完整流程为：加载文档 → 多级分块 → 向量化 → 写入 ChromaDB 持久化存储。

支持构建两个独立的知识库：
1. 课程教材知识库（course_textbook）
   - 数据源：data/textbooks/nlp_textbook.pdf（自然语言处理教材 PDF）
   - 流程：PDF 逐页加载 → 三级分块 → bge-large-zh-v1.5 向量化 → ChromaDB
2. 规章制度知识库（school_regulations）
   - 数据源：data/regulations/*.md（学校规章制度 Markdown 文档）
   - 流程：Markdown 按章节加载 → 三级分块 → 向量化 → ChromaDB

特性：
- 支持增量构建：如果知识库已有数据且未指定强制重建，则跳过
- 支持强制重建：传入 force_rebuild=True 会先删除旧集合再重新构建
- 可作为独立脚本直接运行：python -m knowledge_base.build_kb
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import (
    TEXTBOOK_PDF_PATH,
    REGULATIONS_DIR,
    COURSE_COLLECTION,
    REGULATION_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from knowledge_base.document_loader import load_pdf, load_markdown, load_directory
from knowledge_base.chunking import chunk_documents
from knowledge_base.vector_store import add_documents, collection_count, delete_collection

logger = logging.getLogger(__name__)


def build_course_kb(force_rebuild: bool = False) -> int:
    """
    构建课程教材知识库（从 NLP 教材 PDF）。

    参数：
        force_rebuild: 是否强制删除已有数据并重新构建

    返回：
        知识库中的 chunk 总数
    """
    if not Path(TEXTBOOK_PDF_PATH).exists():
        logger.warning("教材 PDF 未找到: %s，跳过课程知识库构建", TEXTBOOK_PDF_PATH)
        return 0

    # 强制重建时先删除已有集合
    if force_rebuild:
        try:
            delete_collection(COURSE_COLLECTION)
        except Exception:
            pass

    # 增量模式：已有数据则跳过
    existing = collection_count(COURSE_COLLECTION)
    if existing > 0 and not force_rebuild:
        logger.info("课程知识库已有 %d 个 chunk，跳过构建", existing)
        return existing

    logger.info("正在从 %s 构建课程知识库...", TEXTBOOK_PDF_PATH)
    docs = load_pdf(TEXTBOOK_PDF_PATH)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    added = add_documents(COURSE_COLLECTION, chunks)
    logger.info("课程知识库构建完成，共 %d 个 chunk", added)
    return added


def build_regulation_kb(force_rebuild: bool = False) -> int:
    """
    构建学校规章制度知识库（从 Markdown 文档）。

    参数：
        force_rebuild: 是否强制删除已有数据并重新构建

    返回：
        知识库中的 chunk 总数
    """
    reg_dir = Path(REGULATIONS_DIR)
    if not reg_dir.exists():
        logger.warning("规章制度目录未找到: %s", REGULATIONS_DIR)
        return 0

    if force_rebuild:
        try:
            delete_collection(REGULATION_COLLECTION)
        except Exception:
            pass

    existing = collection_count(REGULATION_COLLECTION)
    if existing > 0 and not force_rebuild:
        logger.info("规章制度知识库已有 %d 个 chunk，跳过构建", existing)
        return existing

    logger.info("正在从 %s 构建规章制度知识库...", REGULATIONS_DIR)
    docs = load_directory(REGULATIONS_DIR, "*.md")
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    added = add_documents(REGULATION_COLLECTION, chunks)
    logger.info("规章制度知识库构建完成，共 %d 个 chunk", added)
    return added


def build_all(force_rebuild: bool = False) -> dict[str, int]:
    """
    构建所有知识库（课程 + 规章制度）。

    返回：
        字典，键为知识库名称，值为对应的 chunk 数量
    """
    results = {}
    results["course"] = build_course_kb(force_rebuild)
    results["regulation"] = build_regulation_kb(force_rebuild)
    logger.info("全部知识库构建完成: %s", results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all(force_rebuild=True)
