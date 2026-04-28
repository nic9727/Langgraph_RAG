"""
ChromaDB 向量数据库管理模块 (knowledge_base/vector_store.py)

本模块封装了 ChromaDB 持久化向量数据库的全部操作，是知识库存储层的核心组件。
所有文档的向量化存储、检索、删除等操作均通过本模块完成。

主要功能：
  - get_chroma_client():       获取 ChromaDB 持久化客户端的全局单例实例，
                               数据存储路径由 config.settings.CHROMA_PERSIST_DIR 指定。
  - get_or_create_collection(): 获取或创建指定名称的集合（Collection），
                               使用余弦相似度（cosine）作为 HNSW 索引的距离度量。
  - add_documents():           将分块后的文档批量写入指定集合，包含文本、向量和元数据。
                               支持分批写入（默认每批 100 条），避免内存溢出。
  - query_collection():        通过查询向量在集合中进行最近邻检索，
                               返回文档文本、元数据和距离分数。
  - get_all_documents():       获取集合中的全部文档（用于 BM25 关键词索引的构建）。
  - delete_collection():       删除指定集合（用于知识库强制重建场景）。
  - collection_count():        返回集合中的文档数量。

ChromaDB 持久化说明：
  使用 PersistentClient 模式，数据会自动持久化到磁盘。
  服务重启后无需重新构建知识库，直接加载即可使用。
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from config.settings import CHROMA_PERSIST_DIR
from knowledge_base.document_loader import Document
from knowledge_base.embedding import embed_texts

logger = logging.getLogger(__name__)

# 全局单例：缓存 ChromaDB 持久化客户端
_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    获取 ChromaDB 持久化客户端的全局单例实例。

    首次调用时在指定路径创建客户端并缓存，后续调用直接返回。
    存储路径由 config.settings.CHROMA_PERSIST_DIR 配置。
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info("ChromaDB 客户端已初始化，存储路径: %s", CHROMA_PERSIST_DIR)
    return _chroma_client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """
    获取或创建指定名称的集合（Collection）。

    集合使用余弦相似度（cosine）作为 HNSW 向量索引的距离度量方式，
    这与 bge-large-zh-v1.5 模型输出的归一化向量相匹配。

    参数：
        name: 集合名称（如 "course_textbook"、"school_regulations"）

    返回：
        ChromaDB Collection 实例
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection_name: str,
    documents: list[Document],
    batch_size: int = 100,
) -> int:
    """
    将分块后的文档批量写入指定的 ChromaDB 集合。

    流程：
    1. 按 batch_size 分批处理文档
    2. 对每批文档调用 embed_texts() 进行向量化
    3. 将文档 ID、向量、原文和元数据一起写入集合

    参数：
        collection_name: 目标集合名称
        documents:       待写入的 Document 列表
        batch_size:      每批写入的文档数量，默认 100

    返回：
        成功写入的文档总数
    """
    collection = get_or_create_collection(collection_name)

    total_added = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [doc.content for doc in batch]
        ids = [doc.metadata.get("chunk_id", f"doc_{i+j}") for j, doc in enumerate(batch)]
        metadatas = [doc.metadata for doc in batch]

        # 调用嵌入模块对文本批量向量化
        embeddings = embed_texts(texts).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_added += len(batch)
        logger.info("已写入批次 %d-%d（累计 %d 条）", i, i + len(batch), total_added)

    logger.info(
        "共向集合 '%s' 写入 %d 条文档", collection_name, total_added
    )
    return total_added


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> dict:
    """
    通过查询向量在指定集合中执行最近邻检索。

    参数：
        collection_name: 集合名称
        query_embedding: 查询向量（list[float]），由 embed_query() 生成
        n_results:       返回的最相似文档数量，默认 10
        where:           可选的元数据过滤条件（ChromaDB where 语法）

    返回：
        包含 documents、metadatas、distances 的字典
    """
    collection = get_or_create_collection(collection_name)
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def get_all_documents(collection_name: str) -> dict:
    """
    获取指定集合中的全部文档及其元数据。

    主要用于 BM25 关键词检索器的索引构建——需要获取所有文档文本
    来建立倒排索引。

    参数：
        collection_name: 集合名称

    返回：
        包含 documents 和 metadatas 的字典
    """
    collection = get_or_create_collection(collection_name)
    return collection.get(include=["documents", "metadatas"])


def delete_collection(collection_name: str) -> None:
    """
    删除指定名称的集合。

    用于知识库强制重建场景：先删除旧集合，再重新构建。

    参数：
        collection_name: 要删除的集合名称
    """
    client = get_chroma_client()
    client.delete_collection(collection_name)
    logger.info("已删除集合 '%s'", collection_name)


def collection_count(collection_name: str) -> int:
    """
    返回指定集合中的文档数量。

    参数：
        collection_name: 集合名称

    返回：
        集合中的文档条数
    """
    collection = get_or_create_collection(collection_name)
    return collection.count()
