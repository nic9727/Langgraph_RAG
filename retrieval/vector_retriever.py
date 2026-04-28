"""
向量语义检索器模块 (retrieval/vector_retriever.py)

本模块实现了基于稠密向量的语义检索器，是多路召回策略中的
「向量召回」通道。它通过计算查询文本与知识库文档在高维向量空间中的
余弦相似度来找到语义最相关的文档。

工作流程：
  1. 使用 bge-large-zh-v1.5 模型将用户查询文本编码为 1024 维稠密向量
  2. 在 ChromaDB 集合中执行基于 HNSW 索引的近似最近邻检索
  3. 返回 Top-K 个最相似的文档，附带距离分数和检索方法标识

与 BM25 的互补性：
  - BM25 依赖关键词精确匹配，对同义词、近义词、改写表述无能为力
  - 向量检索通过语义嵌入捕捉深层语义关系，能理解 "机器翻译" 和 "MT" 的等价性
  - 两者结合可以显著提升整体召回率和覆盖度

返回的每个文档元数据中包含：
  - vector_distance: 余弦距离（0~2，越小越相似）
  - vector_score:    相似度得分（1 - distance，越大越相似）
  - retrieval_method: "vector"
"""

from __future__ import annotations

import logging

from knowledge_base.document_loader import Document
from knowledge_base.embedding import embed_query
from knowledge_base.vector_store import query_collection
from config.settings import VECTOR_TOP_K

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    基于稠密向量的语义检索器。

    使用 bge-large-zh-v1.5 嵌入模型 + ChromaDB 向量数据库进行语义最近邻检索。
    """

    def __init__(self, collection_name: str):
        """
        初始化向量检索器。

        参数：
            collection_name: 要检索的 ChromaDB 集合名称
        """
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = VECTOR_TOP_K) -> list[Document]:
        """
        执行向量语义检索。

        参数：
            query:  用户查询文本
            top_k:  返回的最大文档数量，默认由 config.settings.VECTOR_TOP_K 控制

        返回：
            按余弦相似度降序排列的 Document 列表，每个文档的 metadata 中附带：
            - vector_distance: 余弦距离
            - vector_score:    相似度得分（1 - distance）
            - retrieval_method: "vector"
        """
        # 将用户查询编码为稠密向量
        query_vec = embed_query(query)

        # 在 ChromaDB 中执行最近邻检索
        results = query_collection(
            self.collection_name,
            query_embedding=query_vec,
            n_results=top_k,
        )

        # 解析检索结果，附加元数据标识
        documents: list[Document] = []
        if not results or not results.get("documents"):
            return documents

        for texts, metas, dists in zip(
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            for text, meta, dist in zip(texts, metas, dists):
                meta = meta or {}
                meta["vector_distance"] = float(dist)
                meta["vector_score"] = 1.0 - float(dist)
                meta["retrieval_method"] = "vector"
                documents.append(Document(content=text, metadata=meta))

        return documents
