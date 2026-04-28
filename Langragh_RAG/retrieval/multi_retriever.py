"""
多路召回编排模块 (retrieval/multi_retriever.py)

本模块实现了多路召回（Multi-Route Retrieval）的编排逻辑，
是 RAG 检索阶段的核心调度器。它将三种不同的检索策略整合到统一的接口中，
并提供结果合并与去重功能。

三路召回通道：
  1. BM25 关键词检索  — 精确匹配关键词，擅长处理包含专有名词和术语的查询
  2. Vector 向量检索  — 语义相似度匹配，擅长理解查询意图和同义词
  3. HyDE 假设文档检索 — 先生成假设回答再检索，擅长弥补问答语义鸿沟

主要功能：
  - retrieve():              运行所有启用的检索通道，返回按检索方法分组的结果字典
  - merge_and_deduplicate(): 将多路结果合并去重，使用文档内容前 200 字符作为
                             去重键，避免相同文档被多条通道重复返回

设计思路：
  三路召回各有优势和短板，通过多路并行+结果融合的方式，可以显著提升整体召回率。
  每条通道的 Top-K 参数可在 config.settings 中独立配置。
  各通道可通过 retrieve() 的布尔参数灵活开关。
"""

from __future__ import annotations

import logging

from knowledge_base.document_loader import Document
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hyde_retriever import HyDERetriever
from config.settings import BM25_TOP_K, VECTOR_TOP_K, HYDE_TOP_K

logger = logging.getLogger(__name__)


class MultiRetriever:
    """
    多路召回编排器。

    统一管理 BM25、Vector、HyDE 三条检索通道，
    提供一次调用即可完成多路召回和结果合并去重的能力。
    """

    def __init__(self, collection_name: str):
        """
        初始化多路召回编排器，创建三个子检索器实例。

        参数：
            collection_name: 要检索的 ChromaDB 集合名称
        """
        self.bm25 = BM25Retriever(collection_name)
        self.vector = VectorRetriever(collection_name)
        self.hyde = HyDERetriever(collection_name)

    def retrieve(
        self,
        query: str,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_hyde: bool = True,
    ) -> dict[str, list[Document]]:
        """
        运行所有启用的检索通道，返回按检索方法分组的结果。

        参数：
            query:      用户查询文本
            use_bm25:   是否启用 BM25 关键词检索，默认开启
            use_vector: 是否启用向量语义检索，默认开启
            use_hyde:   是否启用 HyDE 假设文档检索，默认开启

        返回：
            字典，键为检索方法名（"bm25"/"vector"/"hyde"），
            值为对应通道返回的 Document 列表
        """
        results: dict[str, list[Document]] = {}

        if use_bm25:
            logger.info("正在执行 BM25 关键词检索...")
            results["bm25"] = self.bm25.retrieve(query, top_k=BM25_TOP_K)
            logger.info("BM25 召回 %d 条结果", len(results["bm25"]))

        if use_vector:
            logger.info("正在执行向量语义检索...")
            results["vector"] = self.vector.retrieve(query, top_k=VECTOR_TOP_K)
            logger.info("向量检索召回 %d 条结果", len(results["vector"]))

        if use_hyde:
            logger.info("正在执行 HyDE 假设文档检索...")
            results["hyde"] = self.hyde.retrieve(query, top_k=HYDE_TOP_K)
            logger.info("HyDE 召回 %d 条结果", len(results["hyde"]))

        return results

    @staticmethod
    def merge_and_deduplicate(
        results: dict[str, list[Document]],
    ) -> list[Document]:
        """
        合并多路召回结果并去重。

        使用文档内容的前 200 个字符作为去重键。当多条通道召回了
        相同的文档时，仅保留首次出现的版本（即优先保留排在前面的通道的结果）。

        参数：
            results: retrieve() 返回的按检索方法分组的结果字典

        返回：
            去重后的 Document 列表（保持原始顺序）
        """
        seen_contents: set[str] = set()
        merged: list[Document] = []

        for method, docs in results.items():
            for doc in docs:
                # 用内容前 200 字符作为去重键，避免完全相同的文档重复出现
                content_key = doc.content[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    merged.append(doc)

        return merged
