"""
BM25 关键词检索器模块 (retrieval/bm25_retriever.py)

本模块实现了基于 BM25 算法的关键词检索器，是多路召回策略中的
「关键词召回」通道。BM25 擅长精确匹配用户查询中的关键词，
与向量语义检索形成互补。

工作原理：
  1. 从 ChromaDB 集合中加载全部文档文本
  2. 使用 jieba 分词库的搜索引擎模式（cut_for_search）对每篇文档进行分词，
     该模式会在精确分词的基础上对长词再进行细粒度切分，提高召回率
  3. 基于分词结果构建 BM25Okapi 倒排索引
  4. 查询时同样对用户输入进行 jieba 搜索分词，然后计算 BM25 得分
  5. 按得分降序排列，返回 Top-K 个得分大于 0 的文档

特点：
  - 懒加载：首次调用 retrieve() 时才构建索引，避免启动时的额外开销
  - 返回的每个文档元数据中会附带 bm25_score 和 retrieval_method 标识，
    便于后续的混合排序（RRF）和结果追溯
"""

from __future__ import annotations

import logging

import jieba
from rank_bm25 import BM25Okapi

from knowledge_base.document_loader import Document
from knowledge_base.vector_store import get_all_documents
from config.settings import BM25_TOP_K

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    基于 BM25Okapi 算法的关键词检索器。

    绑定到一个 ChromaDB 集合，首次检索时自动加载全部文档并构建倒排索引。
    """

    def __init__(self, collection_name: str):
        """
        初始化检索器。

        参数：
            collection_name: 要检索的 ChromaDB 集合名称
        """
        self.collection_name = collection_name
        self._documents: list[Document] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def _load_corpus(self) -> None:
        """
        从 ChromaDB 集合中加载全部文档并构建 BM25 倒排索引。

        流程：
        1. 调用 get_all_documents() 获取集合中的全部文档文本和元数据
        2. 对每篇文档使用 jieba.cut_for_search() 进行搜索引擎模式分词
        3. 用分词结果初始化 BM25Okapi 索引
        """
        data = get_all_documents(self.collection_name)
        docs_list = data.get("documents", [])
        metas_list = data.get("metadatas", [])

        if not docs_list:
            logger.warning("集合 '%s' 中没有文档，无法构建 BM25 索引", self.collection_name)
            return

        self._documents = []
        self._corpus_tokens = []

        for text, meta in zip(docs_list, metas_list):
            self._documents.append(Document(content=text, metadata=meta or {}))
            # 使用搜索引擎模式分词：在精确分词基础上对长词再细切，提高召回率
            tokens = list(jieba.cut_for_search(text))
            self._corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(
            "已为集合 '%s' 构建 BM25 索引，共 %d 篇文档",
            self.collection_name,
            len(self._documents),
        )

    def retrieve(self, query: str, top_k: int = BM25_TOP_K) -> list[Document]:
        """
        执行 BM25 关键词检索。

        参数：
            query:  用户查询文本
            top_k:  返回的最大文档数量，默认由 config.settings.BM25_TOP_K 控制

        返回：
            按 BM25 得分降序排列的 Document 列表，每个文档的 metadata 中
            附带 bm25_score（浮点数）和 retrieval_method（"bm25"）字段
        """
        # 懒加载：首次检索时构建索引
        if self._bm25 is None:
            self._load_corpus()
        if self._bm25 is None or not self._documents:
            return []

        # 对查询文本进行搜索引擎模式分词
        query_tokens = list(jieba.cut_for_search(query))
        scores = self._bm25.get_scores(query_tokens)

        # 按得分降序取 Top-K
        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # 过滤掉得分为 0 的文档，并附加检索方法标识
        results = []
        for idx in scored_indices:
            if scores[idx] > 0:
                doc = self._documents[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                doc.metadata["retrieval_method"] = "bm25"
                results.append(doc)
        return results
