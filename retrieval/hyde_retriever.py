"""
HyDE 假设文档嵌入检索器模块 (retrieval/hyde_retriever.py)

本模块实现了 HyDE（Hypothetical Document Embedding，假设文档嵌入）检索策略，
是多路召回中的第三条通道。HyDE 的核心思想是：先用大模型根据用户问题生成一段
「假设性回答文本」，再将这段假设文本向量化后去知识库中检索真实文档。

与直接用用户原始查询做向量检索相比，HyDE 的优势在于：
  - 假设文档在语义空间中更接近真正的答案文档，而非问题文本
  - 能弥补「问题」与「答案」之间的语义鸿沟（semantic gap）
  - 对短查询、模糊查询的召回效果有显著提升

完整工作流程：
  1. 将用户查询发送给 Qwen-Plus 大模型，要求其扮演教材作者撰写一段假设性回答
  2. 对生成的假设文档进行向量化（bge-large-zh-v1.5）
  3. 用假设文档向量在 ChromaDB 中执行最近邻检索
  4. 返回的真实文档附带 vector_distance、vector_score 和 retrieval_method 元数据

注意事项：
  - HyDE 需要额外调用一次大模型 API，会增加约 1-2 秒延迟
  - temperature 设为 0.5，在创造性和稳定性之间取平衡
  - 生成的假设文档限制在 300 字以内，避免过长导致语义漂移
"""

from __future__ import annotations

import logging

from knowledge_base.document_loader import Document
from knowledge_base.embedding import embed_query
from knowledge_base.vector_store import query_collection
from llm.qwen_client import chat
from config.settings import HYDE_TOP_K

logger = logging.getLogger(__name__)

# HyDE 假设文档生成的系统提示词
HYDE_SYSTEM_PROMPT = """你是一个自然语言处理领域的教材作者。
请根据用户的问题，写一段可能出现在教材中的、能回答该问题的段落。
要求：直接写内容，不要加前缀或解释，不超过300字。"""


class HyDERetriever:
    """
    假设文档嵌入检索器。

    通过大模型生成假设性回答，将其向量化后在知识库中检索语义最相近的真实文档。
    """

    def __init__(self, collection_name: str):
        """
        初始化 HyDE 检索器。

        参数：
            collection_name: 要检索的 ChromaDB 集合名称
        """
        self.collection_name = collection_name

    def _generate_hypothetical_doc(self, query: str) -> str:
        """
        调用 Qwen-Plus 大模型生成假设性文档。

        参数：
            query: 用户原始查询文本

        返回：
            大模型生成的假设性回答段落（不超过 300 字）
        """
        return chat(
            user_message=query,
            system_prompt=HYDE_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=512,
        )

    def retrieve(self, query: str, top_k: int = HYDE_TOP_K) -> list[Document]:
        """
        执行 HyDE 检索流程。

        参数：
            query:  用户查询文本
            top_k:  返回的最大文档数量，默认由 config.settings.HYDE_TOP_K 控制

        返回：
            按向量相似度排序的 Document 列表，每个文档的 metadata 中附带：
            - vector_distance: 余弦距离（越小越相似）
            - vector_score:    相似度得分（1 - distance，越大越相似）
            - retrieval_method: "hyde"
        """
        # 第一步：用大模型生成假设性文档
        hypo_doc = self._generate_hypothetical_doc(query)
        logger.debug("HyDE 生成的假设文档（前100字）: %s", hypo_doc[:100])

        # 第二步：将假设文档向量化，然后在知识库中检索
        hypo_vec = embed_query(hypo_doc)
        results = query_collection(
            self.collection_name,
            query_embedding=hypo_vec,
            n_results=top_k,
        )

        # 第三步：解析检索结果，附加元数据标识
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
                meta["retrieval_method"] = "hyde"
                documents.append(Document(content=text, metadata=meta))

        return documents
