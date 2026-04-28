"""
课程教材问答 Agent 节点模块 (agents/course_agent.py)

本模块实现了 LangGraph 图中"课程问答"分支的三个处理节点：
1. course_retrieve_node  — 从课程教材知识库中进行多路召回（BM25 + 向量 + HyDE）
2. course_rerank_node    — 对召回结果进行 RRF 混排 + CrossEncoder 精排
3. course_generate_node  — 将重排序后的参考文档与学生问题一起发送给 qwen-plus，生成最终回答

数据流：用户问题 → 多路召回 → 重排序 → LLM 生成回答
"""

from __future__ import annotations

import logging

from agents.state import GraphState
from retrieval.multi_retriever import MultiRetriever
from retrieval.reranker import rerank_pipeline
from llm.qwen_client import chat
from config.settings import COURSE_COLLECTION

logger = logging.getLogger(__name__)

# 课程问答的提示词模板，指导大模型基于检索到的参考资料回答学生问题
COURSE_QA_PROMPT = """你是一位自然语言处理课程的助教。请根据以下参考资料回答学生的问题。

## 参考资料
{context}

## 学生问题
{query}

## 要求
1. 基于参考资料回答，如果资料中没有相关内容，请如实说明
2. 回答要准确、清晰、有条理
3. 适当使用专业术语，但要确保学生能理解
4. 如有必要，给出示例来帮助理解"""


def course_retrieve_node(state: GraphState) -> GraphState:
    """
    课程知识库多路召回节点。

    从 ChromaDB 中的课程教材集合进行 BM25 + 向量 + HyDE 三路并行检索，
    然后合并去重，将结果写入 state["retrieved_docs"]。
    """
    query = state["query"]
    logger.info("课程知识库检索: %s", query[:50])

    retriever = MultiRetriever(COURSE_COLLECTION)
    results = retriever.retrieve(query)

    # 合并三路结果并去重
    merged = MultiRetriever.merge_and_deduplicate(results)
    doc_dicts = [{"content": d.content, "metadata": d.metadata} for d in merged]

    return {**state, "retrieved_docs": doc_dicts}


def course_rerank_node(state: GraphState) -> GraphState:
    """
    课程文档重排序节点。

    对召回的文档执行两阶段重排序：
    第一阶段：RRF（Reciprocal Rank Fusion）混排，融合多路召回的排序信息
    第二阶段：CrossEncoder 精排，使用 bge-reranker-v2-m3 做精细语义打分
    最终选出 Top-K 最相关的文档写入 state["reranked_docs"]。
    """
    query = state["query"]
    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        return {**state, "reranked_docs": []}

    from knowledge_base.document_loader import Document

    docs = [Document(content=d["content"], metadata=d["metadata"]) for d in retrieved]
    result_dict = {"merged": docs}
    reranked = rerank_pipeline(query, result_dict)

    reranked_dicts = [{"content": d.content, "metadata": d.metadata} for d in reranked]
    return {**state, "reranked_docs": reranked_dicts}


def course_generate_node(state: GraphState) -> GraphState:
    """
    课程问答回答生成节点。

    将重排序后的参考文档拼接为上下文，连同学生问题一起发送给 qwen-plus 大模型，
    生成基于教材内容的回答。如果没有检索到相关文档，则返回提示信息。
    """
    query = state["query"]
    reranked = state.get("reranked_docs", [])

    if not reranked:
        answer = "抱歉，我在课程教材知识库中没有找到与您问题相关的内容。请确保知识库已构建，或尝试换一种方式提问。"
    else:
        # 拼接参考文档上下文，附带来源和页码信息
        context_parts = []
        for i, doc in enumerate(reranked, 1):
            meta = doc.get("metadata", {})
            source_info = f"[来源: {meta.get('source', '未知')}, 第{meta.get('page', '?')}页]"
            context_parts.append(f"### 参考 {i} {source_info}\n{doc['content']}")
        context = "\n\n".join(context_parts)

        answer = chat(
            user_message=COURSE_QA_PROMPT.format(context=context, query=query),
            system_prompt="你是一位专业的自然语言处理课程助教。",
        )

    return {**state, "answer": answer}