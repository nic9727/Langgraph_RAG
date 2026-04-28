"""
规章制度问答 Agent 节点模块 (agents/regulation_agent.py)

本模块是 LangGraph 状态图中处理「规章制度问答」分支的核心节点集合。
当意图识别判定用户问题属于 regulation_qa 时，流程进入本模块。

处理逻辑分为三个节点，按顺序执行：
1. regulation_skill_node  — 技能分发节点：根据 sub_intent（学分/成绩/实习等）
   调用对应的 Skill 查询 SQLite 数据库或调用大模型获取结构化数据。
   如果 Skill 命中并返回结果，则直接跳到生成节点。
2. regulation_rag_node    — RAG 兜底节点：当没有 Skill 命中时，
   从规章制度知识库（ChromaDB）进行多路召回 + 重排序，检索相关文档。
3. regulation_generate_node — 回答生成节点：综合 Skill 查询结果和/或 RAG 检索结果，
   调用 qwen-plus 大模型生成最终的自然语言回答。

依赖模块：
- skills/skill_router.py  : Skill 路由器，分发到具体技能
- retrieval/              : 多路召回和重排序管线
- llm/qwen_client.py      : 大模型调用
"""

from __future__ import annotations

import logging

from agents.state import GraphState
from skills.skill_router import SkillRouter
from retrieval.multi_retriever import MultiRetriever
from retrieval.reranker import rerank_pipeline
from llm.qwen_client import chat
from config.settings import REGULATION_COLLECTION

logger = logging.getLogger(__name__)

# 全局 Skill 路由器实例（包含学分、成绩、实习等技能）
_skill_router = SkillRouter()

# 规章制度问答的提示词模板
REGULATION_QA_PROMPT = """你是学校教务处的老师。请根据以下学校规章制度信息回答学生的问题。

## 参考资料
{context}

## 学生问题
{query}

## 要求
1. 回答要准确，引用具体的规章制度条款
2. 如涉及数字（学分、百分比等），请给出精确数据
3. 如果问题超出了参考资料范围，请如实说明"""


def regulation_skill_node(state: GraphState) -> GraphState:
    """
    技能分发节点：尝试用专业 Skill 处理查询。

    根据意图识别得到的 sub_intent，将查询分发到对应的 Skill：
    - credit     → CreditSkill（查询 SQLite 学分数据）
    - grade      → GradeSkill（查询 SQLite 成绩评定规则）
    - internship → InternshipSkill（查询实习企业 + 大模型生成建议）

    如果 Skill 成功返回结果，将 skill_result 和 skill_source 写入状态；
    否则留空，后续由条件路由跳转到 RAG 兜底节点。
    """
    query = state["query"]
    sub_intent = state.get("sub_intent", "general")

    result = _skill_router.route(query, sub_intent)

    if result and result.success:
        logger.info("Skill '%s' 成功处理了查询", result.source)
        return {
            **state,
            "skill_result": result.data,
            "skill_source": result.source,
        }

    return {**state, "skill_result": "", "skill_source": ""}


def regulation_rag_node(state: GraphState) -> GraphState:
    """
    RAG 兜底节点：从规章制度知识库中检索相关文档。

    当 Skill 未命中时进入此节点，执行以下流程：
    1. 使用 BM25 + 向量检索进行多路召回（不使用 HyDE，规章类问题关键词匹配更重要）
    2. 合并去重召回结果
    3. 通过 RRF 混排 + CrossEncoder 精排进行两阶段重排序
    4. 将检索和重排结果写入状态
    """
    query = state["query"]
    logger.info("规章制度 RAG 检索: %s", query[:50])

    retriever = MultiRetriever(REGULATION_COLLECTION)
    results = retriever.retrieve(query, use_hyde=False)
    merged = MultiRetriever.merge_and_deduplicate(results)

    doc_dicts = [{"content": d.content, "metadata": d.metadata} for d in merged]
    result_dict = {"merged": merged}
    reranked = rerank_pipeline(query, result_dict)
    reranked_dicts = [{"content": d.content, "metadata": d.metadata} for d in reranked]

    return {**state, "retrieved_docs": doc_dicts, "reranked_docs": reranked_dicts}


def regulation_generate_node(state: GraphState) -> GraphState:
    """
    回答生成节点：综合 Skill 结果和 RAG 结果，调用大模型生成最终回答。

    三种生成策略：
    1. 有 Skill 结果 → 以数据库查询结果为主，RAG 文档为辅，调用大模型润色
    2. 无 Skill 但有 RAG 结果 → 纯文档问答模式，基于检索文档生成回答
    3. 都没有 → 返回兜底提示语
    """
    query = state["query"]
    skill_result = state.get("skill_result", "")
    reranked = state.get("reranked_docs", [])

    if skill_result:
        # 策略 1：Skill 结果为主，RAG 为辅
        context = f"## 数据库查询结果\n{skill_result}"
        if reranked:
            rag_parts = [doc["content"] for doc in reranked[:3]]
            context += "\n\n## 规章制度参考\n" + "\n\n".join(rag_parts)

        answer = chat(
            user_message=REGULATION_QA_PROMPT.format(context=context, query=query),
            system_prompt="你是学校教务处的老师，善于结合数据和规章来回答学生问题。",
        )
    elif reranked:
        # 策略 2：纯 RAG 文档问答
        context_parts = [doc["content"] for doc in reranked]
        context = "\n\n---\n\n".join(context_parts)
        answer = chat(
            user_message=REGULATION_QA_PROMPT.format(context=context, query=query),
            system_prompt="你是学校教务处的老师。",
        )
    else:
        # 策略 3：兜底提示
        answer = "抱歉，我没有找到与您问题相关的规章制度信息。请尝试更具体地描述您的问题。"

    return {**state, "answer": answer}