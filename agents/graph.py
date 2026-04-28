"""
LangGraph 主图编排模块 (agents/graph.py)

本模块是整个校园问答系统的核心编排层，基于 LangGraph 状态图构建。
主要职责：
1. 定义意图识别节点，调用 LLM 对用户问题进行三分类（课程问答/规章制度/笔记查询）
2. 根据识别到的意图，通过条件边将请求路由到对应的处理分支
3. 课程问答分支：多路召回 → 重排序 → LLM 生成回答
4. 规章制度分支：Skill 技能分发 → 若 Skill 无法处理则 fallback 到 RAG → LLM 生成回答
5. 笔记查询分支：关键词搜索笔记 → LLM 生成回答

状态图结构：
    START → 意图识别 → [课程问答 | 规章制度 | 笔记查询] → ... → END
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from agents.state import GraphState
from agents.course_agent import (
    course_retrieve_node,
    course_rerank_node,
    course_generate_node,
)
from agents.regulation_agent import (
    regulation_skill_node,
    regulation_rag_node,
    regulation_generate_node,
)
from agents.notes_agent import notes_search_node
from intent.classifier import classify_intent

logger = logging.getLogger(__name__)


# ── 节点函数 ──────────────────────────────────────────────

def intent_recognition_node(state: GraphState) -> GraphState:
    """意图识别节点：调用 LLM 对用户问题进行分类，输出主意图和子意图。"""
    query = state["query"]
    result = classify_intent(query)
    return {
        **state,
        "intent": result.intent,
        "sub_intent": result.sub_intent,
        "confidence": result.confidence,
        "keywords": result.keywords,
    }


# ── 路由函数 ──────────────────────────────────────────────

def route_by_intent(state: GraphState) -> str:
    """条件路由：根据意图识别结果将请求分发到不同的处理分支。"""
    intent = state.get("intent", "course_qa")
    mapping = {
        "course_qa": "course_retrieve",
        "regulation_qa": "regulation_skill",
        "notes_query": "notes_search",
    }
    return mapping.get(intent, "course_retrieve")


def route_after_skill(state: GraphState) -> str:
    """Skill 后路由：若 Skill 成功处理则直接生成回答，否则 fallback 到 RAG 检索。"""
    skill_result = state.get("skill_result", "")
    if skill_result:
        return "regulation_generate"
    return "regulation_rag"


# ── 状态图构建 ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """构建 LangGraph 状态机，注册所有节点和边。"""
    graph = StateGraph(GraphState)

    # 注册节点
    graph.add_node("intent_recognition", intent_recognition_node)

    # 课程问答路径的三个节点
    graph.add_node("course_retrieve", course_retrieve_node)
    graph.add_node("course_rerank", course_rerank_node)
    graph.add_node("course_generate", course_generate_node)

    # 规章制度路径的三个节点
    graph.add_node("regulation_skill", regulation_skill_node)
    graph.add_node("regulation_rag", regulation_rag_node)
    graph.add_node("regulation_generate", regulation_generate_node)

    # 笔记查询路径的节点
    graph.add_node("notes_search", notes_search_node)

    # 设置入口节点
    graph.set_entry_point("intent_recognition")

    # 意图路由（条件边）：根据 intent 分发到三条分支
    graph.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "course_retrieve": "course_retrieve",
            "regulation_skill": "regulation_skill",
            "notes_search": "notes_search",
        },
    )

    # 课程问答路径：召回 → 重排序 → 生成 → 结束
    graph.add_edge("course_retrieve", "course_rerank")
    graph.add_edge("course_rerank", "course_generate")
    graph.add_edge("course_generate", END)

    # 规章制度路径：Skill 分发 → 条件判断 → 生成/RAG → 结束
    graph.add_conditional_edges(
        "regulation_skill",
        route_after_skill,
        {
            "regulation_generate": "regulation_generate",
            "regulation_rag": "regulation_rag",
        },
    )
    graph.add_edge("regulation_rag", "regulation_generate")
    graph.add_edge("regulation_generate", END)

    # 笔记查询路径：搜索 → 结束
    graph.add_edge("notes_search", END)

    return graph


def get_compiled_graph():
    """编译并返回可执行的 LangGraph 图实例。"""
    graph = build_graph()
    return graph.compile()


def run_query(query: str, chat_history: list[dict] | None = None) -> dict:
    """
    便捷函数：将一条用户查询送入完整的 LangGraph 流程并返回结果。

    参数：
        query:        用户输入的问题文本
        chat_history: 可选的对话历史
    返回：
        包含 intent、answer、retrieved_docs 等字段的状态字典
    """
    compiled = get_compiled_graph()
    initial_state: GraphState = {
        "query": query,
        "intent": "",
        "sub_intent": "",
        "confidence": 0.0,
        "keywords": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "skill_result": "",
        "skill_source": "",
        "answer": "",
        "chat_history": chat_history or [],
    }
    result = compiled.invoke(initial_state)
    return result