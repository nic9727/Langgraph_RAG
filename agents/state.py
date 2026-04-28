"""
LangGraph 全局状态定义模块 (agents/state.py)

本模块定义了校园问答系统中 LangGraph 状态图的全局共享状态结构 GraphState。
所有图节点（意图识别、课程问答、规章制度问答、笔记查询等）都通过读写这个状态
来传递数据。

状态字段说明：
- query:          用户原始输入的问题文本
- intent:         主意图分类结果（course_qa / regulation_qa / notes_query）
- sub_intent:     子意图分类结果（credit / grade / internship / scholarship / general / none）
- confidence:     意图识别的置信度（0.0 ~ 1.0）
- keywords:       从用户问题中提取的关键词列表
- retrieved_docs: 多路召回阶段检索到的原始文档列表
- reranked_docs:  经过 RRF 混排 + CrossEncoder 精排后的文档列表
- skill_result:   Skills 技能执行返回的结构化结果文本
- skill_source:   技能数据来源标识（如 "SQLite:grade_rules"）
- answer:         最终由大模型生成的回答文本
- chat_history:   多轮对话历史，使用 LangGraph 的 add_messages 注解实现自动追加
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class GraphState(TypedDict):
    """LangGraph 状态图的全局共享状态，所有节点通过该状态传递数据。"""

    # 用户输入
    query: str

    # 意图识别结果
    intent: str
    sub_intent: str
    confidence: float
    keywords: list[str]

    # 检索结果
    retrieved_docs: list[dict]
    reranked_docs: list[dict]

    # 技能执行结果
    skill_result: str
    skill_source: str

    # 最终回答
    answer: str

    # 对话历史（自动追加）
    chat_history: Annotated[list[dict], add_messages]