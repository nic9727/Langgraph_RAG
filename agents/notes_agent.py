"""
笔记查询 Agent 节点模块 (agents/notes_agent.py)

本模块实现了 LangGraph 中"笔记查询"分支的处理节点，当意图识别判定
用户问题属于"笔记查询"类别时，流程会路由到本模块。

主要功能：
- 从 mock_notes.json 文件加载模拟笔记数据（带缓存，只加载一次）
- 基于关键词匹配对笔记进行检索打分（标签匹配权重最高，标题次之，内容最低）
- 将匹配到的笔记格式化后交给 qwen-plus 大模型生成综合回答
- 回答中会标注笔记来源（作者、日期、标签等元信息）

依赖模块：
- agents.state    : LangGraph 全局状态定义
- llm.qwen_client : 通义千问大模型调用
- config.settings : 笔记文件路径配置
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.state import GraphState
from llm.qwen_client import chat
from config.settings import NOTES_PATH

logger = logging.getLogger(__name__)

# 笔记数据缓存，避免每次查询都重新读取文件
_notes_cache: list[dict] | None = None

# 笔记问答的提示词模板
NOTES_QA_PROMPT = """你是一个课堂笔记检索助手。根据以下笔记内容回答学生的问题。

## 匹配到的笔记
{notes}

## 学生问题
{query}

## 要求
1. 总结匹配到的笔记要点
2. 如果有多条笔记，综合整理后给出回答
3. 标注笔记来源（作者和标题）"""


def _load_notes() -> list[dict]:
    """
    加载笔记数据（带缓存）。
    首次调用时从 JSON 文件读取，后续调用直接返回缓存。
    """
    global _notes_cache
    if _notes_cache is not None:
        return _notes_cache

    path = Path(NOTES_PATH)
    if not path.exists():
        logger.warning("笔记文件未找到: %s", NOTES_PATH)
        _notes_cache = []
        return _notes_cache

    with open(path, encoding="utf-8") as f:
        _notes_cache = json.load(f)
    logger.info("已加载 %d 条笔记", len(_notes_cache))
    return _notes_cache


def _search_notes(query: str) -> list[dict]:
    """
    基于关键词的简单笔记检索。

    打分规则：
    - 查询中每个字符在笔记可搜索文本中出现：+1 分
    - 笔记标签命中查询关键词：+5 分（权重最高）
    - 笔记标题命中查询中的词：+3 分

    返回得分最高的前 5 条笔记。
    """
    notes = _load_notes()
    query_lower = query.lower()
    scored: list[tuple[float, dict]] = []

    for note in notes:
        score = 0.0
        # 拼接标题、内容、标签为可搜索文本
        searchable = (
            note.get("title", "")
            + " "
            + note.get("content", "")
            + " "
            + " ".join(note.get("tags", []))
        ).lower()

        # 字符级匹配打分
        for char in query_lower:
            if char in searchable:
                score += 1

        # 标签精确匹配打分（权重最高）
        for tag in note.get("tags", []):
            if tag.lower() in query_lower:
                score += 5

        # 标题词匹配打分
        title = note.get("title", "").lower()
        for word in query_lower.split():
            if word in title:
                score += 3

        if score > 0:
            scored.append((score, note))

    # 按得分降序排列，取前 5 条
    scored.sort(key=lambda x: x[0], reverse=True)
    return [note for _, note in scored[:5]]


def notes_search_node(state: GraphState) -> GraphState:
    """
    笔记检索节点：搜索匹配的笔记并调用大模型生成回答。

    这是 LangGraph 图中"笔记查询"分支的唯一节点，
    完成检索和生成两步操作后直接输出最终回答。
    """
    query = state["query"]
    logger.info("笔记检索: %s", query[:50])

    matched = _search_notes(query)

    if not matched:
        answer = "抱歉，没有找到与您问题相关的笔记。当前笔记库中的笔记可能不涵盖您查询的内容。"
        return {**state, "answer": answer}

    # 将匹配到的笔记格式化为文本，包含元信息
    notes_text = ""
    for note in matched:
        notes_text += (
            f"\n### {note['title']}\n"
            f"- 作者：{note.get('author', '未知')}\n"
            f"- 日期：{note.get('created_at', '未知')}\n"
            f"- 标签：{', '.join(note.get('tags', []))}\n"
            f"- 内容：{note['content']}\n"
        )

    # 调用大模型基于笔记内容生成综合回答
    answer = chat(
        user_message=NOTES_QA_PROMPT.format(notes=notes_text, query=query),
        system_prompt="你是一个课堂笔记检索助手。",
    )

    return {**state, "answer": answer}