"""
Chainlit 应用入口模块 (app.py)

本模块是校园智能问答系统的 Web 前端入口，基于 Chainlit 框架构建对话式界面。
主要职责：
- 用户连接时初始化数据库和会话状态
- 接收用户输入的自然语言问题
- 调用 LangGraph 主图执行完整的意图识别 → 检索/技能 → 生成流程
- 将回答、意图识别结果、参考文档来源等信息格式化后返回用户
- 维护对话历史（最近 20 轮），支持多轮对话上下文

启动方式：chainlit run app.py
访问地址：http://localhost:8000
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，确保各模块可以正常导入
sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl

from agents.graph import run_query
from database.init_db import init_database

# 配置全局日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 意图标签映射：将英文意图标识转换为用户友好的中文标签
INTENT_LABELS = {
    "course_qa": "📚 课程教材问答",
    "regulation_qa": "📋 规章制度问答",
    "notes_query": "📝 笔记查询",
}


@cl.on_chat_start
async def on_start():
    """用户连接时的初始化回调：初始化数据库并发送欢迎消息。"""
    init_database()
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content=(
            "# 🎓 校园智能问答系统\n\n"
            "欢迎使用校园问答系统！我可以帮你：\n\n"
            "- **📚 课程问答** — 自然语言处理教材相关问题\n"
            "- **📋 规章制度** — 学分、成绩、实习等校规查询\n"
            "- **📝 笔记查询** — 搜索课堂笔记\n\n"
            "请直接输入你的问题吧！"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户发送的消息：执行问答流程并返回结果。"""
    query = message.content.strip()
    if not query:
        return

    # 获取当前会话的对话历史
    chat_history = cl.user_session.get("chat_history", [])

    # 先发送一条"正在分析"的占位消息，后续更新为最终回答
    thinking_msg = cl.Message(content="🔍 正在分析你的问题...")
    await thinking_msg.send()

    try:
        # 在后台线程中执行同步的 RAG 流水线，避免阻塞事件循环导致 WebSocket 断连
        result = await asyncio.to_thread(run_query, query, chat_history=chat_history)

        # 提取意图识别结果
        intent = result.get("intent", "unknown")
        sub_intent = result.get("sub_intent", "none")
        confidence = result.get("confidence", 0)
        answer = result.get("answer", "抱歉，处理过程中出现了问题。")

        # 构建意图识别的元信息展示
        intent_label = INTENT_LABELS.get(intent, intent)
        meta_info = f"> **意图识别**: {intent_label}"
        if sub_intent and sub_intent != "none":
            meta_info += f" | **子分类**: {sub_intent}"
        meta_info += f" | **置信度**: {confidence:.0%}\n\n"

        # 构建数据来源和参考文档信息
        source_info = ""
        skill_source = result.get("skill_source", "")
        if skill_source:
            source_info += f"\n\n---\n📊 *数据来源: {skill_source}*"

        reranked = result.get("reranked_docs", [])
        if reranked:
            source_info += "\n\n---\n📄 *参考文档:*\n"
            for i, doc in enumerate(reranked[:3], 1):
                meta = doc.get("metadata", {})
                src = meta.get("source", "未知")
                page = meta.get("page", "")
                chapter = meta.get("chapter", "")
                ref = f"{i}. {src}"
                if page:
                    ref += f" (第{page}页)"
                if chapter:
                    ref += f" - {chapter}"
                source_info += ref + "\n"

        # 将占位消息更新为最终的完整回答
        thinking_msg.content = meta_info + answer + source_info
        await thinking_msg.update()

        # 更新对话历史，保留最近 20 轮
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        cl.user_session.set("chat_history", chat_history)

    except Exception as e:
        logger.exception("处理查询时出错")
        thinking_msg.content = f"❌ 处理过程中出现错误：{str(e)}"
        await thinking_msg.update()