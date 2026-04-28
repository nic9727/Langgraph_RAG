"""
通义千问大模型客户端模块 (llm/qwen_client.py)

本模块封装了阿里云百炼平台 qwen-plus 大模型的 API 调用，
采用 OpenAI 兼容接口（dashscope compatible-mode），提供以下能力：
- chat()          : 普通对话，返回纯文本回答
- chat_with_json(): 对话并将回答解析为 JSON 字典
- chat_stream()   : 流式对话，逐块 yield 内容

API Key 和模型名称均已硬编码，无需额外配置即可使用。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ===================== 硬编码的 API 配置 =====================
API_KEY = "你的api"
BASE_URL = "你的url"
MODEL_NAME = "qwen-plus"

# 全局单例客户端
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取 OpenAI 兼容客户端的单例实例。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return _client


def chat(
    user_message: str,
    system_prompt: str = "你是一个有用的校园问答助手。",
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: str | None = None,
) -> str:
    """
    发送对话请求并返回大模型的文本回答。

    参数：
        user_message:  用户输入的问题
        system_prompt: 系统提示词，定义大模型的角色
        history:       对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]
        temperature:   生成温度，值越高回答越随机
        max_tokens:    最大生成 token 数
        model:         可选，覆盖默认模型名称
    返回：
        大模型的回答文本
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = get_client().chat.completions.create(
        model=model or MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_with_json(
    user_message: str,
    system_prompt: str = "你是一个有用的校园问答助手。请以JSON格式回答。",
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.3,
    model: str | None = None,
) -> dict[str, Any]:
    """
    发送对话请求并将回答解析为 JSON 字典。

    如果大模型返回的内容被 Markdown 代码块包裹（```json ... ```），
    会自动去除代码块标记后再解析。解析失败时返回 {"raw": 原始文本}。
    """
    raw = chat(
        user_message=user_message,
        system_prompt=system_prompt,
        history=history,
        temperature=temperature,
        model=model,
    )
    # 清理可能的 Markdown 代码块包裹
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # 去掉开头的 ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # 去掉结尾的 ```
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON 解析失败，原始回答前200字符: %s", raw[:200])
        return {"raw": raw}


def chat_stream(
    user_message: str,
    system_prompt: str = "你是一个有用的校园问答助手。",
    history: list[dict[str, str]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: str | None = None,
):
    """
    流式对话，逐块 yield 生成的内容片段。

    适用于 Chainlit 等需要实时展示回答的前端场景。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    stream = get_client().chat.completions.create(
        model=model or MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
