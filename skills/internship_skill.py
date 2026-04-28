"""
实习查询技能模块 (skills/internship_skill.py)

本模块实现了「实习查询」技能，用于处理学生关于实习企业、实习要求、
岗位推荐等方面的提问。当意图识别器判定用户问题属于
「规章制度 → 实习」类别时，SkillRouter 会将请求分发到本技能。

工作原理（数据库 + 大模型混合策略）：
  1. 从 SQLite 数据库的 internship_companies 表中查询所有合作实习企业信息
  2. 将企业信息（名称、行业、岗位、合作年份、简介）格式化为 Markdown 列表
  3. 将格式化后的企业信息与用户问题一起组装成 Prompt，发送给 Qwen-Plus 大模型
  4. 大模型扮演「就业指导中心老师」的角色，基于真实企业数据为学生提供
     个性化的实习指导和企业推荐

与其他技能的区别：
  - credit_skill 和 grade_skill 仅查询数据库并格式化输出
  - internship_skill 采用「数据库 + 大模型」混合策略：
    数据库提供真实的企业数据（确保信息准确），
    大模型负责理解学生意图并生成个性化建议（提供智能推荐）

数据来源：
  - 结构化数据：SQLite internship_companies 表
  - 智能回答：Qwen-Plus 大模型生成（temperature=0.7，略高以增加回答多样性）
"""

from __future__ import annotations

import logging

from skills.base_skill import BaseSkill, SkillResult
from database.models import get_internship_companies
from llm.qwen_client import chat

logger = logging.getLogger(__name__)

# 实习查询 Prompt 模板：将企业信息和学生问题注入后发送给大模型
INTERNSHIP_PROMPT = """你是校园就业指导中心的老师。根据以下学校合作实习企业信息和学生的问题，给出详细且有帮助的回答。

## 合作企业信息
{company_info}

## 学生问题
{query}

请根据企业信息回答学生的问题。如果问题涉及具体的企业推荐，请根据学生的专业方向给出建议。"""


class InternshipSkill(BaseSkill):
    """
    实习查询技能。

    结合 SQLite 数据库中的真实企业数据和 Qwen-Plus 大模型的智能推荐能力，
    为学生提供个性化的实习指导和企业推荐。
    """

    name = "internship_query"
    description = "查询实习相关信息，包括合作企业、实习要求等"

    def can_handle(self, query: str, sub_intent: str) -> bool:
        """当子意图为 "internship" 时，本技能可以处理。"""
        return sub_intent == "internship"

    def execute(self, query: str, context: dict | None = None) -> SkillResult:
        """
        执行实习信息查询。

        流程：查询数据库获取企业列表 → 格式化为文本 → 大模型生成个性化建议。

        参数：
            query:   用户原始查询文本
            context: 可选上下文（当前未使用，预留扩展）

        返回：
            SkillResult，data 字段为大模型生成的实习指导回答
        """
        # 从数据库查询所有合作实习企业
        companies = get_internship_companies()

        if not companies:
            return SkillResult(
                success=False,
                data="未找到实习企业信息",
                source="SQLite:internship_companies",
            )

        # 将企业信息格式化为 Markdown 列表，作为大模型的参考数据
        company_lines = []
        for c in companies:
            company_lines.append(
                f"- **{c['company_name']}**（{c['industry']}）\n"
                f"  岗位：{c['positions']}\n"
                f"  合作始于：{c.get('cooperation_since', '未知')}年\n"
                f"  简介：{c.get('description', '无')}"
            )
        company_info = "\n".join(company_lines)

        # 调用大模型生成个性化的实习指导回答
        llm_response = chat(
            user_message=INTERNSHIP_PROMPT.format(
                company_info=company_info, query=query
            ),
            system_prompt="你是校园就业指导中心的老师，请给学生提供专业的实习指导。",
            temperature=0.7,
        )

        return SkillResult(
            success=True,
            data=llm_response,
            source="SQLite:internship_companies + LLM",
            raw_data=companies,
        )
