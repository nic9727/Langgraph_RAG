"""
技能路由器模块 (skills/skill_router.py)

本模块实现了技能路由器（SkillRouter），负责根据意图识别的子意图（sub_intent）
将用户的规章制度类问题分发到对应的专门技能进行处理。

路由逻辑：
  1. 遍历所有已注册的技能实例
  2. 调用每个技能的 can_handle() 方法，判断其是否能处理当前子意图
  3. 找到第一个匹配的技能后立即执行其 execute() 方法并返回结果
  4. 若无技能匹配，返回 None，由上层调用方（regulation_agent）回退到 RAG 检索

已注册的技能：
  - CreditSkill     (sub_intent="credit")     → 学分查询，查 SQLite 数据库
  - GradeSkill      (sub_intent="grade")      → 成绩规则查询，查 SQLite 数据库
  - InternshipSkill (sub_intent="internship") → 实习企业查询，查 SQLite + 大模型

扩展方式：
  新增技能只需在 __init__() 的 self.skills 列表中追加新的技能实例即可，
  路由器会自动在路由时检查新技能的 can_handle() 方法。
"""

from __future__ import annotations

import logging

from skills.base_skill import BaseSkill, SkillResult
from skills.credit_skill import CreditSkill
from skills.grade_skill import GradeSkill
from skills.internship_skill import InternshipSkill

logger = logging.getLogger(__name__)


class SkillRouter:
    """
    技能路由器。

    维护一组已注册的技能实例，根据子意图匹配并执行相应技能。
    """

    def __init__(self):
        """初始化路由器，注册所有可用的技能实例。"""
        self.skills: list[BaseSkill] = [
            CreditSkill(),
            GradeSkill(),
            InternshipSkill(),
        ]

    def route(
        self, query: str, sub_intent: str, context: dict | None = None
    ) -> SkillResult | None:
        """
        根据子意图匹配并执行对应的技能。

        按注册顺序依次检查各技能的 can_handle() 方法，
        找到第一个匹配的技能后执行并返回结果。

        参数：
            query:      用户原始查询文本
            sub_intent: 意图分类器识别出的子意图（如 "credit"、"grade"、"internship"）
            context:    可选上下文信息（如学号等），会传递给技能的 execute() 方法

        返回：
            SkillResult 实例（技能匹配成功时），或 None（无技能匹配时）
        """
        for skill in self.skills:
            if skill.can_handle(query, sub_intent):
                logger.info("路由到技能: %s", skill.name)
                return skill.execute(query, context)

        logger.info("子意图 '%s' 无匹配技能，返回 None（将回退到 RAG）", sub_intent)
        return None

    def list_skills(self) -> list[dict[str, str]]:
        """
        列出所有已注册的技能信息。

        返回：
            字典列表，每个字典包含 name（技能名称）和 description（功能描述）
        """
        return [{"name": s.name, "description": s.description} for s in self.skills]
