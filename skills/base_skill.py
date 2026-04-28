"""
技能基类模块 (skills/base_skill.py)

本模块定义了校园问答系统中所有「技能（Skill）」的抽象基类和统一返回格式。
技能是对特定类型规章制度问题的专门处理器，每种技能负责一个细分领域，
例如学分查询、成绩规则查询、实习企业查询等。

核心设计：
  - SkillResult: 技能执行结果的统一数据结构，包含执行状态、文本结果、
    数据来源和可选的原始数据。
  - BaseSkill:   所有技能的抽象基类，定义了两个必须实现的接口方法：
    * can_handle(): 判断当前技能是否能处理给定的查询和子意图
    * execute():    执行技能逻辑，返回 SkillResult

扩展方式：
  新增技能只需继承 BaseSkill，实现 can_handle() 和 execute() 方法，
  然后在 skills/skill_router.py 中注册即可自动被路由器发现和调用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SkillResult:
    """
    技能执行结果的统一数据结构。

    字段：
        success:  技能是否成功处理了问题
        data:     技能返回的文本结果（将作为上下文传给大模型生成最终回答）
        source:   数据来源标识（如 "sqlite_database"、"llm_generation"）
        raw_data: 可选的原始结构化数据（字典或列表），便于后续程序化处理
    """

    success: bool
    data: str
    source: str
    raw_data: dict | list | None = None


class BaseSkill(ABC):
    """
    技能抽象基类。

    所有具体技能（credit_skill、grade_skill、internship_skill 等）
    都必须继承此类并实现以下两个抽象方法。

    类属性：
        name:        技能名称标识（如 "credit"、"grade"、"internship"）
        description: 技能的功能描述（供日志和调试使用）
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def can_handle(self, query: str, sub_intent: str) -> bool:
        """
        判断当前技能是否能处理给定的查询。

        参数：
            query:      用户原始查询文本
            sub_intent: 意图分类器识别出的子意图（如 "credit"、"grade"、"internship"）

        返回：
            True 表示该技能可以处理，False 表示无法处理
        """
        ...

    @abstractmethod
    def execute(self, query: str, context: dict | None = None) -> SkillResult:
        """
        执行技能逻辑并返回结果。

        参数：
            query:   用户原始查询文本
            context: 可选的上下文信息字典（如用户身份、对话历史等）

        返回：
            SkillResult 实例，包含执行状态和结果数据
        """
        ...
