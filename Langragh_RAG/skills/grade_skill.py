"""
成绩规则查询技能模块 (skills/grade_skill.py)

本模块实现了「成绩规则查询」技能，用于处理学生关于成绩评定规则、
绩点计算方式等方面的提问。当意图识别器判定用户问题属于
「规章制度 → 成绩」类别时，SkillRouter 会将请求分发到本技能。

工作原理：
  1. 从 SQLite 数据库的 grade_rules 表中查询所有成绩评定规则
  2. 按规则类别（category）进行分组整理，生成结构化的 Markdown 格式文本
  3. 如果上下文中提供了学号，还会额外查询该学生的课程成绩列表
  4. 将规则说明和个人成绩（如有）合并为完整的文本返回

规则类别示例：
  - 成绩构成：平时成绩占比、期末考试占比、实验报告占比等
  - 绩点转换：百分制到绩点的转换规则
  - 补考重修：补考和重修的相关规定
  - 学业预警：学分不达标时的预警机制

数据来源：
  - 成绩规则：SQLite grade_rules 表
  - 个人成绩：SQLite enrollments + courses 联合查询
"""

from __future__ import annotations

import logging

from skills.base_skill import BaseSkill, SkillResult
from database.models import get_grade_rules, get_student_courses

logger = logging.getLogger(__name__)

# 默认学号：当上下文中未提供学号时使用
DEFAULT_STUDENT_ID = "2021001"


class GradeSkill(BaseSkill):
    """
    成绩规则查询技能。

    查询数据库中的成绩评定规则并按类别组织为 Markdown 格式，
    可选地附加指定学生的个人课程成绩。
    """

    name = "grade_query"
    description = "查询成绩评定规则、绩点计算方式等"

    def can_handle(self, query: str, sub_intent: str) -> bool:
        """当子意图为 "grade" 时，本技能可以处理。"""
        return sub_intent == "grade"

    def execute(self, query: str, context: dict | None = None) -> SkillResult:
        """
        执行成绩规则查询。

        参数：
            query:   用户原始查询文本
            context: 可选上下文，可包含 "student_id" 以附加个人成绩信息

        返回：
            SkillResult，data 字段为 Markdown 格式的成绩规则说明文本
        """
        # 从数据库查询所有成绩评定规则
        rules = get_grade_rules()
        if not rules:
            return SkillResult(
                success=False,
                data="未找到成绩评定规则数据",
                source="SQLite:grade_rules",
            )

        # 按规则类别（category）分组整理
        sections: dict[str, list[str]] = {}
        for rule in rules:
            cat = rule.get("category", "其他")
            desc = rule["rule_name"]
            # 对「成绩构成」类别的规则附加百分比信息
            if rule.get("percentage") is not None:
                desc += f"：{rule['description']}（{rule['percentage']}%）" if cat == "成绩构成" else f"：{rule['description']}"
            else:
                desc += f"：{rule['description']}"
            sections.setdefault(cat, []).append(desc)

        # 组装 Markdown 格式的输出文本
        output_parts = ["## 成绩评定规则\n"]
        for cat, items in sections.items():
            output_parts.append(f"\n### {cat}")
            for item in items:
                output_parts.append(f"- {item}")

        # 如果上下文中包含学号，额外查询并附加该学生的个人课程成绩
        student_id = (context or {}).get("student_id")
        if student_id:
            courses = get_student_courses(student_id)
            if courses:
                output_parts.append(f"\n### 你的课程成绩")
                for c in courses:
                    output_parts.append(
                        f"- {c['course_name']}({c['course_type']}, {c['credits']}学分): {c['score']}分 [{c['semester']}]"
                    )

        return SkillResult(
            success=True,
            data="\n".join(output_parts),
            source="SQLite:grade_rules",
            raw_data=rules,
        )
