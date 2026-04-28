"""
学分查询技能模块 (skills/credit_skill.py)

本模块实现了「学分查询」技能，用于处理学生关于学分情况的提问。
当意图识别器判定用户问题属于「规章制度 → 学分」类别时，
SkillRouter 会将请求分发到本技能进行处理。

工作原理：
  1. 从上下文中获取学号（若未提供则使用默认学号 2021001）
  2. 调用 database.models.get_student_credits() 查询 SQLite 数据库，
     获取该学生的学分统计数据（已修总学分、必修/选修/实践学分、平均成绩等）
  3. 将查询结果与毕业学分要求进行对比，生成结构化的学分情况摘要文本
  4. 返回 SkillResult，摘要文本将作为上下文传给大模型生成最终回答

数据来源：
  SQLite 数据库中的 students、courses、enrollments 三张表，
  通过跨表联合查询汇总学分信息。

毕业学分要求（硬编码）：
  - 总学分要求：160 学分
  - 必修学分：≥ 110 学分
  - 选修学分：≥ 30 学分
  - 实践学分：≥ 20 学分
"""

from __future__ import annotations

import logging

from skills.base_skill import BaseSkill, SkillResult
from database.models import get_student_credits, execute_query

logger = logging.getLogger(__name__)

# 默认学号：当上下文中未提供学号时使用
DEFAULT_STUDENT_ID = "2021001"


class CreditSkill(BaseSkill):
    """
    学分查询技能。

    通过查询 SQLite 数据库获取学生的学分统计信息，
    并与毕业要求进行对比，生成详细的学分情况报告。
    """

    name = "credit_query"
    description = "查询学分相关信息，包括已获学分、学分要求等"

    def can_handle(self, query: str, sub_intent: str) -> bool:
        """当子意图为 "credit" 时，本技能可以处理。"""
        return sub_intent == "credit"

    def execute(self, query: str, context: dict | None = None) -> SkillResult:
        """
        执行学分查询。

        参数：
            query:   用户原始查询文本
            context: 可选上下文，可包含 "student_id" 指定要查询的学生学号

        返回：
            SkillResult，data 字段为格式化的学分情况摘要文本
        """
        # 从上下文中获取学号，若未提供则使用默认学号
        student_id = (context or {}).get("student_id", DEFAULT_STUDENT_ID)

        # 查询数据库获取学分统计
        credit_info = get_student_credits(student_id)
        if not credit_info:
            return SkillResult(
                success=False,
                data=f"未找到学号 {student_id} 的学分信息",
                source="SQLite:credits",
            )

        # 毕业学分要求（各类别最低学分数）
        graduation_total = 160
        required_min = 110
        elective_min = 30
        practice_min = 20

        # 提取查询结果中的各项学分数据
        total = credit_info.get("total_credits", 0) or 0
        required = credit_info.get("required_credits", 0) or 0
        elective = credit_info.get("elective_credits", 0) or 0
        practice = credit_info.get("practice_credits", 0) or 0
        avg_score = credit_info.get("avg_score", 0) or 0

        # 生成学分情况摘要文本
        summary = (
            f"学生 {credit_info.get('name', student_id)}（{credit_info.get('major', '未知')}专业，"
            f"{credit_info.get('grade', '?')}年级）的学分情况：\n\n"
            f"- 已获总学分：{total:.1f} / 毕业要求 {graduation_total} 学分\n"
            f"- 必修学分：{required:.1f} / 要求 {required_min} 学分\n"
            f"- 选修学分：{elective:.1f} / 要求 {elective_min} 学分\n"
            f"- 实践学分：{practice:.1f} / 要求 {practice_min} 学分\n"
            f"- 平均成绩：{avg_score:.1f} 分\n"
            f"- 已修课程数：{credit_info.get('course_count', 0)} 门\n\n"
            f"距毕业还需：{max(0, graduation_total - total):.1f} 学分"
        )

        return SkillResult(
            success=True,
            data=summary,
            source="SQLite:students+enrollments+courses",
            raw_data=dict(credit_info),
        )
