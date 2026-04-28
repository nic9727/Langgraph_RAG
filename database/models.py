"""
SQLite 数据库模型与操作模块 (database/models.py)

本模块负责校园问答系统中所有结构化数据的存储与查询，基于 SQLite 实现。
主要职责：
- 定义数据库表结构（建表 SQL），包括：
    * students      — 学生基本信息（学号、姓名、专业、年级、入学年份）
    * courses       — 课程信息（课程ID、课程名、学分、类型、学期、院系）
    * enrollments   — 选课与成绩记录（学生-课程关联、分数、学期、状态）
    * grade_rules   — 成绩评定规则（规则名称、描述、百分比/绩点、分类）
    * internship_companies — 合作实习企业信息（企业名、行业、岗位、联系方式等）
- 提供数据库连接管理函数
- 提供通用 SQL 查询执行函数
- 提供各类业务查询函数（学分汇总、成绩规则、实习企业、学生课程列表）

Skills 技能模块（credit_skill、grade_skill、internship_skill）通过
调用本模块的查询函数来获取结构化数据。
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Any

from config.settings import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

# ===================== 建表 SQL =====================
# 使用 IF NOT EXISTS 确保重复执行不会报错
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    major TEXT NOT NULL,
    grade INTEGER NOT NULL,
    enrollment_year INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    credits REAL NOT NULL,
    course_type TEXT NOT NULL CHECK(course_type IN ('必修', '选修', '实践')),
    semester TEXT,
    department TEXT
);

CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    score REAL,
    semester TEXT NOT NULL,
    status TEXT DEFAULT '已完成',
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE IF NOT EXISTS grade_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    description TEXT NOT NULL,
    percentage REAL,
    category TEXT
);

CREATE TABLE IF NOT EXISTS internship_companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT NOT NULL,
    industry TEXT NOT NULL,
    positions TEXT NOT NULL,
    contact TEXT,
    cooperation_since INTEGER,
    description TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    """获取 SQLite 数据库连接，自动创建父目录，使用 Row 工厂便于字典访问。"""
    Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_tables() -> None:
    """执行建表 SQL，初始化所有数据库表。"""
    conn = get_connection()
    conn.executescript(CREATE_TABLES_SQL)
    conn.commit()
    conn.close()
    logger.info("数据库表初始化完成")


def execute_query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """
    执行任意 SQL 查询并返回结果列表。

    每一行结果被转换为 dict，键为列名，值为对应数据。
    适用于 SELECT 查询。
    """
    conn = get_connection()
    cursor = conn.execute(sql, params)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_student_credits(student_id: str) -> dict[str, Any]:
    """
    查询指定学生的学分汇总信息。

    返回字典包含：姓名、专业、年级、总学分、必修学分、选修学分、
    实践学分、已修课程数、平均成绩。
    若学生不存在则返回空字典。
    """
    rows = execute_query(
        """
        SELECT
            s.name, s.major, s.grade,
            SUM(c.credits) as total_credits,
            SUM(CASE WHEN c.course_type = '必修' THEN c.credits ELSE 0 END) as required_credits,
            SUM(CASE WHEN c.course_type = '选修' THEN c.credits ELSE 0 END) as elective_credits,
            SUM(CASE WHEN c.course_type = '实践' THEN c.credits ELSE 0 END) as practice_credits,
            COUNT(*) as course_count,
            AVG(e.score) as avg_score
        FROM enrollments e
        JOIN students s ON e.student_id = s.student_id
        JOIN courses c ON e.course_id = c.course_id
        WHERE e.student_id = ? AND e.status = '已完成'
        GROUP BY s.name
        """,
        (student_id,),
    )
    return rows[0] if rows else {}


def get_grade_rules() -> list[dict[str, Any]]:
    """查询所有成绩评定规则，按分类和 ID 排序返回。"""
    return execute_query("SELECT * FROM grade_rules ORDER BY category, id")


def get_internship_companies(industry: str | None = None) -> list[dict[str, Any]]:
    """
    查询合作实习企业列表。

    参数：
        industry: 可选，按行业关键词模糊筛选。为 None 时返回全部企业。
    """
    if industry:
        return execute_query(
            "SELECT * FROM internship_companies WHERE industry LIKE ?",
            (f"%{industry}%",),
        )
    return execute_query("SELECT * FROM internship_companies")


def get_student_courses(student_id: str) -> list[dict[str, Any]]:
    """查询指定学生的所有选课记录及成绩，按学期排序。"""
    return execute_query(
        """
        SELECT c.course_name, c.credits, c.course_type, e.score, e.semester
        FROM enrollments e
        JOIN courses c ON e.course_id = c.course_id
        WHERE e.student_id = ?
        ORDER BY e.semester
        """,
        (student_id,),
    )
