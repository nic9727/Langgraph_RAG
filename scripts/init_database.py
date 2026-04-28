"""
数据库初始化脚本 (scripts/init_database.py)

本脚本用于在命令行中一键初始化 SQLite 数据库，包括：
  - 创建所有数据库表（students、courses、enrollments、grade_rules、internship_companies）
  - 填充预定义的模拟数据，用于开发和测试
  - 初始化完成后打印各表的数据统计信息

本脚本调用 database.init_db 模块完成实际的建表和数据填充工作。
在 Chainlit 应用启动时也会自动调用相同的初始化逻辑，
因此本脚本主要用于：
  - 首次部署时手动初始化数据库
  - 开发调试时快速重置数据库状态
  - 验证数据库表结构和数据是否正确

使用方式：
  python scripts/init_database.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径，确保可以正确导入各模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# 配置日志格式：时间 [模块名] 级别: 消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from database.init_db import init_database
from database.models import execute_query


def main():
    """数据库初始化主函数，执行建表、数据填充并打印统计信息。"""
    print("=" * 60)
    print("  校园问答系统 — 数据库初始化")
    print("=" * 60)

    # 执行数据库初始化（建表 + 填充模拟数据）
    init_database()

    # 查询并打印各表的数据统计
    print("\n📊 数据统计:")
    tables = ["students", "courses", "enrollments", "grade_rules", "internship_companies"]
    for table in tables:
        rows = execute_query(f"SELECT COUNT(*) as cnt FROM {table}")
        count = rows[0]["cnt"] if rows else 0
        print(f"  - {table}: {count} 条记录")

    print("\n✅ 数据库初始化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
