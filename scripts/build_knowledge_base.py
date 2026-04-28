"""
知识库构建脚本 (scripts/build_knowledge_base.py)

本脚本是整个 RAG 知识库的一键构建入口，用于在命令行中执行知识库的
初始化或重建操作。它调用 knowledge_base.build_kb 模块完成实际构建。

支持两种运行模式：
  - 增量模式（默认）：如果知识库中已有数据，则跳过构建，不会重复写入
  - 强制重建模式（--force）：删除现有知识库数据后重新构建

使用方式：
  python scripts/build_knowledge_base.py           # 增量模式
  python scripts/build_knowledge_base.py --force    # 强制重建模式

构建完成后会打印每个知识库的构建结果（chunk 数量或跳过状态）。
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

from knowledge_base.build_kb import build_all


def main():
    """知识库构建主函数，解析命令行参数并执行构建流程。"""
    print("=" * 60)
    print("  校园问答系统 — 知识库构建")
    print("=" * 60)

    # 检查命令行参数是否包含 --force 标志
    force = "--force" in sys.argv
    if force:
        print("\n⚠️  强制重建模式：将删除并重新构建所有知识库\n")
    else:
        print("\n📦 增量模式：已有数据不会被重复构建\n")
        print("  使用 --force 参数强制重建\n")

    # 调用知识库构建模块执行实际构建
    results = build_all(force_rebuild=force)

    # 打印构建结果汇总
    print("\n" + "=" * 60)
    print("  构建结果")
    print("=" * 60)
    for name, count in results.items():
        status = "✅" if count > 0 else "⚠️ 跳过/无数据"
        print(f"  {status} {name}: {count} chunks")
    print("=" * 60)


if __name__ == "__main__":
    main()
