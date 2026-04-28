"""
全局配置模块 (config/settings.py)

本模块集中管理整个校园问答系统的所有配置项，包括：
- 阿里云百炼大模型 API 的连接参数（API Key、Base URL、模型名称）
- 本地词嵌入模型和重排序模型的名称
- ChromaDB 向量数据库和 SQLite 关系数据库的存储路径
- 教材 PDF、规章制度文档、笔记数据等文件路径
- 文档分块（chunking）参数
- 多路召回与重排序的 Top-K 参数

配置优先从 .env 文件加载，若 .env 中未定义则使用默认值。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（config 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# ===================== 大模型配置 =====================
# 阿里云百炼平台 API Key（写死）
DASHSCOPE_API_KEY = "sk-6eab71b446594503ab07642a4d2f9cce"
# OpenAI 兼容接口的 Base URL
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 调用的大模型名称：通义千问 qwen3.5-plus
QWEN_MODEL = "qwen-plus"

# ===================== 词嵌入 / 重排序模型 =====================
# BAAI 中文词嵌入模型，用于文档和查询的向量化
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
# BAAI 重排序模型，用于 CrossEncoder 精排
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ===================== 存储路径 =====================
# ChromaDB 向量数据库持久化目录
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
# SQLite 关系数据库文件路径
SQLITE_DB_PATH = str(BASE_DIR / os.getenv("SQLITE_DB_PATH", "./data/db/school.db"))

# ===================== 数据文件路径 =====================
# 自然语言处理教材 PDF 路径
TEXTBOOK_PDF_PATH = str(
    BASE_DIR / os.getenv("TEXTBOOK_PDF_PATH", "./data/textbooks/nlp_textbook.pdf")
)
# 学校规章制度文档目录
REGULATIONS_DIR = str(
    BASE_DIR / os.getenv("REGULATIONS_DIR", "./data/regulations")
)
# 模拟笔记数据 JSON 路径
NOTES_PATH = str(
    BASE_DIR / os.getenv("NOTES_PATH", "./data/notes/mock_notes.json")
)

# ===================== 文档分块参数 =====================
# 每个 chunk 的最大字符数
CHUNK_SIZE = 800
# 相邻 chunk 之间的重叠字符数
CHUNK_OVERLAP = 150

# ===================== 检索参数 =====================
# BM25 关键词检索返回的 Top-K 数量
BM25_TOP_K = 10
# 向量语义检索返回的 Top-K 数量
VECTOR_TOP_K = 10
# HyDE 假设文档检索返回的 Top-K 数量
HYDE_TOP_K = 10
# RRF 混排后送入精排的候选数量
RERANK_TOP_N = 20
# 最终返回给 LLM 生成的文档数量
FINAL_TOP_K = 5
# RRF 融合公式中的 k 参数
RRF_K = 60

# ===================== ChromaDB 集合名称 =====================
# 课程教材知识库集合
COURSE_COLLECTION = "course_textbook"
# 学校规章制度知识库集合
REGULATION_COLLECTION = "school_regulations"
