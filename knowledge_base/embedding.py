"""
词嵌入（Embedding）模块 (knowledge_base/embedding.py)

本模块封装了基于 BAAI/bge-large-zh-v1.5 模型的文本向量化功能，
是整个 RAG 检索流程中将文本转换为稠密向量的核心组件。

主要功能：
  - get_embedding_model(): 懒加载并缓存 SentenceTransformer 模型实例（全局单例），
    避免重复加载带来的内存和时间开销。
  - embed_texts(): 将一段或多段文本编码为归一化的稠密向量（numpy 数组），
    支持批量编码，超过 100 条时自动显示进度条。
  - embed_query(): 对单条查询文本编码，返回 Python list[float] 格式，
    可直接用于 ChromaDB 的向量查询接口。

模型说明：
  BAAI/bge-large-zh-v1.5 是北京智源人工智能研究院发布的中文通用文本嵌入模型，
  向量维度 1024，在中文语义相似度和检索任务上表现优异。
  模型首次加载时会自动从 HuggingFace 下载并缓存到本地。
"""

from __future__ import annotations

import logging
import os
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# 禁止 transformers / sentence-transformers 每次加载时联网检查模型更新，
# 仅使用本地缓存；首次下载后即可离线运行。
#os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
#os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 全局单例：缓存已加载的嵌入模型，避免重复初始化
_model: SentenceTransformer | None = None


def _select_device() -> str:
    """自动选择推理设备：优先 CUDA GPU，不可用时回退 CPU。"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("检测到 GPU: %s，将使用 CUDA 加速", name)
        return "cuda"
    logger.info("未检测到可用 GPU，使用 CPU 推理")
    return "cpu"


def get_embedding_model() -> SentenceTransformer:
    """
    获取词嵌入模型的全局单例实例。

    首次调用时加载模型并缓存，后续调用直接返回已缓存的实例。
    模型名称从 config.settings.EMBEDDING_MODEL 读取。
    优先使用 GPU 推理，若 CUDA 不可用则自动回退到 CPU。
    """
    global _model
    if _model is None:
        device = _select_device()
        logger.info("正在加载词嵌入模型: %s (device=%s)", EMBEDDING_MODEL, device)
        try:
            _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        except Exception as e:
            if device != "cpu":
                logger.warning("GPU 加载失败 (%s)，回退到 CPU", e)
                _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            else:
                raise
        logger.info("词嵌入模型加载完成 (device=%s)", _model.device.type)
    return _model


def embed_texts(texts: Union[str, list[str]], batch_size: int = 64) -> np.ndarray:
    """
    将文本编码为归一化的稠密向量。

    参数：
        texts:      单条文本字符串或文本列表
        batch_size: 批量编码时的批大小，默认 64

    返回：
        numpy 数组，形状为 (n, dim)，其中 n 为文本数量，dim 为向量维度（1024）。
        所有向量已做 L2 归一化，适合余弦相似度计算。
    """
    if isinstance(texts, str):
        texts = [texts]
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return np.array(embeddings)


def embed_query(query: str) -> list[float]:
    """
    对单条查询文本进行向量化，返回扁平的浮点数列表。

    此函数是 embed_texts 的便捷封装，返回 list[float] 格式，
    可直接传入 ChromaDB 的 query() 接口作为查询向量。

    参数：
        query: 用户查询文本

    返回：
        长度为 dim（1024）的浮点数列表
    """
    vec = embed_texts(query)
    return vec[0].tolist()
