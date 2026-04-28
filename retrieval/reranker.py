"""
重排序模块 (retrieval/reranker.py)

本模块实现了两阶段重排序（Reranking）流水线，是多路召回之后、
大模型生成之前的关键精选环节，负责从大量召回文档中筛选出最相关的 Top-K。

两阶段重排序策略：
  第一阶段 — RRF（Reciprocal Rank Fusion，倒数排名融合）：
    将 BM25、Vector、HyDE 三条通道的排名结果进行无参数融合。
    每篇文档的 RRF 得分 = Σ 1/(k + rank + 1)，其中 k 为平滑常数（默认 60），
    rank 为该文档在各通道中的排名（从 0 开始）。
    同一文档被多条通道召回时得分会累加，体现「多路共识」的思想。

  第二阶段 — CrossEncoder 精排：
    使用 BAAI/bge-reranker-v2-m3 交叉编码器模型对 (query, document) 对
    进行逐一打分。交叉编码器能捕捉 query 与 w 之间更细粒度的交互特征，
    比单独的向量相似度更精准。
    对 RRF 阶段筛选出的候选文档进行精排，返回最终 Top-K。

模型加载：
  - CrossEncoder 模型采用懒加载策略，首次调用时才加载
  - 若 FlagEmbedding 库未安装或模型加载失败，会自动降级为仅使用 RRF 排序
  - 降级时不会中断流程，只是损失精排精度

函数入口：
  - rrf_fusion():         单独执行 RRF 融合
  - cross_encoder_rerank(): 单独执行 CrossEncoder 精排
  - rerank_pipeline():    完整的两阶段重排序流水线（推荐使用）
"""

from __future__ import annotations

import logging
from typing import Optional

from knowledge_base.document_loader import Document
from config.settings import RRF_K, RERANK_TOP_N, FINAL_TOP_K, RERANKER_MODEL

logger = logging.getLogger(__name__)

# 全局单例：缓存已加载的 CrossEncoder 重排序模型
_reranker = None


def _get_reranker():
    """
    懒加载 CrossEncoder 重排序模型。

    使用 FlagEmbedding 库的 FlagReranker 加载 bge-reranker-v2-m3 模型。
    若库未安装或模型加载失败，返回 None 并记录警告日志，
    后续流程会自动降级为仅使用 RRF 排序。
    """
    global _reranker
    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            logger.info("正在加载重排序模型: %s", RERANKER_MODEL)
            _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
            logger.info("重排序模型加载完成")
        except ImportError:
            logger.warning(
                "FlagEmbedding 库未安装，将降级为仅使用 RRF 排序（无 CrossEncoder 精排）"
            )
            _reranker = None
        except Exception as e:
            logger.warning("重排序模型加载失败: %s", e)
            _reranker = None
    return _reranker


def rrf_fusion(
    retrieval_results: dict[str, list[Document]],
    k: int = RRF_K,
) -> list[Document]:
    """
    第一阶段：倒数排名融合（Reciprocal Rank Fusion）。

    将多路召回的排名结果进行无参数融合。公式：
        RRF_score(d) = Σ_{method} 1 / (k + rank_method(d) + 1)

    同一文档被多条通道召回时得分累加，被更多通道认可的文档得分越高。
    使用文档内容前 200 字符作为去重键。

    参数：
        retrieval_results: 按检索方法分组的召回结果字典
        k:                 RRF 平滑常数，默认 60（值越大，排名差异的影响越小）

    返回：
        按 RRF 得分降序排列的 Document 列表，每个文档的 metadata 中附带 rrf_score
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for method, docs in retrieval_results.items():
        for rank, doc in enumerate(docs):
            content_key = doc.content[:200]
            score = 1.0 / (k + rank + 1)

            if content_key in doc_scores:
                # 同一文档被多条通道召回，累加得分
                doc_scores[content_key] += score
            else:
                doc_scores[content_key] = score
                doc_map[content_key] = doc

    # 按 RRF 得分降序排列
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results: list[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = doc_scores[key]
        results.append(doc)

    return results


def cross_encoder_rerank(
    query: str,
    documents: list[Document],
    top_n: int = FINAL_TOP_K,
) -> list[Document]:
    """
    第二阶段：CrossEncoder 精排。

    使用 bge-reranker-v2-m3 交叉编码器对每对 (query, document)
    计算细粒度相关性分数，然后按分数降序取 Top-N。

    若模型未加载成功（降级模式），则直接截取前 top_n 个文档返回。

    参数：
        query:     用户查询文本
        documents: 待精排的候选文档列表（通常来自 RRF 融合结果）
        top_n:     返回的最大文档数量

    返回：
        按精排分数降序排列的 Document 列表，每个文档 metadata 中附带 rerank_score
    """
    reranker = _get_reranker()
    if reranker is None or not documents:
        # 降级模式：无 CrossEncoder，直接按原顺序截取
        return documents[:top_n]

    # 构造 (query, document) 对并批量计算相关性分数
    pairs = [[query, doc.content] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=True)

    # compute_score 在只有一个 pair 时可能返回单个数值而非列表
    if isinstance(scores, (int, float)):
        scores = [scores]

    # 将精排分数写入文档元数据
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)

    # 按精排分数降序排列并截取 Top-N
    reranked = sorted(documents, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
    return reranked[:top_n]


def rerank_pipeline(
    query: str,
    retrieval_results: dict[str, list[Document]],
    rrf_top_n: int = RERANK_TOP_N,
    final_top_k: int = FINAL_TOP_K,
) -> list[Document]:
    """
    完整的两阶段重排序流水线（推荐入口函数）。

    流程：
      1. RRF 融合：将多路召回结果融合并按 RRF 得分排序，取 Top rrf_top_n 作为候选
      2. CrossEncoder 精排：对候选文档进行交叉编码器精排，返回最终 Top final_top_k

    参数：
        query:            用户查询文本
        retrieval_results: 多路召回结果字典（由 MultiRetriever.retrieve() 返回）
        rrf_top_n:        RRF 阶段保留的候选数量，默认由 RERANK_TOP_N 配置
        final_top_k:      最终返回的文档数量，默认由 FINAL_TOP_K 配置

    返回：
        经过两阶段重排序后的最终 Document 列表
    """
    # 第一阶段：RRF 倒数排名融合
    rrf_results = rrf_fusion(retrieval_results)
    candidates = rrf_results[:rrf_top_n]
    logger.info("RRF 融合产出 %d 个候选文档", len(candidates))

    # 第二阶段：CrossEncoder 交叉编码器精排
    final = cross_encoder_rerank(query, candidates, top_n=final_top_k)
    logger.info("CrossEncoder 精排选出 Top %d 文档", len(final))

    return final
