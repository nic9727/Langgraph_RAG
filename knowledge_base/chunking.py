"""
多级文档分块模块 (knowledge_base/chunking.py)

本模块实现了面向教材类长文档的三级分块（chunking）策略，用于将加载后的
原始文档切分成适合向量化和检索的小段文本（chunk）。

三级分块策略：
  L1 — 按章节标题切分：
       通过正则表达式匹配「第X章」「Chapter N」「1. 」等章节标题模式，
       将长文档按章节边界划分为若干大段。
  L2 — 递归字符切分：
       对每个章节内的文本，按照优先级递减的分隔符列表递归切分，
       直到每段不超过 chunk_size（默认 800 字符），相邻段之间保留
       chunk_overlap（默认 150 字符）的重叠以维持上下文连贯。
  L3 — 语义完整性修剪：
       分隔符优先级为 \\n\\n > \\n > 。 > ； > ！ > ？ > . > ; > ! > ? > 空格，
       确保切分点尽量落在句子或段落边界，不在句中截断。

每个生成的 chunk 都附带完整的元数据（metadata），包括：
source（来源文件名）、page（页码）、chapter（章节标题）、
chunk_id（全局唯一编号）、chunk_length（chunk 字符长度）。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from knowledge_base.document_loader import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# L1 层级：章节标题的正则匹配模式列表
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千\d]+[章节篇]", re.MULTILINE),
    re.compile(r"^Chapter\s+\d+", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\d+\.\s+", re.MULTILINE),
]

# L3 层级：分隔符优先级列表（从高到低），用于语义完整性切分
SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", ".", ";", "!", "?", " "]


def _find_chapter_splits(text: str) -> list[tuple[str, str]]:
    """
    L1：按章节标题切分文本。

    扫描所有章节标题正则模式，提取切分点位置和标题文本，
    然后按位置排序、去重后将文本切分为 (标题, 内容) 对的列表。
    如果第一个章节标题之前有前言内容，会作为 "preamble" 插入。
    若未找到任何章节标题，则整篇文本作为一个整体返回。
    """
    split_points: list[tuple[int, str]] = []

    for pattern in CHAPTER_PATTERNS:
        for match in pattern.finditer(text):
            split_points.append((match.start(), match.group()))

    if not split_points:
        return [("full_text", text)]

    split_points.sort(key=lambda x: x[0])
    split_points = _deduplicate_splits(split_points)

    chapters: list[tuple[str, str]] = []
    for i, (pos, title) in enumerate(split_points):
        end = split_points[i + 1][0] if i + 1 < len(split_points) else len(text)
        content = text[pos:end].strip()
        if content:
            chapters.append((title.strip(), content))

    # 如果第一个章节标题之前还有内容（前言），作为 preamble 插入
    if split_points and split_points[0][0] > 0:
        preamble = text[: split_points[0][0]].strip()
        if preamble:
            chapters.insert(0, ("preamble", preamble))

    return chapters


def _deduplicate_splits(
    splits: list[tuple[int, str]], min_gap: int = 10
) -> list[tuple[int, str]]:
    """去除距离过近的切分点，避免在同一位置重复切分。"""
    if not splits:
        return splits
    result = [splits[0]]
    for pos, title in splits[1:]:
        if pos - result[-1][0] >= min_gap:
            result.append((pos, title))
    return result


def _recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """
    L2 + L3：递归字符切分，同时保证语义完整性。

    按分隔符优先级列表依次尝试切分文本：
    1. 如果当前文本长度已在 chunk_size 以内，直接返回
    2. 用当前优先级最高的分隔符切分，尝试拼接不超过 chunk_size 的段
    3. 若某段仍超长，递归使用下一级分隔符继续切分
    4. 所有分隔符都无法使用时，按固定步长硬切（最后兜底手段）
    5. 相邻 chunk 之间保留 chunk_overlap 个字符的重叠
    """
    if separators is None:
        separators = SEPARATORS.copy()

    # 文本已经足够短，直接返回
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []

    if sep and sep in text:
        parts = text.split(sep)
    else:
        # 当前分隔符不存在于文本中，尝试下一级分隔符
        if remaining_seps:
            return _recursive_split(text, chunk_size, chunk_overlap, remaining_seps)
        # 所有分隔符都不可用，按固定步长硬切（最后兜底）
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(part) > chunk_size:
                # 单个分段仍然超长，递归使用更细粒度的分隔符
                sub_chunks = _recursive_split(
                    part, chunk_size, chunk_overlap, remaining_seps
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                # 保留上一个 chunk 尾部的重叠文本，维持上下文连贯
                if chunks and chunk_overlap > 0:
                    overlap_text = chunks[-1][-chunk_overlap:]
                    current = overlap_text + sep + part
                else:
                    current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    对文档列表执行完整的 L1 → L2 → L3 三级分块流水线。

    参数：
        documents:    待分块的文档列表
        chunk_size:   每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 的重叠字符数

    返回：
        分块后的 Document 列表，每个 Document 包含 chunk 文本和完整元数据
    """
    all_chunks: list[Document] = []
    global_id = 0

    for doc in documents:
        # L1：按章节标题切分
        chapters = _find_chapter_splits(doc.content)

        for chapter_title, chapter_content in chapters:
            # L2 + L3：递归字符切分 + 语义完整性修剪
            raw_chunks = _recursive_split(chapter_content, chunk_size, chunk_overlap)

            for chunk_text in raw_chunks:
                if not chunk_text.strip():
                    continue
                metadata = {
                    **doc.metadata,
                    "chapter": chapter_title,
                    "chunk_id": f"chunk_{global_id:05d}",
                    "chunk_length": len(chunk_text),
                }
                all_chunks.append(Document(content=chunk_text, metadata=metadata))
                global_id += 1

    logger.info(
        "将 %d 个文档分块为 %d 个 chunk", len(documents), len(all_chunks)
    )
    return all_chunks
