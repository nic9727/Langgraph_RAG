"""
文档加载器模块 (knowledge_base/document_loader.py)

本模块负责将不同格式的原始文档加载为统一的 Document 数据结构，
是知识库构建流水线的第一个环节。支持以下格式：
- PDF 文件：使用 PyMuPDF (fitz) 逐页提取文本，保留页码元数据
- Markdown 文件：按二级标题（## ）切分为独立章节
- 目录批量加载：扫描指定目录下所有匹配文件并加载

每个加载后的 Document 包含 content（文本内容）和 metadata（来源、页码等）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """统一的文档数据结构，包含文本内容和元数据。"""

    content: str
    metadata: dict = field(default_factory=dict)


def load_pdf(pdf_path: str) -> list[Document]:
    """
    加载 PDF 文件，按页拆分为 Document 列表。

    每个 Document 的 metadata 包含：source（文件名）、page（页码）、total_pages（总页数）。
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件未找到: {pdf_path}")

    doc = fitz.open(str(path))
    documents: list[Document] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source": str(path.name),
                        "page": page_num + 1,
                        "total_pages": len(doc),
                    },
                )
            )
    doc.close()
    logger.info("从 %s 加载了 %d 页内容", path.name, len(documents))
    return documents


def load_markdown(md_path: str) -> list[Document]:
    """
    加载 Markdown 文件，按二级标题（## ）切分为 Document 列表。

    每个 Document 的 metadata 包含：source（文件名）、section（章节标题）。
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown 文件未找到: {md_path}")

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    documents: list[Document] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"^##\s+(.+)", section)
        title = title_match.group(1) if title_match else "引言"
        documents.append(
            Document(
                content=section,
                metadata={"source": str(path.name), "section": title},
            )
        )
    logger.info("从 %s 加载了 %d 个章节", path.name, len(documents))
    return documents


def load_directory(dir_path: str, glob_pattern: str = "*.md") -> list[Document]:
    """
    批量加载指定目录下所有匹配的文件。

    根据文件后缀自动选择对应的加载器（.md 用 load_markdown，.pdf 用 load_pdf）。
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {dir_path}")

    documents: list[Document] = []
    for file_path in sorted(path.glob(glob_pattern)):
        if file_path.suffix == ".md":
            documents.extend(load_markdown(str(file_path)))
        elif file_path.suffix == ".pdf":
            documents.extend(load_pdf(str(file_path)))
    return documents
