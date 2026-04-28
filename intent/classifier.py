"""
意图识别分类器模块 (intent/classifier.py)

本模块负责对用户输入的自然语言问题进行意图分类，是 LangGraph 主图的
第一个处理节点。通过调用 qwen-plus 大模型进行 zero-shot JSON 格式分类，
将用户问题归入以下三个主分类之一：
- course_qa      : 课程教材内容问答（如 NLP 知识、算法原理、概念解释）
- regulation_qa  : 学校规章制度问答（如学分、成绩、考试、实习、奖学金）
- notes_query    : 课堂笔记查询（查找或检索学生笔记、复习资料）

当主分类为 regulation_qa 时，还会进一步识别子分类（sub_intent）：
- credit      : 学分相关
- grade       : 成绩/绩点相关
- internship  : 实习相关
- scholarship : 奖学金相关
- general     : 其他通用规章制度

分类结果以 IntentResult 数据类返回，包含意图、子意图、置信度和关键词。
temperature 设为 0.1 以确保输出稳定一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm.qwen_client import chat_with_json

logger = logging.getLogger(__name__)

# 意图识别的系统提示词，指导大模型按照固定 JSON 格式输出分类结果
INTENT_SYSTEM_PROMPT = """你是一个校园问答系统的意图识别模块。根据学生的问题，判断其意图类别。

## 意图分类规则

**主分类（intent）：**
- course_qa：关于课程教材内容的问题（如NLP知识、算法原理、概念解释等）
- regulation_qa：关于学校规章制度的问题（如学分、成绩、考试、实习、奖学金等）
- notes_query：关于查找或查询课堂笔记、复习资料的问题

**子分类（sub_intent）- 仅当 intent 为 regulation_qa 时需要：**
- credit：涉及学分计算、学分要求、选课学分等
- grade：涉及成绩评定、绩点计算、考试规则、补考重修等
- internship：涉及实习安排、实习企业、实习要求等
- scholarship：涉及奖学金评选、奖学金类型等
- general：其他规章制度相关

## 输出格式

请严格按照以下JSON格式返回，不要包含其他内容：
```json
{
  "intent": "course_qa|regulation_qa|notes_query",
  "sub_intent": "credit|grade|internship|scholarship|general|none",
  "confidence": 0.95,
  "keywords": ["关键词1", "关键词2"]
}
```"""


@dataclass
class IntentResult:
    """意图识别结果数据类，包含主意图、子意图、置信度和提取的关键词。"""
    intent: str
    sub_intent: str
    confidence: float
    keywords: list[str]


def classify_intent(query: str) -> IntentResult:
    """
    调用 qwen-plus 大模型对用户问题进行意图分类。

    参数：
        query: 用户输入的自然语言问题

    返回：
        IntentResult 对象，包含分类结果。
        若大模型返回了无法识别的意图，默认回退为 course_qa。
    """
    result = chat_with_json(
        user_message=query,
        system_prompt=INTENT_SYSTEM_PROMPT,
        temperature=0.1,
    )

    # 从 JSON 结果中提取各字段，设置安全默认值
    intent = result.get("intent", "course_qa")
    sub_intent = result.get("sub_intent", "none")
    confidence = result.get("confidence", 0.5)
    keywords = result.get("keywords", [])

    # 校验主意图是否合法，不合法则回退到课程问答
    if intent not in ("course_qa", "regulation_qa", "notes_query"):
        logger.warning("未知意图 '%s'，回退为 course_qa", intent)
        intent = "course_qa"

    # 非规章制度问答时，子意图强制设为 none
    if intent != "regulation_qa":
        sub_intent = "none"

    logger.info(
        "意图: %s | 子意图: %s | 置信度: %.2f | 关键词: %s",
        intent, sub_intent, confidence, keywords,
    )
    return IntentResult(
        intent=intent,
        sub_intent=sub_intent,
        confidence=confidence,
        keywords=keywords,
    )
