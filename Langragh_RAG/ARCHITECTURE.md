# 系统架构文档

## 1. 总体架构

本系统基于 LangGraph 构建，采用有向状态图进行流程编排。系统接收用户自然语言输入，通过意图识别路由到不同的处理分支，最终由 Qwen-Plus 大模型生成回答。

```
┌─────────────┐
│  Chainlit    │  ← Web 对话界面
│  Frontend    │
└──────┬──────┘
       │ 用户消息
       ▼
┌─────────────┐
│   Intent     │  ← Qwen-Plus Zero-shot 分类
│ Recognition  │     输出: intent + sub_intent
└──────┬──────┘
       │
   ┌───┼────────────────┐
   │   │                │
   ▼   ▼                ▼
┌─────┐ ┌──────────┐ ┌──────┐
│课程 │ │ 规章制度  │ │ 笔记  │
│问答 │ │   问答    │ │ 查询  │
└──┬──┘ └────┬─────┘ └──┬───┘
   │         │          │
   ▼         ▼          ▼
 RAG     Skills      搜索
 管线     路由       匹配
   │    ┌──┼──┐       │
   │    │  │  │       │
   ▼    ▼  ▼  ▼       │
 多路  学分 成绩 实习   │
 召回  查询 查询 查询   │
   │   (DB) (DB) (LLM) │
   ▼                    │
 重排序                  │
   │                    │
   ▼                    ▼
┌─────────────────────────┐
│     Qwen-Plus 生成       │
│     最终回答             │
└─────────────────────────┘
```

## 2. LangGraph 状态图

### 2.1 全局状态 (GraphState)

```python
class GraphState(TypedDict):
    query: str              # 用户原始问题
    intent: str             # 主意图: course_qa / regulation_qa / notes_query
    sub_intent: str         # 子意图: credit / grade / internship / general / none
    confidence: float       # 意图识别置信度
    keywords: list[str]     # 提取的关键词
    retrieved_docs: list    # 召回的文档列表
    reranked_docs: list     # 重排序后的文档
    skill_result: str       # Skill 执行结果
    skill_source: str       # 数据来源标识
    answer: str             # 最终回答
    chat_history: list      # 对话历史
```

### 2.2 节点与边

**节点：**
- `intent_recognition` — 意图识别入口
- `course_retrieve` — 课程知识库多路召回
- `course_rerank` — 课程文档重排序
- `course_generate` — 课程问答生成
- `regulation_skill` — 规章制度 Skill 分发
- `regulation_rag` — 规章制度文档 RAG
- `regulation_generate` — 规章制度回答生成
- `notes_search` — 笔记检索 + 生成

**条件路由：**
1. `intent_recognition` → 按 intent 路由到三条分支
2. `regulation_skill` → 按 skill_result 是否为空决定走 generate 还是 fallback 到 RAG

## 3. 知识库构建

### 3.1 文档加载
- PDF：使用 PyMuPDF 逐页提取文本，保留页码元数据
- Markdown：按二级标题切分，保留章节标题

### 3.2 分块策略（三级）

| 层级 | 策略 | 说明 |
|------|------|------|
| L1 | 章节切分 | 正则匹配「第X章」等标题，按章节边界分割 |
| L2 | 递归字符切分 | chunk_size=800, overlap=150 |
| L3 | 语义边界修剪 | 按 `\n\n > \n > 。 > ；` 优先级确保不在句中截断 |

每个 chunk 附带元数据：`{source, page, chapter, chunk_id, chunk_length}`

### 3.3 向量化与存储
- 模型：BAAI/bge-large-zh-v1.5（本地推理）
- 存储：ChromaDB 持久化，cosine 距离
- 集合：`course_textbook`（课程）、`school_regulations`（规章）

## 4. 多路召回

三条并行召回路径：

| 路径 | 方法 | 适用场景 |
|------|------|---------|
| BM25 | jieba 分词 + rank_bm25 | 精确关键词匹配 |
| Vector | bge-large-zh 编码 + ChromaDB | 语义相似度匹配 |
| HyDE | LLM 生成假设回答 → 编码 → 检索 | 概念性、抽象问题 |

## 5. 重排序

两阶段：

1. **RRF 混排**：Reciprocal Rank Fusion，公式 `score = Σ 1/(k + rank)`，k=60
2. **CrossEncoder 精排**：BAAI/bge-reranker-v2-m3，对 RRF Top-20 做精细排序，输出 Top-5

## 6. Skills 技能系统

基于策略模式，每个 Skill 实现 `can_handle()` 和 `execute()` 接口：

| Skill | 触发条件 | 数据源 |
|-------|---------|--------|
| CreditSkill | sub_intent = credit | SQLite: students + courses + enrollments |
| GradeSkill | sub_intent = grade | SQLite: grade_rules |
| InternshipSkill | sub_intent = internship | SQLite: internship_companies + LLM 生成 |

SkillRouter 按 sub_intent 分发。如果无 Skill 匹配，fallback 到规章文档 RAG。

## 7. 意图识别

使用 Qwen-Plus 进行 zero-shot JSON 格式分类：
- 输入：用户问题
- 输出：`{intent, sub_intent, confidence, keywords}`
- 温度：0.1（确保输出稳定）

## 8. 依赖关系

```
Chainlit (app.py)
  └── agents/graph.py (LangGraph)
        ├── intent/classifier.py (意图识别)
        ├── agents/course_agent.py
        │     ├── retrieval/multi_retriever.py
        │     │     ├── retrieval/bm25_retriever.py
        │     │     ├── retrieval/vector_retriever.py
        │     │     └── retrieval/hyde_retriever.py
        │     └── retrieval/reranker.py
        ├── agents/regulation_agent.py
        │     ├── skills/skill_router.py
        │     │     ├── skills/credit_skill.py → database/models.py
        │     │     ├── skills/grade_skill.py → database/models.py
        │     │     └── skills/internship_skill.py → database/models.py + llm
        │     └── retrieval/* (fallback RAG)
        └── agents/notes_agent.py
              └── data/notes/mock_notes.json
```
