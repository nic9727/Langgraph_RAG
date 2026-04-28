# 校园智能问答系统

基于 **LangGraph** 的校园智能问答系统，集成 RAG（检索增强生成）、多路召回、意图识别和技能路由。

## 功能特性

- **课程教材问答**：基于《自然语言处理》教材的 RAG 问答，支持多路召回（BM25 + 向量 + HyDE）和两阶段重排序
- **规章制度问答**：通过 Skills 技能系统查询学分、成绩、实习等信息，结合规章文档 RAG
- **笔记查询**：搜索和检索课堂笔记内容

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 编排框架 | LangGraph |
| 大模型 | 通义千问 Qwen-Plus（阿里云百炼） |
| 向量数据库 | ChromaDB（持久化） |
| 词嵌入 | BAAI/bge-large-zh-v1.5 |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 关键词检索 | BM25 + jieba 分词 |
| 结构化数据 | SQLite |
| Web 界面 | Chainlit |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，确认 API Key 等配置：

```env
DASHSCOPE_API_KEY=your-api-key-here
```

### 3. 初始化数据库

```bash
python scripts/init_database.py
```

### 4. 构建知识库

将 NLP 教材 PDF 放置到 `data/textbooks/nlp_textbook.pdf`，然后运行：

```bash
python scripts/build_knowledge_base.py
```

> 如果暂时没有 PDF 教材，系统会自动跳过课程知识库构建，规章制度知识库仍可正常使用。

### 5. 启动应用

```bash
chainlit run app.py
```

访问 http://localhost:8000 即可使用。

## 项目结构

```
Langragh_RAG/
├── config/          # 全局配置
├── data/            # 数据文件（教材、规章、笔记、数据库）
├── knowledge_base/  # 知识库构建（加载、分块、向量化、存储）
├── retrieval/       # 检索管线（BM25、图、HyDE、多路融合、重排序）
├── llm/             # LLM 客户端封装
├── intent/          # 意图识别
├── skills/          # 技能系统（学分、成绩、实习查询）
├── agents/          # LangGraph 图定义和 Agent 节点
├── database/        # SQLite 数据库模型和初始化
├── scripts/         # 工具脚本
└── app.py           # Chainlit 应用入口
```

## 系统架构

```
用户输入 → 意图识别 → 路由分发
                        ├── 课程问答 → 多路召回 → 重排序 → LLM 生成
                        ├── 规章制度 → Skill 路由 → 数据库/RAG → LLM 生成
                        └── 笔记查询 → 笔记检索 → LLM 生成
```

详细架构说明见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 示例问题

- "什么是 Transformer 的自注意力机制？"
- "毕业需要多少学分？"
- "我的成绩怎么算绩点？"
- "学校有哪些合作实习企业？"
- "帮我找一下关于 BERT 的笔记"
