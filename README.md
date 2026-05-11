# 🤖 SmartOps Agent - 运维智能体

> 从 RAG 到 Agent：基于 LangChain + Milvus + FastAPI 的生产级运维智能体

## 📖 项目简介

SmartOps Agent 是一个从 RAG 检索增强生成升级为完整智能体的运维问答系统。支持知识库检索、长期记忆、工具自主调用，实现「先检索文档 → 再查历史记忆 → 再调用工具 → 最后生成回答」的完整 Agent 工作流。

## 🏗 架构图（文字版）

```
┌─────────────────────────────────────────────────────────────┐
│                     用户 (Vue3 + SSE)                        │
│                   登录/注册/会话管理/流式对话                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI (app.py)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 认证模块  │  │ 会话管理  │  │ 缓存层   │  │ 模式切换  │   │
│  │ JWT+Redis│  │ Redis Hash│  │ Redis    │  │Agent/Graph│  │
│  └──────────┘  └──────────┘  └──────────┘  └─────┬────┘   │
└──────────────────────────────────────────────────┼─────────┘
                                                   │
                    ┌──────────────────────────────┼─────────┐
                    │         OpsAgent (智能体)      │         │
                    │   AgentExecutor + ToolCalling  │         │
                    │                                │         │
                    │  ┌─────────┐  ┌────────────┐  │         │
                    │  │短期记忆  │  │  长期记忆    │  │         │
                    │  │InMemory │  │Milvus向量库  │  │         │
                    │  │Session  │  │ops_memory   │  │         │
                    │  └─────────┘  └────────────┘  │         │
                    │                                │         │
                    │  ┌──────────────────────────┐  │         │
                    │  │       工具集 (Tools)       │  │         │
                    │  │ ① knowledge_retriever     │  │         │
                    │  │ ② memory_retriever        │  │         │
                    │  │ ③ server_info_query       │  │         │
                    │  │ ④ log_error_stats         │  │         │
                    │  └──────────────────────────┘  │         │
                    └────────────────────────────────┘         │
                                                               │
     ┌─────────────────────────────────────────────────────────┘
     │
     ▼   (兼容回退: LangGraph 工作流)
┌─────────────────────────────────────────────────────────────┐
│  classify → rewrite_query → retrieve → execute_tools → gen  │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │   Milvus   │   │   Redis    │   │ DashScope  │
   │ 知识库+记忆 │   │ 缓存+会话  │   │  通义千问   │
   └────────────┘   └────────────┘   └────────────┘
```

## 🔄 核心流程

```
用户提问 → 缓存命中? → 直接返回
                ↓ 否
         模式判断(USE_AGENT)
           ↓              ↓
     Agent模式          Graph模式(兼容)
           ↓              ↓
   短期记忆加载历史    意图分类→改写→检索
           ↓              ↓
   AgentExecutor调度   工具调用→生成回答
   ┌─────────────┐
   │ LLM自主决策  │
   │ ①检索知识库  │
   │ ②检索历史记忆│
   │ ③查询服务器  │
   │ ④分析日志    │
   └──────┬──────┘
          ↓
   生成结构化回答
          ↓
   保存短期记忆 + 长期记忆 + Redis历史 + 缓存
          ↓
   SSE流式返回前端
```

## 🛠 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Vue 3 + SSE | 单页应用，流式对话，登录注册 |
| **后端** | FastAPI + Uvicorn | 异步API，SSE流式输出 |
| **智能体** | LangChain AgentExecutor | Tool Calling + 自主决策 |
| **工作流** | LangGraph | 意图分类→检索→工具→生成 |
| **短期记忆** | InMemory (LangChain Messages) | 会话隔离，上下文连贯 |
| **长期记忆** | Milvus (ops_memory_store) | 向量化历史问答，跨会话检索 |
| **知识库** | Milvus (ops_knowledge_v2) | BM25+向量混合检索+重排序 |
| **缓存** | Redis | 问答缓存、会话管理、JWT Token |
| **LLM** | DashScope (qwen3.5-plus) | 工具调用+流式生成 |
| **Embedding** | DashScope (text-embedding-v2) | 向量化 |
| **Reranker** | BGE-reranker-v2-m3 | 交叉编码器重排序 |
| **部署** | Docker Compose | Milvus+Redis+FastAPI+Nginx 一键启动 |

## ✨ 改造亮点

1. **双模式架构**：Agent模式（智能体自主决策）+ Graph模式（LangGraph工作流），环境变量一键切换
2. **双层记忆系统**：短期记忆（内存，会话隔离）+ 长期记忆（Milvus向量库，跨会话检索）
3. **工具自主调用**：LLM自主判断是否调用工具，无需硬编码规则
4. **完整认证体系**：JWT Token + Redis存储，注册/登录/退出
5. **会话级缓存清理**：删除会话时级联清理关联的问答缓存
6. **流式Agent输出**：AgentExecutor + astream_events 实现token级流式

## 📂 项目结构

```
SmartOps Assistant/
├── src/
│   ├── app.py                 # FastAPI 主入口（双模式调度）
│   ├── config.py              # 配置管理
│   ├── graph.py               # LangGraph 工作流（兼容回退）
│   ├── retriever.py           # 混合检索器（BM25+向量+重排序）
│   ├── common_tools.py        # 原有工具逻辑
│   ├── memory/                # 🆕 记忆模块
│   │   ├── __init__.py
│   │   ├── short_term.py      # 短期记忆（InMemory，会话隔离）
│   │   └── long_term.py       # 长期记忆（Milvus向量库）
│   ├── tools/                 # 🆕 工具模块
│   │   ├── __init__.py
│   │   ├── server_info.py     # 服务器信息查询工具
│   │   └── log_analyzer.py    # 日志错误统计工具
│   └── agent/                 # 🆕 智能体模块
│       ├── __init__.py
│       └── ops_agent.py       # AgentExecutor + Tool Calling
├── frontend/
│   └── index.html             # Vue 3 前端（登录+会话+流式对话）
├── prompts/
│   └── ops_system.md          # 系统提示词
├── data/                      # 知识库文档
├── model/                     # 本地重排序模型
├── eval/                      # 评估工具
├── middleware/
│   └── docker-compose.yml     # Milvus + Redis 编排
├── start_agent.py             # 🆕 一键启动脚本
├── docker-compose.yml         # 主服务编排
├── Dockerfile                 # 后端容器构建
├── requirements.txt           # Python 依赖
└── README.md
```

## 🚀 快速开始

### Docker 一键部署

```bash
# 1. 启动中间件（Milvus + Redis）
cd middleware && docker compose up -d

# 2. 启动应用
cd .. && docker compose up -d --build

# 3. 访问
# 前端: http://localhost:8080
# API文档: http://localhost:8347/docs
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp Env.env Key.env  # 填入 DASHSCOPE_API_KEY

# 启动
python start_agent.py

# 或指定Graph模式
USE_AGENT=false python start_agent.py
```

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /ask | 智能问答（SSE流式） |
| POST | /auth/register | 用户注册 |
| POST | /auth/login | 用户登录 |
| POST | /auth/logout | 退出登录 |
| GET | /auth/me | 验证Token |
| POST | /new_session | 创建会话 |
| GET | /sessions | 获取会话列表 |
| GET | /sessions/{id} | 获取会话历史 |
| DELETE | /sessions/{id} | 删除会话 |
| PUT | /sessions/{id}/rename | 重命名会话 |
| POST | /clear_history | 清空对话 |
| GET | /mode | 查询当前模式 |

## 🎯 面试高频问题 10 + 标准答案

### Q1: RAG 和 Agent 的区别是什么？为什么从 RAG 升级到 Agent？

**A:** RAG 是「检索增强生成」，流程固定：检索→生成，无法根据问题动态调整策略。Agent 是「智能体」，LLM 可以自主决策调用哪些工具、调用几次、以什么顺序调用。本项目从 RAG 升级到 Agent 的核心原因是：运维场景需要「先查文档 + 再查历史 + 再查服务器状态 + 再分析日志」的复合能力，固定流程无法覆盖。Agent 通过 Tool Calling 让 LLM 自主判断，更灵活、更智能。

### Q2: 短期记忆和长期记忆的区别和实现方式？

**A:** 短期记忆基于 LangChain 的 InMemory ChatMessageHistory，存储当前会话的对话上下文，支持追问和指代消解，会话隔离，进程重启后丢失。长期记忆复用 Milvus 向量库，新增 `ops_memory_store` 集合，将每轮问答向量化保存，提问时先检索 top3 相关历史记忆融合到上下文，解决长会话遗忘和跨会话知识复用问题。记忆带时间戳、用户ID、会话ID元数据。

### Q3: AgentExecutor 的工作原理？Tool Calling 是怎么实现的？

**A:** AgentExecutor 是 LangChain 的智能体执行器，核心循环是：LLM 思考 → 决定调用工具 → 执行工具 → 结果返回 LLM → 继续思考或输出最终答案。Tool Calling 通过 `create_tool_calling_agent` 实现，LLM 根据 tool 的 name 和 description 判断是否需要调用，生成结构化的工具调用请求（含参数），AgentExecutor 解析后执行对应工具函数，将结果注入 agent_scratchpad 供 LLM 继续推理。

### Q4: 为什么用 Milvus 做长期记忆而不是 Redis？

**A:** Redis 适合精确匹配和结构化查询，但长期记忆需要语义检索——用户换一种说法问同样的问题，Redis 无法匹配，而 Milvus 的向量相似度搜索可以。例如用户之前问过"Redis内存溢出"，现在问"缓存服务OOM"，向量检索能召回相关记忆，Redis 做不到。这也是 RAG 的核心价值：语义级别的匹配。

### Q5: 混合检索（BM25 + 向量）+ 重排序的设计思路？

**A:** BM25 擅长关键词精确匹配（如"CPU 100%"），向量检索擅长语义匹配（如"服务器卡"≈"CPU高"），两者互补。用 EnsembleRetriever 按 4:6 权重融合候选集，再用 CrossEncoder 重排序，因为向量检索的余弦相似度不够精确，CrossEncoder 对 query-doc 对做交叉注意力打分，精度更高。这是业界标准的 two-stage retrieval 方案。

### Q6: 如何保证 Agent 不会无限调用工具？

**A:** AgentExecutor 有 `max_iterations` 参数（本项目设为5），超过最大迭代次数自动停止并返回当前结果。同时 `handle_parsing_errors=True` 确保工具调用解析失败时不会崩溃。此外，系统提示词明确规定了工具使用场景，减少不必要的调用。

### Q7: 缓存策略是怎么设计的？删除会话时如何清理关联缓存？

**A:** 问答缓存用 Redis 的 `ops:{query}` 键存储，TTL 7天。同时用 `session_queries:{sid}` Set 追踪每个会话问过哪些问题。删除会话时：1）读取 Set 获取所有 query；2）批量删除 `ops:{query}` 缓存；3）删除历史记录和追踪集合。清空对话同理。这保证了不会出现孤立缓存。

### Q8: 双模式架构（Agent/Graph）的设计意义？

**A:** Agent 模式是主推方案，LLM 自主决策更灵活；Graph 模式是 LangGraph 工作流，流程固定但更可控。保留双模式的意义：1）兼容回退——Agent 出问题时切回 Graph；2）A/B 对比——面试时可以展示两种方案的差异；3）渐进式改造——不破坏原有代码，通过 `USE_AGENT` 环境变量一键切换。

### Q9: Docker 部署的架构和注意事项？

**A:** 分两层 docker-compose：middleware 层（Milvus + etcd + MinIO + Redis + 管理工具）和应用层（FastAPI + Nginx）。注意事项：1）Milvus 依赖 etcd 和 MinIO，需要 healthcheck 确保启动顺序；2）数据卷挂载保证持久化；3）网络用外部 bridge 网络打通两层；4）环境变量区分容器内和本地地址（MILVUS_URL 在容器内用服务名）。

### Q10: 如何评估这个系统的效果？有哪些指标？

**A:** 检索层：Hit Rate（命中率）、MRR（平均倒数排名）、Context Precision（上下文精度），用 RAGAS 框架评估。生成层：Faithfulness（忠实度，是否基于上下文）、Answer Relevancy（答案相关性）。Agent 层：Tool Call Accuracy（工具调用准确率）、Task Completion Rate（任务完成率）。项目 eval/ 目录提供了完整的评估脚本。

## 🎬 现场演示步骤

### 演示1: 多轮对话 + 追问
```
1. 登录系统
2. 问: "Linux服务器CPU使用率持续100%如何排查？"
3. 追问: "具体怎么用top命令定位？"
4. 追问: "刚才那个问题还有其他方法吗？"
→ 展示短期记忆：追问时能理解"刚才那个问题"指代什么
```

### 演示2: 工具调用
```
1. 问: "帮我查一下服务器当前状态"
→ 展示 server_info_query 工具被自动调用
2. 问: "分析一下日志里有什么错误"
→ 展示 log_error_stats 工具被自动调用
```

### 演示3: 长期记忆
```
1. 在会话A问: "Redis内存溢出怎么处理？"
2. 切换到会话B
3. 问: "之前问过缓存相关的问题吗？"
→ 展示长期记忆：能检索到会话A中的问答
```

### 演示4: 模式切换
```
1. 访问 /mode 确认当前是 Agent 模式
2. 设置 USE_AGENT=false 重启
3. 同样的问题，对比两种模式的回答差异
```

### 演示5: Docker 一键部署
```
1. cd middleware && docker compose up -d
2. cd .. && docker compose up -d --build
3. 访问 http://localhost:8080
→ 展示完整的容器化部署能力
```

## 📄 License

MIT
