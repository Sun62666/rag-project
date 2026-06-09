# SmartOps Agent - 智能运维助手

> 基于 LangGraph + Milvus + FastAPI 的生产级运维智能体，融合通用文档 RAG 问答能力

## 项目简介

SmartOps Agent 是一个从 RAG 检索增强生成升级为完整智能体的运维问答系统。支持知识库检索、长期记忆、工具自主调用，实现「先检索文档 → 再查历史记忆 → 再调用工具 → 最后生成回答」的完整 Agent 工作流。同时融合了通用文档 RAG 问答能力（如物业管理条例），支持文档上传、混合检索和 Precision@10 评估。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户 (Vue3 + Element Plus + SSE)         │
│              登录/注册/会话管理/流式对话/文档上传/评估看板       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/SSE
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI (src/app.py 极简入口)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    api/ 路由层                         │   │
│  │  auth / session / chat / ops / evaluate               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  services/ 业务逻辑层                  │   │
│  │  auth_svc / session_svc / chat_svc / evaluate_svc     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   core/ 基础设施层                     │   │
│  │  config (Pydantic) / redis / security / logging       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│  Agent 模式       │              │  Graph 模式(兼容回退)  │
│  (LangGraph       │              │  (LangGraph           │
│   create_react_   │              │   StateGraph          │
│   agent)          │              │   工作流)              │
│                   │              │                       │
│ ┌───────────────┐│              │ classify→rewrite→     │
│ │  tools/ 工具集  ││              │ retrieve→tools→gen    │
│ │ ①knowledge    ││              └──────────────────────┘
│ │ ②document_qa  ││
│ │ ③memory       ││
│ │ ④server_info  ││
│ │ ⑤log_analyzer ││
│ │ ⑥port_check   ││
│ │ ⑦knowledge_   ││
│ │   graph        ││
│ └───────────────┘│
│                   │
│ ┌───────────────┐│
│ │  memory/ 记忆  ││
│ │ 短期: InMemory ││
│ │ 长期: Milvus   ││
│ └───────────────┘│
└──────────────────┘
        │
        ▼
 ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
 │   Milvus   │  │   Redis    │  │ DashScope  │  │   Neo4j    │
 │ 知识库+记忆 │  │ 缓存+会话  │  │  通义千问   │  │ 知识图谱   │
 └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | Vue 3 + Element Plus + Pinia + SSE | SPA 组件化，流式对话，登录注册 |
| **后端** | FastAPI + Uvicorn | 异步API，SSE流式输出 |
| **智能体** | LangGraph create_react_agent | ReAct 模式 + Tool Calling |
| **工作流** | LangGraph StateGraph | 意图分类→检索→工具→生成 |
| **短期记忆** | InMemory (LangChain Messages) | 会话隔离，上下文连贯 |
| **长期记忆** | Milvus (ops_memory_store) | 向量化历史问答，跨会话检索 |
| **知识库** | Milvus (ops_knowledge_v2) | BM25+向量混合检索+重排序 |
| **文档库** | Milvus (property_regulations) | 通用文档 RAG，支持 PDF/TXT/MD/DOCX |
| **缓存** | Redis | 问答缓存、会话管理、JWT Token |
| **LLM** | DashScope (qwen-max) | 工具调用+流式生成 |
| **Embedding** | DashScope (text-embedding-v1) | 向量化 |
| **Reranker** | BGE-reranker-v2-m3 | 交叉编码器重排序 |
| **知识图谱** | Neo4j | 实体关系存储与查询 |
| **配置** | Pydantic Settings | 环境变量+验证+类型安全 |
| **部署** | Docker Compose | Milvus+Redis+FastAPI+Nginx 一键启动 |

## 融合亮点

1. **通用文档 RAG 工具**：融合了 Agent 项目的文档上传+RAG检索能力，作为 `document_qa` 工具集成到运维智能体中
2. **Precision@10 评估**：内置物业管理条例 3 个标准评估问题，基于余弦相似度计算检索精准率
3. **多格式文档支持**：支持 PDF / TXT / MD / DOCX 格式上传，自动切片+向量化入库
4. **Vue3 SPA 前端**：用 Element Plus 组件化替代单文件 HTML，大幅提升可维护性
5. **双模式架构**：Agent模式（LangGraph ReAct智能体）+ Graph模式（LangGraph工作流），环境变量一键切换
6. **双层记忆系统**：短期记忆（内存，会话隔离）+ 长期记忆（Milvus向量库，跨会话检索）
7. **7大工具**：知识库检索、文档QA、历史记忆、服务器信息、日志分析、端口检查、知识图谱

## 项目结构

```
SmartOps Assistant/
├── src/
│   ├── app.py                     # FastAPI 极简入口
│   ├── retriever.py               # 混合检索器（BM25+向量+重排序）
│   ├── mcp_ops_server.py          # MCP 运维服务
│   ├── finetune.py                # 微调工具
│   │
│   ├── core/                      # 基础设施层
│   │   ├── config.py              # Pydantic Settings 配置管理
│   │   ├── redis.py               # Redis 连接管理
│   │   ├── security.py            # JWT认证 + 密码哈希
│   │   ├── logging.py             # 日志配置
│   │   └── milvus_compat.py       # Milvus 2.6.x 兼容性补丁
│   │
│   ├── api/                       # 路由层
│   │   ├── deps.py                # 依赖注入 + 组件初始化
│   │   ├── auth.py                # /auth/* 认证路由
│   │   ├── session.py             # /sessions/* 会话路由
│   │   ├── chat.py                # /ask 对话路由
│   │   ├── ops.py                 # /ops/* 运维工具路由聚合
│   │   ├── documents.py           # /ops/upload 文档上传路由
│   │   ├── logs.py                # /ops/logs 日志查看路由
│   │   ├── knowledge_graph.py     # /ops/knowledge/graph 图谱路由
│   │   └── evaluate.py            # /evaluate 评估路由
│   │
│   ├── services/                  # 业务逻辑层
│   │   ├── auth_service.py        # 注册/登录/登出
│   │   ├── session_service.py     # 会话CRUD + 历史管理
│   │   ├── chat_service.py        # 对话流式处理
│   │   └── evaluate_service.py    # Precision@10 评估服务
│   │
│   ├── agent/                     # 智能体模块
│   │   └── ops_agent.py           # LangGraph ReAct Agent
│   │
│   ├── graph/                     # LangGraph 工作流（兼容回退）
│   │   ├── prompts.py             # 所有 Prompt 模板
│   │   ├── nodes.py               # 各节点工厂函数
│   │   └── workflow.py            # 图构建 + 条件路由
│   │
│   ├── tools/                     # 统一工具模块
│   │   ├── knowledge.py           # 运维知识库检索工具
│   │   ├── document_qa.py         # 通用文档 RAG 问答工具 ← 融合自 Agent
│   │   ├── server_info.py         # 服务器信息查询工具
│   │   ├── port_check.py          # 端口检查工具
│   │   ├── log_analyzer.py        # 日志错误统计工具
│   │   ├── memory_retriever.py    # 历史记忆检索工具
│   │   └── knowledge_graph.py     # 知识图谱工具
│   │
│   └── memory/                    # 记忆模块
│       ├── short_term.py          # 短期记忆（InMemory，会话隔离）
│       └── long_term.py           # 长期记忆（Milvus向量库）
│
├── frontend/                      # Vue3 SPA 前端
│   ├── src/
│   │   ├── main.js                # 入口
│   │   ├── App.vue                # 布局（侧边栏+顶栏+路由）
│   │   ├── api/index.js           # axios 封装
│   │   ├── router/index.js        # 路由守卫
│   │   ├── store/auth.js          # Pinia 认证状态
│   │   └── views/
│   │       ├── Login.vue          # 登录/注册
│   │       ├── Dashboard.vue      # 仪表盘
│   │       ├── Chat.vue           # 智能对话 + SSE 流式
│   │       ├── Tools.vue          # 运维工具台（日志/文档/图谱）
│   │       └── Evaluate.vue       # 评估看板（Precision@10）
│   ├── package.json
│   └── vite.config.js
│
├── prompts/                       # 系统提示词
├── data/                          # 知识库文档
├── model/                         # 本地重排序模型
├── eval/                          # 评估工具
├── tests/                         # 单元测试
├── middleware/
│   └── docker-compose.yml         # Milvus + Redis 编排
│
├── start_agent.py                 # 一键启动脚本
├── docker-compose.yml             # 主服务编排
├── Dockerfile                     # 后端容器构建
├── requirements.txt               # Python 依赖
└── README.md
```

## 快速开始

### 1. 启动中间件（Docker）

```bash
# 启动 Milvus + Redis + Neo4j
cd middleware && docker compose up -d
```

### 2. 配置环境变量

编辑 `Env1.env` 或创建 `Key.env`：

```
DASHSCOPE_API_KEY=你的阿里云API Key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MILVUS_URL=http://localhost:19530
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=smartops123
```

### 3. 启动后端

```bash
pip install -r requirements.txt
python start_agent.py
```

后端运行在 `http://localhost:8347`，Swagger 文档在 `http://localhost:8347/docs`。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端运行在 `http://localhost:3000`，自动代理 API 请求到后端。

### 5. 使用流程

1. **注册/登录** — 访问 `http://localhost:3000/login`
2. **智能对话** — 在「智能对话」页面输入运维问题，Agent 自主调用工具
3. **上传文档** — 在「运维工具 → 文档上传」页面上传法规文件（PDF/TXT/MD/DOCX）
4. **评估效果** — 在「评估看板」页面运行 Precision@10 评估
5. **查看日志** — 在「运维工具 → 系统日志」查看运行日志
6. **知识图谱** — 在「运维工具 → 知识图谱」查看实体关系

## API 接口

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
| POST | /mode | 切换模式 |
| POST | /ops/upload | 文档上传入库 |
| GET | /ops/upload/list | 文档列表 |
| DELETE | /ops/upload/{filename} | 删除文档 |
| GET | /ops/knowledge/stats | 知识库统计 |
| GET | /ops/logs | 查看系统日志 |
| GET | /ops/logs/files | 日志文件列表 |
| GET | /ops/knowledge/graph/stats | 知识图谱统计 |
| GET | /ops/knowledge/graph/vis | 知识图谱可视化 |
| POST | /ops/knowledge/graph/extract | 知识图谱抽取 |
| POST | /evaluate | 运行检索评估 |
| GET | /evaluate/questions | 获取评估问题 |

## Precision@10 评估说明

评估使用 3 个预定义物业管理条例问题，计算检索精准率：

| 问题 | 真值摘要 |
|------|----------|
| 物业服务费的价格是由谁定的？ | 由业主和物业服务企业在合同中约定 |
| 物业挪用专项维修资金的，如何处罚？ | 追回+警告+没收违法所得+2倍以下罚款 |
| 业主在物业管理活动中，享有哪些权利？ | 10项权利（接受服务、提议、投票、选举等） |

**计算方式**：
1. 对每个问题检索 10 条记录
2. 用真值答案的 1536 维向量与每条记录计算余弦相似度
3. 相似度 >= 0.7 为相关（TP），< 0.7 为不相关
4. Precision@10 = TP个数 / 10

## License

MIT
