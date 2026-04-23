# 🤖 SmartOps Assistant

> 基于 RAG (检索增强生成) 的企业级智能运维问答系统

## 📖 项目简介

SmartOps Assistant 是一个智能化的运维知识库问答系统，结合向量检索与大语言模型，为企业提供准确、结构化的故障排查指南和运维操作指导。

### ✨ 核心特性

- 🎯 **精准检索**：混合检索（向量 + BM25）+ BGE 重排序
-  **流式输出**：SSE 实时逐字显示答案
- 💾 **智能缓存**：Redis 缓存高频问题，秒级响应
- 📚 **结构化回答**：严格遵循运维文档规范（故障现象 → 原因 → 排查 → 修复 → 验证）
- 🐳 **容器化部署**：Docker Compose 一键启动
- 🔄 **持续学习**：支持动态更新知识库文档

##  技术架构

### 技术栈

| 组件        | 技术选型                   |
|-----------|------------------------|
| **后端框架**  | FastAPI + Uvicorn      |
| **AI 框架** | LangChain + LangGraph  |
| **向量数据库** | Milvus                 |
| **缓存数据库** | Redis                  |
| **缓存服务**  | Redis Stack            |
| **大语言模型** | DashScope (通义千问)       |
| **重排序模型** | BGE-reranker-v2-m3     |
| **前端**    | Vue 3 + SSE            |
| **部署**    | Docker Compose + Nginx |

## 🚀 快速开始

### 前置条件

- Docker 20.10+
- Docker Compose 2.0+
- 可用的 DashScope API Key

### 安装步骤

#### 1. 克隆项目
bash 
git clone https://github.com/Sun62666/rag-project.git cd SmartOps-Assistant

#### 2. 配置环境变量
bash
复制环境配置模板
cp Env.env .env
编辑 .env 文件，填入您的配置
vim .env
env
DashScope API Key（从阿里云获取）
DASHSCOPE_API_KEY=sk-xxxxx
Milvus 向量数据库地址
MILVUS_URI=http://milvus-standalone:19530
Redis 缓存地址
REDIS_URL=redis://redis:6379/0
缓存过期时间（秒）
CACHE_TTL=3600

#### 3. 准备知识库文档

将运维文档（PDF 格式）放入 `data/` 目录：

bash
ls data/
文档1.pdf 文档2.pdf

#### 4. 启动服务（使用docker部署）
```
bash 

创建中间件middleware
将middleware拖入root目录下
执行 docker compose up -d

创建目录在linux上
mkdir /root/SmartOps Assistant
数据放入，将需要的文件、文件夹拖入目录中(frontend、src、prompts、Env.env、requirements.txt、docker-compose.yml、Dockerfile)
docker compose up -d --build (执行创建镜像和容器)

查看服务状态
docker-compose ps

查看日志
docker-compose logs -f rag-backend
```
#### 5. 访问系统

- 🌐 **前端界面**：http://localhost:8080
- 📚 **API 文档**：http://localhost:8347/docs
-  **Redis 管理**：http://localhost:8001 (RedisInsight)

## 📂 项目结构

```
SmartOps Assistant/
├── src/ # 核心代码 │ 
        ├── app.py # FastAPI 应用主入口 │
        ├── config.py # 配置管理 │ 
        ├── graph.py # LangGraph 工作流定义 │
        └── retriever.py # 混合检索器实现
├── frontend/ # 前端界面 │ 
        └── index.html # Vue 3 单页应用 
├── data/ # 知识库文档 │ 
        ├── 文档1.pdf │
        └── 文档2.pdf 
├── model/ # 本地模型文件 │ 
        └── bge-reranker-v2-m3/ # BGE 重排序模型
├── prompts/ # 系统提示词 │
        └── ops_system.md # 运维助手角色定义 
├── eval/ # 评估工具 │
        ├── run_eval.py # 检索效果评估 │ 
        └── eval_retrievers.py # 生成评估报告 
├── middleware/ # 中间件配置 │ 
        └── docker-compose.yml # Milvus + Redis 编排
├── docker-compose.yml # 主服务编排
├── Dockerfile # 后端容器构建 
├── requirements.txt # Python 依赖 
├── Env.env # 环境变量模板 
└── Key.env # API Key 配置
```
##  API 接口

### POST /ask

智能问答接口，支持 SSE 流式输出。

**请求示例：**
```
bash
curl -X POST http://localhost:8347/ask -H
"Content-Type: application/json" 
-d '{"query": "如何排查 Linux 服务器 CPU 使用率 100%？"}'
```
**响应格式 (SSE)：**
```data: {"type": "status", "message": "正在检索知识库..."} 
data: {"type": "token", "content": "根"} 
data: {"type": "token", "content": "据"}
data: {"type": "token", "content": "系"} ...
data: {"type": "done", "from_cache": false}
```

**响应字段说明：**
- `status`: 系统状态消息
- `token`: 流式输出的文本片段
- `done`: 输出完成标志
  - `from_cache: true` - 来自缓存
  - `from_cache: false` - 实时生成

##  系统提示词规范

系统在 `prompts/ops_system.md` 中定义了严格的回答规范：
```
你是企业级智能运维助手（SmartOps）。请严格遵循以下规则：
仅基于提供的上下文回答。若上下文中无答案，明确回复："当前知识库未覆盖该问题，建议转交人工运维专家。"
回答必须结构化：【故障现象】→【可能原因】→【排查命令】→【修复步骤】→【验证方法】
涉及命令需标注执行环境（如 root@prod-db-01$），危险操作需加 警告
引用来源格式：[文档名:页码]，禁止编造指标或配置
保持专业、简洁，禁止寒暄
```
## 📊 效果评估

### 检索评估

bash
运行检索效果评估
cd eval 
python run_eval.py
生成详细评估报告
python eval_retrievers.py

**评估指标：**
- 检索准确率 (Hit Rate)
- 平均倒数排名 (MRR)
- 上下文相关性 (Context Precision)

### 评估报告输出

-  `ragas_report.csv` - 简化版 CSV 报告
- 📊 `retrieval_eval_report.csv` - 检索评估结果
- 📋 `all_chunks_for_label.json` - 完整评估数据

