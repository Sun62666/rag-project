import logging
import os
from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from src.config import Config
from src.retriever import OpsRetriever
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.common_tools import knowledge_retriever_logic
from src.tools.server_info import server_info_query
from src.tools.log_analyzer import log_error_stats

logger = logging.getLogger(__name__)

retriever_instance: Optional[OpsRetriever] = None
long_term_memory: Optional[LongTermMemory] = None


@tool
def knowledge_retriever(query: str) -> str:
    """从运维知识库检索故障解决方案、配置规范等文档。当用户询问运维故障排查、配置方法、操作规范时调用。
    参数：query-检索关键词"""
    if retriever_instance is None:
        return "知识库未初始化"
    return knowledge_retriever_logic(query, retriever_instance)


@tool
def memory_retriever(query: str) -> str:
    """从历史对话记忆库中检索用户之前问过的相关问题和答案。当用户追问之前的话题、引用历史对话时调用。
    参数：query-检索关键词"""
    if long_term_memory is None:
        return "长期记忆库未初始化"
    result = long_term_memory.format_memory_context(query, top_k=3)
    return result if result else "未检索到相关历史记忆"


AGENT_SYSTEM_PROMPT = """你是资深运维工程师（SmartOps Agent），擅长 Linux/数据库/中间件/云原生运维。

## 核心工作流
处理用户问题时，按以下优先级获取信息：
1. **知识库检索** - 使用 knowledge_retriever 查找运维文档中的解决方案
2. **历史记忆检索** - 使用 memory_retriever 查找用户之前的相关问答
3. **实时系统查询** - 使用 server_info_query 获取服务器实时状态
4. **日志分析** - 使用 log_error_stats 统计分析日志错误

## 回答规范
严格按以下结构回复：
【故障现象】
【可能原因】
【排查命令】
【修复步骤】
【验证方法】

高危操作必须标注 ⚠ 警告。

## 边界规则
- 拒绝闲聊、娱乐、非运维类问题
- 知识库和记忆库均无结果时，回复：当前知识库未覆盖该问题，建议转交人工运维专家。
- 当用户追问之前的话题时，优先使用 memory_retriever 检索历史记忆
"""


class OpsAgent:
    """运维智能体：整合知识库检索、长期记忆、工具调用的 AgentExecutor"""

    def __init__(self, retriever: OpsRetriever, stm: ShortTermMemory, ltm: LongTermMemory):
        global retriever_instance, long_term_memory
        retriever_instance = retriever
        long_term_memory = ltm

        self.cfg = Config()
        self.stm = stm
        self.ltm = ltm

        self.llm = ChatOpenAI(
            model=self.cfg.LLM_MODEL,
            base_url=self.cfg.BASE_URL,
            api_key=self.cfg.DASHSCOPE_API_KEY,
            temperature=0,
            streaming=True,
        )

        self.tools = [
            knowledge_retriever,
            memory_retriever,
            server_info_query,
            log_error_stats,
        ]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

        logger.info(f"[OpsAgent] 初始化完成，工具: {[t.name for t in self.tools]}")

    async def astream(self, query: str, session_id: str, user_id: str = "anonymous"):
        """异步流式执行智能体，返回 token 级别的流"""
        chat_history = self.stm.get_messages(session_id)

        self.stm.add_user_message(session_id, query)

        full_answer = []

        async for event in self.executor.astream_events(
            {"input": query, "chat_history": chat_history},
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chain_start" and event.get("name") == "AgentExecutor":
                logger.info(f"[OpsAgent] 开始执行: {query[:50]}")

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                logger.info(f"[OpsAgent] 调用工具: {tool_name}")
                yield {"type": "status", "message": f"正在调用工具: {tool_name}"}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                logger.info(f"[OpsAgent] 工具完成: {tool_name}")

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_answer.append(chunk.content)
                    yield {"type": "token", "content": chunk.content}

        answer_text = "".join(full_answer)

        self.stm.add_ai_message(session_id, answer_text)

        try:
            self.ltm.save_memory(user_id, session_id, query, answer_text)
        except Exception as e:
            logger.error(f"[OpsAgent] 保存长期记忆失败: {e}")

        yield {"type": "done", "from_cache": False}

    async def ainvoke(self, query: str, session_id: str, user_id: str = "anonymous") -> str:
        """同步执行智能体，返回完整回答"""
        chat_history = self.stm.get_messages(session_id)
        self.stm.add_user_message(session_id, query)

        result = await self.executor.ainvoke({
            "input": query,
            "chat_history": chat_history,
        })

        answer = result.get("output", "")
        self.stm.add_ai_message(session_id, answer)

        try:
            self.ltm.save_memory(user_id, session_id, query, answer)
        except Exception as e:
            logger.error(f"[OpsAgent] 保存长期记忆失败: {e}")

        return answer


def build_agent(retriever: OpsRetriever, stm: ShortTermMemory, ltm: LongTermMemory) -> OpsAgent:
    """构建运维智能体实例"""
    return OpsAgent(retriever, stm, ltm)
