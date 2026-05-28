import logging
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from src.core.config import get_settings
from src.retriever import OpsRetriever
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from langchain_core.messages import HumanMessage
from src.tools import (
    knowledge_retriever,
    server_info_query,
    read_service_log,
    port_check,
    memory_retriever,
    set_retriever,
    set_long_term_memory,
)

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT =   """你是资深运维工程师（SmartOps Agent），擅长 Linux/数据库/中间件/云原生运维。

## 核心工作流
处理用户问题时，按以下优先级获取信息：
1. **历史记忆检索** - 当问题与之前讨论相关时，优先调用 memory_retriever（更高效）
2. **知识库检索** - 全新问题或记忆无结果时，调用 knowledge_retriever
3. **实时系统查询** - 使用 server_info_query 获取服务器状态
4. **日志分析** - 使用 read_service_log 分析日志

## 回答规范
严格按以下结构回复：
【故障现象】
【可能原因】
【排查命令】
【修复步骤】
【验证方法】

高危操作必须标注 ⚠ 警告。

## 边界规则约束
- 拒绝闲聊、娱乐、非运维类问题
- 知识库和记忆库均无结果时，回复：当前知识库未覆盖该问题，建议转交人工运维专家。
- 禁止在回复中包含任何内部推理过程、思维链或标签
"""


class OpsAgent:
    """运维智能体：整合知识库检索、长期记忆、工具调用的 ReAct Agent"""

    def __init__(self, retriever: OpsRetriever, stm: ShortTermMemory, ltm: LongTermMemory):
        set_retriever(retriever)
        set_long_term_memory(ltm)

        self.cfg = get_settings()
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
            read_service_log,
            port_check
        ]

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        logger.info(f"[OpsAgent] 初始化完成，工具: {[t.name for t in self.tools]}")

    async def astream(self, query: str, session_id: str, user_id: str = "anonymous"):
        chat_history = self.stm.get_messages(session_id)
        self.stm.add_user_message(session_id, query)

        full_answer = []

        input_messages = chat_history + [HumanMessage(content=query)]

        async for event in self.agent.astream_events(
            {"messages": input_messages},
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
        chat_history = self.stm.get_messages(session_id)
        self.stm.add_user_message(session_id, query)

        input_messages = chat_history + [HumanMessage(content=query)]

        result = await self.agent.ainvoke({"messages": input_messages})

        messages = result.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                answer = msg.content
                break

        if not answer:
            answer = str(messages[-1].content) if messages else ""

        self.stm.add_ai_message(session_id, answer)

        try:
            self.ltm.save_memory(user_id, session_id, query, answer)
        except Exception as e:
            logger.error(f"[OpsAgent] 保存长期记忆失败: {e}")

        return answer


def build_agent(retriever: OpsRetriever, stm: ShortTermMemory, ltm: LongTermMemory) -> OpsAgent:
    return OpsAgent(retriever, stm, ltm)
