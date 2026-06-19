"""记忆智能体：历史对话检索专家

负责处理用户追问、引用之前话题的问题。
从长期记忆库中检索相关历史对话，辅助上下文理解。
"""

import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_llm
from src.tools.memory_retriever import set_long_term_memory

logger = logging.getLogger(__name__)

MEMORY_AGENT_PROMPT = """你是运维对话记忆专家，负责处理用户的追问和历史话题引用。

## 你的职责
1. 从历史对话记忆中检索相关问题
2. 结合历史上下文理解用户当前意图
3. 基于历史对话给出连贯回答

## 注意
- 当用户追问之前的话题时，必须基于历史对话内容回答
- 不要输出兜底文案
- 回答要连贯，引用之前的讨论内容
"""


class MemoryAgent(BaseAgent):
    """记忆智能体"""

    name = "memory_agent"
    description = "历史对话检索和追问处理专家"

    def __init__(self, ltm=None):
        super().__init__(llm=create_llm())
        if ltm:
            set_long_term_memory(ltm)

        self.chain = ChatPromptTemplate.from_messages([
            ("system", MEMORY_AGENT_PROMPT),
            ("human", "用户问题: {query}\n\n历史对话:\n{context}"),
        ]) | self.llm

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        chat_history = state.get("chat_history", [])
        context_parts = []

        # 第一步：从短期记忆获取历史对话
        if chat_history:
            history_str = "\n".join(chat_history[-6:])
            context_parts.append(f"【近期对话】\n{history_str}")

        # 第二步：从长期记忆库检索
        try:
            from src.tools.memory_retriever import long_term_memory_instance
            if long_term_memory_instance:
                memory_result = long_term_memory_instance.format_memory_context(query, top_k=3)
                if memory_result:
                    context_parts.append(f"【长期记忆】\n{memory_result}")
                    logger.info("[memory_agent] 长期记忆检索完成")
        except Exception as e:
            logger.warning(f"[memory_agent] 长期记忆检索失败: {e}")

        context = "\n\n".join(context_parts) if context_parts else "无历史对话记录"

        response = self.chain.invoke({
            "query": query,
            "context": context,
        })

        result = response.content
        logger.info(f"[memory_agent] 回答完成, 长度: {len(result)}")

        return {
            "memory_result": result,
            "agent_messages": state.get("agent_messages", []) + [
                {"agent": "memory_agent", "message": "历史记忆检索完成"}
            ],
        }
