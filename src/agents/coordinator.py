"""协调者智能体：意图分类和路由分发

负责分析用户问题，决定分发给哪些专家智能体处理。
支持多智能体并行调度（mixed 类型同时分发给多个智能体）。
"""

import logging
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_fast_llm

logger = logging.getLogger(__name__)

# 扩展分类：支持多智能体并行
COORDINATOR_PROMPT = """你是运维问题分类器。根据用户问题判断意图类别，只输出类别名称：

- fault: 需要查阅知识库的故障排查/配置方法/运维规范问题（如"CPU过高怎么办"、"Redis内存溢出"、"Nginx配置负载均衡"）
- system: 需要实时检查服务器状态的问题（如"查看CPU状态"、"检查8080端口"、"查看nginx日志"）
- document: 用户上传的通用文档相关问题（如"物业服务费由谁定"、"物业管理条例规定"）
- mixed: 既需要知识库又需要实时检查的混合问题（如"服务器CPU高怎么排查"）
- followup: 基于历史对话的追问、澄清、引用之前话题的问题（如"上次问了什么"、"刚才那个问题再说一下"、"具体怎么操作"）
- reject: 与运维完全无关的闲聊、娱乐、无关话题（如"讲个笑话"、"今天天气怎么样"）

只输出一个词：fault / system / document / mixed / followup / reject
"""


class CoordinatorAgent(BaseAgent):
    """协调者智能体：分类意图并路由到对应专家"""

    name = "coordinator"
    description = "分析用户意图，路由到合适的专家智能体"

    def __init__(self):
        super().__init__(llm=create_fast_llm())
        self.classify_chain = ChatPromptTemplate.from_messages([
            ("system", COORDINATOR_PROMPT),
            ("human", "用户问题: {query}\n\n最近对话历史:\n{chat_history}"),
        ]) | self.llm

    def classify(self, query: str, chat_history: List[str]) -> str:
        """分类用户意图"""
        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"
        response = self.classify_chain.invoke({
            "query": query,
            "chat_history": history_str,
        })
        intent = response.content.strip().lower()
        valid_intents = ["fault", "system", "document", "mixed", "followup", "reject"]
        if intent not in valid_intents:
            intent = "fault"
        logger.info(f"[coordinator] 意图分类: {intent}")
        return intent

    def get_target_agents(self, intent: str) -> List[str]:
        """根据意图返回需要调度的智能体列表"""
        routing = {
            "fault": ["fault_agent"],
            "system": ["system_agent"],
            "document": ["document_agent"],
            "followup": ["memory_agent"],
            "mixed": ["fault_agent", "system_agent"],
            "reject": [],
        }
        return routing.get(intent, ["fault_agent"])

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        chat_history = state.get("chat_history", [])
        intent = self.classify(query, chat_history)
        targets = self.get_target_agents(intent)

        return {
            "intent": intent,
            "agent_messages": [{"agent": "coordinator", "message": f"意图分类: {intent}, 调度: {targets}"}],
        }
