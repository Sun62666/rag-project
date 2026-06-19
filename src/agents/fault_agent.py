"""故障诊断智能体：运维故障排查专家

负责处理运维技术问题，使用知识库检索和知识图谱查询工具。
擅长 Redis/MySQL/Nginx/K8s/Docker/Linux/Kafka/ES 等组件的故障诊断。
"""

import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_llm
from src.tools import knowledge_retriever, knowledge_graph_query
from src.tools.knowledge import knowledge_retriever_logic, set_retriever

logger = logging.getLogger(__name__)

FAULT_AGENT_PROMPT = """你是资深运维故障诊断专家，擅长 Linux/数据库/中间件/云原生运维故障排查。

## 你的职责
1. 从运维知识库检索故障解决方案
2. 查询知识图谱了解故障连锁影响和修复方案
3. 基于检索结果给出结构化的故障排查建议

## 工作流程
1. 先调用 knowledge_retriever 检索知识库
2. 如果涉及故障影响范围或组件依赖，调用 knowledge_graph_query 查询图谱
3. 综合检索结果给出回答

## 回答格式
【故障现象】
【可能原因】
【排查命令】
【修复步骤】
【验证方法】

高危操作必须标注 ⚠ 警告。
"""


class FaultAgent(BaseAgent):
    """故障诊断智能体"""

    name = "fault_agent"
    description = "运维故障排查专家，使用知识库和知识图谱"

    def __init__(self, retriever=None):
        super().__init__(llm=create_llm())
        if retriever:
            set_retriever(retriever)
        self.tools = [knowledge_retriever, knowledge_graph_query]
        self.bind_tools()

        self.chain = ChatPromptTemplate.from_messages([
            ("system", FAULT_AGENT_PROMPT),
            ("human", "用户问题: {query}\n\n可用上下文:\n{context}"),
        ]) | self.llm

    def run(self, state: Dict) -> Dict:
        query = state["query"]

        # 第一步：直接检索知识库（不依赖 LLM 工具调用，确保稳定性）
        from src.tools.knowledge import retriever_instance
        context_parts = []

        if retriever_instance:
            try:
                kb_result = knowledge_retriever_logic(query, retriever_instance, top_k=3)
                context_parts.append(kb_result)
                logger.info(f"[fault_agent] 知识库检索完成, 长度: {len(kb_result)}")
            except Exception as e:
                logger.warning(f"[fault_agent] 知识库检索失败: {e}")

        # 第二步：查询知识图谱
        try:
            from src.graph.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            if kg.is_available:
                kg_context = kg.format_graph_context(query, depth=2)
                if kg_context:
                    context_parts.append(kg_context)
                    logger.info(f"[fault_agent] 知识图谱查询完成, 长度: {len(kg_context)}")
        except Exception as e:
            logger.warning(f"[fault_agent] 知识图谱查询失败: {e}")

        context = "\n\n".join(context_parts) if context_parts else "无可用上下文"

        # 第三步：LLM 生成回答（带重试和降级逻辑）
        result = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.chain.invoke({
                    "query": query,
                    "context": context,
                })
                result = response.content or ""
                if result.strip():
                    break
                logger.warning(f"[fault_agent] 空内容! finish_reason={response.response_metadata}, "
                               f"tool_calls={getattr(response, 'tool_calls', None)}, "
                               f"context长度={len(context)}, query={query[:80]}")
                logger.warning(f"[fault_agent] LLM 返回空内容, 重试 {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.warning(f"[fault_agent] LLM 调用失败 (尝试 {attempt + 1}): {e}")
                result = ""

        # 降级：LLM 始终返回空时，直接使用检索上下文作为回答
        if not result.strip():
            logger.warning("[fault_agent] LLM 多次返回空内容，降级使用检索上下文")

            if context_parts:
                result = "基于知识库检索结果：\n\n" + "\n\n".join(context_parts)
            else:
                result = "当前知识库未覆盖该问题，建议转交人工运维专家。"

        logger.info(f"[fault_agent] 回答生成完成, 长度: {len(result)}")

        return {
            "fault_result": result,
            "agent_messages": state.get("agent_messages", []) + [
                {"agent": "fault_agent", "message": "故障诊断完成"}
            ],
        }
