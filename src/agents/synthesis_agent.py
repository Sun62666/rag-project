"""综合回答智能体：整合多智能体结果

当多个专家智能体并行处理后，综合智能体负责：
1. 收集各智能体的输出
2. 去重和整合信息
3. 生成结构化的最终回答
"""

import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_llm
from src.prompts import load_system_prompt

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """你是运维综合回答专家，负责整合多个专家智能体的分析结果，生成最终回答。

## 你的职责
1. 收集故障诊断、系统监控、文档问答、历史记忆等专家的分析结果
2. 去重和整合信息，消除矛盾
3. 生成结构化、连贯的最终回答

## 回答格式（运维问题）
【故障现象】
【可能原因】
【排查命令】
【修复步骤】
【验证方法】

## 回答格式（文档问题）
【相关文档】
【文档内容摘要】
【详细解答】

## 回答格式（追问）
直接基于历史对话内容回答，不需要标准格式。

## 重要规则
- 严格基于专家提供的上下文信息生成回答
- 高危操作必须标注 ⚠ 警告
- 如果所有专家都未提供有效信息，输出：当前知识库未覆盖该问题，建议转交人工运维专家。
- 禁止在回复中包含工具调用过程或内部推理
"""


class SynthesisAgent(BaseAgent):
    """综合回答智能体"""

    name = "synthesis_agent"
    description = "整合多智能体结果，生成最终回答"

    def __init__(self):
        super().__init__(llm=create_llm())

        self.chain = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_PROMPT),
            ("human", "用户问题: {query}\n\n各专家分析结果:\n{expert_results}\n\n历史对话:\n{chat_history}"),
        ]) | self.llm

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        chat_history = state.get("chat_history", [])

        # 收集各专家结果
        expert_parts = []

        if state.get("fault_result"):
            expert_parts.append(f"【故障诊断专家】\n{state['fault_result']}")

        if state.get("system_result"):
            expert_parts.append(f"【系统监控专家】\n{state['system_result']}")

        if state.get("document_result"):
            expert_parts.append(f"【文档问答专家】\n{state['document_result']}")

        if state.get("memory_result"):
            expert_parts.append(f"【记忆专家】\n{state['memory_result']}")

        expert_results = "\n\n---\n\n".join(expert_parts) if expert_parts else "无专家分析结果"

        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"

        logger.info(f"[synthesis_agent] 整合 {len(expert_parts)} 个专家结果")

        answer = ""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.chain.invoke({
                    "query": query,
                    "expert_results": expert_results,
                    "chat_history": history_str,
                })
                answer = response.content or ""
                if answer.strip():
                    break
                logger.warning(f"[synthesis_agent] LLM 返回空内容, 重试 {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.warning(f"[synthesis_agent] LLM 调用失败 (尝试 {attempt + 1}): {e}")
                answer = ""

        # 降级：LLM 返回空时，直接使用专家结果
        if not answer.strip():
            logger.warning("[synthesis_agent] LLM 返回空内容，降级使用专家结果")
            if expert_parts:
                answer = "\n\n---\n\n".join(expert_parts)
            else:
                answer = "当前知识库未覆盖该问题，建议转交人工运维专家。"

        logger.info(f"[synthesis_agent] 最终回答生成完成, 长度: {len(answer)}")

        return {
            "answer": answer,
            "agent_messages": state.get("agent_messages", []) + [
                {"agent": "synthesis_agent", "message": "综合回答完成"}
            ],
        }
