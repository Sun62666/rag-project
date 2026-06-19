"""多智能体工作流：基于 LangGraph 编排多个专家智能体

架构:
    START → coordinator(分类) → [fault_agent | system_agent | document_agent | memory_agent]
                                    ↓ (mixed 类型可并行)
                              synthesis_agent(综合) → END

特点:
1. 协调者智能体分类后路由到对应专家
2. mixed 类型支持多专家并行处理
3. 综合智能体整合所有专家结果生成最终回答
4. 支持流式输出（通过 stream_mode="messages"）
"""

import logging
from typing import Dict
from langgraph.graph import StateGraph, START, END
from src.agents.base import MultiAgentState
from src.agents.coordinator import CoordinatorAgent
from src.agents.fault_agent import FaultAgent
from src.agents.system_agent import SystemAgent
from src.agents.document_agent import DocumentAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.synthesis_agent import SynthesisAgent

logger = logging.getLogger(__name__)


def make_coordinator_node(coordinator: CoordinatorAgent):
    def coordinator_node(state: Dict) -> Dict:
        logger.info(f"[workflow] coordinator 处理: {state['query'][:50]}")
        return coordinator.run(state)
    return coordinator_node


def make_fault_node(fault_agent: FaultAgent):
    def fault_node(state: Dict) -> Dict:
        logger.info("[workflow] fault_agent 处理中...")
        return fault_agent.run(state)
    return fault_node


def make_system_node(system_agent: SystemAgent):
    def system_node(state: Dict) -> Dict:
        logger.info("[workflow] system_agent 处理中...")
        return system_agent.run(state)
    return system_node


def make_document_node(document_agent: DocumentAgent):
    def document_node(state: Dict) -> Dict:
        logger.info("[workflow] document_agent 处理中...")
        return document_agent.run(state)
    return document_node


def make_memory_node(memory_agent: MemoryAgent):
    def memory_node(state: Dict) -> Dict:
        logger.info("[workflow] memory_agent 处理中...")
        return memory_agent.run(state)
    return memory_node


def make_synthesis_node(synthesis_agent: SynthesisAgent):
    def synthesis_node(state: Dict) -> Dict:
        logger.info("[workflow] synthesis_agent 整合中...")
        return synthesis_agent.run(state)
    return synthesis_node


def reject_node(state: Dict) -> Dict:
    logger.info("[workflow] 拒绝非运维问题")
    return {"answer": "当前知识库未覆盖该问题，建议转交人工运维专家。"}


def route_by_intent(state: Dict) -> list:
    """根据意图路由到对应专家智能体（支持并行返回多个节点）"""
    intent = state.get("intent", "fault")

    routing = {
        "fault": ["fault_agent"],
        "system": ["system_agent"],
        "document": ["document_agent"],
        "followup": ["memory_agent"],
        "mixed": ["fault_agent", "system_agent"],
        "reject": ["reject"],
    }

    targets = routing.get(intent, ["fault_agent"])
    logger.info(f"[workflow] 路由: {intent} → {targets}")
    return targets


def build_multi_agent_graph(retriever=None, ltm=None):
    """构建多智能体工作流图

    Args:
        retriever: OpsRetriever 实例（用于故障诊断智能体）
        ltm: LongTermMemory 实例（用于记忆智能体）

    Returns:
        编译后的 LangGraph 工作流
    """
    # 初始化所有智能体
    coordinator = CoordinatorAgent()
    fault_agent = FaultAgent(retriever=retriever)
    system_agent = SystemAgent()
    document_agent = DocumentAgent()
    memory_agent = MemoryAgent(ltm=ltm)
    synthesis_agent = SynthesisAgent()

    # 设置 retriever 和 ltm 的全局实例
    if retriever:
        from src.tools.knowledge import set_retriever
        set_retriever(retriever)
    if ltm:
        from src.tools.memory_retriever import set_long_term_memory
        set_long_term_memory(ltm)

    # 构建工作流图
    workflow = StateGraph(MultiAgentState)

    # 添加节点
    workflow.add_node("coordinator", make_coordinator_node(coordinator))
    workflow.add_node("fault_agent", make_fault_node(fault_agent))
    workflow.add_node("system_agent", make_system_node(system_agent))
    workflow.add_node("document_agent", make_document_node(document_agent))
    workflow.add_node("memory_agent", make_memory_node(memory_agent))
    workflow.add_node("synthesis_agent", make_synthesis_node(synthesis_agent))
    workflow.add_node("reject", reject_node)

    # 设置入口
    workflow.add_edge(START, "coordinator")

    # 条件路由：coordinator → 专家智能体（支持并行）
    workflow.add_conditional_edges(
        "coordinator",
        route_by_intent,
        {
            "fault_agent": "fault_agent",
            "system_agent": "system_agent",
            "document_agent": "document_agent",
            "memory_agent": "memory_agent",
            "reject": "reject",
        },
    )

    # 所有专家智能体完成后进入综合智能体
    workflow.add_edge("fault_agent", "synthesis_agent")
    workflow.add_edge("system_agent", "synthesis_agent")
    workflow.add_edge("document_agent", "synthesis_agent")
    workflow.add_edge("memory_agent", "synthesis_agent")

    # 综合智能体和拒绝节点指向结束
    workflow.add_edge("synthesis_agent", END)
    workflow.add_edge("reject", END)

    logger.info("[workflow] 多智能体工作流构建完成")
    return workflow.compile()
