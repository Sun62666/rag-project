"""多智能体系统基础模块

定义共享状态、基础智能体类和通用工具。
每个专家智能体继承 BaseAgent，拥有独立的系统提示词、工具集和 LLM 配置。
"""

import logging
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class MultiAgentState(Dict):
    """多智能体共享状态

    每个字段说明:
    - query: 用户原始问题
    - intent: 协调者分类结果 (fault/system/document/followup/mixed/reject)
    - chat_history: 对话历史列表
    - fault_result: 故障诊断智能体的输出
    - system_result: 系统监控智能体的输出
    - document_result: 文档问答智能体的输出
    - memory_result: 记忆智能体的输出
    - agent_messages: 各智能体的处理过程消息（用于前端展示状态）
    - answer: 最终综合回答
    """

    query: str
    intent: str
    chat_history: list
    fault_result: str
    system_result: str
    document_result: str
    memory_result: str
    agent_messages: List[Dict[str, str]]
    answer: str


def create_llm(temperature: float = 0, streaming: bool = True) -> ChatOpenAI:
    """创建统一的 LLM 实例"""
    cfg = get_settings()
    return ChatOpenAI(
        model=cfg.LLM_MODEL,
        base_url=cfg.BASE_URL,
        api_key=cfg.DASHSCOPE_API_KEY,
        temperature=temperature,
        streaming=streaming,
    )


def create_fast_llm() -> ChatOpenAI:
    """创建快速 LLM（用于分类等轻量任务，关闭流式）"""
    return create_llm(temperature=0, streaming=False)


class BaseAgent:
    """专家智能体基类

    每个子类需实现:
    - name: 智能体名称
    - system_prompt: 系统提示词
    - tools: 该智能体可用的工具列表
    - run(): 执行逻辑
    """

    name: str = "base"
    description: str = ""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or create_llm()
        self.tools: List[BaseTool] = []
        self._tool_map: Dict[str, BaseTool] = {}

    def bind_tools(self):
        """将工具绑定到 LLM"""
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
            self._tool_map = {t.name: t for t in self.tools}

    def execute_tool_calls(self, response) -> Dict[str, str]:
        """执行 LLM 返回的工具调用，返回工具名→结果映射"""
        results = {}
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                if tool_name in self._tool_map:
                    logger.info(f"[{self.name}] 调用工具: {tool_name}, 参数: {tool_args}")
                    result = self._tool_map[tool_name].invoke(tool_args)
                    results[tool_name] = result
                else:
                    logger.warning(f"[{self.name}] 未知工具: {tool_name}")
        return results

    def build_context(self, tool_results: Dict[str, str]) -> str:
        """将工具结果构建为上下文文本"""
        if not tool_results:
            return ""
        parts = []
        for tool_name, result in tool_results.items():
            parts.append(f"【{tool_name} 执行结果】\n{result}")
        return "\n\n".join(parts)

    def run(self, state: Dict) -> str:
        """执行智能体逻辑，返回结果文本"""
        raise NotImplementedError("子类必须实现 run 方法")
