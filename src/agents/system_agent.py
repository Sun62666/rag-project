"""系统监控智能体：实时系统状态检查专家

负责处理服务器状态查询、端口检查、日志分析等实时操作类问题。
"""

import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_llm
from src.tools.server_info import server_info_query, server_system_check_logic
from src.tools.port_check import port_check_logic
from src.tools.log_analyzer import read_service_log_logic, log_error_stats

logger = logging.getLogger(__name__)

SYSTEM_AGENT_PROMPT = """你是系统监控运维专家，负责实时检查服务器状态和日志分析。

## 你的职责
1. 查询服务器 CPU/内存/磁盘/进程状态
2. 检查端口占用情况
3. 读取和分析服务日志
4. 统计日志错误信息

## 回答要求
- 基于实时数据给出分析
- 异常指标需标注 ⚠ 警告
- 给出初步诊断建议
"""


class SystemAgent(BaseAgent):
    """系统监控智能体"""

    name = "system_agent"
    description = "实时系统状态检查和日志分析专家"

    def __init__(self):
        super().__init__(llm=create_llm())

        self.chain = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_AGENT_PROMPT),
            ("human", "用户问题: {query}\n\n实时数据:\n{context}"),
        ]) | self.llm

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        context_parts = []

        # 第一步：获取服务器系统状态
        try:
            sys_info = server_system_check_logic()
            context_parts.append(f"【服务器状态】\n{sys_info}")
            logger.info("[system_agent] 服务器状态获取完成")
        except Exception as e:
            logger.warning(f"[system_agent] 服务器状态获取失败: {e}")

        # 第二步：如果问题涉及端口，检查端口
        import re
        port_match = re.search(r'(\d{2,5})\s*端口|port\s*(\d+)', query, re.IGNORECASE)
        if port_match:
            port = int(port_match.group(1) or port_match.group(2))
            try:
                port_result = port_check_logic(port)
                context_parts.append(f"【端口检查】\n{port_result}")
                logger.info(f"[system_agent] 端口 {port} 检查完成")
            except Exception as e:
                logger.warning(f"[system_agent] 端口检查失败: {e}")

        # 第三步：如果问题涉及日志，读取日志
        if any(kw in query for kw in ["日志", "log", "错误", "error", "异常"]):
            try:
                log_result = read_service_log_logic(lines=30)
                context_parts.append(f"【服务日志】\n{log_result}")
                logger.info("[system_agent] 日志读取完成")
            except Exception as e:
                logger.warning(f"[system_agent] 日志读取失败: {e}")

        context = "\n\n".join(context_parts) if context_parts else "无可用实时数据"

        # LLM 分析
        response = self.chain.invoke({
            "query": query,
            "context": context,
        })

        result = response.content
        logger.info(f"[system_agent] 分析完成, 长度: {len(result)}")

        return {
            "system_result": result,
            "agent_messages": state.get("agent_messages", []) + [
                {"agent": "system_agent", "message": "系统检查完成"}
            ],
        }
