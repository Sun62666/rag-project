import logging
from typing import Optional
from langchain_core.tools import tool
from src.memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

long_term_memory_instance: Optional[LongTermMemory] = None


def set_long_term_memory(ltm: LongTermMemory):
    global long_term_memory_instance
    long_term_memory_instance = ltm


@tool
def memory_retriever(query: str) -> str:
    """从历史对话记忆库中检索用户之前问过的相关问题和答案。

    【调用时机】
    - 用户追问之前讨论过的话题（如"刚才那个问题再说一下"）
    - 用户提出与之前问题相关的场景（如之前问"Docker无法启动"，现在问"Docker挂载权限问题"）
    - 用户补充提问或要求详细解释之前的答案

    【重要】遇到相关问题时必须优先调用此工具，比知识库检索更高效精准

    参数：query-用户当前问题或关键词"""
    if long_term_memory_instance is None:
        return "长期记忆库未初始化"
    result = long_term_memory_instance.format_memory_context(query, top_k=3)
    return result if result else "未检索到相关历史记忆"
