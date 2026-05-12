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
    """从历史对话记忆库中检索用户之前问过的相关问题和答案。当用户追问之前的话题、引用历史对话时调用。
    参数：query-检索关键词"""
    if long_term_memory_instance is None:
        return "长期记忆库未初始化"
    result = long_term_memory_instance.format_memory_context(query, top_k=3)
    return result if result else "未检索到相关历史记忆"
