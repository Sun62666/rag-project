import logging
from typing import Optional
from langchain_core.tools import tool
from src.retriever import OpsRetriever

logger = logging.getLogger(__name__)

retriever_instance: Optional[OpsRetriever] = None


def set_retriever(retriever: OpsRetriever):
    global retriever_instance
    retriever_instance = retriever


def knowledge_retriever_logic(query: str, retriever: OpsRetriever, top_k: int = 3) -> str:
    docs_with_scores = retriever.retriever_and_rerank_with_scores(query, top_k=top_k)
    if not docs_with_scores:
        return "【知识库检索结果】未检索到与问题相关的文档内容，知识库可能未覆盖该问题。"
    results = []
    for doc, score in docs_with_scores:
        relevance = "高" if score > 0.5 else "中" if score > 0.2 else "低"
        source = doc.metadata.get("source", "运维文档")
        results.append(f"[相关性:{relevance}({score:.2f})] {source}\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def knowledge_retriever(query: str) -> str:
    """从运维知识库检索故障解决方案、配置规范等文档。

    【调用时机】
    - 用户提出具体运维故障问题（如："Redis内存占用过大怎么办"、"MySQL连接数满了"）
    - 需要查找标准运维文档、配置规范、最佳实践
    - 涉及具体技术组件的问题（Redis、MySQL、Nginx、K8s、Docker等）
    - memory_retriever 检索后未找到相关历史记忆

    【重要】如果问题涉及具体技术组件的故障排查，应优先调用此工具

    参数：query-检索关键词"""
    if retriever_instance is None:
        return "知识库未初始化"
    return knowledge_retriever_logic(query, retriever_instance)
