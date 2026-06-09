import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _get_kg():
    from src.graph.knowledge_graph import get_knowledge_graph
    return get_knowledge_graph()


@tool
def knowledge_graph_query(query: str) -> str:
    """从运维知识图谱中查询实体关系、故障链路和修复方案。

    【调用时机】
    - 需要了解故障的连锁影响（如"Redis宕机会影响哪些服务"）
    - 需要查找某个故障的所有修复方案
    - 需要了解组件之间的依赖关系
    - 需要进行多跳推理（如A导致B，B导致C的链路分析）

    【重要】当用户问题涉及故障影响范围、组件依赖关系、连锁反应时优先调用此工具

    参数：query-实体名称或故障关键词（如"Redis"、"OOM"、"连接超时"）"""
    kg = _get_kg()
    if not kg.is_available:
        return "知识图谱未连接，无法查询"

    context = kg.format_graph_context(query, depth=2)
    if not context:
        return f"知识图谱中未找到与 '{query}' 相关的信息"

    stats = kg.get_stats()
    return f"{context}\n\n[图谱统计: {stats.get('total_nodes', 0)} 个实体, {stats.get('total_relations', 0)} 个关系]"


@tool
def knowledge_graph_extract(text: str, method: str = "hybrid") -> str:
    """从运维文档文本中抽取实体和关系，写入知识图谱。

    【调用时机】
    - 用户提供了新的运维文档或故障案例，需要将其结构化存入知识图谱
    - 需要从一段文本中提取组件、故障、命令、配置之间的关系

    【抽取方式】
    - hybrid: 混合模式（推荐），先用规则+spaCy快速抽取，不足时用LLM补充
    - rule: 仅规则模板抽取（最快，适合结构化文档）
    - spacy: 仅spaCy NLP抽取（需下载模型）
    - llm: 仅LLM大模型抽取（最准确但最慢）

    参数：
    - text: 需要抽取的运维文档文本
    - method: 抽取方式，默认 hybrid"""
    kg = _get_kg()
    if not kg.is_available:
        return "知识图谱未连接，无法抽取"

    count = kg.extract_and_ingest(text, source="agent_tool", method=method)
    if count == 0:
        return "未能从文本中抽取到有效的实体关系"
    return f"成功抽取并写入 {count} 个三元组到知识图谱 (方式: {method})"
