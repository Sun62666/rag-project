import os
import logging
from typing import Optional
from fastapi import Header
from src.core.security import get_current_user
from src.retriever import OpsRetriever
from src.graph import build_graph
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.agent.ops_agent import OpsAgent, build_agent

logger = logging.getLogger(__name__)

_retriever = None
_graph = None
_stm = None
_ltm = None
_agent = None


def init_components():
    global _retriever, _graph, _stm, _ltm, _agent

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_path = os.path.join(project_root, "data", "文档2.pdf")

    # 如果 PDF 不存在，传空字符串让 OpsRetriever 从 Milvus 加载数据
    if not os.path.exists(pdf_path):
        logger.info(f"PDF 文件不存在: {pdf_path}，将从 Milvus 加载已有数据")
        pdf_path = ""

    logger.info("初始化retriever中。。。。")
    _retriever = OpsRetriever(pdf_path)

    logger.info("构建LangGraph图中。。。。")
    _graph = build_graph(_retriever)

    logger.info("初始化短期记忆中。。。。")
    _stm = ShortTermMemory(max_history=20)

    logger.info("初始化长期记忆中。。。。")
    _ltm = LongTermMemory()

    logger.info("初始化通用文档问答服务中。。。。")
    try:
        from src.tools.document_qa import get_document_qa_service
        doc_qa = get_document_qa_service()
        if doc_qa and doc_qa._ensemble is not None:
            logger.info("✅ 通用文档问答服务初始化成功（property_regulations 集合已就绪）")
        else:
            logger.info("📝 通用文档问答服务已创建（property_regulations 集合为空，等待文档上传）")
    except Exception as e:
        logger.warning(f"通用文档问答服务初始化失败: {e}")

    logger.info("构建Agent智能体中。。。。")
    _agent = build_agent(_retriever, _stm, _ltm)


def cleanup_components():
    global _retriever, _graph, _stm, _ltm, _agent
    import src.core.redis as redis_mod
    if redis_mod._cache_instance is not None:
        try:
            redis_mod._cache_instance.close()
            logger.info("[Redis] 连接已关闭")
        except Exception as e:
            logger.warning(f"[Redis] 关闭异常: {e}")
        redis_mod._cache_instance = None
    _retriever = None
    _graph = None
    _stm = None
    _ltm = None
    _agent = None
    logger.info("[Cleanup] 所有组件已释放")


def get_retriever() -> Optional[OpsRetriever]:
    return _retriever


def get_graph():
    return _graph


def get_stm() -> Optional[ShortTermMemory]:
    return _stm


def get_ltm() -> Optional[LongTermMemory]:
    return _ltm


def get_agent() -> Optional[OpsAgent]:
    return _agent


def get_current_user_dep(authorization: Optional[str] = Header(None)) -> str:
    return get_current_user(authorization)
