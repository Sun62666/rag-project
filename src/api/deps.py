import os
import logging
from functools import lru_cache
from typing import Optional
from fastapi import Depends, Header
from src.core.config import Config, get_settings
from src.core.redis import get_cache
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

    cfg = get_settings()
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "文档2.pdf")

    logger.info("初始化retriever中。。。。")
    _retriever = OpsRetriever(pdf_path)

    logger.info("构建LangGraph图中。。。。")
    _graph = build_graph(_retriever)

    logger.info("初始化短期记忆中。。。。")
    _stm = ShortTermMemory(max_history=20)

    logger.info("初始化长期记忆中。。。。")
    _ltm = LongTermMemory()

    logger.info("构建Agent智能体中。。。。")
    _agent = build_agent(_retriever, _stm, _ltm)


def get_retriever() -> OpsRetriever:
    return _retriever


def get_graph():
    return _graph


def get_stm() -> ShortTermMemory:
    return _stm


def get_ltm() -> LongTermMemory:
    return _ltm


def get_agent() -> OpsAgent:
    return _agent


def get_current_user_dep(authorization: Optional[str] = Header(None)) -> str:
    return get_current_user(authorization)
