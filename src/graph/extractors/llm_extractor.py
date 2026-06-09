"""LLM 抽取器：基于大语言模型从运维文档中抽取实体和关系

使用 LLM 对复杂文本进行深度理解和结构化抽取，
适合处理规则和 NLP 方法难以覆盖的非结构化文本。
"""
import re
import json
import logging
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.config import get_settings
from src.prompts import load_kg_extraction_prompt

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    """延迟加载 LLM"""
    global _llm
    if _llm is None:
        cfg = get_settings()
        _llm = ChatOpenAI(
            model=cfg.LLM_MODEL,
            base_url=cfg.BASE_URL,
            api_key=cfg.DASHSCOPE_API_KEY,
            temperature=0,
        )
    return _llm


def extract_triples(text: str) -> List[Dict]:
    """使用 LLM 从文本中抽取三元组

    Args:
        text: 运维文档文本

    Returns:
        三元组列表，每个元素包含 from_entity, from_type, relation, to_entity, to_type
    """
    llm = _get_llm()
    if llm is None:
        return []

    try:
        # 截取文本避免超出 token 限制
        truncated = text[:3000]
        response = llm.invoke([
            SystemMessage(content=load_kg_extraction_prompt()),
            HumanMessage(content=f"请从以下运维文档中抽取实体和关系：\n\n{truncated}"),
        ])
        content = response.content.strip()
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            triples = json.loads(json_match.group())
            # 验证三元组格式
            valid = []
            for t in triples:
                if all(k in t for k in ("from_entity", "from_type", "relation", "to_entity", "to_type")):
                    valid.append(t)
            if valid:
                logger.info(f"[LLM抽取] 从文本中抽取 {len(valid)} 个三元组")
            return valid
        return []
    except Exception as e:
        logger.error(f"[LLM抽取] 抽取失败: {e}")
        return []
