"""知识图谱实体关系抽取器

提供三种抽取方式：
1. rule_extractor: 基于正则和模式匹配的规则模板抽取
2. spacy_extractor: 基于 spaCy NLP 的依存句法分析抽取
3. llm_extractor: 基于大语言模型的深度理解抽取
"""
from src.graph.extractors.rule_extractor import extract_triples as rule_extract
from src.graph.extractors.spacy_extractor import extract_triples as spacy_extract
from src.graph.extractors.llm_extractor import extract_triples as llm_extract

__all__ = ["rule_extract", "spacy_extract", "llm_extract"]
