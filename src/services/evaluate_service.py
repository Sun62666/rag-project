"""检索效果评估服务: Precision@10

融合自 Agent 项目，基于余弦相似度计算检索精准率。
全程不使用 LLM，仅基于向量相似度判断。
"""
import logging
import numpy as np
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from src.core.config import get_settings

logger = logging.getLogger(__name__)

# 预定义评估问题（物业管理条例）
EVAL_QUESTIONS = [
    {
        "question": "物业服务费的价格是由谁定的？",
        "ground_truth": (
            "由业主和物业服务企业按照国务院价格主管部门会同国务院建设行政主管部门制定的物业服务收费办法，"
            "在物业服务合同中约定。"
        ),
    },
    {
        "question": "物业挪用专项维修资金的，如何处罚？",
        "ground_truth": (
            "违反本条例的规定，挪用专项维修资金的，由县级以上地方人民政府房地产行政主管部门"
            "追回挪用的专项维修资金，给予警告，没收违法所得，可以并处挪用数额2倍以下的罚款；"
            "构成犯罪的，依法追究直接负责的主管人员和其他直接责任人员的刑事责任。"
        ),
    },
    {
        "question": "业主在物业管理活动中，享有哪些权利？",
        "ground_truth": (
            "业主在物业管理活动中，享有下列权利：（一）按照物业服务合同的约定，接受物业服务企业提供的服务；"
            "（二）提议召开业主大会会议，并就物业管理的有关事项提出建议；"
            "（三）提出制定和修改管理规约、业主大会议事规则的建议；"
            "（四）参加业主大会会议，行使投票权；"
            "（五）选举业主委员会成员，并享有被选举权；"
            "（六）监督业主委员会的工作；"
            "（七）监督物业服务企业履行物业服务合同；"
            "（八）对物业共用部位、共用设施设备和相关场地使用情况享有知情权和监督权；"
            "（九）监督物业共用部位、共用设施设备专项维修资金的管理和使用；"
            "（十）法律、法规规定的其他权利。"
        ),
    },
]

# 默认相似度阈值
SIMILARITY_THRESHOLD = 0.7


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _get_embedding(text: str) -> List[float]:
    """获取文本嵌入向量"""
    cfg = get_settings()
    emb = DashScopeEmbeddings(
        model=cfg.EMBED_MODEL,
        dashscope_api_key=cfg.DASHSCOPE_API_KEY
    )
    return emb.embed_query(text)


def evaluate_single(question: str, ground_truth: str, threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """对单个问题执行评估"""
    from src.tools.document_qa import get_document_qa_service

    svc = get_document_qa_service()
    if svc is None:
        return {
            "question": question,
            "error": "文档问答服务未初始化，请先上传文档",
            "precision_at_10": 0,
            "tp_count": 0,
            "chunks": [],
        }

    # Step 1: 将真值答案转为向量
    truth_embedding = _get_embedding(ground_truth)

    # Step 2: 对问题执行混合检索
    retrieved_chunks = svc.search_raw(question, top_k=10)

    if not retrieved_chunks:
        return {
            "question": question,
            "precision_at_10": 0,
            "tp_count": 0,
            "chunks": [],
        }

    # Step 3: 计算每个切片与真值向量的余弦相似度
    chunk_details = []
    tp_count = 0
    for chunk in retrieved_chunks:
        chunk_embedding = _get_embedding(chunk["content"])
        sim = cosine_similarity(truth_embedding, chunk_embedding)
        is_relevant = sim >= threshold
        if is_relevant:
            tp_count += 1
        chunk_details.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"][:200] + ("..." if len(chunk["content"]) > 200 else ""),
            "similarity": round(sim, 4),
            "relevant": is_relevant,
        })

    # Step 4: 计算 Precision@10
    precision = tp_count / max(len(chunk_details), 1)

    return {
        "question": question,
        "precision_at_10": round(precision, 4),
        "tp_count": tp_count,
        "chunks": chunk_details,
    }


def run_evaluation(threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """运行全部评估"""
    results = []
    for eq in EVAL_QUESTIONS:
        result = evaluate_single(eq["question"], eq["ground_truth"], threshold)
        results.append(result)

    avg_precision = sum(r["precision_at_10"] for r in results) / max(len(results), 1)

    return {
        "results": results,
        "avg_precision_at_10": round(avg_precision, 4),
        "threshold": threshold,
    }
