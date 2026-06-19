"""检索效果评估服务: 多策略验证

支持以下评估策略:
1. Precision@K - 前 K 个结果中相关文档的比例
2. Recall@K - 前 K 个结果中召回的相关文档比例
3. MRR (Mean Reciprocal Rank) - 第一个相关文档的倒数排名
4. NDCG@K (Normalized Discounted Cumulative Gain) - 归一化折损累计增益
5. F1@K - Precision 和 Recall 的调和平均
6. Reranker 效果评估 - 对比重排序前后的排序质量

全程不使用 LLM，仅基于向量相似度判断。
"""
import logging
import math
import numpy as np
from typing import List, Dict, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from src.core.config import get_settings

logger = logging.getLogger(__name__)

# ============================================================================
# 评估问题集：覆盖运维领域 + 通用文档领域
# ============================================================================

# 运维领域评估问题
OPS_EVAL_QUESTIONS = [
    {
        "question": "Redis 内存溢出如何排查？",
        "ground_truth": (
            "Redis内存溢出通常由未设置maxmemory或大Key堆积导致。"
            "排查步骤：1. redis-cli info memory 查看内存使用 2. redis-cli --bigkeys 扫描大Key "
            "3. 检查淘汰策略 config get maxmemory-policy。"
            "修复：config set maxmemory-policy allkeys-lru，定期清理大Key。"
        ),
        "domain": "ops",
    },
    {
        "question": "MySQL 主从同步延迟怎么处理？",
        "ground_truth": (
            "MySQL主从延迟排查：show slave status\\G 查看 Seconds_Behind_Master。"
            "修复方案：1. 开启多线程复制 set global slave_parallel_workers=4 "
            "2. 拆分大事务 3. 升级从库硬件 4. 使用半同步复制。"
        ),
        "domain": "ops",
    },
    {
        "question": "Nginx 502 Bad Gateway 如何排查？",
        "ground_truth": (
            "Nginx 502 表示后端服务不可用。排查：1. curl 后端健康检查接口 "
            "2. tail -f /var/log/nginx/error.log 3. netstat -tlnp | grep 端口。"
            "修复：重启后端服务，调整 proxy_read_timeout，配置 max_fails 健康检查。"
        ),
        "domain": "ops",
    },
    {
        "question": "Kubernetes Pod CrashLoopBackOff 如何解决？",
        "ground_truth": (
            "CrashLoopBackOff 排查：kubectl describe pod 查看事件，"
            "kubectl logs --previous 查看崩溃日志。"
            "常见原因：OOMKilled、启动命令错误、配置缺失、健康检查配置不当。"
            "修复：调整 resources.limits.memory，修复配置，调整 initialDelaySeconds。"
        ),
        "domain": "ops",
    },
    {
        "question": "Linux 磁盘空间满如何清理？",
        "ground_truth": (
            "磁盘满排查：df -h 查看使用率，du -sh /* | sort -rh 定位大目录，"
            "lsof | grep deleted 查找已删除但被占用的文件。"
            "清理：删除过期日志，清理临时文件，docker system prune。"
        ),
        "domain": "ops",
    },
    {
        "question": "Docker 容器网络不通怎么排查？",
        "ground_truth": (
            "Docker网络排查：docker network ls 查看网络，docker network inspect 检查配置，"
            "docker exec ping 测试连通性。"
            "修复：加入同一网络，重建网桥，重启Docker恢复iptables，指定DNS。"
        ),
        "domain": "ops",
    },
    {
        "question": "CPU 使用率突然飙高怎么排查？",
        "ground_truth": (
            "CPU飙高排查：top -c 定位高CPU进程，top -H -p PID 分析线程，"
            "vmstat 1 5 查看系统级CPU。常见原因：进程死循环、突发流量、定时任务、内存不足导致swap。"
        ),
        "domain": "ops",
    },
    {
        "question": "Kafka 消费积压如何处理？",
        "ground_truth": (
            "Kafka消费积压排查：kafka-consumer-groups.sh --describe 查看 Lag。"
            "修复：增加消费者实例数，优化消费逻辑，扩容分区，调整 max.poll.interval.ms。"
        ),
        "domain": "ops",
    },
]

# 通用文档领域评估问题（物业管理条例）
DOC_EVAL_QUESTIONS = [
    {
        "question": "物业服务费的价格是由谁定的？",
        "ground_truth": (
            "由业主和物业服务企业按照国务院价格主管部门会同国务院建设行政主管部门制定的物业服务收费办法，"
            "在物业服务合同中约定。"
        ),
        "domain": "document",
    },
    {
        "question": "物业挪用专项维修资金的，如何处罚？",
        "ground_truth": (
            "违反本条例的规定，挪用专项维修资金的，由县级以上地方人民政府房地产行政主管部门"
            "追回挪用的专项维修资金，给予警告，没收违法所得，可以并处挪用数额2倍以下的罚款；"
            "构成犯罪的，依法追究直接负责的主管人员和其他直接责任人员的刑事责任。"
        ),
        "domain": "document",
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
        "domain": "document",
    },
]

# 全部评估问题
EVAL_QUESTIONS = OPS_EVAL_QUESTIONS + DOC_EVAL_QUESTIONS

# 默认相似度阈值
SIMILARITY_THRESHOLD = 0.7


# ============================================================================
# 向量相似度计算
# ============================================================================

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
    emb = OllamaEmbeddings(
        model=cfg.EMBED_MODEL,
        base_url=cfg.OLLAMA_URL
    )
    return emb.embed_query(text)


# ============================================================================
# 多策略评估指标计算
# ============================================================================

def compute_precision_at_k(relevance_flags: List[bool], k: int = 10) -> float:
    """Precision@K: 前 K 个结果中相关文档的比例"""
    top_k = relevance_flags[:k]
    if not top_k:
        return 0.0
    tp = sum(1 for r in top_k if r)
    return tp / len(top_k)


def compute_recall_at_k(relevance_flags: List[bool], total_relevant: int, k: int = 10) -> float:
    """Recall@K: 前 K 个结果中召回的相关文档比例"""
    if total_relevant == 0:
        return 0.0
    top_k = relevance_flags[:k]
    tp = sum(1 for r in top_k if r)
    return tp / total_relevant


def compute_mrr(relevance_flags: List[bool]) -> float:
    """MRR: 第一个相关文档的倒数排名"""
    for i, r in enumerate(relevance_flags):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """NDCG@K: 归一化折损累计增益

    Args:
        relevance_scores: 每个文档的相关性分数（0-1）
        k: 截断位置
    """
    top_k = relevance_scores[:k]
    if not top_k:
        return 0.0

    # DCG
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(top_k))

    # IDCG (理想排序)
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_f1_at_k(precision: float, recall: float) -> float:
    """F1@K: Precision 和 Recall 的调和平均"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# 单问题评估
# ============================================================================

def evaluate_single(
    question: str,
    ground_truth: str,
    threshold: float = SIMILARITY_THRESHOLD,
    k: int = 10,
    use_reranker: bool = False,
) -> dict:
    """对单个问题执行多策略评估

    Args:
        question: 评估问题
        ground_truth: 标准答案
        threshold: 相似度阈值，超过则判定为相关
        k: 截断位置（默认10）
        use_reranker: 是否使用 Reranker 重排序后再评估

    Returns:
        包含所有指标的评估结果
    """
    from src.tools.document_qa import get_document_qa_service

    svc = get_document_qa_service()
    if svc is None:
        return {
            "question": question,
            "error": "文档问答服务未初始化，请先上传文档",
            "precision_at_k": 0,
            "recall_at_k": 0,
            "mrr": 0,
            "ndcg_at_k": 0,
            "f1_at_k": 0,
            "tp_count": 0,
            "chunks": [],
        }

    # Step 1: 将真值答案转为向量
    truth_embedding = _get_embedding(ground_truth)

    # Step 2: 对问题执行混合检索
    retrieved_chunks = svc.search_raw(question, top_k=k)

    if not retrieved_chunks:
        return {
            "question": question,
            "precision_at_k": 0,
            "recall_at_k": 0,
            "mrr": 0,
            "ndcg_at_k": 0,
            "f1_at_k": 0,
            "tp_count": 0,
            "chunks": [],
        }

    # Step 3: 计算每个切片与真值向量的余弦相似度
    chunk_details = []
    relevance_flags = []
    relevance_scores = []
    tp_count = 0

    for chunk in retrieved_chunks:
        chunk_embedding = _get_embedding(chunk["content"])
        sim = cosine_similarity(truth_embedding, chunk_embedding)
        is_relevant = sim >= threshold
        if is_relevant:
            tp_count += 1
        relevance_flags.append(is_relevant)
        relevance_scores.append(sim)
        chunk_details.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"][:200] + ("..." if len(chunk["content"]) > 200 else ""),
            "similarity": round(sim, 4),
            "relevant": is_relevant,
        })

    # Step 4: Reranker 重排序评估（可选）
    reranker_improvement = None
    if use_reranker:
        try:
            reranker_result = _evaluate_with_reranker(question, retrieved_chunks, ground_truth, threshold, k)
            reranker_improvement = reranker_result
        except Exception as e:
            logger.warning(f"Reranker 评估失败: {e}")

    # Step 5: 计算多策略指标
    total_relevant = tp_count
    precision = compute_precision_at_k(relevance_flags, k)
    recall = compute_recall_at_k(relevance_flags, total_relevant, k)
    mrr = compute_mrr(relevance_flags)
    ndcg = compute_ndcg_at_k(relevance_scores, k)
    f1 = compute_f1_at_k(precision, recall)

    result = {
        "question": question,
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "f1_at_k": round(f1, 4),
        "tp_count": tp_count,
        "total_chunks": len(retrieved_chunks),
        "chunks": chunk_details,
    }

    if reranker_improvement:
        result["reranker_evaluation"] = reranker_improvement

    return result


def _evaluate_with_reranker(
    question: str,
    chunks: List[Dict],
    ground_truth: str,
    threshold: float,
    k: int,
) -> dict:
    """使用 Reranker 重排序后评估指标变化

    对比重排序前后的 Precision@K 和 MRR，衡量 Reranker 的效果。
    """
    from sentence_transformers import CrossEncoder
    import os

    cfg = get_settings()
    rerank_model_path = cfg.RERANK_MODEL

    if not os.path.exists(rerank_model_path):
        return {"error": f"Reranker 模型不存在: {rerank_model_path}"}

    model = CrossEncoder(rerank_model_path, num_labels=1)

    # 获取所有文档内容
    docs = [c["content"] for c in chunks]

    # Reranker 打分
    pairs = [(question, doc) for doc in docs]
    scores = model.predict(pairs)

    # 按分数排序
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_chunks = [chunks[i] for i in ranked_indices]
    ranked_scores = [float(scores[i]) for i in ranked_indices]

    # 计算重排序后的相似度
    truth_embedding = _get_embedding(ground_truth)
    reranked_relevance_flags = []
    reranked_relevance_scores = []
    reranked_tp = 0

    for chunk in ranked_chunks:
        chunk_embedding = _get_embedding(chunk["content"])
        sim = cosine_similarity(truth_embedding, chunk_embedding)
        is_relevant = sim >= threshold
        if is_relevant:
            reranked_tp += 1
        reranked_relevance_flags.append(is_relevant)
        reranked_relevance_scores.append(sim)

    # 计算重排序后的指标
    reranked_precision = compute_precision_at_k(reranked_relevance_flags, k)
    reranked_mrr = compute_mrr(reranked_relevance_flags)
    reranked_ndcg = compute_ndcg_at_k(reranked_relevance_scores, k)

    # 计算原始指标
    original_relevance_flags = []
    original_relevance_scores = []
    original_tp = 0
    for chunk in chunks:
        chunk_embedding = _get_embedding(chunk["content"])
        sim = cosine_similarity(truth_embedding, chunk_embedding)
        is_relevant = sim >= threshold
        if is_relevant:
            original_tp += 1
        original_relevance_flags.append(is_relevant)
        original_relevance_scores.append(sim)

    original_precision = compute_precision_at_k(original_relevance_flags, k)
    original_mrr = compute_mrr(original_relevance_flags)
    original_ndcg = compute_ndcg_at_k(original_relevance_scores, k)

    return {
        "original_precision_at_k": round(original_precision, 4),
        "reranked_precision_at_k": round(reranked_precision, 4),
        "precision_delta": round(reranked_precision - original_precision, 4),
        "original_mrr": round(original_mrr, 4),
        "reranked_mrr": round(reranked_mrr, 4),
        "mrr_delta": round(reranked_mrr - original_mrr, 4),
        "original_ndcg_at_k": round(original_ndcg, 4),
        "reranked_ndcg_at_k": round(reranked_ndcg, 4),
        "ndcg_delta": round(reranked_ndcg - original_ndcg, 4),
        "reranker_scores": [round(s, 4) for s in ranked_scores[:k]],
    }


# ============================================================================
# 批量评估
# ============================================================================

def run_evaluation(
    threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "all",
    use_reranker: bool = False,
    k: int = 10,
) -> dict:
    """运行全部评估

    Args:
        threshold: 相似度阈值
        domain: 评估领域 ("all"=全部, "ops"=运维, "document"=通用文档)
        use_reranker: 是否启用 Reranker 评估
        k: 截断位置

    Returns:
        包含所有问题和汇总指标的评估结果
    """
    # 筛选领域
    if domain == "ops":
        questions = OPS_EVAL_QUESTIONS
    elif domain == "document":
        questions = DOC_EVAL_QUESTIONS
    else:
        questions = EVAL_QUESTIONS

    results = []
    for eq in questions:
        result = evaluate_single(
            eq["question"],
            eq["ground_truth"],
            threshold=threshold,
            k=k,
            use_reranker=use_reranker,
        )
        result["domain"] = eq.get("domain", "unknown")
        results.append(result)

    # 汇总指标
    valid_results = [r for r in results if "error" not in r]
    n = max(len(valid_results), 1)

    summary = {
        "avg_precision_at_k": round(sum(r["precision_at_k"] for r in valid_results) / n, 4),
        "avg_recall_at_k": round(sum(r["recall_at_k"] for r in valid_results) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in valid_results) / n, 4),
        "avg_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in valid_results) / n, 4),
        "avg_f1_at_k": round(sum(r["f1_at_k"] for r in valid_results) / n, 4),
        "total_questions": len(results),
        "valid_questions": len(valid_results),
        "threshold": threshold,
        "k": k,
        "domain": domain,
        "use_reranker": use_reranker,
    }

    # 如果启用了 Reranker，汇总 Reranker 效果
    if use_reranker:
        reranker_results = [r["reranker_evaluation"] for r in valid_results if "reranker_evaluation" in r and "error" not in r.get("reranker_evaluation", {})]
        if reranker_results:
            rn = max(len(reranker_results), 1)
            summary["reranker_summary"] = {
                "avg_precision_delta": round(sum(r["precision_delta"] for r in reranker_results) / rn, 4),
                "avg_mrr_delta": round(sum(r["mrr_delta"] for r in reranker_results) / rn, 4),
                "avg_ndcg_delta": round(sum(r["ndcg_delta"] for r in reranker_results) / rn, 4),
            }

    return {
        "results": results,
        "summary": summary,
    }
