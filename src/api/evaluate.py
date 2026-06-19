"""检索效果评估 API

支持多策略评估:
- Precision@K, Recall@K, MRR, NDCG@K, F1@K
- Reranker 效果对比
- 按领域筛选 (ops/document/all)
- 自定义问题评估
"""
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from src.api.deps import get_current_user_dep
from src.services.evaluate_service import (
    run_evaluation,
    evaluate_single,
    EVAL_QUESTIONS,
    OPS_EVAL_QUESTIONS,
    DOC_EVAL_QUESTIONS,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["评估"])


class EvaluateRequest(BaseModel):
    threshold: float = 0.7
    domain: str = "all"  # all / ops / document
    use_reranker: bool = False
    k: int = 10
    selected_indices: Optional[List[int]] = None  # 选择指定问题的索引


class CustomEvaluateRequest(BaseModel):
    question: str
    ground_truth: str
    threshold: float = 0.7
    k: int = 10


@router.post("/evaluate")
async def evaluate(
    req: EvaluateRequest = EvaluateRequest(),
    _user: str = Depends(get_current_user_dep),
):
    """运行多策略检索效果评估

    - Precision@K: 前 K 个结果中相关文档比例
    - Recall@K: 召回率
    - MRR: 平均倒数排名
    - NDCG@K: 归一化折损累计增益
    - F1@K: 调和平均
    - Reranker 效果对比（可选）
    - 支持按领域筛选和指定问题索引
    """
    try:
        # 如果选择了指定问题，筛选问题集
        if req.selected_indices is not None and len(req.selected_indices) > 0:
            from src.services.evaluate_service import OPS_EVAL_QUESTIONS, DOC_EVAL_QUESTIONS, EVAL_QUESTIONS

            if req.domain == "ops":
                all_questions = OPS_EVAL_QUESTIONS
            elif req.domain == "document":
                all_questions = DOC_EVAL_QUESTIONS
            else:
                all_questions = EVAL_QUESTIONS

            selected_questions = []
            for idx in req.selected_indices:
                if 0 <= idx < len(all_questions):
                    selected_questions.append(all_questions[idx])

            if not selected_questions:
                return {"status": "error", "message": "未选择有效问题"}

            # 手动执行评估
            results = []
            for eq in selected_questions:
                result = evaluate_single(
                    eq["question"],
                    eq["ground_truth"],
                    threshold=req.threshold,
                    k=req.k,
                    use_reranker=req.use_reranker,
                )
                result["domain"] = eq.get("domain", "unknown")
                results.append(result)

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
                "threshold": req.threshold,
                "k": req.k,
                "domain": req.domain,
                "use_reranker": req.use_reranker,
            }

            return {"status": "ok", "results": results, "summary": summary}

        result = run_evaluation(
            threshold=req.threshold,
            domain=req.domain,
            use_reranker=req.use_reranker,
            k=req.k,
        )
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"评估失败: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/evaluate/custom")
async def evaluate_custom(
    req: CustomEvaluateRequest,
    _user: str = Depends(get_current_user_dep),
):
    """评估自定义问题

    用户输入问题和标准答案，系统执行检索评估
    """
    try:
        result = evaluate_single(
            req.question,
            req.ground_truth,
            threshold=req.threshold,
            k=req.k,
            use_reranker=False,
        )
        result["domain"] = "custom"
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"自定义评估失败: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/evaluate/questions")
async def get_eval_questions(
    domain: str = "all",
    _user: str = Depends(get_current_user_dep),
):
    """获取预定义评估问题列表

    Args:
        domain: all=全部, ops=运维, document=通用文档
    """
    if domain == "ops":
        questions = OPS_EVAL_QUESTIONS
    elif domain == "document":
        questions = DOC_EVAL_QUESTIONS
    else:
        questions = EVAL_QUESTIONS

    return {
        "total": len(questions),
        "questions": [
            {
                "index": i,
                "question": eq["question"],
                "ground_truth": eq["ground_truth"][:100] + "...",
                "domain": eq.get("domain", "unknown"),
            }
            for i, eq in enumerate(questions)
        ],
    }
