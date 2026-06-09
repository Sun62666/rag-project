"""检索效果评估 API"""
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.api.deps import get_current_user_dep
from src.services.evaluate_service import run_evaluation, EVAL_QUESTIONS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["评估"])


class EvaluateRequest(BaseModel):
    threshold: float = 0.7


@router.post("/evaluate")
async def evaluate(
    req: EvaluateRequest = EvaluateRequest(),
    _user: str = Depends(get_current_user_dep),
):
    """运行检索效果评估（Precision@10）"""
    try:
        result = run_evaluation(threshold=req.threshold)
        return {"status": "ok", **result}
    except Exception as e:
        logger.error(f"评估失败: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/evaluate/questions")
async def get_eval_questions(_user: str = Depends(get_current_user_dep)):
    """获取预定义评估问题列表"""
    return {
        "questions": [
            {"question": eq["question"], "ground_truth": eq["ground_truth"]}
            for eq in EVAL_QUESTIONS
        ]
    }
