import logging
from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from src.core.config import get_settings
from src.api.deps import get_current_user_dep, get_agent, get_graph, get_stm
from src.services.chat_service import handle_cached_answer, ask_agent, ask_graph, handle_lora_fallback

logger = logging.getLogger(__name__)

router = APIRouter(tags=["对话"])


class Query(BaseModel):
    query: str
    session_id: str = "default"


@router.post("/ask")
async def ask(
    req: Query,
    bg_tasks: BackgroundTasks,
    username: str = Depends(get_current_user_dep),
):
    try:
        stm = get_stm()
        cached_response = await handle_cached_answer(req.query, req.session_id, username, stm, bg_tasks)
        if cached_response:
            return cached_response
    except Exception as e:
        logger.error(f"Redis缓存异常：{e}")

    cfg = get_settings()
    stm = get_stm()

    try:
        if cfg.USE_AGENT:
            return await ask_agent(req, username, bg_tasks, get_agent(), stm)
        else:
            return await ask_graph(req, username, bg_tasks, get_graph(), stm)
    except Exception as e:
        logger.error(f"主模型调用失败: {e}，尝试LoRA降级...")
        lora_response = await handle_lora_fallback(req.query, req.session_id, username, stm, bg_tasks)
        if lora_response:
            return lora_response
        raise


class ModeSwitch(BaseModel):
    use_agent: bool


@router.get("/mode")
async def get_mode():
    cfg = get_settings()
    return {"mode": "agent" if cfg.USE_AGENT else "graph", "use_agent": cfg.USE_AGENT}


@router.post("/mode")
async def switch_mode(body: ModeSwitch):
    cfg = get_settings()
    cfg.USE_AGENT = body.use_agent
    mode_name = "agent" if body.use_agent else "graph"
    logger.info(f"[Mode] 切换为 {mode_name} 模式")
    return {"mode": mode_name, "use_agent": body.use_agent}
