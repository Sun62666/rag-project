import logging
from fastapi import APIRouter, Depends, Header
from typing import Optional
from pydantic import BaseModel
from src.api.deps import get_current_user_dep, get_stm
from src.services.session_service import (
    create_session, get_user_sessions, get_chat_history,
    clear_history, delete_session, rename_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["会话"])


class RenameSession(BaseModel):
    title: str


@router.post("/new_session")
async def new_session(username: str = Depends(get_current_user_dep)):
    return create_session(username)


@router.post("/clear_history")
async def clear_history_endpoint(session_id: str = "default"):
    stm = get_stm()
    return clear_history(session_id, stm)


@router.get("/sessions")
async def get_sessions(username: str = Depends(get_current_user_dep)):
    sessions = get_user_sessions(username)
    return {"status": "ok", "sessions": sessions}


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    stm = get_stm()
    return delete_session(session_id, stm)


@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    history = get_chat_history(session_id)
    return {"status": "ok", "history": history}


@router.put("/sessions/{session_id}/rename")
async def rename_session_endpoint(session_id: str, body: RenameSession):
    return rename_session(session_id, body.title)
