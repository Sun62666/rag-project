import logging
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from pydantic import BaseModel
from src.api.deps import get_current_user_dep
from src.services.auth_service import register_user, login_user, logout_user, get_user_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["认证"])


class UserAuth(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(user: UserAuth):
    try:
        return register_user(user.username, user.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(user: UserAuth):
    try:
        return login_user(user.username, user.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    return logout_user(authorization)


@router.get("/me")
async def me(authorization: Optional[str] = Header(None)):
    try:
        return get_user_info(authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
