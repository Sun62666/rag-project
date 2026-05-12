import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.logging import setup_logging
from src.core.config import get_settings
from src.api.deps import init_components
from src.api.auth import router as auth_router
from src.api.session import router as session_router
from src.api.chat import router as chat_router

setup_logging()

app = FastAPI(title="SmartOps Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(session_router)
app.include_router(chat_router)


@app.on_event("startup")
async def startup():
    init_components()
