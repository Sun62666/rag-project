from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.logging import setup_logging
from src.api.deps import init_components, cleanup_components
from src.api.auth import router as auth_router
from src.api.session import router as session_router
from src.api.chat import router as chat_router
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PyMilvusDeprecationWarning.*")

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_components()
    yield
    cleanup_components()


app = FastAPI(title="SmartOps Agent API", lifespan=lifespan)

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
