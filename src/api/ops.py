"""运维工具路由聚合

将各子模块路由统一挂载到 /ops 前缀下。
子模块：
  - logs.py           日志查看
  - documents.py      文档上传与向量化
  - knowledge_graph.py 知识图谱
  - vis_utils.py      可视化辅助（内部模块，无路由）
"""
from fastapi import APIRouter

from src.api.logs import router as logs_router
from src.api.documents import router as documents_router
from src.api.knowledge_graph import router as kg_router

router = APIRouter(prefix="/ops", tags=["运维工具"])
router.include_router(logs_router)
router.include_router(documents_router)
router.include_router(kg_router)
