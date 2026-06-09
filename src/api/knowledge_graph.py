"""知识图谱 API"""
import logging
from fastapi import APIRouter, Depends, Query

from src.api.deps import get_current_user_dep
from src.api.vis_utils import build_vis_data, build_full_vis_data

logger = logging.getLogger(__name__)

router = APIRouter(tags=["运维工具-知识图谱"])


@router.get("/knowledge/graph/stats")
async def knowledge_graph_stats(_user: str = Depends(get_current_user_dep)):
    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        return kg.get_stats()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.post("/knowledge/graph/extract")
async def knowledge_graph_extract_from_doc(
    text: str = Query(..., description="需要抽取的运维文档文本"),
    method: str = Query(default="hybrid", description="抽取方式: rule/spacy/llm/hybrid"),
    _user: str = Depends(get_current_user_dep),
):
    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if not kg.is_available:
            return {"status": "error", "message": "知识图谱未连接"}
        count = kg.extract_and_ingest(text, source="api_manual", method=method)
        return {"status": "ok", "triples_extracted": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/knowledge/graph/vis")
async def knowledge_graph_visualization(
    entity: str = Query(default="", description="中心实体名称，为空则显示全图概览"),
    depth: int = Query(default=2, ge=1, le=4, description="查询深度"),
    _user: str = Depends(get_current_user_dep),
):
    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if not kg.is_available:
            return {"available": False, "nodes": [], "edges": []}

        if entity:
            return build_vis_data(kg, entity, depth)
        return build_full_vis_data(kg)
    except Exception as e:
        return {"available": False, "error": str(e), "nodes": [], "edges": []}
