"""知识图谱 API"""
import logging
import os
import tempfile
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form

from src.api.deps import get_current_user_dep
from src.api.vis_utils import build_vis_data, build_full_vis_data

logger = logging.getLogger(__name__)

router = APIRouter(tags=["运维工具-知识图谱"])


def _load_file(file_path: str, ext: str):
    """根据文件扩展名解析文档内容"""
    from langchain_core.documents import Document
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
    return []


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


@router.post("/knowledge/graph/upload")
async def knowledge_graph_upload_file(
    file: UploadFile = File(..., description="上传文档文件（PDF/TXT/MD/DOCX）"),
    method: str = Form(default="hybrid", description="抽取方式: rule/spacy/llm/hybrid"),
    _user: str = Depends(get_current_user_dep),
):
    """上传文档文件，自动解析文本后抽取知识图谱三元组"""
    if not file.filename:
        return {"status": "error", "message": "文件名不能为空"}

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md", ".docx"]:
        return {"status": "error", "message": f"不支持的文件格式: {ext}，仅支持 PDF/TXT/MD/DOCX"}

    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if not kg.is_available:
            return {"status": "error", "message": "知识图谱未连接（Neo4j 不可用）"}

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 解析文档
            docs = _load_file(tmp_path, ext)
            if not docs:
                return {"status": "error", "message": "文件内容为空或无法解析"}

            # 合并所有文档文本
            full_text = "\n\n".join([d.page_content for d in docs if d.page_content])
            if not full_text.strip():
                return {"status": "error", "message": "文档内容为空"}

            # 分段抽取（每段不超过 2000 字符，避免 LLM 超时）
            total_triples = 0
            chunk_size = 2000
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
            for i, chunk in enumerate(chunks):
                count = kg.extract_and_ingest(chunk, source=file.filename, method=method)
                total_triples += count
                logger.info(f"[知识图谱上传] 第 {i+1}/{len(chunks)} 段抽取 {count} 个三元组")

            return {
                "status": "ok",
                "triples_extracted": total_triples,
                "chunks_processed": len(chunks),
                "message": f"从 {file.filename} 中抽取并写入 {total_triples} 个三元组",
            }
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"[知识图谱上传] 处理失败: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/knowledge/graph/relations")
async def knowledge_graph_relations(
    entity: str = Query(default="", description="中心实体名称，为空则返回全部"),
    depth: int = Query(default=2, ge=1, le=4, description="查询深度"),
    _user: str = Depends(get_current_user_dep),
):
    """查询知识图谱关系，返回文字列表格式"""
    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        if not kg.is_available:
            return {"available": False, "relations": []}

        all_records = []
        seen = set()

        if entity:
            records = kg.query_related(entity, depth)
            for r in records:
                key = f"{r['source']}-{r['relation']}-{r['target']}"
                if key not in seen:
                    seen.add(key)
                    all_records.append(r)
        else:
            # 返回全图关系
            with kg._driver.session() as session:
                result = session.run(
                    "MATCH (n)-[r]->(m) "
                    "RETURN n.name AS source, labels(n)[0] AS source_type, "
                    "type(r) AS relation, m.name AS target, labels(m)[0] AS target_type "
                    "LIMIT 500"
                )
                for record in result:
                    r = {
                        "source": record["source"],
                        "source_type": record["source_type"] or "",
                        "relation": record["relation"],
                        "target": record["target"],
                        "target_type": record["target_type"] or "",
                    }
                    key = f"{r['source']}-{r['relation']}-{r['target']}"
                    if key not in seen:
                        seen.add(key)
                        all_records.append(r)

        return {"available": True, "relations": all_records}
    except Exception as e:
        return {"available": False, "relations": [], "error": str(e)}


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
