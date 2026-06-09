"""通用文档 RAG 问答工具

融合自 Agent 项目，作为 SmartOps Agent 的附属工具。
支持对任意上传文档（如物业管理条例）进行混合检索+重排序+LLM答案生成。
"""
import logging
from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_milvus import Milvus as MilvusVS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# 全局实例
_document_qa_instance: Optional["DocumentQAService"] = None


class DocumentQAService:
    """通用文档 RAG 服务：支持独立的 collection 进行文档检索和问答"""

    def __init__(self, collection_name: str = "property_regulations"):
        self.cfg = get_settings()
        self.collection_name = collection_name
        self._emb = DashScopeEmbeddings(
            model=self.cfg.EMBED_MODEL,
            dashscope_api_key=self.cfg.DASHSCOPE_API_KEY
        )
        self._vs = None
        self._bm25 = None
        self._ensemble = None
        self._reranker = None
        self._splits: List[Document] = []

        self._init_reranker()
        self._init_milvus()

    def _init_reranker(self):
        try:
            rerank_model_path = self.cfg.LORA_RERANK_MODEL if self.cfg.LORA_RERANK_MODEL else self.cfg.RERANK_MODEL
            self._reranker = CrossEncoder(rerank_model_path)
            logger.info(f"[DocumentQA] 重排序模型加载成功: {rerank_model_path}")
        except Exception as e:
            logger.warning(f"[DocumentQA] 重排序模型加载失败: {e}")
            self._reranker = None

    def _init_milvus(self):
        try:
            from src.core.milvus_compat import ensure_milvus_connection, get_collection_count
            uri = self.cfg.MILVUS_URI
            if not uri.startswith("http://") and not uri.startswith("https://"):
                uri = f"http://{uri}"

            client = ensure_milvus_connection(uri)
            if client.has_collection(self.collection_name):
                count = get_collection_count(client, self.collection_name)
                if count > 0:
                    self._vs = MilvusVS(
                        embedding_function=self._emb,
                        collection_name=self.collection_name,
                        connection_args={"uri": uri},
                        auto_id=True,
                        enable_dynamic_field=True
                    )
                    logger.info(f"[DocumentQA] 加载已有集合 {self.collection_name}，共 {count} 条")
                    self._init_ensemble()
                    return
            logger.info(f"[DocumentQA] 集合 {self.collection_name} 不存在或为空，等待文档上传")
        except Exception as e:
            logger.error(f"[DocumentQA] Milvus 初始化失败: {e}")

    def _init_ensemble(self):
        if self._vs is None:
            return
        try:
            vec_retr = self._vs.as_retriever(search_kwargs={"k": 10})
            if self._bm25 is not None:
                self._ensemble = EnsembleRetriever(
                    retrievers=[self._bm25, vec_retr],
                    weights=[0.4, 0.6]
                )
            else:
                self._ensemble = vec_retr
            logger.info(f"[DocumentQA] 混合检索器初始化成功")
        except Exception as e:
            logger.error(f"[DocumentQA] 混合检索器初始化失败: {e}")
            self._ensemble = None

    def rebuild_bm25(self, splits: List[Document]):
        """重建 BM25 索引（文档上传后调用）"""
        if splits:
            self._splits = splits
            self._bm25 = BM25Retriever.from_documents(splits)
            self._bm25.k = 10
            self._init_ensemble()
            logger.info(f"[DocumentQA] BM25 索引重建完成，{len(splits)} 个切片")

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """混合检索 + 重排序，返回 (Document, score) 列表"""
        if self._ensemble is None:
            logger.warning("[DocumentQA] 检索器未初始化，无法检索")
            return []

        try:
            docs = self._ensemble.invoke(query)
            docs = self._deduplicate(docs)

            if not docs:
                return []

            if self._reranker:
                pairs = [(query, d.page_content) for d in docs]
                scores = self._reranker.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return [(doc, score) for doc, score in ranked[:top_k]]
            else:
                return [(doc, 0.8) for doc in docs[:top_k]]
        except Exception as e:
            logger.error(f"[DocumentQA] 检索失败: {e}")
            return []

    def search_raw(self, query: str, top_k: int = 10) -> List[dict]:
        """检索并返回字典格式（用于评估）"""
        results = self.search(query, top_k)
        return [
            {
                "chunk_id": doc.metadata.get("doc_id", str(i)),
                "content": doc.page_content,
                "score": float(score),
            }
            for i, (doc, score) in enumerate(results)
        ]

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for doc in docs:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)
        return unique


def get_document_qa_service() -> Optional[DocumentQAService]:
    global _document_qa_instance
    if _document_qa_instance is None:
        try:
            _document_qa_instance = DocumentQAService()
        except Exception as e:
            logger.error(f"[DocumentQA] 服务初始化失败: {e}")
    return _document_qa_instance


def rebuild_document_qa_bm25(splits):
    """文档上传后重建 BM25"""
    svc = get_document_qa_service()
    if svc:
        svc.rebuild_bm25(splits)


@tool
def document_qa(query: str) -> str:
    """从通用文档知识库（如物业管理条例等上传文档）检索相关内容。

    【调用时机】
    - 用户询问关于物业管理、法规条例等非运维领域的问题
    - 用户明确提到"文档"、"条例"、"法规"等关键词
    - knowledge_retriever 未检索到相关运维知识时，可尝试此工具

    【重要】此工具用于检索用户上传的通用文档，与运维知识库(knowledge_retriever)互补

    参数：query-检索关键词或问题"""
    svc = get_document_qa_service()
    if svc is None:
        return "文档问答服务未初始化"

    results = svc.search(query, top_k=5)
    if not results:
        return "【文档检索结果】未检索到与问题相关的文档内容，文档库可能未上传或未覆盖该问题。"

    output_parts = ["【通用文档检索结果】"]
    for i, (doc, score) in enumerate(results):
        relevance = "高" if score > 0.5 else "中" if score > 0.2 else "低"
        source = doc.metadata.get("source", "上传文档")
        output_parts.append(f"[{i+1}] [相关性:{relevance}({score:.2f})] {source}\n{doc.page_content}")

    return "\n\n".join(output_parts)
