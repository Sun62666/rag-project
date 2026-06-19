import hashlib
import logging
import re
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus as MilvusVS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from typing import List, Optional
from src.core.config import get_settings
from src.core.milvus_compat import ensure_milvus_connection, get_collection_count

logger = logging.getLogger(__name__)
class OpsRetriever:
    _instance = None
    def __new__(cls, pdf_path: str = ""):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, pdf_path: str = ""):

        if self.initialized:
            return

        self.cfg = get_settings()
        logger.info("初始化配置成功。。。。")

        self.splits = []
        self._milvus_client = None
        if pdf_path:
            self._split_docs(pdf_path)
            logger.info("划分文档成功。。。。")
            self._inject_doc_ids()
            logger.info("注入doc_id成功。。。。")
        else:
            logger.info("pdf_path为空，将从 Milvus 加载已有数据构建 BM25")

        self._init_retrievers()
        logger.info("检索器初始化成功。。。。")

        rerank_model_path = self.cfg.LORA_RERANK_MODEL if self.cfg.LORA_RERANK_MODEL else self.cfg.RERANK_MODEL
        self.reranker = CrossEncoder(rerank_model_path)
        logger.info(f"重排序模型创建成功: {rerank_model_path}")

        self.kg = None
        try:
            from src.graph.knowledge_graph import get_knowledge_graph
            self.kg = get_knowledge_graph()
            if self.kg.is_available:
                logger.info("✅ 知识图谱已连接，检索时将融合图谱上下文")
            else:
                logger.info("📝 知识图谱未连接，仅使用 RAG 检索")
        except Exception as e:
            logger.warning(f"知识图谱初始化失败，仅使用 RAG 检索: {e}")
            self.kg = None

        self.initialized = True

    def _split_docs(self,path: str):
        docs = PyPDFLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=60,separators=["\n案例 ","\n案例","案例","\n\n","\n","。"," ",""])

        raw_splits = splitter.split_documents(docs)
        logger.info(f"合并前长度为： {len(raw_splits)}")
        merged = self.merge_chunks(raw_splits)
        if merged:
            self.splits = merged
        else:
            self.splits = raw_splits
        logger.info(f"长度为： {len(self.splits)} ")

    def _init_retrievers(self):
        collection_exists = False
        collection_count = 0

        # 使用兼容性补丁连接 Milvus（pymilvus 2.6.x ConnectionManager 与 ORM 不兼容）
        # 保存连接为实例属性，后续 _load_all_from_milvus / rebuild_bm25 复用
        try:
            self._milvus_client = ensure_milvus_connection(self.cfg.MILVUS_URI)
            logger.info("✅ Milvus 连接成功！")
            if self._milvus_client.has_collection(self.cfg.COLLECTION_NAME):
                collection_count = get_collection_count(self._milvus_client, self.cfg.COLLECTION_NAME)
                if collection_count > 0:
                    collection_exists = True
                    logger.info(f"√ 检测到现有向量库（{collection_count} 条数据），跳过重建")
        except Exception as e:
            logger.error(f"❌ Milvus 连接失败，将仅使用 BM25 检索: {e}")
            self._milvus_client = None
            if self.splits:
                self.bm25 = BM25Retriever.from_documents(self.splits)
                self.bm25.k = 10
            else:
                self.bm25 = None
            self.vs = None
            self.vec_retr = None
            self.ensemble = None
            return

        self.vs = None
        self.vec_retr = None
        self.bm25 = None

        # 初始化 Milvus 向量检索（必须先初始化，后续可能从中加载数据）
        try:
            emb = OllamaEmbeddings(model=self.cfg.EMBED_MODEL,base_url=self.cfg.OLLAMA_URL)

            if collection_exists:
                self.vs = MilvusVS(
                    embedding_function=emb,
                    collection_name=self.cfg.COLLECTION_NAME,
                    connection_args={"uri": self.cfg.MILVUS_URI},
                    auto_id=True,
                    enable_dynamic_field=True
                )
            else:
                if not self.splits:
                    logger.warning("文档切片为空且 Milvus 无数据，无法初始化检索器")
                    self.ensemble = None
                    return
                logger.info("创建新的 Milvus 集合并导入数据...")
                self.vs = MilvusVS.from_documents(
                    self.splits,
                    emb,
                    collection_name=self.cfg.COLLECTION_NAME,
                    connection_args={"uri": self.cfg.MILVUS_URI},
                    enable_dynamic_field=True
                )
                logger.info("集合创建完成")

            self.vec_retr = self.vs.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            logger.error(f"⚠ Milvus 初始化失败，仅使用 BM25 检索: {e}")
            self.vs = None
            self.vec_retr = None

        # 初始化 BM25：始终从 Milvus 全量加载，确保与向量检索数据源一致
        # 即使有本地 PDF splits，Milvus 可能包含更多上传数据，BM25 必须覆盖全量
        if self.vs is not None:
            logger.info("📝 从 Milvus 全量加载构建 BM25（确保与向量检索数据源一致）...")
            milvus_splits = self._load_all_from_milvus()
            if milvus_splits:
                self.splits = milvus_splits
                self.bm25 = BM25Retriever.from_documents(milvus_splits)
                self.bm25.k = 10
                logger.info(f"✅ BM25 索引构建完成（{len(milvus_splits)} 个切片来自 Milvus 全量）")
            elif self.splits:
                # Milvus 查询失败时降级使用本地 splits
                self.bm25 = BM25Retriever.from_documents(self.splits)
                self.bm25.k = 10
                logger.warning(f"⚠ Milvus 加载失败，降级使用本地文档（{len(self.splits)} 个切片，可能不完整）")
            else:
                logger.warning("⚠ Milvus 中无数据且无本地文档，BM25 不可用")
                self.bm25 = None
        elif self.splits:
            # Milvus 不可用时的降级方案
            self.bm25 = BM25Retriever.from_documents(self.splits)
            self.bm25.k = 10
            logger.warning(f"⚠ Milvus 不可用，BM25 仅基于本地文档（{len(self.splits)} 个切片，可能不完整）")
        else:
            logger.warning("⚠ 文档切片为空，BM25 检索不可用")

        # 构建混合检索器
        if self.bm25 and self.vec_retr:
            self.ensemble = EnsembleRetriever(retrievers=[self.bm25, self.vec_retr], weights=[0.4, 0.6])
            logger.info("✅ 混合检索器初始化成功（BM25 + Milvus 向量）")
        elif self.vec_retr:
            self.ensemble = self.vec_retr
            logger.info("✅ 仅向量检索器可用")
        elif self.bm25:
            self.ensemble = None  # 降级到 BM25
            logger.info("✅ 仅 BM25 检索可用")
        else:
            self.ensemble = None
            logger.warning("⚠ 无可用检索器")

    def _load_all_from_milvus(self) -> List[Document]:
        """从 Milvus 全量加载数据构建 BM25 索引，确保 BM25 与向量检索数据源一致。
        复用初始化时保存的 MilvusClient 连接，避免频繁创建/销毁连接。"""
        if self._milvus_client is None:
            logger.error("[Milvus加载] MilvusClient 未初始化，无法加载数据")
            return []

        try:
            if not self._milvus_client.has_collection(self.cfg.COLLECTION_NAME):
                return []

            # 查询全量数据（只取 text 和 metadata 字段）
            results = self._milvus_client.query(
                collection_name=self.cfg.COLLECTION_NAME,
                filter="",
                output_fields=["text", "source", "doc_id"],
                limit=10000,
            )

            docs = []
            for row in results:
                text = row.get("text", "")
                if not text:
                    continue
                metadata = {}
                if row.get("source"):
                    metadata["source"] = row["source"]
                if row.get("doc_id"):
                    metadata["doc_id"] = row["doc_id"]
                docs.append(Document(page_content=text, metadata=metadata))

            logger.info(f"[Milvus加载] 从 {self.cfg.COLLECTION_NAME} 加载 {len(docs)} 条数据")
            return docs
        except Exception as e:
            logger.error(f"[Milvus加载] 从 Milvus 加载数据失败: {e}")
            return []

    def rebuild_bm25(self):
        """重建 BM25 索引（文档上传后调用），从 Milvus 全量加载确保数据源一致"""
        logger.info("[OpsRetriever] 开始重建 BM25 索引...")
        milvus_splits = self._load_all_from_milvus()
        if milvus_splits:
            self.splits = milvus_splits
            self.bm25 = BM25Retriever.from_documents(milvus_splits)
            self.bm25.k = 10
            # 重建混合检索器
            if self.vec_retr:
                self.ensemble = EnsembleRetriever(retrievers=[self.bm25, self.vec_retr], weights=[0.4, 0.6])
            logger.info(f"[OpsRetriever] ✅ BM25 索引重建完成（{len(milvus_splits)} 个切片）")
        else:
            logger.warning("[OpsRetriever] ⚠ Milvus 中无数据，BM25 重建失败")

    def retriever_and_rerank(self, query: str, top_k: int = 3) -> List[str]:

        docs = self.get_ensemble_rerank_docs(query,top_k)
        results = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source","位置文档")
            results.append(f"{source} {content}")
        return results

    def retriever_and_rerank_with_scores(self, query: str, top_k: int = 3) -> List[tuple]:
        if self.ensemble is not None:
            try:
                docs = self.ensemble.invoke(query)
                docs = self._deduplicate(docs)
                if docs:
                    kg_context = self._get_kg_context(query)
                    if kg_context:
                        kg_doc = Document(
                            page_content=kg_context,
                            metadata={"source": "知识图谱", "doc_id": "kg_context"}
                        )
                        docs = [kg_doc] + docs
                    pairs = [(query, d.page_content) for d in docs]
                    scores = self.reranker.predict(pairs)
                    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                    logger.info(f"\nrerank_scores: {[f'{score:.4f}' for _, score in ranked[:top_k]]}")
                    return [(doc, score) for doc, score in ranked[:top_k]]
            except Exception as e:
                logger.error(f"⚠ 混合检索失败，降级到 BM25: {e}")
        
        logger.info("📝 使用纯 BM25 检索...")
        docs = self.get_bm25_docs(query, top_k)
        return [(doc, 0.8) for doc in docs]

    def _get_kg_context(self, query: str) -> str:
        if not self.kg or not self.kg.is_available:
            return ""
        try:
            return self.kg.format_graph_context(query, depth=2)
        except Exception as e:
            logger.warning(f"知识图谱查询失败: {e}")
            return ""

    def merge_chunks(self,chunks):
        merge = []
        current = None
        for chunk in chunks:
            if re.match(r'^案例\s*\d+[: ：]',chunk.page_content.strip()):
                if current:
                    merge.append(current)
                current = chunk
            else:
                if current:
                    current.page_content += "\n" + chunk.page_content
        if current:
            merge.append(current)
        return merge

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """与线上完全一致的去重逻辑，评估时复用保证公平"""
        seen, unique = set(), []
        # print(f"\ndocs: {docs}")
        for doc in docs:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)
        if not unique:
            logger.info("\n❌ 无检索结果")
            return []
        # print(f"\nseen: {seen}")
        # print(f"\nunique: {unique[:2]}")
        return unique

    def _inject_doc_ids(self):
        """初始化时自动为每个 chunk 注入稳定 ID（仅执行一次）"""
        # print(f"\n更新前没加doc_id的len(splits): {len(self.splits)} splits: {self.splits[:2]}")
        for i, doc in enumerate(self.splits):
            # 若已存在则跳过，否则基于 source+索引 生成稳定 hash
            # doc.metadata.setdefault("doc_id", hashlib.md5(
            #     f"{doc.metadata.get('source', '')}_{i}".encode()
            # ).hexdigest()[:12]) # md5哈希算法这里生成任意长度的哈希值（字节），再用hexdigest转为字符串取前12个

            # 增强（如果改变chunk顺序会改变排序结果导致相同的内容有不同的哈希）
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
            doc.metadata.setdefault("doc_id",doc_id)
        # print(f"\n更后加doc_id的len(splits): {len(self.splits)} splits: {self.splits[:2]}")


    def get_bm25_docs(self, query: str, top_k: int = 3) -> List[Document]:
        if self.bm25 is None:
            return []
        docs = self.bm25.invoke(query)
        return self._deduplicate(docs)[:top_k]

    def get_vector_docs(self, query: str, top_k: int = 3) -> List[Document]:
        if self.vec_retr is None:
            return []
        docs = self.vec_retr.invoke(query)
        return self._deduplicate(docs)[:top_k]

    def get_ensemble_rerank_docs(self, query: str, top_k: int = 3) -> List[Document]:
        if self.ensemble is None:
            return self.get_bm25_docs(query, top_k)

        docs = self.ensemble.invoke(query)
        docs = self._deduplicate(docs)
        if not docs: return []

        # 复用线上重排序逻辑
        logger.info(f"\n==================== 重排序后最终结果 ====================")
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        # print(f"\npairs: {pairs[:2]}")
        logger.info(f"\nrerank_scores: {[score for _,score in ranked[:top_k]]}")
        # print(f"\nranked_docs: {ranked[:2]}")

        result = [doc for doc, score in ranked[:top_k]]
        # print(f"\nresult_docs: {[d.page_content[:50] for d in result]}")
        # print(f"len(result): {len(result)}")
        return result

