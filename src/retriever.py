import hashlib
import logging
import re
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from pymilvus import connections,Collection,utility
from typing import List
from src.config import Config
logger = logging.getLogger(__name__)
class OpsRetriever:
    _instance = None
    def __new__(cls, pdf_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self,pdf_path: str):

        if self.initialized:
            return

        self.cfg = Config()
        logger.info("初始化配置成功。。。。")

        if pdf_path:
            self._split_docs(pdf_path)
            logger.info("划分文档成功。。。。")

            self._inject_doc_ids()
            logger.info("注入doc_id成功。。。。")

        else:
            logger.info("pdf_path为None，划分文档失败。。。。")

        self._init_retrievers()
        logger.info("检索其初始化成功。。。。")

        self.reranker = CrossEncoder(self.cfg.RERANK_MODEL)
        logger.info("重排序模型创建成功。。。。")

        self.initialized = True

    def _split_docs(self,path: str):
        docs = PyPDFLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=60,separators=["\n案例 ","\n案例","案例","\n\n","\n","。"," ",""])

        self.splits = splitter.split_documents(docs)
        # print(f"原来未合并文档： {self.splits[:2]}")
        self.splits = self.merge_chunks(self.splits)
        # print(f"已经合并文档： {self.splits[:2]}")
        logger.info(f"长度为： {len(self.splits)}  划分数据成果： {self.splits[:2]} ")

    def _init_retrievers(self):
        max_retries = 10
        for i in range(max_retries):
            try:
                logger.info(f"尝试连接 Milvus ({i+1}/{max_retries})...")
                connections.connect(
                    alias="default",
                    uri=self.cfg.MILVUS_URI  # 和你的配置完全一致
                )
                logger.info("✅ Milvus 连接成功！")
                break
            except Exception as e:
                if i < max_retries - 1:
                    logger.error(f"⚠ Milvus 连接失败，3秒后重试: {e}")
                    time.sleep(3)
                else:
                    logger.error("⚠ Milvus 连接失败，将仅使用 BM25 检索！")
        self.vs = None
        self.vec_retr = None
        collection_exists = False

        try:
            if utility.has_collection(self.cfg.COLLECTION_NAME):

                logger.info("---检测到现有向量库--")
                collection = Collection(self.cfg.COLLECTION_NAME)

                if collection.num_entities > 0:
                    collection_exists = True
                    logger.info(f"√ 检测到现有向量库（{collection.num_entities} 条数据），跳过重建")

        except Exception as e:
            logger.error(f"⚠ 检查集合失败：{e}")

        self.bm25 = BM25Retriever.from_documents(self.splits)
        self.bm25.k = 10

        try:

            emb = DashScopeEmbeddings(model = self.cfg.EMBED_MODEL,dashscope_api_key=self.cfg.DASHSCOPE_API_KEY)

            if collection_exists:
                self.vs = Milvus(
                    embedding_function=emb,
                    collection_name=self.cfg.COLLECTION_NAME,
                    connection_args={"uri":self.cfg.MILVUS_URI}
                )

            else:
                logger.info("🆕 创建新的 Milvus 集合并导入数据...")
                self.vs = Milvus.from_documents(self.splits,
                                                emb,
                                                collection_name=self.cfg.COLLECTION_NAME,
                                                connection_args={"uri": self.cfg.MILVUS_URI}
                                                )

            self.vec_retr = self.vs.as_retriever(search_kwargs={"k":10})
            self.ensemble = EnsembleRetriever(retrievers=[self.bm25,self.vec_retr],weights=[0.4,0.6])
            logger.info("✅ 混合检索器初始化成功（BM25 + Milvus 向量）")

        except Exception as e:
            logger.error(f"⚠ Milvus 初始化失败，仅使用 BM25 检索: {e}")
            self.ensemble = None

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
                    pairs = [(query, d.page_content) for d in docs]
                    scores = self.reranker.predict(pairs)
                    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                    logger.info(f"\nrerank_scores: {[f'{score:.4f}' for _, score in ranked[:top_k]]}")
                    return [(doc, score) for doc, score in ranked[:top_k]]
            except Exception as e:
                logger.error(f"⚠ 混合检索失败，降级到 BM25: {e}")
        
        logger.info("📝 使用纯 BM25 检索...")
        docs = self.get_bm25_docs(query, top_k)
        # 给 BM25 结果一个默认的高分数（0.8）
        return [(doc, 0.8) for doc in docs]

    def merge_chunks(self,chunks):
        merge = []
        current = None
        for chunk in chunks:
            if re.match(r'^案例\s+\d+[: ：]',chunk.page_content.strip()):
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
        docs = self.bm25.invoke(query)
        return self._deduplicate(docs)[:top_k]

    def get_vector_docs(self, query: str, top_k: int = 3) -> List[Document]:
        docs = self.vec_retr.invoke(query)
        return self._deduplicate(docs)[:top_k]

    def get_ensemble_rerank_docs(self, query: str, top_k: int = 3) -> List[Document]:

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

