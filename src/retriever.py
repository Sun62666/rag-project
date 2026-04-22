import hashlib
import re
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
        print("\n初始化配置成功。。。。")
        if pdf_path:
            self._split_docs(pdf_path)
            print("划分文档成功。。。。")
            self._inject_doc_ids()
            print("注入doc_id成功。。。。")
        else:
            print("pdf_path为None，划分文档失败。。。。")
        self._init_retrievers()
        print("检索其初始化成功。。。。")
        self.reranker = CrossEncoder(self.cfg.RERANK_MODEL)
        print("重排序成功。。。。")
        self.initialized = True
    def _split_docs(self,path: str):
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=60,separators=["\n案例 ","\n案例","案例","\n\n","\n","。"," ",""])
        self.splits = splitter.split_documents(docs)
        # print(f"原来未合并文档： {self.splits[:2]}")
        self.splits = self.merge_chunks(self.splits)
        # print(f"已经合并文档： {self.splits[:2]}")
        print(f"\n长度为： {len(self.splits)}  划分数据成果： {self.splits[:2]} ")

    def _init_retrievers(self):
        try:
            connections.connect(
                alias="default",
                uri=self.cfg.MILVUS_URI  # 和你的配置完全一致
            )
        except Exception as e:
            print("⚠ 连接向量库失败！")
        collection_exists = False

        try:
            if utility.has_collection(self.cfg.COLLECTION_NAME):
                print("---检测到现有向量库--")
                collection = Collection(self.cfg.COLLECTION_NAME)
                if collection.num_entities > 0:
                    collection_exists = True
                    print(f"√ 检测到现有向量库（{collection.num_entities} 条数据），跳过重建")
        except Exception as e:
            print(f"⚠ 检查集合失败：{e}")

        emb = DashScopeEmbeddings(model = self.cfg.EMBED_MODEL,dashscope_api_key=self.cfg.DASHSCOPE_API_KEY)
        if collection_exists:
            self.vs = Milvus(
                embedding_function=emb,
                collection_name=self.cfg.COLLECTION_NAME,
                connection_args={"uri":self.cfg.MILVUS_URI}
            )
        else:
            print("🆕 创建新的 Milvus 集合并导入数据...")
            self.vs = Milvus.from_documents(self.splits,
                                            emb,
                                            collection_name=self.cfg.COLLECTION_NAME,
                                            connection_args={"uri": self.cfg.MILVUS_URI}
                                            )
        self.bm25 = BM25Retriever.from_documents(self.splits)
        self.bm25.k = 10
        self.vec_retr = self.vs.as_retriever(search_kwargs={"k":10})
        # print(f"\nbm25关键词检索： {self.bm25}")
        # print(f"\nb向量库vec_retr检索： {self.vec_retr}")
        self.ensemble = EnsembleRetriever(retrievers=[self.bm25,self.vec_retr],weights=[0.4,0.6])

    def retriever_and_rerank(self, query: str, top_k: int = 3) -> List[str]:
        docs = self.get_ensemble_rerank_docs(query,top_k)
        result = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source","位置文档")
            result.append(f"{source} {content}")
        # print(f"\n转换为字符串格式的结果: {result}")
        return result
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
            print("\n❌ 无检索结果")
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
        print(f"\n==================== 重排序后最终结果 ====================")
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        # print(f"\npairs: {pairs[:2]}")
        print(f"\nrerank_scores: {scores[:top_k]}")
        # print(f"\nranked_docs: {ranked[:2]}")

        result = [doc for doc, score in ranked[:top_k]]
        # print(f"\nresult_docs: {[d.page_content[:50] for d in result]}")
        # print(f"len(result): {len(result)}")
        return result

