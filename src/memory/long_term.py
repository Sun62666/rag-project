import logging
import time
import uuid
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_milvus import Milvus
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility, Collection
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class LongTermMemory:
    """长期记忆管理器：复用 Milvus 向量库，保存和检索对话历史记忆，解决长会话遗忘问题"""

    COLLECTION_NAME = "ops_memory_store"
    _MILVUS_PLACEHOLDER = "milvus_uri"

    def __init__(self):
        self.cfg = get_settings()
        self._vs: Optional[Milvus] = None

        # 检测 Milvus 是否已配置
        if not self.cfg.MILVUS_URI or self.cfg.MILVUS_URI == self._MILVUS_PLACEHOLDER:
            logger.warning(f"[长期记忆] MILVUS_URI 未配置（当前值: {self.cfg.MILVUS_URI!r}），长期记忆不可用")
            return

        self._emb = DashScopeEmbeddings(
            model=self.cfg.EMBED_MODEL,
            dashscope_api_key=self.cfg.DASHSCOPE_API_KEY
        )
        self._init_collection()

    def _init_collection(self):
        max_retries = 3
        last_err = None
        for i in range(max_retries):
            try:
                connections.connect(alias="default", uri=self.cfg.MILVUS_URI)
                # 验证连接是否真正可用
                utility.has_collection(self.COLLECTION_NAME)
                logger.info(f"[长期记忆] Milvus 连接成功")
                break
            except Exception as e:
                last_err = e
                if i < max_retries - 1:
                    logger.warning(f"[长期记忆] Milvus 连接失败，3秒后重试: {e}")
                    time.sleep(3)
                else:
                    logger.error(f"[长期记忆] Milvus 连接失败，长期记忆不可用: {last_err}")
                    return

        try:
            if utility.has_collection(self.COLLECTION_NAME):
                col = Collection(self.COLLECTION_NAME)
                if col.num_entities > 0:
                    self._vs = Milvus(
                        embedding_function=self._emb,
                        collection_name=self.COLLECTION_NAME,
                        connection_args={"uri": self.cfg.MILVUS_URI},
                    )
                    logger.info(f"[长期记忆] 加载已有记忆集合，共 {col.num_entities} 条")
                    return
        except Exception as e:
            logger.warning(f"[长期记忆] 检查集合失败: {e}")

        logger.info(f"[长期记忆] 记忆集合不存在或为空，将在首次写入时创建")

    def save_memory(self, user_id: str, session_id: str, user_msg: str, assistant_msg: str):
        qa_text = f"问: {user_msg}\n答: {assistant_msg}"
        metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "qa_memory"
        }
        doc = Document(page_content=qa_text, metadata=metadata)

        try:
            if self._vs is None:
                self._vs = Milvus.from_documents(
                    [doc],
                    self._emb,
                    collection_name=self.COLLECTION_NAME,
                    connection_args={"uri": self.cfg.MILVUS_URI},
                    auto_id=True,
                )
                logger.info(f"[长期记忆] 创建记忆集合并写入首条记忆")
            else:
                memory_id = str(uuid.uuid4())
                self._vs.add_documents([doc], ids=[memory_id])
                logger.info(f"[长期记忆] 写入记忆: session={session_id}")
        except Exception as e:
            logger.error(f"[长期记忆] 保存记忆失败: {e}")

    def search_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        if self._vs is None:
            logger.warning(f"[长期记忆] 向量库未初始化，无法检索")
            return []
        try:
            retriever = self._vs.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            logger.info(f"[长期记忆] 检索到 {len(results)} 条相关记忆")
            return results
        except Exception as e:
            logger.error(f"[长期记忆] 检索记忆失败: {e}")
            return []

    def format_memory_context(self, query: str, top_k: int = 3) -> str:
        memories = self.search_memory(query, top_k)
        if not memories:
            return ""
        parts = ["【历史记忆检索结果】"]
        for i, m in enumerate(memories, 1):
            meta = m.get("metadata", {})
            ts = meta.get("timestamp", "未知时间")
            uid = meta.get("user_id", "未知用户")
            parts.append(f"[记忆{i}] (时间:{ts}, 用户:{uid})\n{m['content']}")
        logger.info(f"获取当前长期记忆： {'\n\n'.join(parts)}")
        return "\n\n".join(parts)
