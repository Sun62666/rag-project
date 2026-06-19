"""文档问答智能体：通用文档检索专家

负责处理用户上传的通用文档（如物业管理条例、法规等）相关问题。
使用独立的 Milvus 集合进行文档检索。
"""

import logging
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from src.agents.base import BaseAgent, create_llm

logger = logging.getLogger(__name__)

DOCUMENT_AGENT_PROMPT = """你是通用文档问答专家，负责回答用户上传的文档相关问题。

## 你的职责
1. 从通用文档知识库（如物业管理条例、法规）中检索相关内容
2. 基于文档内容给出准确回答
3. 引用文档来源

## 回答格式
【相关文档】
【文档内容摘要】
【详细解答】

## 注意
- 严格基于文档内容回答，不要编造
- 如果文档中未涉及，明确告知
"""


class DocumentAgent(BaseAgent):
    """文档问答智能体"""

    name = "document_agent"
    description = "通用文档检索和问答专家"

    def __init__(self):
        super().__init__(llm=create_llm())

        self.chain = ChatPromptTemplate.from_messages([
            ("system", DOCUMENT_AGENT_PROMPT),
            ("human", "用户问题: {query}\n\n文档检索结果:\n{context}"),
        ]) | self.llm

    def run(self, state: Dict) -> Dict:
        query = state["query"]
        context = "无文档检索结果"

        # 从通用文档 QA 服务检索
        try:
            from src.tools.document_qa import get_document_qa_service
            doc_qa = get_document_qa_service()
            if doc_qa and doc_qa._ensemble is not None:
                # 使用文档 QA 服务的检索+重排序
                docs = doc_qa._ensemble.invoke(query)
                if docs:
                    # 去重
                    seen = set()
                    unique_docs = []
                    for doc in docs:
                        if doc.page_content not in seen:
                            unique_docs.append(doc)
                            seen.add(doc.page_content)

                    # 重排序
                    if doc_qa._reranker and unique_docs:
                        pairs = [(query, d.page_content) for d in unique_docs[:10]]
                        scores = doc_qa._reranker.predict(pairs)
                        ranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
                        unique_docs = [d for d, _ in ranked[:3]]

                    parts = []
                    for doc in unique_docs:
                        source = doc.metadata.get("source", "文档")
                        parts.append(f"[来源: {source}]\n{doc.page_content}")
                    context = "\n\n".join(parts)
                    logger.info(f"[document_agent] 文档检索完成, {len(unique_docs)} 条结果")
                else:
                    context = "文档知识库中未检索到相关内容"
            else:
                context = "文档问答服务未初始化（property_regulations 集合为空）"
        except Exception as e:
            logger.warning(f"[document_agent] 文档检索失败: {e}")
            context = f"文档检索异常: {e}"

        response = self.chain.invoke({
            "query": query,
            "context": context,
        })

        result = response.content
        logger.info(f"[document_agent] 回答完成, 长度: {len(result)}")

        return {
            "document_result": result,
            "agent_messages": state.get("agent_messages", []) + [
                {"agent": "document_agent", "message": "文档检索完成"}
            ],
        }
