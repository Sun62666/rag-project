from typing import List,TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.retriever import OpsRetriever


class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str

def build_graph(retriever: OpsRetriever):
    cfg = Config()
    llm = ChatOpenAI(
        model = cfg.LLM_MODEL,
        base_url = cfg.BASE_URL,
        api_key = cfg.DASHSCOPE_API_KEY,
        temperature=0.1,
        streaming=True
    )
    print("构建提示词LLM中。。。。")
    import os
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"prompts","ops_system.md")
    with open(prompt_path,"r",encoding="utf-8") as f:
        sys_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system',sys_prompt),
            ('human','上下文:\n{context}\n\n问题: {query}')
        ]
    )

    chain = prompt | llm

    def retrieve(state: AgentState):
        print(f"\n...开始检索中")
        ctx = retriever.retriever_and_rerank(state["query"],top_k=3)
        return {"context": ctx}

    def generate(state:AgentState):
        print("\n...开始调用大模型中")
        res = chain.invoke({"context": "\n".join(state["context"]),"query":state["query"]})
        print(f"\nres: {res}")
        return {"answer": res.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve",retrieve)
    workflow.add_node("generate",generate)
    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve","generate")
    workflow.add_edge("generate",END)
    return workflow.compile()