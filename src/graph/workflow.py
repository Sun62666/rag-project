from typing import Dict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from src.core.config import Config
from src.retriever import OpsRetriever
from src.tools import server_system_check, port_check, read_service_log, set_retriever
from src.graph.nodes import (
    make_classify_node,
    make_rewrite_query_node,
    make_retrieve_node,
    make_execute_tools_node,
    make_generate_node,
    reject,
)


class AgentState(Dict):
    query: str
    intent: str
    rewritten_query: str
    retrieved_context: str
    tool_results: Dict[str, str]
    answer: str
    chat_history: list


def route_by_intent(state: Dict) -> str:
    intent = state.get("intent", "fault")
    if intent == "reject":
        return "reject"
    elif intent == "followup":
        return "generate"
    elif intent == "fault":
        return "rewrite_query"
    elif intent == "system":
        return "execute_tools"
    elif intent == "mixed":
        return "rewrite_query"
    return "rewrite_query"


def route_after_rewrite(state: Dict) -> str:
    return "retrieve"


def route_after_retrieve(state: Dict) -> str:
    if state.get("intent") == "mixed":
        return "execute_tools"
    return "generate"


def build_graph(retriever: OpsRetriever):
    set_retriever(retriever)

    cfg = Config()

    llm = ChatOpenAI(
        model=cfg.LLM_MODEL,
        base_url=cfg.BASE_URL,
        api_key=cfg.DASHSCOPE_API_KEY,
        temperature=0,
        streaming=True,
    )

    fast_llm = ChatOpenAI(
        model=cfg.LLM_MODEL,
        base_url=cfg.BASE_URL,
        api_key=cfg.DASHSCOPE_API_KEY,
        temperature=0,
        streaming=False,
    )

    system_tools = [server_system_check, port_check, read_service_log]
    llm_with_system_tools = llm.bind_tools(system_tools)
    system_tools_map = {t.name: t for t in system_tools}

    classify = make_classify_node(fast_llm)
    rewrite_query = make_rewrite_query_node(fast_llm)
    retrieve = make_retrieve_node(retriever)
    execute_tools = make_execute_tools_node(llm_with_system_tools, system_tools_map)
    generate = make_generate_node(llm)

    workflow = StateGraph(AgentState)

    workflow.add_node("classify", classify)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("generate", generate)
    workflow.add_node("reject", reject)

    workflow.add_edge(START, "classify")

    workflow.add_conditional_edges("classify", route_by_intent, {
        "rewrite_query": "rewrite_query",
        "execute_tools": "execute_tools",
        "generate": "generate",
        "reject": "reject",
    })

    workflow.add_conditional_edges("rewrite_query", route_after_rewrite, {
        "retrieve": "retrieve",
    })

    workflow.add_conditional_edges("retrieve", route_after_retrieve, {
        "execute_tools": "execute_tools",
        "generate": "generate",
    })

    workflow.add_edge("execute_tools", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("reject", END)

    return workflow.compile()
