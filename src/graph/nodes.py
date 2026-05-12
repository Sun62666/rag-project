import logging
from typing import Dict
from src.graph.prompts import get_classify_prompt, get_rewrite_prompt, get_generate_prompt, get_tool_call_prompt
from src.tools import knowledge_retriever_logic

logger = logging.getLogger(__name__)


def make_classify_node(fast_llm):
    classify_chain = get_classify_prompt() | fast_llm

    def classify(state: Dict) -> Dict:
        logger.info(f"[classify] 分类中... query: {state['query']}")
        chat_history = state.get("chat_history", [])
        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"
        response = classify_chain.invoke({
            "query": state["query"],
            "chat_history": history_str
        })
        intent = response.content.strip().lower()
        logger.info(f"[classify] 非默认结果: {intent}")
        if intent not in ["fault", "system", "mixed", "followup", "reject"]:
            intent = "fault"
        logger.info(f"[classify] 最终结果: {intent}")
        return {"intent": intent}

    return classify


def make_rewrite_query_node(fast_llm):
    rewrite_chain = get_rewrite_prompt() | fast_llm

    def rewrite_query(state: Dict) -> Dict:
        logger.info(f"[rewrite_query] 改写中... query: {state['query']}")
        response = rewrite_chain.invoke({"query": state["query"]})
        rewritten = response.content.strip()
        logger.info(f"[rewrite_query] 结果: {rewritten}")
        return {"rewritten_query": rewritten}

    return rewrite_query


def make_retrieve_node(retriever):
    def retrieve(state: Dict) -> Dict:
        query = state.get("rewritten_query") or state["query"]
        logger.info(f"[retrieve] 检索中... query: {query}")
        results = knowledge_retriever_logic(query, retriever)
        logger.info(f"[retrieve] 完成, 结果长度: {len(results)}")
        return {"retrieved_context": results}

    return retrieve


def make_execute_tools_node(llm_with_system_tools, system_tools_map):
    tool_chain = get_tool_call_prompt() | llm_with_system_tools

    def execute_tools(state: Dict) -> Dict:
        logger.info(f"[execute_tools] 执行中... query: {state['query']}")
        response = tool_chain.invoke({"query": state["query"]})
        results = {}
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_fn = system_tools_map[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                results[tc["name"]] = result
                logger.info(f"[execute_tools] 工具: {tc['name']}, 参数: {tc['args']}")
        logger.info(f"[execute_tools] 完成, 调用 {len(results)} 个工具")
        return {"tool_results": results}

    return execute_tools


def make_generate_node(llm):
    generate_chain = get_generate_prompt() | llm

    def generate(state: Dict) -> Dict:
        logger.info(f"[generate] 生成回答中...")
        context_parts = []
        if state.get("retrieved_context"):
            context_parts.append(f"【知识库检索结果】\n{state['retrieved_context']}")
        if state.get("tool_results"):
            for tool_name, result in state["tool_results"].items():
                context_parts.append(f"【工具 {tool_name} 执行结果】\n{result}")
        context = "\n\n".join(context_parts) if context_parts else "无可用上下文信息"
        chat_history = state.get("chat_history", [])
        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"
        response = generate_chain.invoke({
            "query": state["query"],
            "context": context,
            "chat_history": history_str
        })
        logger.info(f"[generate] 完成")
        return {"answer": response.content}

    return generate


def reject(state: Dict) -> Dict:
    logger.info(f"[reject] 拒绝非运维问题")
    return {"answer": "当前知识库未覆盖该问题，建议转交人工运维专家。"}
