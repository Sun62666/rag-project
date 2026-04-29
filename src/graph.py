import os
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.retriever import OpsRetriever
from langchain_core.tools import tool
from src.common_tools import (
    server_system_check_logic,
    port_check_logic,
    read_service_log_logic,
    knowledge_retriever_logic
)

retriever_instance = None


class AgentState(TypedDict):
    query: str
    intent: str
    rewritten_query: str
    retrieved_context: str
    tool_results: Dict[str, str]
    answer: str
    chat_history: List[str]


@tool
def server_system_check() -> str:
    """检查服务器CPU、内存、磁盘使用率和运行进程数。"""
    return server_system_check_logic()


@tool
def port_check(port: int) -> str:
    """检查指定端口占用情况。参数：port-端口号(1-65535)"""
    return port_check_logic(port)


@tool
def read_service_log(log_path: str = "/var/log/syslog", lines: int = 20) -> str:
    """读取服务器日志文件。参数：log_path-日志路径，lines-读取行数"""
    return read_service_log_logic(log_path, lines)


@tool
def knowledge_retriever(query: str) -> str:
    """从运维知识库检索故障解决方案、配置规范等文档。参数：query-检索关键词"""
    return knowledge_retriever_logic(query, retriever_instance)


def build_graph(retriever: OpsRetriever):
    global retriever_instance
    retriever_instance = retriever

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

    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "ops_system.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_prompt = f.read()

    classify_prompt = ChatPromptTemplate.from_messages([
        ('system', """你是运维问题分类器。根据用户问题判断意图类别，只输出类别名称：

        - fault: 需要查阅知识库的故障排查/配置方法/运维规范问题（如"CPU过高怎么办"、"Redis内存溢出"、"Nginx配置负载均衡"）
        - system: 需要实时检查服务器状态的问题（如"查看CPU状态"、"检查8080端口"、"查看nginx日志"）
        - mixed: 既需要知识库又需要实时检查的混合问题（如"服务器CPU高怎么排查"）
        - followup: 基于历史对话的追问、澄清、引用之前话题的问题（如"上次问了什么"、"刚才那个问题再说一下"、"具体怎么操作"、"还有其他方法吗"、"帮我详细解释一下"）
        - reject: 与运维完全无关的闲聊、娱乐、无关话题（如"讲个笑话"、"今天天气怎么样"）

        只输出一个词：fault / system / mixed / followup / reject"""),
        ('human', '用户问题: {query}\n\n最近对话历史:\n{chat_history}')
    ])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ('system', """你是运维检索专家。将用户问题改写为更适合知识库检索的关键词。

        要求：
        1. 提取核心技术名词和故障关键词
        2. 补充同义词和专业术语（如"内存满"→"OOM out-of-memory"）
        3. 输出2-5个检索关键词，用空格分隔
        4. 只输出改写后的关键词，不要解释

        示例：
        - "Redis内存满了怎么办" → "Redis OOM 内存溢出 maxmemory 淘汰策略"
        - "服务器CPU使用率100%" → "CPU使用率过高 CPU满载 进程占用 top"
        - "MySQL连接超时" → "MySQL连接超时 connection_timeout wait_timeout" """),
        ('human', '{query}')
    ])

    generate_prompt = ChatPromptTemplate.from_messages([
        ('system', sys_prompt + "\n\n特别注意：当用户问题是对之前对话的追问或引用时，必须基于历史对话内容回答，不要输出兜底文案。"),
        ('human', '用户问题: {query}\n\n可用上下文:\n{context}\n\n历史对话:\n{chat_history}')
    ])

    tool_call_prompt = ChatPromptTemplate.from_messages([
        ('system', '你是运维工具调用助手。根据用户问题调用合适的系统工具获取实时数据，只调用与问题相关的工具。'),
        ('human', '用户问题: {query}')
    ])

    def classify(state: AgentState):
        print(f"\n[classify] 分类中... query: {state['query']}")
        chat_history = state.get("chat_history", [])
        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"
        response = fast_llm.invoke(classify_prompt.format_messages(
            query=state["query"],
            chat_history=history_str
        ))
        intent = response.content.strip().lower()
        if intent not in ["fault", "system", "mixed", "followup", "reject"]:
            intent = "fault"
        print(f"[classify] 结果: {intent}")
        return {"intent": intent}

    def rewrite_query(state: AgentState):
        print(f"\n[rewrite_query] 改写中... query: {state['query']}")
        response = fast_llm.invoke(rewrite_prompt.format_messages(query=state["query"]))
        rewritten = response.content.strip()
        print(f"[rewrite_query] 结果: {rewritten}")
        return {"rewritten_query": rewritten}

    def retrieve(state: AgentState):
        query = state.get("rewritten_query") or state["query"]
        print(f"\n[retrieve] 检索中... query: {query}")
        results = knowledge_retriever_logic(query, retriever_instance)
        print(f"[retrieve] 完成, 结果长度: {len(results)} \n 结果为: {results}")
        return {"retrieved_context": results}

    def execute_tools(state: AgentState):
        print(f"\n[execute_tools] 执行中... query: {state['query']}")
        response = llm_with_system_tools.invoke(
            tool_call_prompt.format_messages(query=state["query"])
        )
        results = {}
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_fn = system_tools_map[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                results[tc["name"]] = result
                print(f"[execute_tools] 工具: {tc['name']}, 参数: {tc['args']}")
        print(f"[execute_tools] 完成, 调用 {len(results)} 个工具")
        return {"tool_results": results}

    def generate(state: AgentState):
        print(f"\n[generate] 生成回答中...")
        context_parts = []
        if state.get("retrieved_context"):
            context_parts.append(f"【知识库检索结果】\n{state['retrieved_context']}")
        if state.get("tool_results"):
            for tool_name, result in state["tool_results"].items():
                context_parts.append(f"【工具 {tool_name} 执行结果】\n{result}")
        context = "\n\n".join(context_parts) if context_parts else "无可用上下文信息"

        chat_history = state.get("chat_history", [])
        history_str = "\n".join(chat_history[-6:]) if chat_history else "无历史对话"

        messages = generate_prompt.format_messages(
            query=state["query"],
            context=context,
            chat_history=history_str
        )
        response = llm.invoke(messages)
        print(f"[generate] 完成")
        return {"answer": response.content}

    def reject(state: AgentState):
        print(f"\n[reject] 拒绝非运维问题")
        return {"answer": "当前知识库未覆盖该问题，建议转交人工运维专家。"}

    def route_by_intent(state: AgentState):
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

    def route_after_rewrite(state: AgentState):
        return "retrieve"

    def route_after_retrieve(state: AgentState):
        if state.get("intent") == "mixed":
            return "execute_tools"
        return "generate"

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
