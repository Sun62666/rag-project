from typing import List,TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.retriever import OpsRetriever
import subprocess
import psutil
from langchain_core.tools import tool, render_text_description


class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    tool_calls: List[str]
    tool_results: List[str]

# 【服务器cpu/内存/磁盘巡检】
@tool
def server_system_check() -> str:
    """检查当前服务器的系统资源状态，包括CPU使用率、内存使用率、磁盘使用率和运行进程数。

    适用场景：
    - 用户询问服务器性能问题（如"服务器卡顿"、"响应慢"、"负载高"）
    - 用户要求查看系统资源使用情况（如"查看CPU/内存/磁盘状态"）
    - 用户询问服务器健康状态（如"服务器是否正常"、"系统巡检"）
    - 故障排查时需要了解系统资源瓶颈

    返回格式：包含CPU、内存、磁盘的使用百分比及总量信息"""
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return f"""
        【服务器系统巡检报告】
        CPU使用率： {cpu_usage}%
        内存使用率： {mem.percent}% (总内存： {round(mem.total/1024/1024/1024,2)}G)
        磁盘使用率： {disk.percent}% (总内存： {round(disk.total/1024/1024/1024,2)}G)
        运行进程数： {len(psutil.pids())}
        """
    except Exception as e:
        return f"系统巡检失败： {str(e)}"

# 【端口占用检测】
@tool
def port_check(port:int) -> str:
    """检查指定端口是否被占用，返回占用该端口的进程信息和PID。

    适用场景：
    - 用户询问某个端口是否可用（如"8080端口能用吗"、"3306端口是否被占用"）
    - 服务启动失败需要排查端口冲突（如"Tomcat启动失败"、"MySQL无法启动"）
    - 用户想知道哪个程序在使用某个端口（如"谁在用80端口"）
    - 网络服务部署前的端口检查

    参数说明：port - 要检查的端口号（整数，范围1-65535）
    返回格式：端口占用信息及对应的进程名称和PID"""
    try:
        result = subprocess.check_output(f"net stat -tulpn | grep :{port}",shell=True,text=True)
        return f"端口{port}占用信息:\n{result}"

    except Exception as e:
        return f"端口{port}未被占用"

# 真实日志读取
@tool
def read_service_log(log_path: str = "/var/log/syslog",lines: int = 20) -> str:
    """读取服务器上的日志文件内容，用于故障排查和问题分析。

        适用场景：
        - 用户要求查看服务日志（如"查看nginx日志"、"显示最近的应用日志"）
        - 服务异常需要分析错误信息（如"为什么服务崩溃了"、"查看报错日志"）
        - 用户询问特定时间段的日志（如"查看最近的20行日志"）
        - 排查服务启动失败、连接超时等问题

        参数说明：
        - log_path - 日志文件路径（默认/var/log/syslog，常见路径：/var/log/nginx/error.log、/var/log/mysql/error.log等）
        - lines - 读取的行数（默认20行，可根据需要调整）

        返回格式：日志文件的最后N行内容"""
    try:
        result = subprocess.check_output(f"tail -n {lines} {log_path}",shell=True,text=True)
        return f"【日志内容】\n {result}"
    except Exception as e:
        return f"读取日志失败：{str(e)}"

# 调用rag检索结结果
@tool
def knowledge_retrirver(query: str) -> str:
    """从运维知识库中检索相关的故障解决方案、配置规范、最佳实践等文档内容。

        适用场景：
        - 用户询问故障处理方法（如"CPU使用率过高怎么办"、"MySQL连接超时如何解决"）
        - 用户需要了解配置规范（如"Nginx如何配置负载均衡"、"Redis集群搭建步骤"）
        - 用户查询运维标准和流程（如"数据库备份策略"、"服务器安全加固指南"）
        - 任何需要参考历史经验或文档知识的运维问题

        参数说明：query - 检索关键词或问题描述（应简洁明确，如"CPU过高"、"MySQL主从同步"）
        返回格式：匹配的知识库文档片段（可能包含多个相关文档）

        注意：当用户问题涉及具体故障处理、配置方法、运维规范时，应优先调用此工具获取专业知识"""
    ctx = retriever_instance.retriever_and_rerank(query,top_k=3)
    return "\n".join(ctx)

def build_graph(retriever: OpsRetriever):
    global retriever_instance
    retriever_instance = retriever

    cfg = Config()
    llm = ChatOpenAI(
        model = cfg.LLM_MODEL,
        base_url = cfg.BASE_URL,
        api_key = cfg.DASHSCOPE_API_KEY,
        temperature=0.1,
        streaming=True,
        # model_kwargs={"lora_weights":"./lora-ops-model"}
    )
    tools = [server_system_check,port_check,read_service_log,knowledge_retrirver]
    llm_with_tools = llm.bind_tools(tools)
    tools_map = {tool.name: tool for tool in tools}
    print("构建提示词LLM中。。。。")
    import os
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"prompts","ops_system.md")
    with open(prompt_path,"r",encoding="utf-8") as f:
        sys_prompt = f.read()

    print(f"可以使用的工具： {render_text_description(tools)}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system',sys_prompt + "\n\n你需要使用以下工具:\n" + render_text_description(tools)),
            ('human','用户问题: {query}\n工具调用结果: {tool_results}')
        ]
    )

    def generate(state:AgentState):
        """LLM决策节点：判断是否需要调用工具，或直接生成最终回答"""
        print("\n...开始调用大模型中(tools已准备好)")
        response = llm_with_tools.invoke(prompt.format_messages(
            query=state["query"],
            tool_results=state.get("tool_results",[])
        ))
        print(f"本次LLM节点调用工具为： {response.tool_calls}")
        if response.tool_calls:
            return {"tool_calls":response.tool_calls}
        else:
            return {"answer":response.content,"tool_calls":[]}

    def tool_node(state:AgentState):
        """工具执行节点： 执行LLM决定的工具调用，并返回结果"""
        results = []
        for tool_call in state["tool_calls"]:
            tool = tools_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            results.append(result)
        print(f"调用工具完毕，输出： {results}")
        return {"tool_results":results,"tool_calls":[]}

    def should_continue(state: AgentState):
        if state.get("tool_calls"):
            return "tools"
        return END
    workflow = StateGraph(AgentState)
    workflow.add_node("generate",generate)
    workflow.add_node("tools",tool_node)
    workflow.add_edge(START,"generate")
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "tools":"tools",
            END:END
        }

    )
    workflow.add_edge("tools","generate")
    return workflow.compile()