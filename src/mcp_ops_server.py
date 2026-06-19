import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mcp.server.fastmcp import FastMCP
from src.tools import (
    server_system_check_logic,
    port_check_logic,
    read_service_log_logic,
    knowledge_retriever_logic,
    set_retriever,
)
from src.retriever import OpsRetriever

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "文档2.pdf")
retriever = OpsRetriever(PDF_PATH)
set_retriever(retriever)
mcp = FastMCP("smartops-assistant")


@mcp.tool()
def server_system_check() -> str:
    """检查服务器CPU、内存、磁盘使用率和运行进程数。"""
    return server_system_check_logic()


@mcp.tool()
def port_check(port: int) -> str:
    """检查指定端口占用情况。参数：port-端口号(1-65535)"""
    return port_check_logic(port)


@mcp.tool()
def read_service_log(log_path: str = "/var/log/syslog", lines: int = 20) -> str:
    """读取服务器日志文件。参数：log_path-日志路径，lines-读取行数"""
    return read_service_log_logic(log_path, lines)


@mcp.tool()
def knowledge_retriever(query: str) -> str:
    """从运维知识库检索故障解决方案、配置规范等文档。参数：query-检索关键词"""
    return knowledge_retriever_logic(query, retriever)


if __name__ == "__main__":
    mcp.run(transport="stdio")
