import psutil
import subprocess
from src.retriever import OpsRetriever

def server_system_check_logic() -> str:
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('C:\\')
        return (
            f"【服务器系统巡检报告】\n"
            f"CPU使用率：{cpu_usage}%\n"
            f"内存使用率：{mem.percent}% (总内存：{round(mem.total/1024/1024/1024,2)}G)\n"
            f"磁盘使用率：{disk.percent}% (总磁盘：{round(disk.total/1024/1024/1024,2)}G)\n"
            f"运行进程数：{len(psutil.pids())}"
        )
    except Exception as e:
        return f"系统巡检失败：{str(e)}"

def port_check_logic(port: int) -> str:
    try:
        result = subprocess.check_output(f"netstat -tulpn | grep :{port}", shell=True, text=True)
        return f"端口{port}占用信息:\n{result}"
    except Exception:
        return f"端口{port}未被占用"

def read_service_log_logic(log_path: str = "/var/log/syslog", lines: int = 20) -> str:
    try:
        result = subprocess.check_output(f"tail -n {lines} {log_path}", shell=True, text=True)
        return f"【日志内容】\n{result}"
    except Exception as e:
        return f"读取日志失败：{str(e)}"

def knowledge_retriever_logic(query: str, retriever: OpsRetriever, top_k: int = 3) -> str:
    docs_with_scores = retriever.retriever_and_rerank_with_scores(query, top_k=top_k)
    if not docs_with_scores:
        return "【知识库检索结果】未检索到与问题相关的文档内容，知识库可能未覆盖该问题。"
    results = []
    for doc, score in docs_with_scores:
        relevance = "高" if score > 0.5 else "中" if score > 0.2 else "低"
        source = doc.metadata.get("source", "运维文档")
        results.append(f"[相关性:{relevance}({score:.2f})] {source}\n{doc.page_content}")
    return "\n\n".join(results)
