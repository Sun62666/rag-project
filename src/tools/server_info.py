import logging
import platform
import psutil
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def server_info_query(server_name: str = "localhost") -> str:
    """查询服务器系统信息，包括CPU使用率、内存使用率、磁盘使用率、运行进程数、系统负载等。
    当用户询问服务器状态、系统资源、CPU/内存/磁盘使用情况时调用此工具。
    参数：server_name-服务器名称（默认localhost）"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        pids = len(psutil.pids())
        os_info = f"{platform.system()} {platform.release()}"
        load_avg = "N/A"
        if hasattr(psutil, "getloadavg"):
            try:
                load_avg = ", ".join(f"{x:.2f}" for x in psutil.getloadavg())
            except Exception:
                pass
        top_procs = []
        for proc in sorted(psutil.process_iter(["pid", "name", "cpu_percent"]),
                           key=lambda p: p.info.get("cpu_percent") or 0, reverse=True)[:5]:
            top_procs.append(f"  PID={proc.info['pid']}  {proc.info['name']}  CPU={proc.info.get('cpu_percent', 0)}%")

        result = (
            f"【服务器系统信息报告 - {server_name}】\n"
            f"操作系统: {os_info}\n"
            f"CPU: {cpu_percent}% 使用率 ({cpu_count} 核)\n"
            f"系统负载: {load_avg}\n"
            f"内存: {mem.percent}% 使用率 (总量 {round(mem.total/1024/1024/1024, 1)}GB, "
            f"可用 {round(mem.available/1024/1024/1024, 1)}GB)\n"
            f"磁盘: {disk.percent}% 使用率 (总量 {round(disk.total/1024/1024/1024, 1)}GB, "
            f"可用 {round(disk.free/1024/1024/1024, 1)}GB)\n"
            f"运行进程数: {pids}\n"
            f"Top5 进程:\n" + "\n".join(top_procs)
        )
        logger.info(f"[工具] server_info_query 执行成功: CPU={cpu_percent}%, MEM={mem.percent}%")
        return result
    except Exception as e:
        logger.error(f"[工具] server_info_query 执行失败: {e}")
        return f"服务器信息查询失败: {str(e)}"
