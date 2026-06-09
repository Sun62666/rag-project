import logging
import re
import os
import subprocess
from collections import Counter
from langchain_core.tools import tool

from src.core import get_settings

logger = logging.getLogger(__name__)

SAMPLE_LOG = """2026-05-07 10:12:01 [ERROR] ConnectionTimeout: Redis connection failed after 30s
2026-05-07 10:12:03 [WARN] Retry attempt 1/3 for Redis connection
2026-05-07 10:12:05 [ERROR] OutOfMemory: Java heap space - allocated 4096MB
2026-05-07 10:12:06 [WARN] GC overhead limit exceeded
2026-05-07 10:12:08 [ERROR] NullPointerException at com.ops.ServiceHandler.process(ServiceHandler.java:142)
2026-05-07 10:12:10 [INFO] Health check passed for nginx
2026-05-07 10:12:12 [ERROR] DiskFull: No space left on device /dev/sda1
2026-05-07 10:12:14 [WARN] CPU usage exceeded 90% threshold
2026-05-07 10:12:16 [ERROR] ConnectionRefused: MySQL port 3306 not responding
2026-05-07 10:12:18 [ERROR] SSLHandshakeFailed: certificate expired for api.example.com
2026-05-07 10:12:20 [INFO] Backup completed successfully
2026-05-07 10:12:22 [ERROR] PermissionDenied: cannot write to /var/log/app.log
2026-05-07 10:12:24 [WARN] Memory usage at 85%
2026-05-07 10:12:26 [ERROR] Timeout: API request to payment-service exceeded 60s
2026-05-07 10:12:28 [INFO] User login successful: admin
2026-05-07 10:12:30 [ERROR] SegmentationFault: worker process pid=28451 crashed
2026-05-07 10:12:32 [WARN] Slow query detected: SELECT * FROM orders (12.5s)
2026-05-07 10:12:34 [ERROR] DNSResolutionFailed: cannot resolve db-master.internal
2026-05-07 10:12:36 [INFO] Cron job cleanup executed
2026-05-07 10:12:38 [ERROR] PortAlreadyInUse: 8080 occupied by process java(28432)
"""
cfg = get_settings()
_DEFAULT_LOG = getattr(cfg, 'LOG_PATH', os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "smartops.log"
))
_DEFAULT_ERROR_LOG = getattr(cfg, 'ERROR_LOG_PATH', os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "error.log"
))


def read_service_log_logic(log_path: str = "", lines: int = 20) -> str:
    if not log_path:
        log_path = _DEFAULT_LOG
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        content = "".join(all_lines[-lines:])
        return f"【日志内容】\n{content}"
    except Exception as e:
        return f"读取日志失败：{str(e)}"


@tool
def read_service_log(log_path: str = "", lines: int = 20) -> str:
    """读取服务器日志文件。参数：log_path-日志路径（默认为项目logs/smartops.log），lines-读取行数"""
    return read_service_log_logic(log_path, lines)


@tool
def log_error_stats(log_path: str = _DEFAULT_ERROR_LOG, lines: int = 100) -> str:
    """统计日志文件中的错误信息，包括错误类型分布、高频错误TOP5、时间趋势等。
    当用户询问日志错误统计、错误分析、日志排查时调用此工具。
    参数：log_path-日志文件路径（为空则使用示例数据），lines-读取行数"""
    log_content = ""

    if log_path and os.path.exists(log_path):
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), log_path],
                capture_output=True, text=True, timeout=10
            )
            log_content = result.stdout if result.returncode == 0 else ""
        except Exception:
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    all_lines = f.readlines()
                    log_content = "".join(all_lines[-lines:])
            except Exception as e:
                log_content = SAMPLE_LOG
                logger.warning(f"[工具] 读取日志失败，使用示例数据: {e}")
    else:
        log_content = SAMPLE_LOG
        logger.info("[工具] 未指定日志路径，使用示例数据演示")

    if not log_content.strip():
        return "日志内容为空，无法统计。"

    error_pattern = re.compile(r'\[ERROR\]\s*(.+?)(?:\n|$)', re.IGNORECASE)
    warn_pattern = re.compile(r'\[WARN\]\s*(.+?)(?:\n|$)', re.IGNORECASE)

    errors = error_pattern.findall(log_content)
    warnings = warn_pattern.findall(log_content)

    error_categories = Counter()
    for err in errors:
        if "Timeout" in err or "timeout" in err:
            error_categories["超时错误(Timeout)"] += 1
        elif "Connection" in err or "connection" in err:
            error_categories["连接错误(Connection)"] += 1
        elif "Memory" in err or "memory" in err or "OutOfMemory" in err:
            error_categories["内存错误(Memory)"] += 1
        elif "Disk" in err or "disk" in err or "space" in err.lower():
            error_categories["磁盘错误(Disk)"] += 1
        elif "Permission" in err or "permission" in err:
            error_categories["权限错误(Permission)"] += 1
        elif "SSL" in err or "certificate" in err:
            error_categories["安全错误(Security)"] += 1
        elif "DNS" in err:
            error_categories["网络错误(Network/DNS)"] += 1
        elif "Port" in err or "port" in err:
            error_categories["端口错误(Port)"] += 1
        elif "Segmentation" in err or "crash" in err:
            error_categories["进程崩溃(Crash)"] += 1
        else:
            error_categories["其他错误(Other)"] += 1

    top_errors = Counter(errors).most_common(5)

    result_parts = [
        f"【日志错误统计报告】",
        f"分析日志行数: {len(log_content.splitlines())}",
        f"错误(ERROR)总数: {len(errors)}",
        f"警告(WARN)总数: {len(warnings)}",
        f"",
        f"--- 错误类型分布 ---",
    ]
    for cat, count in error_categories.most_common():
        result_parts.append(f"  {cat}: {count}次")

    result_parts.append(f"\n--- 高频错误 TOP5 ---")
    for i, (err_msg, count) in enumerate(top_errors, 1):
        result_parts.append(f"  {i}. [{count}次] {err_msg[:80]}")

    if len(errors) > 0:
        result_parts.append(f"\n--- 建议 ---")
        top_cat = error_categories.most_common(1)[0][0]
        result_parts.append(f"  主要问题类型: {top_cat}")
        if "超时" in top_cat:
            result_parts.append("  建议: 检查网络连通性、服务健康状态、超时配置")
        elif "连接" in top_cat:
            result_parts.append("  建议: 检查目标服务是否运行、端口是否开放、防火墙规则")
        elif "内存" in top_cat:
            result_parts.append("  建议: 检查JVM堆配置、内存泄漏、增加物理内存")
        elif "磁盘" in top_cat:
            result_parts.append("  建议: 清理日志文件、临时文件，扩容磁盘")
        elif "权限" in top_cat:
            result_parts.append("  建议: 检查文件/目录权限、运行用户身份")

    logger.info(f"[工具] log_error_stats 执行成功: {len(errors)}个错误, {len(warnings)}个警告")
    return "\n".join(result_parts)
