import logging
import subprocess
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def port_check_logic(port: int) -> str:
    try:
        result = subprocess.check_output(f"netstat -tulpn | grep :{port}", shell=True, text=True)
        return f"端口{port}占用信息:\n{result}"
    except Exception:
        return f"端口{port}未被占用"


@tool
def port_check(port: int) -> str:
    """检查指定端口占用情况。参数：port-端口号(1-65535)"""
    return port_check_logic(port)
