"""日志查看 API"""
import os
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.deps import get_current_user_dep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["运维工具-日志"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# ==================== 数据模型 ====================

class LogLine(BaseModel):
    line_no: int
    content: str
    level: str = ""


class LogResponse(BaseModel):
    log_path: str
    total_lines: int
    lines: List[LogLine]
    error_count: int = 0
    warn_count: int = 0


# ==================== 接口 ====================

@router.get("/logs", response_model=LogResponse)
async def get_system_logs(
    log_name: str = Query(default="smartops.log", description="日志文件名"),
    lines: int = Query(default=100, ge=1, le=1000, description="读取行数"),
    level: str = Query(default="all", description="过滤级别: all/error/warn/info"),
    _user: str = Depends(get_current_user_dep),
):
    log_path = os.path.join(LOG_DIR, log_name)
    if not os.path.exists(log_path):
        for f in os.listdir(LOG_DIR) if os.path.isdir(LOG_DIR) else []:
            if f.endswith(".log"):
                log_path = os.path.join(LOG_DIR, f)
                break
        else:
            return LogResponse(log_path=log_path, total_lines=0, lines=[])

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {e}")

    total = len(all_lines)
    recent = all_lines[-lines:]

    result = []
    error_count = 0
    warn_count = 0

    for i, line in enumerate(recent):
        line_level = ""
        if "ERROR" in line:
            line_level = "error"
            error_count += 1
        elif "WARN" in line:
            line_level = "warn"
            warn_count += 1
        elif "INFO" in line:
            line_level = "info"

        if level != "all" and line_level != level:
            continue

        result.append(LogLine(
            line_no=total - lines + i + 1,
            content=line.rstrip(),
            level=line_level,
        ))

    return LogResponse(
        log_path=log_path,
        total_lines=total,
        lines=result,
        error_count=error_count,
        warn_count=warn_count,
    )


@router.get("/logs/files")
async def list_log_files(_user: str = Depends(get_current_user_dep)):
    if not os.path.isdir(LOG_DIR):
        return {"files": []}
    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    file_info = []
    for f in files:
        fp = os.path.join(LOG_DIR, f)
        size = os.path.getsize(fp)
        mtime = os.path.getmtime(fp)
        file_info.append({"name": f, "size": size, "modified": mtime})
    return {"files": file_info}
