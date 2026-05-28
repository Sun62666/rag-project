"""
SmartOps Agent 一键启动脚本
直接运行: python start_agent.py
"""
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8347"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    print("=" * 50)
    print("  SmartOps Agent 智能运维智能体")
    print(f"  模式: {'Agent' if os.getenv('USE_AGENT', 'true').lower() == 'true' else 'Graph'}")
    print(f"  地址: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs")
    print("=" * 50)

    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

