"""
Milvus 连接兼容性补丁

pymilvus 2.6.x 引入了新的 ConnectionManager，与旧版 pymilvus.connections 不兼容。
MilvusClient 使用 ConnectionManager 管理连接，但 langchain_milvus 内部的 Collection ORM
仍依赖 pymilvus.connections 查找连接，导致 ConnectionNotExistException。

此模块提供 ensure_milvus_connection() 函数，预先创建 MilvusClient 并将其 handler
注册到 pymilvus.connections，使 langchain_milvus 能正常工作。
"""

import logging
from pymilvus import MilvusClient, connections
from pymilvus.orm.connections import Connections

logger = logging.getLogger(__name__)

_patched = False


def ensure_milvus_connection(uri: str) -> MilvusClient:
    """创建 MilvusClient 并将其 handler 注册到 pymilvus.connections。

    必须在创建 langchain_milvus.Milvus 实例之前调用此函数。

    Args:
        uri: Milvus 连接 URI，如 "http://192.168.100.128:19530"

    Returns:
        MilvusClient 实例（可用于检查集合等操作，不要 close）
    """
    global _patched

    # 创建 MilvusClient（使用 ConnectionManager 管理连接）
    client = MilvusClient(uri=uri)
    alias = client._using  # 如 "cm-2524965309856"
    handler = client._handler

    # 将 handler 注册到 pymilvus.connections，使 Collection ORM 能找到它
    conn_instance = Connections.get_instance() if hasattr(Connections, 'get_instance') else connections
    if alias not in conn_instance._alias_handlers:
        conn_instance._alias_handlers[alias] = handler
        conn_instance._alias_config[alias] = {
            'address': client._config.address,
            'uri': client._config.uri,
        }
        logger.debug(f"[Milvus补丁] 已注册连接别名 {alias} 到 pymilvus.connections")

    if not _patched:
        _patched = True
        logger.info(f"[Milvus补丁] pymilvus 2.6.x 连接兼容性补丁已生效，别名: {alias}")

    return client


def get_collection_count(client: MilvusClient, collection_name: str) -> int:
    """使用 MilvusClient 获取集合的实体数量。

    Milvus 2.4+ 不允许 count(*) 带分页参数（limit），
    必须只传 filter 和 output_fields，不传 limit。
    """
    if not client.has_collection(collection_name):
        return 0
    try:
        # count(*) 不能带 limit 参数，否则报 "count entities with pagination is not allowed"
        result = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if result and len(result) > 0:
            count_val = list(result[0].values())[0] if result[0] else 0
            return int(count_val) if count_val else 0
    except Exception:
        pass
    return 0
