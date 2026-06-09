import os
from pydantic_settings import BaseSettings
from pydantic import Field,SecretStr
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_env_files = [
    os.path.join(_BASE_DIR, ".env"),
    os.path.join(_BASE_DIR, "Key.env"),
    os.path.join(_BASE_DIR, "Env.env"),
    os.path.join(_BASE_DIR, "Env1.env"),
]
for _ef in _env_files:
    if os.path.exists(_ef):
        load_dotenv(_ef, override=False)
        break


class Config(BaseSettings):
    DASHSCOPE_API_KEY: str = Field(default="", description="DashScope API密钥")
    BASE_URL: str = Field(default="", description="LLM API基础URL")
    MILVUS_URI: str = Field(default="http://192.168.100.128:19530", alias="MILVUS_URL", description="Milvus连接URI")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis连接URL")
    LLM_MODEL: str = Field(default="qwen-max", description="LLM模型名称")
    EMBED_MODEL: str = Field(default="text-embedding-v4", description="Embedding模型名称")
    COLLECTION_NAME: str = Field(default="ops_knowledge_v2", description="Milvus知识库集合名")
    DOC_COLLECTION_NAME: str = Field(default="property_regulations", description="通用文档知识库集合名(物业法规等)")
    RERANK_MODEL: str = Field(
        default=os.path.join(_BASE_DIR, "model", "bge-reranker-v2-m3"),
        description="Rerank模型路径"
    )
    CACHE_TTL_SHORT: int = Field(default=3600 * 24 * 7, description="短期缓存TTL(秒)")
    CACHE_TTL_LONG: int = Field(default=3600 * 24 * 30, description="长期缓存TTL(秒)")
    USE_AGENT: bool = Field(default=True, description="是否使用Agent模式")
    HOST: str = Field(default="0.0.0.0", description="服务监听地址")
    PORT: int = Field(default=8347, description="服务监听端口")
    LOG_PATH: str = Field(default="", description="服务日志路径")
    ERROR_LOG_PATH: str = Field(default="", description="服务错误日志路径")
    NEO4J_URI: str = Field(default="bolt://192.168.100.128:7687", description="Neo4j连接URI")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j用户名")
    NEO4J_PASSWORD: str = Field(default="smartops123", description="Neo4j密码")
    LORA_BASE_MODEL: str = Field(default="", description="LoRA基座模型路径")
    LORA_WEIGHTS: str = Field(default="", description="LoRA微调权重路径")
    USE_LORA: bool = Field(default=False, description="是否使用LoRA微调模型(作为本地降级)")
    LORA_RERANK_MODEL: str = Field(default="", description="微调后的Reranker模型路径")
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


_settings_instance = None


def get_settings() -> Config:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Config()
    return _settings_instance
