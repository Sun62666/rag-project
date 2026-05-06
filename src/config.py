import os
from dotenv import load_dotenv
abpath = os.path.dirname(__file__)
env_path = os.path.join(os.path.dirname(abpath),"Key.env")
rerank_path = os.path.join(os.path.dirname(abpath),"model","bge-reranker-v2-m3")
load_dotenv(env_path)

class Config:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    MILVUS_URI = os.getenv("MILVUS_URL")
    LLM_MODEL = "qwen3.5-plus"
    REDIS_URL = os.getenv("REDIS_URL")
    COLLECTION_NAME = "ops_knowledge_v2"
    EMBED_MODEL = "text-embedding-v2"
    RERANK_MODEL = rerank_path
    CACHE_TTL_SHOT = 3600*24*7
    CACHE_TTL_LONG = 3600*24*30