import os
import json
import logging
import hashlib
import argparse
import sys
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logging.getLogger("neo4j").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CRAWLED_DIR = os.path.join(BASE_DIR, "data", "crawled")

for ef in [".env", "Key.env", "Env.env", "Env1.env"]:
    p = os.path.join(BASE_DIR, ef)
    if os.path.exists(p):
        load_dotenv(p, override=False)
        break

MILVUS_URI = os.getenv("MILVUS_URL", "http://192.168.100.128:19530")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v1")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ops_knowledge_v2")

# 统一 metadata 字段：所有文档只保留这些 key，避免 schema 不匹配
UNIFIED_METADATA_KEYS = ["source", "source_name", "doc_id", "type"]


def _normalize_uri(uri: str) -> str:
    if not uri.startswith("http://") and not uri.startswith("https://"):
        uri = f"http://{uri}"
    return uri


def _unify_metadata(doc: Document) -> Document:
    """统一 metadata，只保留 UNIFIED_METADATA_KEYS 中的字段"""
    old_meta = doc.metadata
    new_meta = {}
    for key in UNIFIED_METADATA_KEYS:
        if key in old_meta:
            val = old_meta[key]
            if isinstance(val, str) and len(val) > 200:
                val = val[:200]
            new_meta[key] = val
    if "source" not in new_meta and "source_name" not in new_meta:
        new_meta["source"] = "unknown"
    doc.metadata = new_meta
    return doc


def _get_vectorstore_class():
    """获取 Milvus 向量存储类，优先使用 langchain_milvus"""
    try:
        from langchain_milvus import Milvus
        return Milvus
    except ImportError:
        logger.warning("langchain-milvus 未安装，请运行: pip install langchain-milvus")
        raise


def load_pdf(path: str) -> List[Document]:
    return PyPDFLoader(path).load()


def load_text(path: str) -> List[Document]:
    return TextLoader(path, encoding="utf-8").load()


def load_markdown(path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(path).load()
    except (ImportError, Exception):
        return TextLoader(path, encoding="utf-8").load()


def load_json_qa(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item, dict):
            if "messages" in item:
                q, a = "", ""
                for msg in item["messages"]:
                    if msg.get("role") == "user":
                        q = msg.get("content", "")
                    elif msg.get("role") == "assistant":
                        a = msg.get("content", "")
                if q and a:
                    content = f"问: {q}\n答: {a}"
                    docs.append(Document(page_content=content, metadata={"source": os.path.basename(path), "type": "qa"}))
            elif "input" in item and "output" in item:
                content = f"问: {item['input']}\n答: {item['output']}"
                docs.append(Document(page_content=content, metadata={"source": os.path.basename(path), "type": "qa"}))
            elif "query" in item:
                query = item["query"]
                pos = item.get("positive_contexts", [])
                if pos:
                    content = f"问: {query}\n相关文档: {pos[0]}"
                    docs.append(Document(page_content=content, metadata={"source": os.path.basename(path), "type": "rerank"}))
    return docs


def load_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_markdown,
        ".json": load_json_qa,
    }
    loader = loaders.get(ext)
    if not loader:
        return []
    try:
        docs = loader(path)
        for doc in docs:
            doc.metadata["file_path"] = path
        logger.info(f"加载 {os.path.basename(path)}: {len(docs)} 个文档")
        return docs
    except Exception as e:
        logger.error(f"加载失败 {path}: {e}")
        return []


def load_directory(dir_path: str, recursive: bool = True) -> List[Document]:
    all_docs = []
    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".pdf", ".txt", ".md", ".json"):
                    all_docs.extend(load_file(fpath))
    else:
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                all_docs.extend(load_file(fpath))
    return all_docs


def split_docs(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 60) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n案例 ", "\n案例", "案例", "\n问：", "\n问:", "\n## ", "\n\n", "\n", "。", " ", ""],
    )
    splits = splitter.split_documents(docs)
    for i, doc in enumerate(splits):
        doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        doc.metadata.setdefault("doc_id", doc_id)
        source = doc.metadata.get("file_path", doc.metadata.get("source", ""))
        if source:
            parts = source.replace("\\", "/").split("/")
            for part in reversed(parts):
                if part not in ("data", "crawled", "docs"):
                    doc.metadata.setdefault("source_name", part)
                    break
        # 统一 metadata
        _unify_metadata(doc)
    return splits


def import_to_milvus(splits: List[Document], collection_name: str, uri: str, drop_existing: bool = False, batch_size: int = 200, append_only: bool = False):
    """分批导入 Milvus，使用 langchain_milvus.Milvus，带详细进度
    
    append_only: 仅追加，集合不存在时报错
    """
    uri = _normalize_uri(uri)
    emb = DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)
    MilvusVS = _get_vectorstore_class()

    # 使用兼容性补丁建立连接（pymilvus 2.6.x ConnectionManager 与 ORM 不兼容）
    from pymilvus import MilvusClient, connections
    client = MilvusClient(uri=uri)

    total = len(splits)
    if total == 0:
        logger.error("没有文档可导入")
        return

    # 删除已有集合
    if drop_existing and client.has_collection(collection_name):
        logger.info(f"删除已有集合: {collection_name}")
        client.drop_collection(collection_name)

    need_create = not client.has_collection(collection_name)
    
    if append_only and need_create:
        logger.error(f"集合 {collection_name} 不存在，append_only 模式下无法追加")
        return

    # 注册 MilvusClient 的 handler 到 pymilvus.connections（不要 close client）
    alias = client._using
    handler = client._handler
    if alias not in connections._alias_handlers:
        connections._alias_handlers[alias] = handler
        connections._alias_config[alias] = {
            'address': client._config.address,
            'uri': client._config.uri,
        }

    if need_create:
        first_batch = min(batch_size, total)
        logger.info(f"[1/{(total + batch_size - 1) // batch_size}] 创建集合并导入前 {first_batch} 个切片...")
        vs = MilvusVS.from_documents(
            splits[:first_batch],
            emb,
            collection_name=collection_name,
            connection_args={"uri": uri},
            enable_dynamic_field=True
        )
        logger.info(f"  已导入 {first_batch}/{total} ({first_batch * 100 // total}%)")
        remaining = splits[first_batch:]
        if not remaining:
            logger.info(f"导入完成! 集合: {collection_name}, 总计: {total} 个切片")
            return
    else:
        logger.info(f"集合 {collection_name} 已存在，追加数据...")
        remaining = splits
        vs = MilvusVS(
            collection_name=collection_name,
            embedding_function=emb,
            connection_args={"uri": uri},
            auto_id=True,
            enable_dynamic_field=True
        )

    # 分批追加
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    start_idx = total - len(remaining)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining[start:end]

        current_global = start_idx + end
        pct = current_global * 100 // total

        try:
            vs.add_documents(batch)
            logger.info(f"[{batch_idx + 1}/{total_batches}] 已导入 {current_global}/{total} ({pct}%) - 本批 {len(batch)} 个切片")
        except Exception as e:
            logger.error(f"[{batch_idx + 1}/{total_batches}] 导入失败: {e}")
            try:
                vs = MilvusVS(
                    collection_name=collection_name,
                    embedding_function=emb,
                    connection_args={"uri": uri},
                    auto_id=True,
                    enable_dynamic_field=True
                )
                vs.add_documents(batch)
                logger.info(f"  重试成功: {len(batch)} 个切片")
            except Exception as e2:
                logger.error(f"  重试也失败，跳过本批: {e2}")

        if batch_idx < total_batches - 1:
            time.sleep(0.3)

    logger.info(f"导入完成! 集合: {collection_name}, 总计: {total} 个切片")



def import_crawled_data(collection_name: str, uri: str, max_per_source: int = 0, chunk_size: int = 500, drop: bool = False, batch_size: int = 200):
    """逐源加载并分批导入 Milvus，每完成一个源显示进度"""
    if not os.path.isdir(CRAWLED_DIR):
        logger.error(f"爬取数据目录不存在: {CRAWLED_DIR}")
        return

    sources = sorted([d for d in os.listdir(CRAWLED_DIR) if os.path.isdir(os.path.join(CRAWLED_DIR, d))])
    if not sources:
        logger.error("没有爬取数据")
        return

    logger.info(f"发现 {len(sources)} 个数据源: {', '.join(sources)}")

    # 先统计各源切片数
    source_splits_map = {}
    total_splits = 0
    for source_name in sources:
        source_dir = os.path.join(CRAWLED_DIR, source_name)
        docs = load_directory(source_dir, recursive=True)
        if not docs:
            logger.warning(f"  {source_name}: 无文档")
            continue
        splits = split_docs(docs, chunk_size=chunk_size)
        if max_per_source and len(splits) > max_per_source:
            splits = splits[:max_per_source]
        source_splits_map[source_name] = splits
        total_splits += len(splits)
        logger.info(f"  {source_name}: {len(docs)} 文档 -> {len(splits)} 切片")

    if not source_splits_map:
        logger.error("没有可导入的数据")
        return

    logger.info(f"总计 {total_splits} 个切片，分批导入 Milvus (每批 {batch_size})...")

    # 如果 drop，先删除集合
    if drop:
        uri_n = _normalize_uri(uri)
        client = MilvusClient(uri=uri_n)
        if client.has_collection(collection_name):
            logger.info(f"删除已有集合: {collection_name}")
            client.drop_collection(collection_name)
        client.close()

    # 逐源导入
    emb = DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=DASHSCOPE_API_KEY)
    MilvusVS = _get_vectorstore_class()

    uri_n = _normalize_uri(uri)
    imported_total = 0
    vs = None

    for src_idx, (source_name, splits) in enumerate(source_splits_map.items(), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"[{src_idx}/{len(source_splits_map)}] 导入源: {source_name} ({len(splits)} 切片)")
        logger.info(f"{'='*50}")

        # 首个源创建集合
        if vs is None:
            first_batch = min(batch_size, len(splits))
            logger.info(f"  创建集合并导入前 {first_batch} 个切片...")
            vs = MilvusVS.from_documents(
                splits[:first_batch],
                emb,
                collection_name=collection_name,
                connection_args={"uri": uri_n},
            )
            imported_total += first_batch
            logger.info(f"  进度: {imported_total}/{total_splits} ({imported_total * 100 // total_splits}%)")
            remaining = splits[first_batch:]
        else:
            remaining = splits

        # 分批追加当前源的剩余切片
        if remaining:
            num_batches = (len(remaining) + batch_size - 1) // batch_size
            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, len(remaining))
                batch = remaining[start:end]

                try:
                    vs.add_documents(batch)
                    imported_total += len(batch)
                    pct = imported_total * 100 // total_splits
                    logger.info(f"  [{b_idx + 1}/{num_batches}] {source_name}: 已导入 {imported_total}/{total_splits} ({pct}%)")
                except Exception as e:
                    logger.error(f"  导入失败: {e}")
                    try:
                        vs = MilvusVS(
                            collection_name=collection_name,
                            embedding_function=emb,
                            connection_args={"uri": uri_n},
                            auto_id=True,
                        )
                        vs.add_documents(batch)
                        imported_total += len(batch)
                        logger.info(f"  重试成功: {imported_total}/{total_splits}")
                    except Exception as e2:
                        logger.error(f"  重试也失败，跳过: {e2}")

                if b_idx < num_batches - 1:
                    time.sleep(0.3)

    logger.info(f"\n{'='*50}")
    logger.info(f"全部导入完成! 总计: {imported_total}/{total_splits} 个切片 -> {collection_name}")
    logger.info(f"{'='*50}")


def import_to_knowledge_graph(docs: List[Document], batch_size: int = 10):
    try:
        from src.graph.knowledge_graph import get_knowledge_graph
    except ImportError:
        logger.error("无法导入知识图谱模块，请检查 src/graph/knowledge_graph.py")
        return 0

    kg = get_knowledge_graph()
    if not kg.is_available:
        logger.error("知识图谱不可用 (Neo4j 未连接)")
        return 0

    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        for doc in batch:
            text = doc.page_content
            source = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
            count = kg.extract_and_ingest(text, source=source)
            total += count

        if (i + batch_size) % 50 == 0:
            logger.info(f"  知识图谱进度: {min(i + batch_size, len(docs))}/{len(docs)}")

        time.sleep(0.5)

    logger.info(f"知识图谱导入完成: {total} 个三元组")
    return total


def show_stats(collection_name: str, uri: str):
    uri = _normalize_uri(uri)
    client = MilvusClient(uri=uri)
    if client.has_collection(collection_name):
        stats = client.get_collection_stats(collection_name)
        row_count = stats.get("row_count", 0)
        info = client.describe_collection(collection_name)
        field_names = [f["name"] for f in info["fields"]]
        logger.info(f"Milvus 集合 {collection_name}: {row_count} 条数据, 字段: {field_names}")
    else:
        logger.info(f"Milvus 集合 {collection_name} 不存在")
    client.close()

    try:
        from src.graph.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        kg_stats = kg.get_stats()
        if kg_stats.get("available"):
            logger.info(f"Neo4j 知识图谱: {kg_stats['total_nodes']} 节点, {kg_stats['total_relations']} 关系")
            for et, cnt in kg_stats.get("entity_types", {}).items():
                logger.info(f"  {et}: {cnt}")
        else:
            logger.info("Neo4j 知识图谱: 不可用")
    except Exception as e:
        logger.info(f"Neo4j 知识图谱: 查询失败 ({e})")


def main():
    parser = argparse.ArgumentParser(description="SmartOps 数据导入工具")
    parser.add_argument("action", choices=["import", "append", "crawled", "graph", "stats", "all"], help="操作类型")
    parser.add_argument("--path", type=str, help="文件或目录路径")
    parser.add_argument("--collection", type=str, default=COLLECTION_NAME, help="Milvus 集合名")
    parser.add_argument("--uri", type=str, default=MILVUS_URI, help="Milvus URI")
    parser.add_argument("--drop", action="store_true", help="导入前删除已有集合")
    parser.add_argument("--chunk-size", type=int, default=500, help="切片大小")
    parser.add_argument("--chunk-overlap", type=int, default=60, help="切片重叠")
    parser.add_argument("--max-per-source", type=int, default=0, help="每个源最大切片数(0=不限制)")
    parser.add_argument("--batch-size", type=int, default=200, help="每批导入切片数(默认200)")
    parser.add_argument("--no-graph", action="store_true", help="跳过知识图谱导入(all模式)")

    args = parser.parse_args()

    if args.action == "stats":
        show_stats(args.collection, args.uri)
        return

    if args.action == "crawled":
        import_crawled_data(
            args.collection, args.uri,
            max_per_source=args.max_per_source,
            chunk_size=args.chunk_size,
            drop=args.drop,
            batch_size=args.batch_size,
        )
        return

    if args.action == "graph":
        path = args.path or CRAWLED_DIR
        if not os.path.exists(path):
            logger.error(f"路径不存在: {path}")
            return
        docs = load_directory(path, recursive=True) if os.path.isdir(path) else load_file(path)
        if not docs:
            logger.error("没有加载到文档")
            return
        logger.info(f"共加载 {len(docs)} 个文档，开始导入知识图谱...")
        import_to_knowledge_graph(docs)
        return

    if args.action == "all":
        logger.info("=" * 50)
        logger.info("  全量导入: 爬取数据 -> Milvus + Neo4j")
        logger.info("=" * 50)

        import_crawled_data(
            args.collection, args.uri,
            max_per_source=args.max_per_source,
            chunk_size=args.chunk_size,
            drop=args.drop,
            batch_size=args.batch_size,
        )

        if not args.no_graph:
            logger.info("\n开始导入知识图谱...")
            docs = load_directory(CRAWLED_DIR, recursive=True)
            if docs:
                import_to_knowledge_graph(docs)

        show_stats(args.collection, args.uri)
        return

    path = args.path or DATA_DIR
    if not os.path.exists(path):
        logger.error(f"路径不存在: {path}")
        return

    if os.path.isdir(path):
        docs = load_directory(path, recursive=True)
    else:
        docs = load_file(path)

    if not docs:
        logger.error("没有加载到任何文档")
        return

    logger.info(f"共加载 {len(docs)} 个原始文档")
    splits = split_docs(docs, args.chunk_size, args.chunk_overlap)
    logger.info(f"切片后共 {len(splits)} 个文档块")

    if args.action == "import":
        import_to_milvus(splits, args.collection, args.uri, drop_existing=args.drop, batch_size=args.batch_size)
    elif args.action == "append":
        import_to_milvus(splits, args.collection, args.uri, append_only=True, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
