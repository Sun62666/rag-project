import os
import sys
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

for ef in [".env", "Key.env", "Env.env", "Env1.env"]:
    p = os.path.join(BASE_DIR, ef)
    if os.path.exists(p):
        load_dotenv(p, override=False)
        break


def import_from_directory(dir_path: str, batch_size: int = 5):
    from src.graph.knowledge_graph import OpsKnowledgeGraph
    kg = OpsKnowledgeGraph()
    if not kg.is_available:
        logger.error("知识图谱未连接，请先启动 Neo4j")
        return

    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    all_docs = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(fpath).load()
            elif ext in [".txt", ".md"]:
                docs = TextLoader(fpath, encoding="utf-8").load()
            else:
                continue
            all_docs.extend(docs)
            logger.info(f"加载 {fname}: {len(docs)} 页")
        except Exception as e:
            logger.error(f"加载失败 {fname}: {e}")

    if not all_docs:
        logger.error("没有加载到任何文档")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)
    logger.info(f"共 {len(splits)} 个文档块，开始抽取实体关系...")

    total_triples = 0
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        for j, chunk in enumerate(batch):
            count = kg.extract_and_ingest(chunk.page_content, source=chunk.metadata.get("source", f"chunk_{i+j}"))
            total_triples += count
            logger.info(f"批次 {i//batch_size + 1}, 块 {i+j+1}/{len(splits)}: 抽取 {count} 个三元组")

    stats = kg.get_stats()
    logger.info(f"✅ 导入完成! 总三元组: {total_triples}, 图谱统计: {stats}")
    kg.close()


def import_from_json(json_path: str):
    from src.graph.knowledge_graph import OpsKnowledgeGraph
    kg = OpsKnowledgeGraph()
    if not kg.is_available:
        logger.error("知识图谱未连接，请先启动 Neo4j")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    count = 0
    for t in triples:
        try:
            kg.add_triple(
                t["from_entity"], t["from_type"],
                t["relation"],
                t["to_entity"], t["to_type"],
            )
            count += 1
        except Exception as e:
            logger.warning(f"写入三元组失败: {e}")

    logger.info(f"✅ 从 JSON 导入 {count} 个三元组")
    kg.close()


def import_sample_data():
    from src.graph.knowledge_graph import OpsKnowledgeGraph
    kg = OpsKnowledgeGraph()
    if not kg.is_available:
        logger.error("知识图谱未连接，请先启动 Neo4j")
        return

    sample_triples = [
        ("Redis", "Component", "causes", "OOM", "Fault"),
        ("Redis", "Component", "causes", "连接超时", "Fault"),
        ("maxmemory", "Config", "configures", "Redis", "Component"),
        ("redis-cli info memory", "Command", "checks", "Redis", "Component"),
        ("redis-cli ping", "Command", "checks", "Redis", "Component"),
        ("重启Redis", "Command", "fixes", "OOM", "Fault"),
        ("调整maxmemory", "Config", "fixes", "OOM", "Fault"),
        ("OOM", "Fault", "causes", "服务不可用", "Fault"),
        ("MySQL", "Component", "causes", "连接超时", "Fault"),
        ("wait_timeout", "Config", "configures", "MySQL", "Component"),
        ("max_connections", "Config", "configures", "MySQL", "Component"),
        ("连接数满", "Fault", "causes", "连接超时", "Fault"),
        ("SHOW PROCESSLIST", "Command", "checks", "MySQL", "Component"),
        ("mysql -u root -p", "Command", "checks", "MySQL", "Component"),
        ("Nginx", "Component", "causes", "502", "Fault"),
        ("worker_connections", "Config", "configures", "Nginx", "Component"),
        ("nginx -t", "Command", "checks", "Nginx", "Component"),
        ("systemctl reload nginx", "Command", "restarts", "Nginx", "Component"),
        ("502", "Fault", "causes", "服务不可用", "Fault"),
        ("Docker", "Component", "causes", "磁盘满", "Fault"),
        ("docker system prune", "Command", "fixes", "磁盘满", "Fault"),
        ("docker logs", "Command", "checks", "Docker", "Component"),
        ("Kubernetes", "Component", "depends_on", "etcd", "Component"),
        ("Kubernetes", "Component", "causes", "Pod CrashLoopBackOff", "Fault"),
        ("kubectl describe pod", "Command", "checks", "Kubernetes", "Component"),
        ("kubectl logs", "Command", "checks", "Kubernetes", "Component"),
        ("CPU满载", "Fault", "indicates", "进程异常", "Fault"),
        ("top", "Command", "checks", "CPU满载", "Fault"),
        ("内存泄漏", "Fault", "causes", "OOM", "Fault"),
        ("free -h", "Command", "checks", "内存泄漏", "Fault"),
        ("payment-service", "Service", "depends_on", "Redis", "Component"),
        ("payment-service", "Service", "depends_on", "MySQL", "Component"),
        ("user-service", "Service", "depends_on", "MySQL", "Component"),
        ("user-service", "Service", "depends_on", "Redis", "Component"),
        ("order-service", "Service", "depends_on", "MySQL", "Component"),
        ("order-service", "Service", "depends_on", "Kafka", "Component"),
        ("Redis宕机", "Fault", "causes", "服务不可用", "Fault"),
        ("服务不可用", "Fault", "causes", "订单失败", "Fault"),
        ("Prometheus", "Component", "monitors", "Redis", "Component"),
        ("Prometheus", "Component", "monitors", "MySQL", "Component"),
        ("Prometheus", "Component", "monitors", "Nginx", "Component"),
    ]

    count = 0
    for from_e, from_t, rel, to_e, to_t in sample_triples:
        kg.add_triple(from_e, from_t, rel, to_e, to_t)
        count += 1

    stats = kg.get_stats()
    logger.info(f"✅ 导入 {count} 个示例三元组, 图谱统计: {stats}")
    kg.close()


def main():
    parser = argparse.ArgumentParser(description="SmartOps 知识图谱导入工具")
    parser.add_argument("action", choices=["import", "json", "sample", "stats"], help="操作类型")
    parser.add_argument("--path", type=str, help="文档目录路径 (import) 或 JSON 文件路径 (json)")
    parser.add_argument("--batch-size", type=int, default=5, help="LLM抽取批处理大小")

    args = parser.parse_args()

    if args.action == "sample":
        import_sample_data()
    elif args.action == "stats":
        from src.graph.knowledge_graph import OpsKnowledgeGraph
        kg = OpsKnowledgeGraph()
        print(json.dumps(kg.get_stats(), ensure_ascii=False, indent=2))
        kg.close()
    elif args.action == "json":
        if not args.path:
            print("请指定 --path 为 JSON 文件路径")
            return
        import_from_json(args.path)
    elif args.action == "import":
        if not args.path:
            print("请指定 --path 为文档目录路径")
            return
        import_from_directory(args.path, args.batch_size)


if __name__ == "__main__":
    main()
